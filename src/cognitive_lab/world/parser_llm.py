"""P2: a learned language module (LFM + LoRA) that only reads, and the exact solver judges.

Interface between modules (the "common format"): for each utterance the language
module outputs either `<door>, <key>, 긍정|부정` or `없음`, plus a confidence (the
probability of its own greedy output). The speaker is metadata and is passed through
without language processing. The question's door is found by matching the given door
names. The solver (S1) then judges from these events, exactly as with gold events.

Training uses the same split, seed, base model, and LoRA settings as L1-LoRA; only the
target changes from "final answer" to "per-utterance fact". Training examples are the
distinct (entity names, utterance) pairs in the training part, labeled from gold events.

Connection policy (`p2-lora-abstain`): after training, a confidence threshold is chosen
on validation episodes; if any utterance in an episode was parsed below it, the
pipeline answers 모름 instead of risking a wrong answer.

python -m cognitive_lab.world.parser_llm --split iid --seed 42 --device cuda
"""

import argparse
import json
import random
import statistics
import time
from pathlib import Path

import torch

from cognitive_lab.world.llm_agent import DEFAULT_MODEL_DIR, END_OF_TURN, load_model
from cognitive_lab.world.lora import apply_lora, load_lora, lora_state_dict
from cognitive_lab.world.scoreboard import score_answer
from cognitive_lab.world.solver import solve
from cognitive_lab.world.templates import UNKNOWN

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "artifacts" / "world-v1" / "data"
CHECKPOINT_DIR = PROJECT_ROOT / "artifacts" / "checkpoints"
RESULTS_DIR = PROJECT_ROOT / "artifacts" / "results"
NONE = "없음"
POLARITY = {True: "긍정", False: "부정"}

SYSTEM_PROMPT = (
    "너는 문장에서 사실만 뽑는 해석기다.\n"
    "문장이 어떤 문이 어떤 열쇠로 열린다(또는 안 열린다)고 말하면 '문 이름, 열쇠 이름, 긍정' "
    "또는 '문 이름, 열쇠 이름, 부정' 형식으로 한 줄만 쓴다.\n"
    "앞말을 정정하는 문장이면 정정한 내용을 쓴다.\n"
    "문과 열쇠에 대한 사실이 없으면 '없음'이라고만 쓴다."
)


def parser_path(split: str, seed: int) -> Path:
    return CHECKPOINT_DIR / f"world-v1_p2-parser_{split}_seed-{seed}.pt"


def load_part(split: str, seed: int, part: str) -> list[dict]:
    path = DATA_DIR / split / f"seed-{seed}" / f"{part}.jsonl"
    if not path.exists():
        raise SystemExit(f"missing {path}; run `python -m cognitive_lab.world generate --seed {seed}` first")
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file]


def messages_for(text: str, doors: list[str], keys: list[str]) -> list[dict]:
    user = f"문: {', '.join(doors)}\n열쇠: {', '.join(keys)}\n문장: {text}"
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


def target_for(event: dict) -> str:
    if event["type"] == "chatter":
        return NONE
    return f"{event['door']} 문, {event['key']} 열쇠, {POLARITY[event['polarity']]}"


def read_output(text: str, doors: list[str], keys: list[str]) -> tuple[str, str, bool] | None:
    """Turn the module's reply into (door label, key label, polarity); None for 없음/invalid."""
    parts = [part.strip() for part in text.strip().split(",")]
    if len(parts) != 3 or parts[0] not in doors or parts[1] not in keys or parts[2] not in POLARITY.values():
        return None
    return parts[0].removesuffix(" 문"), parts[1].removesuffix(" 열쇠"), parts[2] == "긍정"


def utterance_examples(episodes: list[dict]) -> list[tuple[str, list[str], list[str], str]]:
    """Distinct (text, doors, keys) -> target pairs from gold events."""
    seen: dict[tuple, str] = {}
    for episode in episodes:
        doors, keys = episode["entities"]["doors"], episode["entities"]["keys"]
        for utterance, event in zip(episode["utterances"], episode["gold_events"]):
            seen.setdefault((utterance["text"], tuple(doors), tuple(keys)), target_for(event))
    return [(text, list(doors), list(keys), target) for (text, doors, keys), target in seen.items()]


class LanguageModule:
    """Greedy structured reader with per-output confidence and a cache of seen inputs."""

    def __init__(self, tokenizer, model, device: torch.device):
        self.tokenizer, self.model, self.device = tokenizer, model, device
        self.cache: dict[tuple, tuple[str, float]] = {}
        self.calls = 0
        self.cache_hits = 0
        self.invalid = 0

    def read(self, text: str, doors: list[str], keys: list[str]) -> tuple[str, float]:
        key = (text, tuple(doors), tuple(keys))
        if key in self.cache:
            self.cache_hits += 1
            return self.cache[key]
        encoded = self.tokenizer.apply_chat_template(
            messages_for(text, doors, keys), add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt",
        ).to(self.device)
        with torch.inference_mode():
            output = self.model.generate(
                **encoded, do_sample=False, max_new_tokens=24, pad_token_id=self.tokenizer.pad_token_id,
                output_scores=True, return_dict_in_generate=True,
            )
            log_probs = self.model.compute_transition_scores(
                output.sequences, output.scores, normalize_logits=True
            )
        self.calls += 1
        new_tokens = output.sequences[0, encoded["input_ids"].shape[1]:]
        confidence = float(torch.exp(log_probs[0].float().sum()).item())
        reply = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        self.cache[key] = (reply, confidence)
        return reply, confidence


def parse_episode(module: LanguageModule, episode: dict) -> tuple[list[dict], float, int]:
    """Return solver events, the lowest utterance confidence, and the invalid-reply count."""
    doors, keys = episode["entities"]["doors"], episode["entities"]["keys"]
    events, lowest, invalid = [], 1.0, 0
    for step, utterance in enumerate(episode["utterances"]):
        reply, confidence = module.read(utterance["text"], doors, keys)
        lowest = min(lowest, confidence)
        fact = read_output(reply, doors, keys)
        if fact is None:
            invalid += reply != NONE
            events.append({"step": step, "type": "chatter", "source": utterance["source"]})
            continue
        door, key, polarity = fact
        events.append({"step": step, "type": "assert" if polarity else "negate", "door": door,
                       "key": key, "polarity": polarity, "source": utterance["source"]})
    return events, lowest, invalid


def judge(events: list[dict], episode: dict) -> str:
    asked = [door for door in episode["entities"]["doors"] if door in episode["question"]]
    if len(asked) != 1:
        return UNKNOWN
    keys = [key.removesuffix(" 열쇠") for key in episode["entities"]["keys"]]
    answer = solve(events, asked[0].removesuffix(" 문"), keys)
    return UNKNOWN if answer == UNKNOWN else f"{answer} 열쇠"


class P2Agent:
    def __init__(self, device: str, model_dir: Path, adapter: Path, abstain: bool):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        tokenizer, model = load_model(model_dir, self.device)
        checkpoint = torch.load(adapter, map_location="cpu")
        apply_lora(model, checkpoint["lora"]["rank"], checkpoint["lora"]["alpha"], 0.0)
        load_lora(model, checkpoint["state"])
        model.to(self.device).eval()
        self.module = LanguageModule(tokenizer, model, self.device)
        self.threshold = checkpoint["abstain_threshold"] if abstain else None
        self.adapter = str(adapter)
        self.abstained = 0
        self.invalid = 0

    def __call__(self, episode: dict) -> str:
        events, lowest, invalid = parse_episode(self.module, episode)
        self.invalid += invalid
        if self.threshold is not None and lowest < self.threshold:
            self.abstained += 1
            return UNKNOWN
        return judge(events, episode)

    def cost_report(self) -> dict:
        return {
            "language_module": "LFM2.5-1.2B-Instruct + LoRA (utterance -> fact)",
            "judge": "exact solver S1",
            "adapter": self.adapter,
            "abstain_threshold": self.threshold,
            "abstained_episodes": self.abstained,
            "invalid_parser_replies": self.invalid,
            "language_module_calls": self.module.calls,
            "language_module_cache_hits": self.module.cache_hits,
            "peak_vram_gib": round(torch.cuda.max_memory_allocated(self.device) / 2**30, 3)
            if self.device.type == "cuda" else None,
        }


def parse_accuracy(module: LanguageModule, examples: list[tuple]) -> float:
    correct = sum(module.read(text, doors, keys)[0] == target for text, doors, keys, target in examples)
    return round(correct / len(examples), 4)


def choose_threshold(module: LanguageModule, episodes: list[dict]) -> tuple[float, list[dict]]:
    """Pick the abstention threshold that maximizes validation mean calibrated score."""
    rows = []
    for episode in episodes:
        events, lowest, _ = parse_episode(module, episode)
        rows.append((lowest, score_answer(judge(events, episode), episode["answer"]),
                     score_answer(UNKNOWN, episode["answer"])))
    candidates = sorted({0.0} | {round(lowest, 4) + 1e-6 for lowest, _, _ in rows})
    table = []
    for threshold in candidates:
        scores = [abstain if lowest < threshold else answered for lowest, answered, abstain in rows]
        table.append({"threshold": threshold, "mean_score": round(statistics.mean(scores), 4)})
    best = max(table, key=lambda row: (row["mean_score"], -row["threshold"]))
    return best["threshold"], table


def main() -> None:
    parser = argparse.ArgumentParser(description="Train P2's language module (utterance -> fact)")
    parser.add_argument("--split", default="iid", choices=("iid", "phrasing", "composition", "entity"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=16.0)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--accumulation", type=int, default=4)
    parser.add_argument("--validation-episodes", type=int, default=300)
    args = parser.parse_args()

    device_name = args.device
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested, but PyTorch cannot access a CUDA device")
    device = torch.device(device_name)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_examples = utterance_examples(load_part(args.split, args.seed, "train"))
    validation_episodes = load_part(args.split, args.seed, "validation")[: args.validation_episodes]
    validation_examples = utterance_examples(validation_episodes)
    print(f"distinct training utterances: {len(train_examples)}, validation: {len(validation_examples)}", flush=True)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    tokenizer, model = load_model(args.model_dir, device)
    trainable = apply_lora(model, args.rank, args.alpha, args.dropout)
    model.to(device)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.learning_rate, weight_decay=0.0
    )

    def encode(text: str, doors: list[str], keys: list[str], target: str) -> tuple[list[int], list[int]]:
        prompt = tokenizer.apply_chat_template(messages_for(text, doors, keys), add_generation_prompt=True, tokenize=False)
        return (tokenizer(prompt, add_special_tokens=False)["input_ids"],
                tokenizer(target + END_OF_TURN, add_special_tokens=False)["input_ids"])

    encoded = [encode(*example) for example in train_examples]

    model.eval()
    history = [{"epoch": 0, "validation_parse_accuracy":
                parse_accuracy(LanguageModule(tokenizer, model, device), validation_examples)}]
    print(f"epoch 0: {history[0]}", flush=True)
    best = (history[0]["validation_parse_accuracy"], 0, None)
    started = time.perf_counter()
    order = list(range(len(encoded)))
    shuffler = random.Random(args.seed)

    for epoch in range(1, args.epochs + 1):
        model.train()
        shuffler.shuffle(order)
        losses = []
        optimizer.zero_grad(set_to_none=True)
        for step, index in enumerate(order, 1):
            prompt_ids, target_ids = encoded[index]
            input_ids = torch.tensor([prompt_ids + target_ids], device=device)
            logits = model(input_ids=input_ids, logits_to_keep=len(target_ids) + 1).logits[0, :-1]
            loss = torch.nn.functional.cross_entropy(logits.float(), torch.tensor(target_ids, device=device))
            (loss / args.accumulation).backward()
            losses.append(loss.item())
            if step % args.accumulation == 0 or step == len(order):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        model.eval()
        accuracy = parse_accuracy(LanguageModule(tokenizer, model, device), validation_examples)
        record = {"epoch": epoch, "train_loss": round(statistics.mean(losses), 4),
                  "validation_parse_accuracy": accuracy}
        history.append(record)
        print(f"epoch {epoch}: {record}", flush=True)
        if accuracy > best[0]:
            best = (accuracy, epoch, lora_state_dict(model))

    if best[2] is None:
        best = (history[-1]["validation_parse_accuracy"], args.epochs, lora_state_dict(model))
    load_lora(model, best[2])
    model.eval()
    module = LanguageModule(tokenizer, model, device)
    threshold, threshold_table = choose_threshold(module, validation_episodes)
    print(f"selected epoch {best[1]}, abstain threshold {threshold}", flush=True)

    path = parser_path(args.split, args.seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    settings = {"rank": args.rank, "alpha": args.alpha, "dropout": args.dropout, "targets": "default"}
    torch.save({"lora": settings, "state": best[2], "epoch": best[1], "abstain_threshold": threshold}, path)
    result = {
        "world": "v1",
        "experiment": "p2-language-module-finetune",
        "split": args.split,
        "seed": args.seed,
        "model": args.model_dir.name,
        "distinct_training_utterances": len(train_examples),
        "distinct_validation_utterances": len(validation_examples),
        "lora": settings,
        "trainable_parameters": trainable,
        "optimizer": {"name": "AdamW", "learning_rate": args.learning_rate, "accumulation": args.accumulation},
        "epochs": args.epochs,
        "selected_epoch": best[1],
        "selection_metric": "validation utterance parse exact-match accuracy",
        "history": history,
        "abstain_threshold": threshold,
        "abstain_threshold_table": threshold_table[:: max(1, len(threshold_table) // 40)],
        "training_seconds": round(time.perf_counter() - started, 1),
        "peak_vram_gib": round(torch.cuda.max_memory_allocated(device) / 2**30, 3) if device.type == "cuda" else None,
        "adapter": str(path.relative_to(PROJECT_ROOT)),
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"world-v1_p2-parser-train_{args.split}_seed-{args.seed}.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved adapter: {path}\nSaved: {out}")


if __name__ == "__main__":
    main()
