"""P2-nli: a small NLI encoder (mDeBERTa, 279M) as the language module, same solver.

The module plugs into the same common format as P2's LFM reader: for each utterance
it returns `<door>, <key>, 긍정|부정` or `없음`, with a confidence. Only the reader
changes; `parser_llm.parse_episode`, the solver, and the abstention rule are reused.

Reading an utterance = 9 NLI checks, premise = utterance, hypothesis = "<door>은 <key>로
열린다." for every door/key pair:
- an entailed pair (p >= 0.5)        -> positive fact, confidence = p(entailment)
- else a contradicted pair (p >= 0.5) -> negative fact, confidence = p(contradiction)
- else                               -> 없음, confidence = 1 - max(entail, contradict)

Fine-tuning labels follow the world axiom: a positive fact entails its pair and
contradicts the other keys for that door; a negative fact contradicts only its pair;
everything else is neutral. Word embeddings stay frozen (they hold most of the
parameters and keep pretrained representations of names unseen in training).

python -m cognitive_lab.world.nli_parser --split iid --seed 42 --device cuda
"""

import argparse
import json
import random
import statistics
import time
from pathlib import Path

import torch

from cognitive_lab.world.parser_llm import (
    NONE,
    POLARITY,
    choose_threshold,
    judge,
    load_part,
    parse_accuracy,
    parse_episode,
    utterance_examples,
)
from cognitive_lab.world.templates import UNKNOWN

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_NLI_DIR = PROJECT_ROOT / "model" / "mDeBERTa-v3-base-mnli-xnli"
CHECKPOINT_DIR = PROJECT_ROOT / "artifacts" / "checkpoints"
RESULTS_DIR = PROJECT_ROOT / "artifacts" / "results"
ENTAIL, NEUTRAL, CONTRADICT = 0, 1, 2  # mDeBERTa-v3-base-mnli-xnli id2label order
THRESHOLD = 0.5


def nli_path(split: str, seed: int) -> Path:
    return CHECKPOINT_DIR / f"world-v1_p2-nli_{split}_seed-{seed}.pt"


def hypothesis(door: str, key: str) -> str:
    return f"{door}은 {key}로 열린다."


def pair_labels(target: str, doors: list[str], keys: list[str]) -> list[int]:
    """NLI labels for the 9 (door, key) hypotheses implied by a target fact string."""
    labels = [NEUTRAL] * (len(doors) * len(keys))
    if target == NONE:
        return labels
    door, key, polarity = [part.strip() for part in target.split(",")]
    for i, d in enumerate(doors):
        for j, k in enumerate(keys):
            if d != door:
                continue
            if k == key:
                labels[i * len(keys) + j] = ENTAIL if polarity == POLARITY[True] else CONTRADICT
            elif polarity == POLARITY[True]:
                labels[i * len(keys) + j] = CONTRADICT
    return labels


def load_nli(model_dir: Path, device: torch.device, dtype: torch.dtype) -> tuple:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir, dtype=dtype)
    return tokenizer, model.to(device)


class NLIModule:
    """Same `read(text, doors, keys) -> (reply, confidence)` contract as parser_llm.LanguageModule."""

    def __init__(self, tokenizer, model, device: torch.device):
        self.tokenizer, self.model, self.device = tokenizer, model, device
        self.cache: dict[tuple, tuple[str, float]] = {}
        self.calls = 0
        self.cache_hits = 0

    def probabilities(self, text: str, doors: list[str], keys: list[str]) -> torch.Tensor:
        hypotheses = [hypothesis(d, k) for d in doors for k in keys]
        batch = self.tokenizer([text] * len(hypotheses), hypotheses, return_tensors="pt", padding=True).to(self.device)
        return self.model(**batch).logits.float().softmax(dim=-1)

    def read(self, text: str, doors: list[str], keys: list[str]) -> tuple[str, float]:
        cache_key = (text, tuple(doors), tuple(keys))
        if cache_key in self.cache:
            self.cache_hits += 1
            return self.cache[cache_key]
        with torch.inference_mode():
            probs = self.probabilities(text, doors, keys).cpu()
        self.calls += 1
        entail, contradict = probs[:, ENTAIL], probs[:, CONTRADICT]
        best_entail, best_contradict = int(entail.argmax()), int(contradict.argmax())
        if entail[best_entail] >= THRESHOLD:
            index, polarity, confidence = best_entail, True, float(entail[best_entail])
        elif contradict[best_contradict] >= THRESHOLD:
            index, polarity, confidence = best_contradict, False, float(contradict[best_contradict])
        else:
            result = (NONE, 1.0 - float(max(entail.max(), contradict.max())))
            self.cache[cache_key] = result
            return result
        door, key = doors[index // len(keys)], keys[index % len(keys)]
        result = (f"{door}, {key}, {POLARITY[polarity]}", confidence)
        self.cache[cache_key] = result
        return result


class P2NLIAgent:
    def __init__(self, device: str, model_dir: Path, checkpoint: Path | None, abstain: bool):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        tokenizer, model = load_nli(model_dir, self.device, torch.float32)
        self.threshold = None
        self.checkpoint = None
        if checkpoint is not None:
            saved = torch.load(checkpoint, map_location="cpu")
            model.load_state_dict(saved["state"], strict=False)
            self.threshold = saved["abstain_threshold"] if abstain else None
            self.checkpoint = str(checkpoint)
        model.to(self.device, dtype=dtype).eval()
        self.module = NLIModule(tokenizer, model, self.device)
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
            "language_module": "mDeBERTa-v3-base-mnli-xnli (9 NLI checks per utterance)"
            + (" + fine-tuning" if self.checkpoint else ", zero-shot"),
            "judge": "exact solver S1",
            "checkpoint": self.checkpoint,
            "abstain_threshold": self.threshold,
            "abstained_episodes": self.abstained,
            "language_module_calls": self.module.calls,
            "language_module_cache_hits": self.module.cache_hits,
            "peak_vram_gib": round(torch.cuda.max_memory_allocated(self.device) / 2**30, 3)
            if self.device.type == "cuda" else None,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune the NLI language module for P2-nli")
    parser.add_argument("--split", default="iid", choices=("iid", "phrasing", "composition", "entity"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_NLI_DIR)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--validation-episodes", type=int, default=300)
    parser.add_argument("--train-limit", type=int, default=0, help="use only the first N distinct utterances")
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
    if args.train_limit:
        train_examples = train_examples[: args.train_limit]
    validation_episodes = load_part(args.split, args.seed, "validation")[: args.validation_episodes]
    validation_examples = utterance_examples(validation_episodes)
    pairs = [
        (text, hypothesis(d, k), label)
        for text, doors, keys, target in train_examples
        for (d, k), label in zip([(d, k) for d in doors for k in keys], pair_labels(target, doors, keys))
    ]
    print(f"distinct training utterances: {len(train_examples)} -> {len(pairs)} NLI pairs", flush=True)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    tokenizer, model = load_nli(args.model_dir, device, torch.float32)
    model.deberta.embeddings.word_embeddings.weight.requires_grad_(False)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=0.01)

    model.eval()
    history = [{"epoch": 0, "validation_parse_accuracy":
                parse_accuracy(NLIModule(tokenizer, model, device), validation_examples)}]
    print(f"epoch 0 (zero-shot): {history[0]}", flush=True)
    best = (history[0]["validation_parse_accuracy"], 0, None)
    started = time.perf_counter()
    shuffler = random.Random(args.seed)

    def trainable_state() -> dict[str, torch.Tensor]:
        return {name: p.detach().cpu().clone() for name, p in model.named_parameters() if p.requires_grad}

    for epoch in range(1, args.epochs + 1):
        model.train()
        shuffler.shuffle(pairs)
        losses = []
        for start in range(0, len(pairs), args.batch_size):
            chunk = pairs[start : start + args.batch_size]
            batch = tokenizer([c[0] for c in chunk], [c[1] for c in chunk], return_tensors="pt", padding=True).to(device)
            labels = torch.tensor([c[2] for c in chunk], device=device)
            loss = torch.nn.functional.cross_entropy(model(**batch).logits.float(), labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        model.eval()
        accuracy = parse_accuracy(NLIModule(tokenizer, model, device), validation_examples)
        record = {"epoch": epoch, "train_loss": round(statistics.mean(losses), 4), "validation_parse_accuracy": accuracy}
        history.append(record)
        print(f"epoch {epoch}: {record}", flush=True)
        if accuracy > best[0]:
            best = (accuracy, epoch, trainable_state())

    if best[2] is None:
        best = (history[-1]["validation_parse_accuracy"], args.epochs, trainable_state())
    model.load_state_dict(best[2], strict=False)
    model.eval()
    threshold, table = choose_threshold(NLIModule(tokenizer, model, device), validation_episodes)
    print(f"selected epoch {best[1]}, abstain threshold {threshold}", flush=True)

    path = nli_path(args.split, args.seed)
    if args.train_limit:
        path = path.with_name(path.stem + f"_train-limit-{args.train_limit}.pt")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state": best[2], "epoch": best[1], "abstain_threshold": threshold}, path)
    result = {
        "world": "v1",
        "experiment": "p2-nli-language-module-finetune",
        "split": args.split,
        "seed": args.seed,
        "model": args.model_dir.name,
        "distinct_training_utterances": len(train_examples),
        "nli_training_pairs": len(pairs),
        "trainable_parameters": sum(p.numel() for p in trainable),
        "frozen": "word embeddings",
        "optimizer": {"name": "AdamW", "learning_rate": args.learning_rate, "batch_size": args.batch_size},
        "epochs": args.epochs,
        "selected_epoch": best[1],
        "selection_metric": "validation utterance parse exact-match accuracy",
        "history": history,
        "abstain_threshold": threshold,
        "abstain_threshold_table": table[:: max(1, len(table) // 40)],
        "training_seconds": round(time.perf_counter() - started, 1),
        "peak_vram_gib": round(torch.cuda.max_memory_allocated(device) / 2**30, 3) if device.type == "cuda" else None,
        "checkpoint": str(path.relative_to(PROJECT_ROOT)),
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    suffix = f"_train-limit-{args.train_limit}" if args.train_limit else ""
    out = RESULTS_DIR / f"world-v1_p2-nli-train_{args.split}_seed-{args.seed}{suffix}.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved checkpoint: {path}\nSaved: {out}")


if __name__ == "__main__":
    main()
