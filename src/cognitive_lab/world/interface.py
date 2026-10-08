"""Can modules learn their own common format? (design: design/interface-learning-v1.md)

Three systems on rule world v1, all with the same small neural judge:

- P2-N  (designed format): each utterance becomes the designed one-hot fact
        [door slot(3), key slot(3), polarity(2), none(1)]. The v1 language modules read
        every test split with 100% accuracy, so gold structure stands in for their output.
- E2E-C (learned, continuous): a mDeBERTa encoder maps each utterance to a 16-d message.
- E2E-D (learned, discrete): the encoder emits K=3 slots x V=8 symbols through a
        straight-through Gumbel-softmax, so the message is a short symbol string.

E2E systems are trained from final answers only; gold events are never used. Speakers
are passed as metadata (one-hot source), as in P2. The judge is a 2-layer transformer
over [message, source, is_question] tokens with position embeddings; it predicts one of
the 3 listed keys or 모름 (4 classes) from the question token.

Engineering notes: v1 has few distinct utterances, so each batch encodes its distinct
texts once and scatters the messages back to their positions (exactly equivalent to
encoding every occurrence, far cheaper). On an 8 GB GPU the first version filled VRAM and
spilled into shared memory (0% utilization), so the encoder runs under bf16 autocast with
gradient checkpointing and a 64-token limit.

`--curriculum` (development track D2) orders the *training* lessons from easy to hard;
validation and test always use full episodes, and messages never see gold structure:
  stage 1: only utterances about the queried door, simple cases only   (learn: key in message)
  stage 2: only utterances about the queried door, all cases            (order, negation)
  stage 3: plus 1-4 random distractor utterances                        (learn: door in message)
  stage 4: full episodes
Gold events are used only to choose which utterances a lesson shows (like a teacher
picking easy examples), not as message targets.

The question token sits at position 0 and utterances follow at positions 1..n. The first
curriculum runs put the question last: short lessons trained only positions 1-4, so the
full episodes of stage 4 moved the question to untrained positions and training collapsed.

python -m cognitive_lab.world.interface --system e2e-discrete --split composition --seed 42
"""

import argparse
import json
import random
import statistics
import time
from pathlib import Path

import torch
from torch import nn

from cognitive_lab.world.templates import SOURCES, UNKNOWN

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "artifacts" / "world-v1" / "data"
CHECKPOINT_DIR = PROJECT_ROOT / "artifacts" / "checkpoints"
RESULTS_DIR = PROJECT_ROOT / "artifacts" / "results"
NLI_DIR = PROJECT_ROOT / "model" / "mDeBERTa-v3-base-mnli-xnli"
# Encoders for the NLI-knowledge control (research report, 2026-10-07): "nli" was fine-tuned on
# MNLI+XNLI; "base" is the same mDeBERTa-v3 before any NLI training (identical spm.model);
# "e5" is multilingual-e5-small (a bi-encoder with weak English NLI exposure via SimCSE data).
ENCODER_DIRS = {
    "nli": NLI_DIR,
    "base": PROJECT_ROOT / "model" / "mdeberta-v3-base",
    "e5": PROJECT_ROOT / "model" / "multilingual-e5-small",
}


def tokenizer_dir(encoder_dir: Path) -> Path:
    """mdeberta-v3-base ships only spm.model; its vocabulary is byte-identical to the NLI
    checkpoint's, so reuse that checkpoint's fast tokenizer instead of converting."""
    return encoder_dir if (encoder_dir / "tokenizer.json").exists() else NLI_DIR
SYSTEMS = ("p2-neural", "e2e-continuous", "e2e-discrete")
DESIGNED_SIZE = 9
MAX_TOKENS = 11  # up to 10 utterances + the question


def load_part(split: str, seed: int, part: str) -> list[dict]:
    with (DATA_DIR / split / f"seed-{seed}" / f"{part}.jsonl").open(encoding="utf-8") as file:
        return [json.loads(line) for line in file]


def label_of(episode: dict) -> int:
    return 3 if episode["answer"] == UNKNOWN else episode["entities"]["keys"].index(episode["answer"])


def utterance_text(episode: dict, text: str, question: bool = False) -> str:
    doors, keys = ", ".join(episode["entities"]["doors"]), ", ".join(episode["entities"]["keys"])
    return f"문: {doors}\n열쇠: {keys}\n{'질문' if question else '문장'}: {text}"


def designed_message(episode: dict, event: dict) -> list[float]:
    vector = [0.0] * DESIGNED_SIZE
    if event["type"] == "chatter":
        vector[8] = 1.0
        return vector
    doors = [d.removesuffix(" 문") for d in episode["entities"]["doors"]]
    keys = [k.removesuffix(" 열쇠") for k in episode["entities"]["keys"]]
    vector[doors.index(event["door"])] = 1.0
    vector[3 + keys.index(event["key"])] = 1.0
    vector[6 if event["polarity"] else 7] = 1.0
    return vector


def designed_question(episode: dict) -> list[float]:
    vector = [0.0] * DESIGNED_SIZE
    doors = [d.removesuffix(" 문") for d in episode["entities"]["doors"]]
    vector[doors.index(episode["gold_question"]["door"])] = 1.0
    return vector


class Judge(nn.Module):
    def __init__(self, message_size: int, width: int = 64, layers: int = 2, heads: int = 4):
        super().__init__()
        self.embed = nn.Linear(message_size + len(SOURCES) + 1, width)
        self.position = nn.Embedding(MAX_TOKENS, width)
        layer = nn.TransformerEncoderLayer(width, heads, dim_feedforward=4 * width, dropout=0.1, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers)
        self.out = nn.Linear(width, 4)

    def forward(self, messages: torch.Tensor, sources: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """messages [B,T,M] (question at position 0), sources [B,T,3], mask [B,T] True = real token."""
        batch, tokens, _ = messages.shape
        is_question = torch.zeros(batch, tokens, 1, device=messages.device)
        is_question[:, 0, 0] = 1.0
        x = self.embed(torch.cat([messages, sources, is_question], -1))
        x = x + self.position(torch.arange(tokens, device=messages.device)).unsqueeze(0)
        h = self.encoder(x, src_key_padding_mask=~mask)
        return self.out(h[:, 0])


class MessageEncoder(nn.Module):
    """mDeBERTa (word embeddings frozen) -> mean-pooled state -> message."""

    def __init__(self, discrete: bool, size: int = 16, slots: int = 3, symbols: int = 8,
                 encoder_dir: Path = NLI_DIR):
        super().__init__()
        from transformers import AutoModel, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir(encoder_dir))
        self.backbone = AutoModel.from_pretrained(encoder_dir, dtype=torch.float32)
        self.backbone.embeddings.word_embeddings.weight.requires_grad_(False)
        self.backbone.gradient_checkpointing_enable()
        self.discrete, self.slots, self.symbols = discrete, slots, symbols
        hidden = self.backbone.config.hidden_size
        self.head = nn.Linear(hidden, slots * symbols if discrete else size)
        self.message_size = slots * symbols if discrete else size

    def forward(self, texts: list[str]) -> torch.Tensor:
        batch = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=64)
        batch = batch.to(self.head.weight.device)
        with torch.autocast(device_type=self.head.weight.device.type, dtype=torch.bfloat16,
                            enabled=self.head.weight.device.type == "cuda"):
            states = self.backbone(**batch).last_hidden_state
        states = states.float()
        mask = batch["attention_mask"].unsqueeze(-1).float()
        pooled = (states * mask).sum(1) / mask.sum(1)
        logits = self.head(pooled)
        if not self.discrete:
            return torch.tanh(logits)
        logits = logits.view(-1, self.slots, self.symbols)
        if self.training:
            codes = nn.functional.gumbel_softmax(logits, tau=1.0, hard=True)
        else:
            codes = nn.functional.one_hot(logits.argmax(-1), self.symbols).float()
        return codes.reshape(len(texts), -1)

    def symbols_of(self, texts: list[str]) -> list[tuple[int, ...]]:
        with torch.no_grad():
            codes = self.forward(texts).view(len(texts), self.slots, self.symbols).argmax(-1)
        return [tuple(row.tolist()) for row in codes]


def source_onehots(episode: dict) -> list[list[float]]:
    return [[0.0] * len(SOURCES)] + [[1.0 if u["source"] == s else 0.0 for s in SOURCES] for u in episode["utterances"]]


def batch_inputs(episodes: list[dict], encoder: MessageEncoder | None, device: torch.device):
    """Pad episodes into judge tensors; encoder=None uses designed messages from gold events."""
    tokens = max(len(e["utterances"]) + 1 for e in episodes)
    if encoder is None:
        size = DESIGNED_SIZE
        rows = [[designed_question(e)] + [designed_message(e, ev) for ev in e["gold_events"]] for e in episodes]
        messages = torch.zeros(len(episodes), tokens, size)
        for b, row in enumerate(rows):
            messages[b, : len(row)] = torch.tensor(row)
        messages = messages.to(device)
    else:
        texts = [[utterance_text(e, e["question"], True)] + [utterance_text(e, u["text"]) for u in e["utterances"]]
                 for e in episodes]
        distinct = sorted({t for row in texts for t in row})
        position = {t: i for i, t in enumerate(distinct)}
        encoded = encoder(distinct)  # [U, M]
        index = torch.zeros(len(episodes), tokens, dtype=torch.long, device=device)
        for b, row in enumerate(texts):
            index[b, : len(row)] = torch.tensor([position[t] for t in row], device=device)
        messages = encoded[index]  # padding positions read message 0 and are masked out
    sources = torch.zeros(len(episodes), tokens, len(SOURCES), device=device)
    mask = torch.zeros(len(episodes), tokens, dtype=torch.bool, device=device)
    for b, e in enumerate(episodes):
        sources[b, : len(e["utterances"]) + 1] = torch.tensor(source_onehots(e), device=device)
        mask[b, : len(e["utterances"]) + 1] = True
    labels = torch.tensor([label_of(e) for e in episodes], device=device)
    return messages, sources, mask, labels


CURRICULUM_STAGES = (1, 2, 3, 4)


def lesson(episode: dict, stage: int, rng: random.Random) -> dict | None:
    """A training view of `episode` for a curriculum stage (None = not used in this stage)."""
    if stage >= 4:
        return episode
    if stage == 1 and episode["split_keys"]["case"] != "simple":
        return None
    door = episode["gold_question"]["door"]
    about = [i for i, ev in enumerate(episode["gold_events"]) if ev["type"] != "chatter" and ev["door"] == door]
    keep = set(about)
    if stage == 3:
        others = [i for i in range(len(episode["utterances"])) if i not in keep]
        keep |= set(rng.sample(others, min(len(others), rng.randint(1, 4))))
    order = sorted(keep)
    view = dict(episode)
    view["utterances"] = [episode["utterances"][i] for i in order]
    view["gold_events"] = [episode["gold_events"][i] for i in order]
    return view


def calibrated(predictions: list[int], labels: list[int]) -> float:
    """+1 correct, 0 for 모름 when an answer exists, -1 otherwise (v1 scoring)."""
    scores = [1 if p == y else (0 if p == 3 else -1) for p, y in zip(predictions, labels)]
    return round(statistics.mean(scores), 4)


def evaluate(judge: Judge, encoder, episodes: list[dict], device, batch_size: int = 64) -> dict:
    judge.eval()
    if encoder is not None:
        encoder.eval()
    predictions, labels, cases = [], [], []
    with torch.no_grad():
        for start in range(0, len(episodes), batch_size):
            chunk = episodes[start : start + batch_size]
            messages, sources, mask, y = batch_inputs(chunk, encoder, device)
            predictions += judge(messages, sources, mask).argmax(-1).tolist()
            labels += y.tolist()
            cases += [e["split_keys"]["case"] for e in chunk]
    by_case = {}
    for case in sorted(set(cases)):
        idx = [i for i, c in enumerate(cases) if c == case]
        by_case[case] = calibrated([predictions[i] for i in idx], [labels[i] for i in idx])
    return {"mean_score": calibrated(predictions, labels),
            "accuracy": round(sum(p == y for p, y in zip(predictions, labels)) / len(labels), 4),
            "by_case": by_case}


def probe(encoder: MessageEncoder, train: list[dict], test: list[dict], device) -> dict:
    """Linear probes: can door slot / key slot / polarity / none be read off the messages?"""
    def table(episodes):
        seen = {}
        for e in episodes:
            for u, ev in zip(e["utterances"], e["gold_events"]):
                seen.setdefault(utterance_text(e, u["text"]), designed_message(e, ev))
        return list(seen), torch.tensor(list(seen.values()))

    encoder.eval()
    results = {}
    with torch.no_grad():
        texts_tr, y_tr = table(train)
        texts_te, y_te = table(test)
        x_tr = torch.cat([encoder(texts_tr[i : i + 64]) for i in range(0, len(texts_tr), 64)]).cpu()
        x_te = torch.cat([encoder(texts_te[i : i + 64]) for i in range(0, len(texts_te), 64)]).cpu()
    targets = {"none": (8, 9), "polarity": (6, 8), "key": (3, 6), "door": (0, 3)}
    for name, (lo, hi) in targets.items():
        fact_tr = y_tr[:, 8] == 0 if name != "none" else torch.ones(len(y_tr), dtype=torch.bool)
        fact_te = y_te[:, 8] == 0 if name != "none" else torch.ones(len(y_te), dtype=torch.bool)
        if name == "none":
            labels_tr, labels_te = y_tr[:, 8].long(), y_te[:, 8].long()
        else:
            labels_tr, labels_te = y_tr[fact_tr, lo:hi].argmax(-1), y_te[fact_te, lo:hi].argmax(-1)
        classifier = nn.Linear(x_tr.shape[1], int(labels_tr.max()) + 1)
        optimizer = torch.optim.Adam(classifier.parameters(), lr=0.05)
        for _ in range(300):
            optimizer.zero_grad()
            nn.functional.cross_entropy(classifier(x_tr[fact_tr]), labels_tr).backward()
            optimizer.step()
        with torch.no_grad():
            accuracy = (classifier(x_te[fact_te]).argmax(-1) == labels_te).float().mean().item()
        results[name] = round(accuracy, 4)
    results["distinct_test_utterances"] = len(texts_te)
    if encoder.discrete:
        codes = encoder.symbols_of(texts_te[:64])
        results["symbol_examples"] = [{"text": t.split("문장: ")[-1], "symbols": c} for t, c in zip(texts_te[:12], codes[:12])]
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Learned vs designed module interfaces on rule world v1")
    parser.add_argument("--system", choices=SYSTEMS, required=True)
    parser.add_argument("--split", default="composition", choices=("iid", "phrasing", "composition", "entity"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--epochs", type=int, default=0, help="default: 30 for p2-neural, 6 for e2e")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--judge-lr", type=float, default=1e-3)
    parser.add_argument("--encoder-lr", type=float, default=2e-5)
    parser.add_argument("--encoder", choices=sorted(ENCODER_DIRS), default="nli")
    parser.add_argument("--curriculum", action="store_true",
                        help="train easy-to-hard lessons (stage 1: 5 epochs, stages 2-3: 2 each, then full episodes)")
    args = parser.parse_args()

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else
                          ("cpu" if args.device == "auto" else args.device))
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    train = load_part(args.split, args.seed, "train")
    validation = load_part(args.split, args.seed, "validation")
    test = load_part(args.split, args.seed, "test")

    encoder = None if args.system == "p2-neural" else MessageEncoder(
        discrete=args.system == "e2e-discrete", encoder_dir=ENCODER_DIRS[args.encoder]).to(device)
    judge = Judge(DESIGNED_SIZE if encoder is None else encoder.message_size).to(device)
    groups = [{"params": judge.parameters(), "lr": args.judge_lr}]
    if encoder is not None:
        # The message head is new, so it learns at the judge's rate; only the pretrained
        # backbone uses the small encoder rate. (The first runs put the head at the
        # encoder rate, 50x slower, and the messages never picked up door/key information.)
        groups.append({"params": encoder.head.parameters(), "lr": args.judge_lr})
        groups.append({"params": [p for p in encoder.backbone.parameters() if p.requires_grad], "lr": args.encoder_lr})
    optimizer = torch.optim.AdamW(groups, weight_decay=0.01)
    epochs = args.epochs or (30 if encoder is None else 6)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started, history, best = time.perf_counter(), [], (-2.0, 0, None)
    stage_epochs = {1: 5, 2: 2, 3: 2}  # stage 1 has few lessons; remaining epochs use full episodes

    def stage_of(epoch: int) -> int:
        if not args.curriculum:
            return 4
        boundary = 0
        for stage in (1, 2, 3):
            boundary += stage_epochs[stage]
            if epoch <= boundary:
                return stage
        return 4

    for epoch in range(1, epochs + 1):
        stage = stage_of(epoch)
        lesson_rng = random.Random(args.seed * 1000 + epoch)
        lessons = [view for e in train if (view := lesson(e, stage, lesson_rng)) is not None]
        order = list(range(len(lessons)))
        judge.train()
        if encoder is not None:
            encoder.train()
        random.Random(args.seed + epoch).shuffle(order)
        losses = []
        epoch_started = time.perf_counter()
        for step, start in enumerate(range(0, len(order), args.batch_size), 1):
            chunk = [lessons[i] for i in order[start : start + args.batch_size]]
            messages, sources, mask, labels = batch_inputs(chunk, encoder, device)
            loss = nn.functional.cross_entropy(judge(messages, sources, mask), labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            if encoder is not None and step % 25 == 0:
                print(f"  epoch {epoch} step {step} loss {statistics.mean(losses[-25:]):.4f} "
                      f"({time.perf_counter() - epoch_started:.0f}s)", flush=True)
        metrics = evaluate(judge, encoder, validation, device)
        history.append({"epoch": epoch, "stage": stage, "lessons": len(lessons),
                        "train_loss": round(statistics.mean(losses), 4), "validation": metrics["mean_score"]})
        print(history[-1], flush=True)
        if metrics["mean_score"] > best[0]:
            state = {"judge": {k: v.detach().cpu().clone() for k, v in judge.state_dict().items()}}
            if encoder is not None:
                state["encoder"] = {k: v.detach().cpu().clone() for k, v in encoder.state_dict().items()}
            best = (metrics["mean_score"], epoch, state)

    judge.load_state_dict(best[2]["judge"])
    if encoder is not None:
        encoder.load_state_dict(best[2]["encoder"])
    test_metrics = evaluate(judge, encoder, test, device)
    result = {
        "world": "v1", "experiment": "interface-learning", "system": args.system, "split": args.split,
        "seed": args.seed, "epochs": epochs, "curriculum": args.curriculum, "encoder": args.encoder,
        "selected_epoch": best[1],
        "selection_metric": "validation calibrated score (validation holds out the split's axis)",
        "trained_on_gold_events": encoder is None, "history": history, "test": test_metrics,
        "probe": probe(encoder, train, test, device) if encoder is not None else None,
        "training_seconds": round(time.perf_counter() - started, 1),
        "peak_vram_gib": round(torch.cuda.max_memory_allocated(device) / 2**30, 3) if device.type == "cuda" else None,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    tag = ("-curriculum" if args.curriculum else "") + ("" if args.encoder == "nli" else f"-enc-{args.encoder}")
    out = RESULTS_DIR / f"world-v1_interface-{args.system}{tag}_{args.split}_seed-{args.seed}.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"test {test_metrics}\nprobe {result['probe']}\nSaved: {out}")


if __name__ == "__main__":
    main()
