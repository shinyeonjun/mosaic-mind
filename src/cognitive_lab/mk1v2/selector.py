"""Thinking satellite, part 1: a hop-by-hop paragraph selector learned from HotpotQA (design/mk1-v2-spec.md, 8).

A cross-encoder (starting from the mDeBERTa NLI model) scores one paragraph against what has been read so far:
  hop 1   ("question", paragraph)                       -> is this a supporting paragraph?
  hop 2   ("question || paragraph read at hop 1", other) -> is this the other supporting paragraph?
Training: HotpotQA distractor train (huggingface.co/datasets/hotpotqa/hotpot_qa, CC BY-SA 4.0), a fixed random subset
of bridge and comparison questions; per question both supporting paragraphs (positives) and distractors (negatives)
at each hop. Selection: HotpotQA validation. MuSiQue (the exam) is never used here.

At inference the selector chains: pick the best hop-1 paragraph, then the best hop-2 paragraph given it (and a third
for longer questions while it scores high). The chain's scores are also the "is there a full chain of evidence?"
signal: MuSiQue's unanswerable twins miss a needed paragraph.

python -m cognitive_lab.mk1v2.selector --questions 5000
"""

import argparse
import json
import random
import time

import pyarrow.parquet as pq
import torch

ROOT = __import__("pathlib").Path(__file__).resolve().parents[3]
DATA = ROOT / "data" / "hotpotqa"
BASE = ROOT / "model" / "mDeBERTa-v3-base-mnli-xnli"
CHECKPOINT = ROOT / "artifacts" / "checkpoints" / "mk1v2-selector.pt"
MAX_LENGTH = 384


def questions(part: str) -> list[dict]:
    files = sorted(DATA.glob("train-*.parquet")) if part == "train" else [DATA / "validation-00000-of-00001.parquet"]
    out = []
    for f in files:
        for r in pq.read_table(f).to_pylist():
            paragraphs = {t: " ".join(s) for t, s in zip(r["context"]["title"], r["context"]["sentences"])}
            support = list(dict.fromkeys(r["supporting_facts"]["title"]))
            if len(support) == 2 and all(t in paragraphs for t in support):
                out.append({"id": r["id"], "question": r["question"], "answer": r["answer"], "type": r["type"],
                            "paragraphs": paragraphs, "support": support})
    return out


def text(title: str, body: str) -> str:
    return f"{title}: {body}"


def pairs(q: dict, rng: random.Random, negatives: int = 4) -> list[tuple[str, str, int]]:
    """(first text, paragraph, label) for both hops."""
    titles = list(q["paragraphs"])
    others = [t for t in titles if t not in q["support"]]
    rng.shuffle(others)
    out = []
    for t in q["support"]:
        out.append((q["question"], text(t, q["paragraphs"][t]), 1))
    for t in others[:negatives]:
        out.append((q["question"], text(t, q["paragraphs"][t]), 0))
    for a, b in (q["support"], q["support"][::-1]):
        first = q["question"] + " || " + text(a, q["paragraphs"][a])
        out.append((first, text(b, q["paragraphs"][b]), 1))
        for t in others[:negatives // 2]:
            out.append((first, text(t, q["paragraphs"][t]), 0))
    return out


class Selector:
    def __init__(self, device: torch.device, checkpoint=CHECKPOINT):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(BASE)
        self.model = AutoModelForSequenceClassification.from_pretrained(BASE, num_labels=1, ignore_mismatched_sizes=True)
        if checkpoint is not None:
            self.model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
        self.model.to(device).eval()

    @torch.no_grad()
    def score(self, firsts: list[str], paragraphs: list[str], batch: int = 16) -> list[float]:
        out = []
        for s in range(0, len(firsts), batch):
            x = self.tokenizer(firsts[s:s + batch], paragraphs[s:s + batch], truncation="longest_first", max_length=MAX_LENGTH,
                               padding=True, return_tensors="pt").to(self.device)
            out += torch.sigmoid(self.model(**x).logits.float().squeeze(-1)).tolist()
        return out

    def chain(self, question: str, docs: list[str], hops: int = 3, keep: float = 0.5) -> list[tuple[int, float]]:
        """Greedy chain of (doc index, score): hop 1 always, then further hops while the best score passes `keep`."""
        scores = self.score([question] * len(docs), docs)
        first = max(range(len(docs)), key=lambda i: scores[i])
        picked = [(first, scores[first])]
        context = question + " || " + docs[first]
        for _ in range(hops - 1):
            rest = [i for i in range(len(docs)) if i not in {p for p, _ in picked}]
            if not rest:
                break
            s = self.score([context] * len(rest), [docs[i] for i in rest])
            j = max(range(len(rest)), key=lambda k: s[k])
            if picked and len(picked) >= 2 and s[j] < keep:
                break
            picked.append((rest[j], s[j]))
            context += " || " + docs[rest[j]]
        return picked


def validate(selector: Selector, qs: list[dict]) -> dict:
    """Both supporting paragraphs in the first two picks (out of 10)."""
    both = 0
    for q in qs:
        titles = list(q["paragraphs"])
        docs = [text(t, q["paragraphs"][t]) for t in titles]
        picked = {titles[i] for i, _ in selector.chain(q["question"], docs, hops=2)}
        both += set(q["support"]) <= picked
    return {"questions": len(qs), "both_supporting_found": round(both / len(qs), 4)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", type=int, default=5000)
    parser.add_argument("--validation", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(0)
    torch.manual_seed(0)
    train = questions("train")
    random.Random("hotpot-subset").shuffle(train)
    train = train[:args.questions]
    valid = questions("validation")
    random.Random("hotpot-valid").shuffle(valid)
    valid = valid[:args.validation]
    rng = random.Random(0)
    examples = [p for q in train for p in pairs(q, rng)]
    rng.shuffle(examples)
    print(f"train {len(train)} questions, {len(examples)} pairs ({sum(e[2] for e in examples)} positive)", flush=True)
    selector = Selector(device, checkpoint=None)
    history = [{"stage": "before (NLI model, untrained head)", **validate(selector, valid)}]
    print(history[-1], flush=True)
    model = selector.model
    model.train()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    steps = (len(examples) + args.batch_size - 1) // args.batch_size
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: min(1.0, (s + 1) / max(1, steps // 10)) * max(0.0, (steps - s) / steps))
    started = time.perf_counter()
    for step, s in enumerate(range(0, len(examples), args.batch_size), 1):
        chunk = examples[s:s + args.batch_size]
        x = selector.tokenizer([a for a, _, _ in chunk], [b for _, b, _ in chunk], truncation="longest_first", max_length=MAX_LENGTH,
                               padding=True, return_tensors="pt").to(device)
        y = torch.tensor([float(c) for _, _, c in chunk], device=device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model(**x).logits.squeeze(-1)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits.float(), y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        if step % 200 == 0:
            rate = step / (time.perf_counter() - started)
            print(f"step {step}/{steps} loss {loss.item():.3f} ({(steps - step) / rate / 60:.0f} min left)", flush=True)
    model.eval()
    CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), CHECKPOINT)
    history.append({"stage": "after", **validate(selector, valid)})
    print(history[-1], flush=True)
    (ROOT / "artifacts" / "results" / "mk1v2-selector-train.json").write_text(
        json.dumps({"questions": len(train), "pairs": len(examples), "history": history,
                    "minutes": round((time.perf_counter() - started) / 60, 1)}, indent=2), encoding="utf-8")
    print("Saved:", CHECKPOINT)


if __name__ == "__main__":
    main()
