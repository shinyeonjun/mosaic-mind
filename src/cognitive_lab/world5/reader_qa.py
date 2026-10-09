"""The reading specialist for stage 5: mdeberta-v3-base trained to find the answer span on KLUE-MRC.

Training data: KLUE train minus our judge-train (3,000) and validation (500) questions, so the
reader's outputs on those are out-of-sample when the MK1 judge learns from them. Unanswerable
questions point at [CLS] (SQuAD 2.0 style). Windows of 384 tokens, stride 128.

Prediction per question (over all windows): best span (start+end logits, at most 30 tokens),
its score, and the no-answer score (start+end logit at [CLS], best over windows). Cached as
JSON lines for train / validation / test.

python -m cognitive_lab.world5.reader_qa train          # about an hour on the laptop GPU
python -m cognitive_lab.world5.reader_qa predict        # a few minutes
"""

import argparse
import json
import math
import random
import time

import torch

from cognitive_lab.world5 import klue

BASE_DIR = klue.PROJECT_ROOT / "model" / "mdeberta-v3-base"
TOKENIZER_DIR = klue.PROJECT_ROOT / "model" / "mDeBERTa-v3-base-mnli-xnli"  # same vocabulary, fast tokenizer
CHECKPOINT = klue.PROJECT_ROOT / "artifacts" / "checkpoints" / "world-v5_reader-qa.pt"
CACHE = klue.PROJECT_ROOT / "artifacts" / "cache" / "world-v5_reader-qa_predictions.jsonl"
MAX_LENGTH, STRIDE, MAX_ANSWER_TOKENS = 384, 128, 30


def reader_training_rows() -> list[dict]:
    held = {q["guid"] for part in ("train", "validation") for q in klue.load(part)}
    rows = []
    for r in klue._read("train"):
        if r["guid"] in held:
            continue
        rows.append({"guid": r["guid"], "context": r["context"], "question": r["question"],
                     "impossible": bool(r["is_impossible"]),
                     "starts": list(r["answers"]["answer_start"]), "texts": list(r["answers"]["text"])})
    return rows


class Windows(dict):
    """Windows of [CLS] question [SEP] context-chunk [SEP], built by hand: the fast tokenizer's
    return_overflowing_tokens stopped after one overflow (a 1,183-token passage gave 2 windows
    and the rest of the passage was never read)."""

    def sequence_ids(self, i: int) -> list:
        return self["sequence"][i]


def encode(tokenizer, questions: list[str], contexts: list[str]) -> Windows:
    out = Windows(input_ids=[], attention_mask=[], offset_mapping=[], sequence=[], overflow_to_sample_mapping=[])
    for n, (question, context) in enumerate(zip(questions, contexts)):
        q_ids = tokenizer(question, add_special_tokens=False)["input_ids"][:64]
        c = tokenizer(context, add_special_tokens=False, return_offsets_mapping=True)
        c_ids, c_off = c["input_ids"], c["offset_mapping"]
        room = MAX_LENGTH - len(q_ids) - 3
        step = room - STRIDE
        start = 0
        while True:
            chunk = range(start, min(start + room, len(c_ids)))
            ids = [tokenizer.cls_token_id] + q_ids + [tokenizer.sep_token_id] + [c_ids[k] for k in chunk] + [tokenizer.sep_token_id]
            offsets = [(0, 0)] * (len(q_ids) + 2) + [tuple(c_off[k]) for k in chunk] + [(0, 0)]
            sequence = [None] + [0] * len(q_ids) + [None] + [1] * len(chunk) + [None]
            pad = MAX_LENGTH - len(ids)
            out["input_ids"].append(ids + [tokenizer.pad_token_id] * pad)
            out["attention_mask"].append([1] * len(ids) + [0] * pad)
            out["offset_mapping"].append(offsets + [(0, 0)] * pad)
            out["sequence"].append(sequence + [None] * pad)
            out["overflow_to_sample_mapping"].append(n)
            if start + room >= len(c_ids):
                break
            start += step
    return out


def training_features(tokenizer, rows: list[dict]) -> dict[str, torch.Tensor]:
    enc = encode(tokenizer, [r["question"] for r in rows], [r["context"] for r in rows])
    starts, ends = [], []
    for i, offsets in enumerate(enc["offset_mapping"]):
        row = rows[enc["overflow_to_sample_mapping"][i]]
        sequence = enc.sequence_ids(i)
        start = end = 0  # [CLS]
        if not row["impossible"]:
            a = row["starts"][0]
            b = a + len(row["texts"][0])
            ctx = [k for k, s in enumerate(sequence) if s == 1]
            if offsets[ctx[0]][0] <= a and offsets[ctx[-1]][1] >= b:
                start = next(k for k in ctx if offsets[k][0] <= a < offsets[k][1] or offsets[k][0] >= a)
                end = next(k for k in reversed(ctx) if offsets[k][0] < b <= offsets[k][1] or offsets[k][1] <= b)
        starts.append(start)
        ends.append(end)
    return {"input_ids": torch.tensor(enc["input_ids"]), "attention_mask": torch.tensor(enc["attention_mask"]),
            "start_positions": torch.tensor(starts), "end_positions": torch.tensor(ends)}


def load_model(device: torch.device, trained: bool, checkpoint=None):
    from transformers import AutoModelForQuestionAnswering

    model = AutoModelForQuestionAnswering.from_pretrained(BASE_DIR, dtype=torch.float32)
    if trained:
        model.load_state_dict(torch.load(checkpoint or CHECKPOINT, map_location="cpu"))
    return model.to(device)


def train(args, device: torch.device) -> None:
    from transformers import AutoTokenizer

    random.seed(0)
    torch.manual_seed(0)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
    rows = reader_training_rows()
    if args.limit:
        rows = rows[:args.limit]
    features = training_features(tokenizer, rows)
    count = len(features["input_ids"])
    print(f"reader training: {len(rows)} questions, {count} windows, "
          f"{int((features['start_positions'] == 0).sum())} pointing at [CLS]", flush=True)
    model = load_model(device, trained=False)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    steps = args.epochs * math.ceil(count / args.batch_size)
    warmup = int(0.1 * steps)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: min(1.0, (s + 1) / max(1, warmup)) * max(0.0, (steps - s) / max(1, steps - warmup)))
    started, step = time.perf_counter(), 0
    model.train()
    for epoch in range(args.epochs):
        order = torch.randperm(count)
        for start in range(0, count, args.batch_size):
            batch = {k: v[order[start:start + args.batch_size]].to(device) for k, v in features.items()}
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                loss = model(**batch).loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            step += 1
            if step % 100 == 0:
                rate = step / (time.perf_counter() - started)
                print(f"epoch {epoch + 1} step {step}/{steps} loss {loss.item():.3f} "
                      f"({(steps - step) / rate / 60:.0f} min left)", flush=True)
    CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), CHECKPOINT)
    print(f"Saved: {CHECKPOINT} ({(time.perf_counter() - started) / 60:.0f} min)")


@torch.no_grad()
def predict(device: torch.device, parts: tuple[str, ...] = ("train", "validation", "test")) -> None:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
    model = load_model(device, trained=True).eval()
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    with CACHE.open("w", encoding="utf-8") as out:
        for part in parts:
            questions = klue.load(part)
            for start in range(0, len(questions), 32):
                chunk = questions[start:start + 32]
                enc = encode(tokenizer, [q["question"] for q in chunk], [q["context"] for q in chunk])
                ids = torch.tensor(enc["input_ids"]).to(device)
                mask = torch.tensor(enc["attention_mask"]).to(device)
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    output = model(input_ids=ids, attention_mask=mask)
                start_logits, end_logits = output.start_logits.float().cpu(), output.end_logits.float().cpu()
                best = {}
                for i in range(len(ids)):
                    q = enc["overflow_to_sample_mapping"][i]
                    sequence, offsets = enc.sequence_ids(i), enc["offset_mapping"][i]
                    ctx = [k for k, s in enumerate(sequence) if s == 1]
                    null = (start_logits[i, 0] + end_logits[i, 0]).item()
                    s_logits, e_logits = start_logits[i, ctx], end_logits[i, ctx]
                    span = s_logits[:, None] + e_logits[None, :]
                    allowed = torch.ones_like(span, dtype=torch.bool).triu() & ~torch.ones_like(span, dtype=torch.bool).triu(MAX_ANSWER_TOKENS)
                    span = span.masked_fill(~allowed, -1e9)
                    flat = int(span.argmax())
                    a, b = ctx[flat // len(ctx)], ctx[flat % len(ctx)]
                    record = best.setdefault(q, {"span_score": -1e9, "null_score": -1e9})
                    record["null_score"] = max(record["null_score"], null)
                    if span.max().item() > record["span_score"]:
                        record["span_score"] = span.max().item()
                        record["answer"] = chunk[q]["context"][offsets[a][0]:offsets[b][1]].strip()
                for q, record in best.items():
                    out.write(json.dumps({"guid": chunk[q]["guid"], "part": part, "answer": record["answer"],
                                          "span_score": round(record["span_score"], 4),
                                          "null_score": round(record["null_score"], 4)}, ensure_ascii=False) + "\n")
            print(f"predicted {part}: {len(questions)}", flush=True)


def load_predictions() -> dict[str, dict]:
    return {r["guid"]: r for r in map(json.loads, CACHE.read_text(encoding="utf-8").splitlines()) if r}


def main() -> None:
    parser = argparse.ArgumentParser(description="Reading specialist (extractive QA) for KLUE-MRC")
    parser.add_argument("command", choices=("train", "predict"))
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--limit", type=int, default=0, help="train on the first N questions (smoke test)")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    device = torch.device("cuda" if args.device != "cpu" and torch.cuda.is_available() else "cpu")
    train(args, device) if args.command == "train" else predict(device)


if __name__ == "__main__":
    main()


class Reader:
    """The trained reading specialist for single passages (rule world v6 uses it on report versions).

    Reads in fp32 by default (MK1): in bf16 an answer could change with what else was in the batch
    (10 of 300 test answers did). `precise=False` is the bf16 reading the v5/v6 caches were made with.
    `precision="fp16"` is the fast mode (about 2.4x on the laptop GPU; see design/mk1-integration.md)."""

    def __init__(self, device: torch.device, precise: bool = True, precision: str | None = None, checkpoint=None):
        from transformers import AutoTokenizer

        self.device = device
        self.precision = precision or ("fp32" if precise else "bf16")
        self.precise = self.precision == "fp32"
        self.half = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(self.precision)
        self.tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
        self.model = load_model(device, trained=True, checkpoint=checkpoint).eval()

    @torch.no_grad()
    def spans(self, question: str, context: str, k: int = 1, max_tokens: int = MAX_ANSWER_TOKENS):
        """Top-k (score, answer text) spans over all windows, and the best no-answer score."""
        enc = encode(self.tokenizer, [question], [context])
        ids = torch.tensor(enc["input_ids"]).to(self.device)
        mask = torch.tensor(enc["attention_mask"]).to(self.device)
        with torch.autocast(device_type=self.device.type, dtype=self.half or torch.bfloat16,
                            enabled=self.device.type == "cuda" and self.half is not None):
            output = self.model(input_ids=ids, attention_mask=mask)
        starts, ends = output.start_logits.float().cpu(), output.end_logits.float().cpu()
        found, null = [], -1e9
        for i in range(len(ids)):
            ctx = [t for t, s in enumerate(enc.sequence_ids(i)) if s == 1]
            offsets = enc["offset_mapping"][i]
            null = max(null, (starts[i, 0] + ends[i, 0]).item())
            span = starts[i, ctx][:, None] + ends[i, ctx][None, :]
            ones = torch.ones_like(span, dtype=torch.bool)
            span = span.masked_fill(~(ones.triu() & ~ones.triu(max_tokens)), -1e9).flatten()
            for f in span.topk(min(k, len(span))).indices.tolist():
                a, b = ctx[f // len(ctx)], ctx[f % len(ctx)]
                found.append((span[f].item(), context[offsets[a][0]:offsets[b][1]].strip()))
        found.sort(reverse=True)
        return found[:k], null

    @torch.no_grad()
    def read_many(self, pairs: list[tuple[str, str]], batch: int = 16) -> list[dict]:
        """`read` for many (question, passage) pairs at once: windows of all passages go through the model
        in batches. Windows are padded to a fixed length, so in fp32 each window's logits do not depend on
        what it was batched with, and the answers equal one-at-a-time reading."""
        enc = encode(self.tokenizer, [q for q, _ in pairs], [c for _, c in pairs])
        ids_all = torch.tensor(enc["input_ids"])
        mask_all = torch.tensor(enc["attention_mask"])
        best = [None] * len(pairs)
        null = [-1e9] * len(pairs)
        for start in range(0, len(ids_all), batch):
            ids = ids_all[start:start + batch].to(self.device)
            mask = mask_all[start:start + batch].to(self.device)
            with torch.autocast(device_type=self.device.type, dtype=self.half or torch.bfloat16,
                                enabled=self.device.type == "cuda" and self.half is not None):
                output = self.model(input_ids=ids, attention_mask=mask)
            starts, ends = output.start_logits.float().cpu(), output.end_logits.float().cpu()
            for j in range(len(ids)):
                i = start + j
                n = enc["overflow_to_sample_mapping"][i]
                ctx = [t for t, s in enumerate(enc.sequence_ids(i)) if s == 1]
                offsets = enc["offset_mapping"][i]
                null[n] = max(null[n], (starts[j, 0] + ends[j, 0]).item())
                span = starts[j, ctx][:, None] + ends[j, ctx][None, :]
                ones = torch.ones_like(span, dtype=torch.bool)
                span = span.masked_fill(~(ones.triu() & ~ones.triu(MAX_ANSWER_TOKENS)), -1e9).flatten()
                f = int(span.argmax())
                a, b = ctx[f // len(ctx)], ctx[f % len(ctx)]
                found = (span[f].item(), pairs[n][1][offsets[a][0]:offsets[b][1]].strip())
                if best[n] is None or found > best[n]:
                    best[n] = found
        return [{"answer": b[1], "span_score": round(b[0], 4), "null_score": round(z, 4)} for b, z in zip(best, null)]

    def read(self, question: str, context: str) -> dict:
        (best,), null = self.spans(question, context, k=1)
        return {"answer": best[1], "span_score": round(best[0], 4), "null_score": round(null, 4)}
