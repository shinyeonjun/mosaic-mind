"""Satellite A: an English reading satellite that also says when a passage has no answer.

Starts from the KLUE-trained reading specialist (world5/reader_qa.py, kept unchanged in its own checkpoint) and
continues on SQuAD 2.0 (rajpurkar.github.io, about a third of the questions have no answer in the passage), so it
learns both the answer span and the no-answer score in English. Training: a fixed random subset of the SQuAD 2.0
training questions, 1 epoch. Selection and reporting: SQuAD 2.0 dev. RGB is never touched here.

python -m cognitive_lab.mk1v2.reader_en --train-questions 30000
"""

import argparse
import json
import math
import random
import time

import torch

from cognitive_lab.world5 import reader_qa
from cognitive_lab.world5.klue import PROJECT_ROOT

SQUAD = PROJECT_ROOT / "data" / "squad2"
CHECKPOINT = PROJECT_ROOT / "artifacts" / "checkpoints" / "mk1v2-reader-en.pt"


def rows(part: str) -> list[dict]:
    data = json.loads((SQUAD / f"{part}-v2.0.json").read_text(encoding="utf-8"))["data"]
    out = []
    for article in data:
        for p in article["paragraphs"]:
            for qa in p["qas"]:
                answers = qa["answers"] if not qa["is_impossible"] else []
                out.append({"guid": qa["id"], "context": p["context"], "question": qa["question"], "impossible": qa["is_impossible"],
                            "starts": [a["answer_start"] for a in answers], "texts": [a["text"] for a in answers]})
    return out


def evaluate(reader: reader_qa.Reader, questions: list[dict]) -> dict:
    """SQuAD 2.0 style: answer when span score > no-answer score, else no answer; exact match and no-answer accuracy."""
    import re

    def norm(t):
        return re.sub(r"\b(a|an|the)\b", " ", re.sub(r"[^\w\s]", " ", t.lower())).split()

    reads = reader.read_many([(q["question"], q["context"]) for q in questions])
    exact = no_answer_right = 0
    for q, r in zip(questions, reads):
        abstain = r["null_score"] >= r["span_score"]
        if q["impossible"]:
            exact += abstain
            no_answer_right += abstain
        else:
            no_answer_right += not abstain
            exact += (not abstain) and any(norm(r["answer"]) == norm(t) for t in q["texts"])
    return {"questions": len(questions), "exact_match": round(100 * exact / len(questions), 2),
            "has_answer_vs_no_answer_accuracy": round(100 * no_answer_right / len(questions), 2)}


def main() -> None:
    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser(description="Train the English reading satellite on SQuAD 2.0")
    parser.add_argument("--train-questions", type=int, default=30000)
    parser.add_argument("--dev-questions", type=int, default=2000)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(0)
    torch.manual_seed(0)
    train = rows("train")
    random.Random("squad2-subset").shuffle(train)
    train = train[:args.train_questions]
    dev = rows("dev")
    random.Random("squad2-dev").shuffle(dev)
    dev = dev[:args.dev_questions]
    tokenizer = AutoTokenizer.from_pretrained(reader_qa.TOKENIZER_DIR)
    features = reader_qa.training_features(tokenizer, train)
    count = len(features["input_ids"])
    print(f"train {len(train)} questions ({sum(r['impossible'] for r in train)} without answer), {count} windows", flush=True)
    before = reader_qa.Reader(device)  # the KLUE-trained specialist, as it is
    history = [{"stage": "KLUE reader, before", **evaluate(before, dev)}]
    print(history[-1], flush=True)
    del before
    torch.cuda.empty_cache()
    model = reader_qa.load_model(device, trained=True)  # continue from the KLUE checkpoint
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    steps = math.ceil(count / args.batch_size)
    warmup = int(0.1 * steps)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: min(1.0, (s + 1) / max(1, warmup)) * max(0.0, (steps - s) / max(1, steps - warmup)))
    started = time.perf_counter()
    model.train()
    order = torch.randperm(count)
    for step, start in enumerate(range(0, count, args.batch_size), 1):
        batch = {k: v[order[start:start + args.batch_size]].to(device) for k, v in features.items()}
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            loss = model(**batch).loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        if step % 200 == 0:
            rate = step / (time.perf_counter() - started)
            print(f"step {step}/{steps} loss {loss.item():.3f} ({(steps - step) / rate / 60:.0f} min left)", flush=True)
    CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), CHECKPOINT)
    del model
    torch.cuda.empty_cache()
    after = reader_qa.Reader(device, checkpoint=CHECKPOINT)
    history.append({"stage": "English reader, after", **evaluate(after, dev)})
    print(history[-1], flush=True)
    summary = {"train_questions": len(train), "windows": count, "history": history, "minutes": round((time.perf_counter() - started) / 60, 1)}
    (PROJECT_ROOT / "artifacts" / "results" / "mk1v2-reader-en-train.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("Saved:", CHECKPOINT)


if __name__ == "__main__":
    main()
