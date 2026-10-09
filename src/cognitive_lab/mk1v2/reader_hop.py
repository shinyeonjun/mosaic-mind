"""Thinking satellite, part 2: a reader for answers spread over a chain of paragraphs, learned from HotpotQA.

Continues the English reading satellite (mk1v2/reader_en.py) on HotpotQA distractor train: context = the two
supporting paragraphs (in random order) joined, answer = the answer span (yes/no comparison questions and answers not
found verbatim are skipped: MuSiQue has neither). Reporting: HotpotQA validation, the same way. MuSiQue is never used.

python -m cognitive_lab.mk1v2.reader_hop --questions 10000
"""

import argparse
import json
import random
import time

import torch

from cognitive_lab.mk1v2.reader_en import CHECKPOINT as READER_EN
from cognitive_lab.mk1v2.selector import questions, text
from cognitive_lab.world5 import reader_qa
from cognitive_lab.world5.klue import PROJECT_ROOT

CHECKPOINT = PROJECT_ROOT / "artifacts" / "checkpoints" / "mk1v2-reader-hop.pt"


def rows(qs: list[dict], seed: str) -> list[dict]:
    rng = random.Random(seed)
    out = []
    for q in qs:
        if q["answer"].lower() in ("yes", "no"):
            continue
        order = q["support"][:]
        rng.shuffle(order)
        context = "\n".join(text(t, q["paragraphs"][t]) for t in order)
        start = context.find(q["answer"])
        if start < 0:
            continue
        out.append({"guid": q["id"], "context": context, "question": q["question"], "impossible": False,
                    "starts": [start], "texts": [q["answer"]]})
    return out


def evaluate(reader, rs: list[dict]) -> dict:
    reads = reader.read_many([(r["question"], r["context"]) for r in rs])
    right = sum(any(t.lower() in x["answer"].lower() and len(x["answer"]) <= len(t) + 15 for t in r["texts"]) for r, x in zip(rs, reads))
    return {"questions": len(rs), "answer_right": round(right / len(rs), 4)}


def main() -> None:
    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", type=int, default=10000)
    parser.add_argument("--validation", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    train = questions("train")
    random.Random("hotpot-reader-subset").shuffle(train)
    train_rows = rows(train[:args.questions], "train")
    valid = questions("validation")
    random.Random("hotpot-valid").shuffle(valid)
    valid_rows = rows(valid[:args.validation], "valid")
    tokenizer = AutoTokenizer.from_pretrained(reader_qa.TOKENIZER_DIR)
    features = reader_qa.training_features(tokenizer, train_rows)
    count = len(features["input_ids"])
    print(f"train {len(train_rows)} questions, {count} windows; validation {len(valid_rows)}", flush=True)
    history = [{"stage": "English reader, before", **evaluate(reader_qa.Reader(device, checkpoint=READER_EN), valid_rows)}]
    print(history[-1], flush=True)
    torch.cuda.empty_cache()
    model = reader_qa.load_model(device, trained=True, checkpoint=READER_EN)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    steps = (count + args.batch_size - 1) // args.batch_size
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: min(1.0, (s + 1) / max(1, steps // 10)) * max(0.0, (steps - s) / steps))
    started = time.perf_counter()
    model.train()
    order = torch.randperm(count)
    for step, s in enumerate(range(0, count, args.batch_size), 1):
        batch = {k: v[order[s:s + args.batch_size]].to(device) for k, v in features.items()}
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
    torch.save(model.state_dict(), CHECKPOINT)
    del model
    torch.cuda.empty_cache()
    history.append({"stage": "hop reader, after", **evaluate(reader_qa.Reader(device, checkpoint=CHECKPOINT), valid_rows)})
    print(history[-1], flush=True)
    (PROJECT_ROOT / "artifacts" / "results" / "mk1v2-reader-hop-train.json").write_text(
        json.dumps({"questions": len(train_rows), "windows": count, "history": history,
                    "minutes": round((time.perf_counter() - started) / 60, 1)}, indent=2), encoding="utf-8")
    print("Saved:", CHECKPOINT)


if __name__ == "__main__":
    main()
