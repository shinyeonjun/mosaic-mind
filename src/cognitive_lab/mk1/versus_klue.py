"""J2 (reading): LoRA-train the language model on exactly the reading specialist's training data (design/mk1-versus.md).

The reading specialist (world5/reader_qa.py) learned from the KLUE-MRC rows of `reader_training_rows()` (14,054
questions). The language model gets the same rows, in the KorQuAD exam's prompt format, and learns to write the
answer (모름 for questions without one). Model selection uses our KLUE validation questions (held out from both);
KorQuAD is never touched here. Then: python -m cognitive_lab.mk1.korquad --system llm --adapter <saved adapter>

python -m cognitive_lab.mk1.versus_klue
"""

import argparse
import json
import random
import statistics
import time

import torch

from cognitive_lab.mk1.korquad import PROMPT, char_f1
from cognitive_lab.mk1.versus import UNKNOWN, LanguageModel
from cognitive_lab.world2.integrated import CHECKPOINT_DIR, RESULTS_DIR
from cognitive_lab.world5 import klue

MAX_PROMPT_TOKENS = 1536  # longer passages are skipped in training (the reading specialist reads windows instead)


def messages(context: str, question: str) -> list[dict]:
    return [{"role": "system", "content": PROMPT}, {"role": "user", "content": f"글:\n{context}\n\n질문: {question}"}]


def validate(model: LanguageModel, questions: list[dict]) -> float:
    """Mean character F1 on answerable questions, with 모름 right only for unanswerable ones (as a 0/1 score)."""
    scores = []
    for q in questions:
        answer = model.generate(messages(q["context"], q["question"]))
        if q["impossible"]:
            scores.append(float(answer == UNKNOWN))
        else:
            scores.append(0.0 if answer == UNKNOWN else char_f1(answer, q["answers"]))
    return statistics.mean(scores)


def main() -> None:
    from cognitive_lab.world.lora import apply_lora, lora_state_dict
    from cognitive_lab.world5.reader_qa import reader_training_rows

    parser = argparse.ArgumentParser(description="LoRA-train the language model on the reading specialist's KLUE data")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=16.0)
    parser.add_argument("--accumulation", type=int, default=8)
    parser.add_argument("--validation", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    started = time.perf_counter()
    model = LanguageModel(device)
    tokenizer, net = model.tokenizer, model.model
    rows = reader_training_rows()
    encoded, skipped = [], 0
    for r in rows:
        prompt = tokenizer.apply_chat_template(messages(r["context"], r["question"]), add_generation_prompt=True, tokenize=False)
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        if len(prompt_ids) > MAX_PROMPT_TOKENS:
            skipped += 1
            continue
        target = UNKNOWN if r["impossible"] else r["texts"][0]
        encoded.append((prompt_ids, tokenizer(target + "<|im_end|>", add_special_tokens=False)["input_ids"]))
    validation = klue.load("validation")[:args.validation]
    print(f"train {len(encoded)} (skipped {skipped} long), validation {len(validation)} ({time.perf_counter() - started:.0f}s)", flush=True)
    trainable = apply_lora(net, args.rank, args.alpha, 0.05)
    net.to(device)
    net.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})  # fits in 8GB with room to spare
    net.config.use_cache = False
    optimizer = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=args.learning_rate, weight_decay=0.0)
    net.eval()
    history = [{"epoch": 0, "validation": round(validate(model, validation), 4)}]
    print(history[-1], flush=True)
    best = (history[0]["validation"], 0, None)
    order = list(range(len(encoded)))
    shuffler = random.Random(args.seed)
    tokens = 0
    for epoch in range(1, args.epochs + 1):
        net.train()
        shuffler.shuffle(order)
        losses = []
        optimizer.zero_grad(set_to_none=True)
        for step, i in enumerate(order, 1):
            prompt_ids, target_ids = encoded[i]
            ids = torch.tensor([prompt_ids + target_ids], device=device)
            logits = net(input_ids=ids, logits_to_keep=len(target_ids) + 1).logits[0, :-1]
            loss = torch.nn.functional.cross_entropy(logits.float(), torch.tensor(target_ids, device=device))
            (loss / args.accumulation).backward()
            losses.append(loss.item())
            tokens += ids.shape[1]
            if step % args.accumulation == 0 or step == len(order):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            if step % 500 == 0:
                print(f"epoch {epoch} step {step}/{len(order)} loss {statistics.mean(losses[-500:]):.4f} "
                      f"({time.perf_counter() - started:.0f}s)", flush=True)
        net.eval()
        value = round(validate(model, validation), 4)
        history.append({"epoch": epoch, "train_loss": round(statistics.mean(losses), 4), "validation": value})
        print(history[-1], flush=True)
        if value > best[0]:
            best = (value, epoch, lora_state_dict(net))
    path = CHECKPOINT_DIR / "mk1-klue-lora.pt"
    torch.save({"lora": {"rank": args.rank, "alpha": args.alpha}, "state": best[2] if best[2] is not None else lora_state_dict(net),
                "epoch": best[1]}, path)
    summary = {"trainable_parameters": trainable, "train_questions": len(encoded), "skipped_long": skipped, "train_tokens": tokens,
               "history": history, "selected_epoch": best[1], "seconds": round(time.perf_counter() - started, 1),
               "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2) if device.type == "cuda" else None}
    (RESULTS_DIR / "mk1-klue-lora-train.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(summary, f"\nSaved: {path}")


if __name__ == "__main__":
    main()
