"""Fine-tune L1 (LFM2.5-1.2B) with LoRA on a split's training episodes.

The model sees the same zero-shot prompt as `l1-choice`/`l1-generate` and learns to
emit the answer; loss covers only the answer tokens. After each epoch the adapter is
scored on validation episodes (option log-probability, calibrated score), and the best
epoch's adapter is kept. Validation never touches the test part.

python -m cognitive_lab.world.finetune --split iid --seed 42 --device cuda
"""

import argparse
import json
import random
import statistics
import time
from pathlib import Path

import torch

from cognitive_lab.world.llm_agent import (
    DEFAULT_MODEL_DIR,
    END_OF_TURN,
    LLMAgent,
    build_messages,
    load_model,
)
from cognitive_lab.world.lora import apply_lora, lora_state_dict
from cognitive_lab.world.scoreboard import score_answer

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "artifacts" / "world-v1" / "data"
CHECKPOINT_DIR = PROJECT_ROOT / "artifacts" / "checkpoints"
RESULTS_DIR = PROJECT_ROOT / "artifacts" / "results"


def adapter_path(split: str, seed: int) -> Path:
    return CHECKPOINT_DIR / f"world-v1_l1-lora_{split}_seed-{seed}.pt"


def load_part(split: str, seed: int, part: str) -> list[dict]:
    path = DATA_DIR / split / f"seed-{seed}" / f"{part}.jsonl"
    if not path.exists():
        raise SystemExit(f"missing {path}; run `python -m cognitive_lab.world generate --seed {seed}` first")
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file]


def encode(tokenizer, episode: dict) -> tuple[list[int], list[int]]:
    messages, _ = build_messages(episode)
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    target_ids = tokenizer(episode["answer"] + END_OF_TURN, add_special_tokens=False)["input_ids"]
    return prompt_ids, target_ids


def validation_score(tokenizer, model, device: torch.device, episodes: list[dict]) -> dict:
    agent = LLMAgent("choice", str(device.type), loaded=(tokenizer, model))
    scores = [score_answer(agent(episode), episode["answer"]) for episode in episodes]
    return {
        "episodes": len(episodes),
        "mean_score": round(statistics.mean(scores), 4),
        "accuracy": round(sum(s == 1 for s in scores) / len(scores), 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="LoRA fine-tuning of L1 on rule world v1")
    parser.add_argument("--split", default="iid", choices=("iid", "phrasing", "composition", "entity"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=16.0)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--accumulation", type=int, default=8, help="episodes per optimizer step")
    parser.add_argument("--train-limit", type=int, default=0, help="use only the first N training episodes")
    parser.add_argument("--validation-limit", type=int, default=300, help="validation episodes per epoch")
    args = parser.parse_args()

    device_name = args.device
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested, but PyTorch cannot access a CUDA device")
    device = torch.device(device_name)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    train = load_part(args.split, args.seed, "train")
    if args.train_limit:
        train = train[: args.train_limit]
    validation = load_part(args.split, args.seed, "validation")[: args.validation_limit]

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    tokenizer, model = load_model(args.model_dir, device)
    trainable = apply_lora(model, args.rank, args.alpha, args.dropout)
    model.to(device)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.learning_rate, weight_decay=0.0
    )
    encoded = [encode(tokenizer, episode) for episode in train]

    model.eval()
    history = [{"epoch": 0, "validation": validation_score(tokenizer, model, device, validation)}]
    print(f"epoch 0 (base model) validation: {history[0]['validation']}", flush=True)
    best_score, best_epoch, best_state = history[0]["validation"]["mean_score"], 0, None
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
            loss = torch.nn.functional.cross_entropy(
                logits.float(), torch.tensor(target_ids, device=device)
            )
            (loss / args.accumulation).backward()
            losses.append(loss.item())
            if step % args.accumulation == 0 or step == len(order):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            if step % 500 == 0:
                print(f"epoch {epoch} step {step}/{len(order)} loss {statistics.mean(losses[-500:]):.4f}", flush=True)

        model.eval()
        metrics = validation_score(tokenizer, model, device, validation)
        record = {"epoch": epoch, "train_loss": round(statistics.mean(losses), 4), "validation": metrics}
        history.append(record)
        print(f"epoch {epoch}: {record}", flush=True)
        if metrics["mean_score"] > best_score:
            best_score, best_epoch, best_state = metrics["mean_score"], epoch, lora_state_dict(model)

    elapsed = round(time.perf_counter() - started, 1)
    if best_state is None:
        print("No epoch beat the base model on validation; saving the last epoch's adapter anyway.")
        best_state, best_epoch = lora_state_dict(model), args.epochs
    path = adapter_path(args.split, args.seed)
    if args.train_limit:
        path = path.with_name(path.stem + f"_train-limit-{args.train_limit}.pt")
    path.parent.mkdir(parents=True, exist_ok=True)
    settings = {"rank": args.rank, "alpha": args.alpha, "dropout": args.dropout, "targets": "default"}
    torch.save({"lora": settings, "state": best_state, "epoch": best_epoch}, path)

    result = {
        "world": "v1",
        "experiment": "l1-lora-finetune",
        "split": args.split,
        "seed": args.seed,
        "model": args.model_dir.name,
        "train_episodes": len(train),
        "validation_episodes_per_epoch": len(validation),
        "lora": settings,
        "trainable_parameters": trainable,
        "optimizer": {"name": "AdamW", "learning_rate": args.learning_rate, "accumulation": args.accumulation},
        "epochs": args.epochs,
        "selected_epoch": best_epoch,
        "selection_metric": "validation mean calibrated score (option log-probability)",
        "history": history,
        "training_seconds": elapsed,
        "peak_vram_gib": round(torch.cuda.max_memory_allocated(device) / 2**30, 3) if device.type == "cuda" else None,
        "adapter": str(path.relative_to(PROJECT_ROOT)),
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    suffix = f"_train-limit-{args.train_limit}" if args.train_limit else ""
    out = RESULTS_DIR / f"world-v1_l1-lora-train_{args.split}_seed-{args.seed}{suffix}.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved adapter: {path}\nSaved: {out}")


if __name__ == "__main__":
    main()
