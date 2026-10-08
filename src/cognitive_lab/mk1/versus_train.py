"""J2: the language model trained (LoRA) on the same training worlds MK1's parts learned from (design/mk1-versus.md).

Same prompts as `versus.py`. Training problems come only from training parts (v3 train, v6 train sessions, v10
train sessions); targets are what can be known from the given information: the key (or 모름) for chains; the true
answer if some source's article contains it, else 모름, for articles; for doors the true key when the board states
it or some article contains the answer, else 모름. Loss on the answer tokens only. The adapter with the best
validation score (validation parts, never the test samples) is kept.

python -m cognitive_lab.mk1.versus_train
"""

import argparse
import random
import statistics
import time

import torch

from cognitive_lab.mk1 import versus
from cognitive_lab.world2.integrated import CHECKPOINT_DIR, RESULTS_DIR
from cognitive_lab.world3 import chains as v3

TRAIN = {"chains": 2000, "articles": 25, "doors": 25}  # chains: problems; others: sessions of 40
VALIDATION = {"chains": 150, "articles": 4, "doors": 4}


def build(part: str, sizes: dict, seed: int) -> list[tuple[str, list[dict], str, list[str] | None, dict]]:
    """(family, messages, target, options, episode) for every problem of a part."""
    from cognitive_lab.mk1.articles import article_sessions
    from cognitive_lab.mk1.system import chain_sessions
    from cognitive_lab.world6 import world as v6
    from cognitive_lab.world10 import world as v10

    out = []
    for s in chain_sessions(v3.generate(part, 45)[:sizes["chains"]]):
        e = s["episodes"][0]
        options = versus.options_for(e, e["query"] + part + str(len(out)))
        target = e["answer"] if e["answer"] in e["keys"] else versus.UNKNOWN
        out.append(("chains", versus.chain_prompt(e, options), target, options, e))
    for s in article_sessions(v6.generate(part, seed, sizes["articles"]), part):
        history = []
        for e in s["episodes"]:
            known = any(versus.contains(r["passage"], e["answers"]) for r in e["reports"])
            out.append(("articles", versus.article_prompt(e, history), e["answers"][0] if known else versus.UNKNOWN, None, e))
            history.append((e["question"], e["reports"], e["answers"]))
    for s in v10.generate(part, seed, sizes["doors"]):
        history = []
        for e in s["episodes"]:
            a = e["article"]
            options = versus.options_for(e, a["question"])
            known = e["tag"] == "C" or any(versus.contains(r["passage"], a["answers"]) for r in a["reports"])
            out.append(("doors", versus.door_prompt(e, history, options), e["answer"] if known else versus.UNKNOWN, options, e))
            history.append((a["question"], a["reports"], a["answers"]))
    return out


def validate(model: versus.LanguageModel, problems: list) -> float:
    from cognitive_lab.mk1.system import episode_score

    scores = []
    for family, messages, _, options, e in problems:
        answer = model.generate(messages) if family == "articles" else model.choose(messages, options)
        scores.append(episode_score(e, answer))
    return statistics.mean(scores)


def main() -> None:
    from cognitive_lab.world.lora import apply_lora, lora_state_dict

    parser = argparse.ArgumentParser(description="LoRA-train the language model on MK1's training worlds")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=16.0)
    parser.add_argument("--accumulation", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    started = time.perf_counter()
    train, validation = build("train", TRAIN, args.seed), build("validation", VALIDATION, 0)
    print(f"train {len(train)} problems, validation {len(validation)} ({time.perf_counter() - started:.0f}s)", flush=True)
    model = versus.LanguageModel(device)
    tokenizer, net = model.tokenizer, model.model
    trainable = apply_lora(net, args.rank, args.alpha, 0.05)
    net.to(device)
    net.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})  # 2-5k-token prompts fit in 8GB
    net.config.use_cache = False
    optimizer = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=args.learning_rate, weight_decay=0.0)
    encoded = []
    for _, messages, target, _, _ in train:
        prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        encoded.append((tokenizer(prompt, add_special_tokens=False)["input_ids"],
                        tokenizer(target + "<|im_end|>", add_special_tokens=False)["input_ids"]))
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
            if step % 250 == 0:
                print(f"epoch {epoch} step {step}/{len(order)} loss {statistics.mean(losses[-250:]):.4f} "
                      f"({time.perf_counter() - started:.0f}s)", flush=True)
        net.eval()
        value = round(validate(model, validation), 4)
        history.append({"epoch": epoch, "train_loss": round(statistics.mean(losses), 4), "validation": value})
        print(history[-1], flush=True)
        if value > best[0]:
            best = (value, epoch, lora_state_dict(net))
    path = CHECKPOINT_DIR / "mk1-versus-lora.pt"
    state = best[2] if best[2] is not None else lora_state_dict(net)
    torch.save({"lora": {"rank": args.rank, "alpha": args.alpha}, "state": state, "epoch": best[1]}, path)
    summary = {"trainable_parameters": trainable, "train_problems": len(train), "train_tokens": tokens, "history": history,
               "selected_epoch": best[1], "seconds": round(time.perf_counter() - started, 1),
               "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2) if device.type == "cuda" else None}
    (RESULTS_DIR / "mk1-versus-lora-train.json").write_text(__import__("json").dumps(summary, indent=2), encoding="utf-8")
    print(summary, f"\nSaved: {path}")


if __name__ == "__main__":
    main()
