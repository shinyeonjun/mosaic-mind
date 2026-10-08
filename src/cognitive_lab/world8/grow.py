"""Growing the next-key branch on a frozen thinker, and checking it.

python -m cognitive_lab.world8.grow --seed 45      # offline check: can the branch be learned?
"""

import argparse
import json
import random

import torch
from torch import nn

from cognitive_lab.world2.integrated import CHECKPOINT_DIR, RESULTS_DIR
from cognitive_lab.world3.thinker import episode_tensors, evaluate, step_loss
from cognitive_lab.world8.branch import BranchedThinker
from cognitive_lab.world8.chains import FAMILIES, generate
from cognitive_lab.world8.language import next_table, thinker_table

TRAIN_STEPS, TEST_STEPS = 8, 16


def load_branched(seed: int, device: torch.device) -> tuple[BranchedThinker, dict]:
    table = thinker_table(device)
    model = BranchedThinker(table["features"].to(device), next_table(device, table["sentences"]).to(device),
                            reader_head=table["head"]).to(device)
    missing, unexpected = model.load_state_dict(
        torch.load(CHECKPOINT_DIR / f"world-v3_thinker-grown_seed-{seed}.pt")["state"], strict=False)
    branch = {n for n, _ in model.named_parameters() if n.startswith(("next_head", "pair_score", "relation_score"))}
    if set(missing) != branch or unexpected:
        raise RuntimeError(f"unexpected load: missing {missing}, unexpected {unexpected}")
    for name, p in model.named_parameters():
        p.requires_grad_(name in branch)
    return model.eval(), table


def train_branch(model: BranchedThinker, data: tuple, validation: tuple, seed: int, epochs: int = 30,
                 learning_rate: float = 2e-3, batch_size: int = 32) -> int:
    """Train only the branch from (episodes, answers); keep the epoch with the best validation score."""
    torch.manual_seed(seed)
    optimizer = torch.optim.Adam(model.branch_parameters(), lr=learning_rate)
    index, query, target, _ = data
    best = (-9.0, 0, {k: v.detach().clone() for k, v in model.state_dict().items()})
    model.eval()
    with torch.no_grad():
        best = (evaluate(model, validation, TEST_STEPS)["adaptive"], 0, best[2])
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(query), device=query.device)
        for start in range(0, len(order), batch_size):
            batch = order[start:start + batch_size]
            logits, _ = model.think(index[batch], query[batch], TRAIN_STEPS)
            loss = step_loss(logits, target[batch]).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.branch_parameters(), 1.0)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            value = evaluate(model, validation, TEST_STEPS)["adaptive"]
        if value > best[0]:
            best = (value, epoch, {k: v.detach().clone() for k, v in model.state_dict().items()})
    model.load_state_dict(best[2])
    model.eval()
    return best[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline check: grow the next-key branch")
    parser.add_argument("--seed", type=int, default=45, help="stage-3 thinker seed")
    parser.add_argument("--episodes", type=int, default=1000)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(args.seed)
    model, table = load_branched(args.seed, device)
    def tensors(episodes):
        return tuple(t.to(device) for t in episode_tensors(episodes, table))
    half = args.episodes // 2
    train = generate("next", "train", args.seed, half) + generate("same", "train", args.seed, half)
    validation = generate("next", "validation", args.seed, 200) + generate("same", "validation", args.seed, 200)
    before = {f: evaluate(model, tensors(generate(f, "test", args.seed, 600)), TEST_STEPS)["adaptive"] for f in FAMILIES}
    epoch = train_branch(model, tensors(train), tensors(validation), args.seed)
    after = {f: evaluate(model, tensors(generate(f, "test", args.seed, 600)), TEST_STEPS)["adaptive"] for f in FAMILIES}
    model.branch_enabled = False
    disabled = {f: evaluate(model, tensors(generate(f, "test", args.seed, 600)), TEST_STEPS)["adaptive"] for f in FAMILIES}
    result = {"seed": args.seed, "train_episodes": args.episodes, "selected_epoch": epoch,
              "branch_parameters": sum(p.numel() for p in model.branch_parameters()),
              "untrained_branch": before, "grown_branch": after, "branch_switched_off": disabled}
    print(json.dumps(result, ensure_ascii=False))
    (RESULTS_DIR / f"world-v8_branch-offline_seed-{args.seed}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2),
                                                                              encoding="utf-8")


if __name__ == "__main__":
    main()
