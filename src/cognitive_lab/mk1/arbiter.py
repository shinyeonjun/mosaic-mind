"""MK1 stage F: an arbiter that learns which channel to believe (design/mk1-integration.md, F).

Channels are parts, not people: the board (through the thinker's first pass) and the article part (through
the real-text trust part). For the article door, each channel says a key or nothing. Like the stage-1
trust part, the arbiter keeps a memory per channel, updated after every question from whether that
channel was right, and weighs what each channel says by its memory. Weights are shared over keys and
channels (no channel names), so it can only learn "trust whoever has been right this session".
Output: logits over the 3 keys and 모름; answer a key when its probability passes 0.5.

python -m cognitive_lab.mk1.arbiter --seed 42      # train (article readings from the v6 cache)
"""

import argparse
import json
import time

import torch
from torch import nn

from cognitive_lab.world2.integrated import CHECKPOINT_DIR, RESULTS_DIR

BOARD_ACCURACIES = (1.0, 0.75, 0.5)


class Arbiter(nn.Module):
    def __init__(self, memory: int = 16, width: int = 32):
        super().__init__()
        self.memory_initial = nn.Parameter(torch.zeros(memory))
        self.memory_cell = nn.GRUCell(3, memory)
        self.contribution = nn.Sequential(nn.Linear(1 + memory, width), nn.Tanh(), nn.Linear(width, 1))
        self.none = nn.Sequential(nn.Linear(1 + memory, width), nn.Tanh(), nn.Linear(width, 1))

    def initial_memory(self, batch: int, channels: int = 2) -> torch.Tensor:
        return self.memory_initial.expand(batch, channels, -1)

    def step(self, said: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        """said [B,C,3] one-hot (zeros = nothing), memory [B,C,M] -> logits [B,4] (3 keys, 모름)."""
        spoke = said.amax(-1)  # [B,C]
        mem = memory[:, :, None, :].expand(-1, -1, said.shape[-1], -1)
        contribution = self.contribution(torch.cat([said[..., None], mem], -1)).squeeze(-1) * spoke[..., None]
        none = self.none(torch.cat([spoke[..., None], memory], -1)).squeeze(-1).mean(1, keepdim=True)
        return torch.cat([contribution.sum(1), none], -1)

    def update(self, memory: torch.Tensor, said: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
        """After the answer is revealed: each channel remembers whether it spoke and was right."""
        batch, channels = said.shape[:2]
        spoke = said.amax(-1)
        right = said.gather(-1, truth[:, None, None].expand(-1, channels, 1)).squeeze(-1)  # [B,C]
        update = torch.stack([spoke, right, spoke - right], -1)
        return self.memory_cell(update.reshape(-1, 3), memory.reshape(batch * channels, -1)).view(batch, channels, -1)

    def forward(self, says: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
        """says [B,E,C,3] one-hot per channel (all zero = said nothing), truth [B,E] (feedback after each
        question) -> logits [B,E,4] (3 keys, 모름)."""
        memory = self.initial_memory(says.shape[0], says.shape[2])
        out = []
        for t in range(says.shape[1]):
            out.append(self.step(says[:, t], memory))
            memory = self.update(memory, says[:, t], truth[:, t])
        return torch.stack(out, 1)


def door_scores(logits: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
    """+1 right key, -1 wrong key, 0 모름 (the article door always has a key)."""
    p = logits.softmax(-1)[..., :3]
    best, pick = p.max(-1)
    return torch.where(best > 0.5, torch.where(pick == truth, 1.0, -1.0), torch.zeros_like(best))


def main() -> None:
    from cognitive_lab.mk1.system import MK1
    from cognitive_lab.world10 import world

    parser = argparse.ArgumentParser(description="Train the channel arbiter (MK1 stage F)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-sessions", type=int, default=600)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--tag", default="", help="checkpoint name suffix (revisions)")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    started = time.perf_counter()
    model = MK1(device, articles_from_cache=True)  # training reads the v6 cache (bf16); the test reads live
    views = {}
    for part, seed, count in (("train", args.seed, args.train_sessions), ("validation", 0, None)):
        sessions = world.generate(part, seed, count, board_accuracies=BOARD_ACCURACIES)
        views[part] = {k: v.to(device) for k, v in model.channel_views(sessions).items()}
        print(f"{part}: {len(sessions)} sessions ({time.perf_counter() - started:.0f}s)", flush=True)
    arbiter = Arbiter().to(device)
    optimizer = torch.optim.Adam(arbiter.parameters(), lr=3e-3)
    generator = torch.Generator().manual_seed(args.seed)
    train, valid = views["train"], views["validation"]
    best, history = (-9.0, 0, None), []
    for epoch in range(1, args.epochs + 1):
        arbiter.train()
        order = torch.randperm(len(train["truth"]), generator=generator).to(device)
        for start in range(0, len(order), 32):
            batch = order[start:start + 32]
            logits = arbiter(train["says"][batch], train["truth"][batch])
            loss = nn.functional.cross_entropy(logits.reshape(-1, 4), train["truth"][batch].reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(arbiter.parameters(), 1.0)
            optimizer.step()
        arbiter.eval()
        with torch.no_grad():
            value = door_scores(arbiter(valid["says"], valid["truth"]), valid["truth"]).mean().item()
        history.append({"epoch": epoch, "loss": round(loss.item(), 4), "validation_door_score": round(value, 4)})
        print(history[-1], flush=True)
        if value > best[0]:
            best = (value, epoch, {k: v.detach().cpu().clone() for k, v in arbiter.state_dict().items()})
    path = CHECKPOINT_DIR / f"mk1-arbiter{args.tag}_seed-{args.seed}.pt"
    torch.save({"state": best[2], "epoch": best[1]}, path)
    out = {"part": "arbiter", "seed": args.seed, "selected_epoch": best[1], "validation": round(best[0], 4),
           "parameters": sum(p.numel() for p in arbiter.parameters()), "history": history,
           "seconds": round(time.perf_counter() - started, 1)}
    (RESULTS_DIR / f"mk1-f_arbiter{args.tag}-train_seed-{args.seed}.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"selected epoch {best[1]} (validation {best[0]:.4f}); saved {path}")


if __name__ == "__main__":
    main()
