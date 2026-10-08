"""MK1 stage F: an arbiter that learns which channel to believe (design/mk1-integration.md, F).

Channels are parts, not people: the board (through the thinker's first pass) and the article part (through
the real-text trust part). For the article door, each channel says a key or nothing. Like the stage-1
trust part, the arbiter keeps a memory per channel, updated after every question from whether that
channel was right, and weighs what each channel says by its memory. Weights are shared over keys and
channels (no channel names), so it can only learn "trust whoever has been right this session".
Output: logits over the 3 keys and 모름; answer a key when its probability passes 0.5.

python -m cognitive_lab.mk1.arbiter --seed 42                   # train (article readings from the v6 cache)
python -m cognitive_lab.mk1.arbiter --seed 42 --kind counting   # stage I: fit only the counting arbiter's prior
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


class CountingArbiter(nn.Module):
    """The same job without learned weights: per channel, count how often it spoke and was right this session;
    its accuracy is the Beta(1,1) posterior mean (right + 1) / (spoke + 2), any value in (0, 1). A channel saying
    key k makes k `a` times as likely and each other key (1 - a) / 2 times, so a channel right less often than
    chance (a < 1/3) counts against what it says. Nothing here depends on which accuracies training showed:
    the learned arbiter failed on a board right only 25% of the time (design/audit-2026-10-08.md, 8).
    With `learn_prior`, the starting belief Beta(alpha, beta) (shared by all channels: no names) is fitted on
    training sessions, so experience sets how much a new channel is trusted at first, while the counts can still
    take it anywhere in (0, 1)."""

    def __init__(self, learn_prior: bool = False):
        super().__init__()
        self.log_prior = nn.Parameter(torch.zeros(2), requires_grad=learn_prior)  # log alpha, log beta (1, 1)

    def initial_memory(self, batch: int, channels: int = 2) -> torch.Tensor:
        return torch.zeros(batch, channels, 2, device=self._device)  # [right, spoke]

    @property
    def _device(self):
        return getattr(self, "device", torch.device("cpu"))

    def to(self, *args, **kwargs):
        device = next((a for a in args if isinstance(a, (torch.device, str))), kwargs.get("device"))
        if device is not None:
            self.device = torch.device(device)
        return super().to(*args, **kwargs)

    def accuracy(self, memory: torch.Tensor) -> torch.Tensor:
        """Each channel's estimated accuracy [B,C]: the Beta posterior mean of its counts."""
        alpha, beta = self.log_prior.exp()
        return (memory[..., 0] + alpha) / (memory[..., 1] + alpha + beta)

    def step(self, said: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        accuracy = self.accuracy(memory)[..., None]  # [B,C,1]
        spoke = said.amax(-1, keepdim=True)  # [B,C,1]
        keys = said.shape[-1]
        likelihood = torch.where(said > 0, accuracy, (1 - accuracy) / (keys - 1))
        log_p = torch.where(spoke > 0, likelihood.log(), torch.zeros_like(likelihood)).sum(1)  # [B,3]
        return torch.cat([log_p, torch.full_like(log_p[:, :1], -1e4)], -1)  # never 모름 by itself: p > 0.5 decides

    def update(self, memory: torch.Tensor, said: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
        spoke = said.amax(-1)
        right = said.gather(-1, truth[:, None, None].expand(-1, said.shape[1], 1)).squeeze(-1)
        return memory + torch.stack([right, spoke], -1)

    def forward(self, says: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
        memory = self.initial_memory(says.shape[0], says.shape[2]).to(says.device)
        out = []
        for t in range(says.shape[1]):
            out.append(self.step(says[:, t], memory))
            memory = self.update(memory, says[:, t], truth[:, t])
        return torch.stack(out, 1)


class ChangeArbiter(CountingArbiter):
    """The counting arbiter that also notices when a channel's reliability changes (MK1 stage K).

    Bayesian online change-point detection (Adams & MacKay 2007) per channel: the memory holds a distribution over
    the run length r (observations of this channel since its last change) and, per r, the counts of that run. The
    channel's accuracy is the mixture over r of the Beta(alpha, beta) posterior means. Each observation either
    continues every run (prob 1 - h) or starts a new one (prob h); an observation the current runs found unlikely
    moves the mass to short runs, so old evidence stops counting: unexpected uncertainty, the role Yu & Dayan give
    norepinephrine (design/neuroscience-review-2026-10-09.md, 4). Learned numbers: alpha, beta, h.
    The learned prior Beta(alpha, beta) is only for the run that began with the session (a channel met for the first
    time); a run that begins at a change starts from Beta(1, 1): after a betrayal, old expectations are dropped. (With
    the learned prior for every run, a trusted board that began to mislead kept being followed for many questions.)
    Run lengths go up to `max_run - 1`; sessions are 40 questions, so 41 bins never overflow."""

    def __init__(self, learn: bool = False, max_run: int = 41):
        super().__init__(learn_prior=learn)
        self.logit_hazard = nn.Parameter(torch.tensor(-3.0), requires_grad=learn)  # h = sigmoid(-3) = 0.05
        self.max_run = max_run

    def initial_memory(self, batch: int, channels: int = 2) -> torch.Tensor:
        memory = torch.zeros(batch, channels, self.max_run, 4, device=self._device)  # [weight, right, spoke, first run?]
        memory[:, :, 0, 0] = 1.0
        memory[:, :, 0, 3] = 1.0
        return memory

    def _run_accuracy(self, memory: torch.Tensor) -> torch.Tensor:
        alpha, beta = self.log_prior.exp()
        first = memory[..., 3]
        a, b = first * alpha + (1 - first), first * beta + (1 - first)  # learned prior for the first run, else Beta(1, 1)
        return (memory[..., 1] + a) / (memory[..., 2] + a + b)  # [B,C,L]

    def accuracy(self, memory: torch.Tensor) -> torch.Tensor:
        return (memory[..., 0] * self._run_accuracy(memory)).sum(-1)

    def update(self, memory: torch.Tensor, said: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
        spoke = said.amax(-1)  # [B,C]
        right = said.gather(-1, truth[:, None, None].expand(-1, said.shape[1], 1)).squeeze(-1)  # [B,C]
        h = torch.sigmoid(self.logit_hazard)
        run_accuracy = self._run_accuracy(memory)
        joint = memory[..., 0] * torch.where(right[..., None] > 0, run_accuracy, 1 - run_accuracy)  # [B,C,L]
        weight = torch.cat([joint.sum(-1, keepdim=True) * h, joint[..., :-1] * (1 - h)], -1)  # new run | runs grow by 1
        weight = weight / weight.sum(-1, keepdim=True)
        counts = torch.cat([torch.zeros_like(memory[..., :1, 1:3]), memory[..., :-1, 1:3]], -2)  # shift runs by one
        counts = counts + torch.stack([right, torch.ones_like(right)], -1)[..., None, :]  # all runs see this observation
        first = torch.cat([torch.zeros_like(memory[..., :1, 3:]), memory[..., :-1, 3:]], -2)  # a new run is not the first
        new = torch.cat([weight[..., None], counts, first], -1)
        return torch.where(spoke[..., None, None] > 0, new, memory)  # a silent channel learns nothing


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
    parser.add_argument("--kind", choices=("learned", "counting", "change"), default="learned")
    parser.add_argument("--switch", action="store_true",
                        help="stage K: training sessions where the board's accuracy may change mid-session")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    started = time.perf_counter()
    model = MK1(device, seed=args.seed, articles_from_cache=True)  # training reads the v6 cache (bf16); the test reads live
    views = {}
    for part, seed, count in (("train", args.seed, args.train_sessions), ("validation", 0, None)):
        sessions = world.generate(part, seed, count, board_accuracies=BOARD_ACCURACIES)
        if args.switch:  # stage K experience: a third no change, a third mild changes, a third betrayals
            third = len(sessions) // 3
            sessions = (sessions[:third]
                        + world.generate(part, seed + 1000, third, board_accuracies=BOARD_ACCURACIES, switch=True)
                        + world.generate(part, seed + 2000, len(sessions) - 2 * third, board_accuracies=(1.0, 0.75),
                                         switch=True, switch_to=(0.25,)))
        views[part] = {k: v.to(device) for k, v in model.channel_views(sessions).items()}
        print(f"{part}: {len(sessions)} sessions ({time.perf_counter() - started:.0f}s)", flush=True)
    if args.kind in ("counting", "change"):  # stage I/K: alpha, beta (and the change rate h), fitted full-batch
        arbiter = (CountingArbiter(learn_prior=True) if args.kind == "counting" else ChangeArbiter(learn=True)).to(device)
        optimizer = torch.optim.Adam(arbiter.parameters(), lr=0.05)
        train = views["train"]
        for _ in range(150):
            loss = nn.functional.cross_entropy(arbiter(train["says"], train["truth"]).reshape(-1, 4), train["truth"].reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        alpha, beta = arbiter.log_prior.exp().tolist()
        with torch.no_grad():
            value = door_scores(arbiter(views["validation"]["says"], views["validation"]["truth"]), views["validation"]["truth"]).mean().item()
        path = CHECKPOINT_DIR / f"mk1-arbiter-{args.kind}{'-switch' if args.switch else ''}_seed-{args.seed}.pt"
        hazard = torch.sigmoid(arbiter.logit_hazard).item() if args.kind == "change" else None
        torch.save({"state": arbiter.state_dict(), "alpha": alpha, "beta": beta, "hazard": hazard}, path)
        print(f"prior Beta({alpha:.2f}, {beta:.2f}), hazard {hazard}, validation door score {value:.4f}; saved {path}")
        return
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
