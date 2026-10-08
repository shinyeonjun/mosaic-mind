"""Rule world v6: a learned trust part on top of the frozen reading specialist.

Inputs per question: for each source, which candidate it reported (at most two candidates: the
true-version and false-version readings) and the specialist's scores on its passage (span,
no-answer, margin). One memory per source (shared GRU), updated after the answer is revealed with
[spoke, its report was right, was wrong, its margin]. For candidate k:
    logit_k = sum over speaking sources s of f([says k, margin, span, no-answer] * says k, memory_s)
and a "no candidate is right" logit from the pooled source memories. Equivariant in sources and
candidates. Answer the top candidate when its probability > 0.5, else 모름 (expected score).
It can therefore use both trust (who lies) and reading confidence (when the specialist misreads).

python -m cognitive_lab.world6.trust --seed 42
"""

import argparse
import json
import random
import statistics
import time

import torch
from torch import nn

from cognitive_lab.world5 import klue
from cognitive_lab.world6 import world

RESULTS_DIR = klue.PROJECT_ROOT / "artifacts" / "results"
CHECKPOINT_DIR = klue.PROJECT_ROOT / "artifacts" / "checkpoints"
SLOTS = 2  # candidates per question


def tensors(sessions: list[dict]) -> dict[str, torch.Tensor]:
    batch, episodes, speakers = len(sessions), world.SESSION_LENGTH, world.SPEAKERS
    says = torch.full((batch, episodes, speakers), -1, dtype=torch.long)
    scores = torch.zeros(batch, episodes, speakers, 3)
    right = torch.zeros(batch, episodes, SLOTS, dtype=torch.bool)
    valid = torch.zeros(batch, episodes, SLOTS, dtype=torch.bool)
    target = torch.full((batch, episodes), SLOTS, dtype=torch.long)  # SLOTS = no candidate is right
    for b, session in enumerate(sessions):
        for t, episode in enumerate(session["episodes"]):
            k = len(episode["candidates"])
            if k > SLOTS:
                raise ValueError("more than two candidates")
            valid[b, t, :k] = True
            right[b, t, :k] = torch.tensor(episode["right"])
            if any(episode["right"]):
                target[b, t] = episode["right"].index(True)
            for r in episode["reports"]:
                says[b, t, r["speaker"]] = r["candidate"]
                scores[b, t, r["speaker"]] = torch.tensor([r["span_score"] / 10, r["null_score"] / 10,
                                                           (r["span_score"] - r["null_score"]) / 10])
    return {"says": says, "scores": scores, "right": right, "valid": valid, "target": target}


class TrustPart(nn.Module):
    def __init__(self, memory: int = 16, width: int = 32):
        super().__init__()
        self.memory_initial = nn.Parameter(torch.zeros(memory))
        self.memory_cell = nn.GRUCell(4, memory)
        self.contribution = nn.Sequential(nn.Linear(4 + memory, width), nn.Tanh(), nn.Linear(width, 1))
        self.none = nn.Sequential(nn.Linear(1 + memory, width), nn.Tanh(), nn.Linear(width, 1))

    def forward(self, data: dict) -> torch.Tensor:
        """Logits [B,E,SLOTS+1] for each question, memory carried through the session."""
        says, scores, right, valid = data["says"], data["scores"], data["right"], data["valid"]
        batch, episodes, speakers = says.shape
        memory = self.memory_initial.expand(batch, speakers, -1)
        out = []
        for t in range(episodes):
            spoke = (says[:, t] >= 0).float()  # [B,S]
            chosen = torch.nn.functional.one_hot(says[:, t].clamp(min=0), SLOTS).float() * spoke[..., None]  # [B,S,K]
            features = torch.cat([chosen[..., None], chosen[..., None] * scores[:, t, :, None, :]], -1)  # [B,S,K,4]
            mem = memory[:, :, None, :].expand(-1, -1, SLOTS, -1)
            contribution = self.contribution(torch.cat([features, mem], -1)).squeeze(-1) * chosen  # [B,S,K]
            logits = contribution.sum(1).masked_fill(~valid[:, t], -1e4)
            none = self.none(torch.cat([spoke[..., None], memory], -1)).squeeze(-1)  # [B,S]
            none = (none * spoke).sum(1, keepdim=True) / spoke.sum(1, keepdim=True).clamp(min=1)
            out.append(torch.cat([logits, none], -1))
            said_right = (right[:, t].gather(1, says[:, t].clamp(min=0)).float()) * spoke
            update = torch.stack([spoke, said_right, spoke - said_right, scores[:, t, :, 2] * spoke], -1)
            memory = self.memory_cell(update.reshape(-1, 4), memory.reshape(batch * speakers, -1)).view(batch, speakers, -1)
        return torch.stack(out, 1)


def choices(logits: torch.Tensor) -> torch.Tensor:
    """Candidate index to answer, or -1 (모름): answer when P(candidate) > 0.5."""
    p = logits.softmax(-1)[..., :SLOTS]
    best, index = p.max(-1)
    return torch.where(best > 0.5, index, torch.full_like(index, -1))


def episode_scores(choice: torch.Tensor, data: dict) -> torch.Tensor:
    picked = data["right"].gather(-1, choice.clamp(min=0)[..., None]).squeeze(-1)
    return torch.where(choice < 0, torch.zeros_like(picked, dtype=torch.float), torch.where(picked, 1.0, -1.0))


def bootstrap(difference: torch.Tensor, seed: int, draws: int = 10000) -> list[float]:
    generator = torch.Generator().manual_seed(seed)
    index = torch.randint(len(difference), (draws, len(difference)), generator=generator)
    means = difference[index].mean(1).sort().values
    return [round(means[int(0.025 * draws)].item(), 4), round(means[int(0.975 * draws)].item(), 4)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Learned trust part on the reading specialist (v6)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    device = torch.device("cuda" if args.device != "cpu" and torch.cuda.is_available() else "cpu")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    started = time.perf_counter()

    sessions = {"train": world.generate("train", args.seed), "validation": world.generate("validation", 0),
                "test": world.generate("test", args.seed)}
    data = {part: {k: v.to(device) for k, v in tensors(s).items()} for part, s in sessions.items()}
    model = TrustPart().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    generator = torch.Generator().manual_seed(args.seed)
    history, best = [], (-9.0, 0, None)
    train = data["train"]
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = torch.randperm(len(train["target"]), generator=generator).to(device)
        losses = []
        for start in range(0, len(order), args.batch_size):
            batch = {k: v[order[start:start + args.batch_size]] for k, v in train.items()}
            loss = nn.functional.cross_entropy(model(batch).reshape(-1, SLOTS + 1), batch["target"].reshape(-1))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())
        model.eval()
        with torch.no_grad():
            value = episode_scores(choices(model(data["validation"])), data["validation"]).mean().item()
        history.append({"epoch": epoch, "train_loss": round(statistics.mean(losses), 4), "validation": round(value, 4)})
        print(history[-1], flush=True)
        if value > best[0]:
            best = (value, epoch, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()})

    model.load_state_dict(best[2])
    model.eval()
    with torch.no_grad():
        learned = episode_scores(choices(model(data["test"])), data["test"]).flatten().cpu()
    test_episodes = [e for s in sessions["test"] for e in s["episodes"]]
    threshold = world.tune_confident(sessions["validation"])
    confidence = torch.tensor([world.score(world.confident(e, threshold), e) for e in test_episodes], dtype=torch.float)
    rates = world.reading_rates("train")
    by_quarter = [round(learned.view(-1, world.SESSION_LENGTH)[:, i:i + 10].mean().item(), 4) for i in range(0, 40, 10)]
    result = {"world": "v6", "experiment": "learned-trust", "seed": args.seed, "selected_epoch": best[1],
              "parameters": sum(p.numel() for p in model.parameters()), "history": history,
              "test": {"learned": round(learned.mean().item(), 4), "learned_by_10": by_quarter,
                       "reading_confidence": round(confidence.mean().item(), 4), "confidence_threshold": round(threshold, 4),
                       "majority": world.run_majority(sessions["test"], 99),
                       "formula_learner": world.run_bayes(sessions["test"], True, rates),
                       "knows_accuracies": world.run_bayes(sessions["test"], False, rates)},
              "learned_minus_confidence": {"mean": round((learned - confidence).mean().item(), 4),
                                           "bootstrap_95": bootstrap(learned - confidence, args.seed)},
              "seconds": round(time.perf_counter() - started, 1)}
    path = CHECKPOINT_DIR / f"world-v6_trust_seed-{args.seed}.pt"
    torch.save({"state": best[2], "epoch": best[1]}, path)
    out = RESULTS_DIR / f"world-v6_trust_seed-{args.seed}.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("test", "learned_minus_confidence", "selected_epoch", "parameters", "seconds")},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
