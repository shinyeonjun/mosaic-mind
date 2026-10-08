"""Stage 2 of the ladder: the integrated system learns when to ask, and whom (rule world v2-A).

The stage-1 system (frozen reader body, learned connection head, memory, judge; trained in
v2-H) is frozen. A new small part, the curiosity head, looks at the judge's state before it
answers and predicts, for each speaker, how much the episode score would rise if that speaker
were asked. The system asks the speaker with the largest prediction if it exceeds the cost.

Training: for every training episode and speaker, the asked answer is simulated and read by
the frozen judge, so the realized gain score(after) - score(before) is known. The curiosity
head regresses it (MSE). Speakers are treated by one shared head (no names, no order).
Reading the answer is one more GRU step for that speaker, because the answer comes last.

Memory: training targets follow the no-question trajectory; at test the memory is updated
from what was actually heard, including answers. (Training memories are slightly less
informed than test memories.)

python -m cognitive_lab.world2.asking_system --seed 42
"""

import argparse
import json
import random
import statistics
import time

import torch
from torch import nn

from cognitive_lab.world2 import asking, hedged
from cognitive_lab.world2.hedged_system import ConnectedSystem, checkpoint_path, feature_table, session_index
from cognitive_lab.world2.integrated import CHECKPOINT_DIR, PROJECT_ROOT, RESULTS_DIR, queried_door

COST = 0.2


def ask_index(sessions: list[dict], rows: list) -> tuple[torch.Tensor, torch.Tensor]:
    """answer entry index [B,E,S,3] (pad = N where nothing new is said), silent [B,E,S]."""
    lookup = {row: i for i, row in enumerate(rows)}
    batch, episodes, speakers = len(sessions), len(sessions[0]["episodes"]), len(sessions[0]["speakers"])
    index = torch.full((batch, episodes, speakers, 3), len(rows), dtype=torch.long)
    silent = torch.zeros(batch, episodes, speakers, dtype=torch.bool)
    for b, session in enumerate(sessions):
        for t, episode in enumerate(session["episodes"]):
            door, keys = queried_door(episode), episode["entities"]["keys"]
            for s, name in enumerate(session["speakers"]):
                answer = asking.ask(session, episode, name)
                if answer is not None:
                    silent[b, t, s] = True
                    index[b, t, s] = torch.tensor([lookup[(answer[0]["text"], door, key)] for key in keys])
    return index, silent


def decision_score(logits: torch.Tensor, truths: torch.Tensor) -> torch.Tensor:
    q = logits.softmax(-1)
    best, choice = q.max(-1)
    return torch.where(best > 0.5, torch.where(choice == truths, 1.0, -1.0), torch.zeros_like(best))


class CuriosityHead(nn.Module):
    """Per speaker: [memory, mean_k h, sum_k q_k h, spoke, sorted q] -> predicted gain."""

    def __init__(self, width: int = 32, memory: int = 32, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(memory + 2 * width + 1 + 3, hidden), nn.Tanh(),
                                 nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1))

    @staticmethod
    def features(memory, h, spoke, logits) -> torch.Tensor:
        q = logits.softmax(-1)  # [B,3]
        speakers = h.shape[1]
        weighted = (h * q[:, None, :, None]).sum(2)
        ranked = q.sort(-1, descending=True).values[:, None].expand(-1, speakers, -1)
        return torch.cat([memory, h.mean(2), weighted, spoke.unsqueeze(-1), ranked], -1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)


def episode_step(system: ConnectedSystem, table, memory, base, answers, truth, t):
    """Judge state before asking and the logits after asking each speaker."""
    judge = system.judge
    index, speakers = base[0][:, t], base[1][:, t]
    h, spoke = judge.read(table[index], speakers)
    before = judge.logits(h, spoke, memory)
    answer = judge.embed(table[answers[0][:, t]])  # [B,S,3,W]
    batch, count, keys, width = h.shape
    stepped = judge.reader(answer.reshape(-1, width), h.reshape(-1, width)).view(batch, count, keys, width)
    silent = answers[1][:, t]
    after_h, after_spoke = [], []
    for s in range(count):
        new_h, new_spoke = h.clone(), spoke.clone()
        new_h[:, s] = torch.where(silent[:, s, None, None], stepped[:, s], h[:, s])
        new_spoke[:, s] = torch.where(silent[:, s], torch.ones_like(spoke[:, s]), spoke[:, s])
        after_h.append(new_h)
        after_spoke.append(new_spoke)
    after = torch.stack([judge.logits(a, b, memory) for a, b in zip(after_h, after_spoke)], 1)  # [B,S,3]
    return h, spoke, before, after, after_h, after_spoke


@torch.no_grad()
def training_targets(system, table, base, answers):
    """Curiosity features [B*E*S, F] and realized gains [B*E*S] on the no-question trajectory."""
    truths = base[2]
    memory = system.judge.initial_memory(truths.shape[0])
    features, gains = [], []
    for t in range(truths.shape[1]):
        h, spoke, before, after, _, _ = episode_step(system, table, memory, base, answers, truths[:, t], t)
        score = decision_score(before, truths[:, t])
        gain = decision_score(after, truths[:, t, None].expand(-1, after.shape[1])) - score[:, None]
        features.append(CuriosityHead.features(memory, h, spoke, before))
        gains.append(gain)
        memory = system.judge.update(memory, h, spoke, truths[:, t])
    return torch.stack(features, 1).flatten(0, 2), torch.stack(gains, 1).flatten()


@torch.no_grad()
def run_policy(system, table, base, answers, policy: str, head: CuriosityHead | None = None,
               cost: float = COST, seed: int = 0, trace: list | None = None) -> dict:
    """Net score per episode under a policy: never / hand-rule / curiosity / hindsight-best."""
    truths = base[2]
    batch, episodes = truths.shape
    memory = system.judge.initial_memory(batch)
    generator = torch.Generator().manual_seed(seed)
    nets, asked, chosen_silent = [], [], []
    for t in range(episodes):
        h, spoke, before, after, after_h, after_spoke = episode_step(system, table, memory, base, answers,
                                                                     truths[:, t], t)
        silent = answers[1][:, t]
        target = torch.full((batch,), -1, dtype=torch.long, device=truths.device)
        if policy == "hand-rule":
            unsure = before.softmax(-1).max(-1).values <= 0.5
            noise = torch.rand(silent.shape, generator=generator).to(silent.device)
            pick = torch.where(silent, noise, torch.full_like(noise, -1.0)).argmax(-1)
            target = torch.where(unsure & silent.any(-1), pick, target)
        elif policy == "curiosity":
            predicted = head(CuriosityHead.features(memory, h, spoke, before))
            best_gain, best = predicted.max(-1)
            target = torch.where(best_gain > cost, best, target)
        elif policy == "hindsight-best":  # knows the realized gains: an upper bound, not a policy
            realized = decision_score(after, truths[:, t, None].expand(-1, after.shape[1]))
            gain = realized - decision_score(before, truths[:, t])[:, None]
            best_gain, best = gain.max(-1)
            target = torch.where(best_gain > cost, best, target)
        if trace is not None:
            trace.append(target.cpu())
        rows = torch.arange(batch, device=truths.device)
        pick = target.clamp(min=0)
        logits = torch.where((target >= 0)[:, None], after[rows, pick], before)
        new_h = torch.stack(after_h, 1)[rows, pick]
        new_spoke = torch.stack(after_spoke, 1)[rows, pick]
        h = torch.where((target >= 0)[:, None, None, None], new_h, h)
        spoke = torch.where((target >= 0)[:, None], new_spoke, spoke)
        nets.append(decision_score(logits, truths[:, t]) - cost * (target >= 0).float())
        asked.append((target >= 0).float())
        chosen_silent.append(torch.where(target >= 0, silent[rows, pick].float(), torch.nan))
        memory = system.judge.update(memory, h, spoke, truths[:, t])
    net, ask = torch.stack(nets, 1), torch.stack(asked, 1)
    hit = torch.stack(chosen_silent, 1)
    return {"net": round(net.mean().item(), 4), "ask_rate": round(ask.mean().item(), 3),
            "asked_silent_share": round(hit[~hit.isnan()].mean().item(), 3) if (~hit.isnan()).any() else None,
            "ask_rate_by_10": [round(ask[:, i:i + 10].mean().item(), 3) for i in range(0, episodes, 10)],
            "net_by_10": [round(net[:, i:i + 10].mean().item(), 4) for i in range(0, episodes, 10)]}


def load_stage1(seed: int, device: torch.device) -> tuple[ConnectedSystem, list]:
    saved = torch.load(checkpoint_path(seed, 8), map_location="cpu")
    rows, features, head = feature_table(saved["reader"], device)
    system = ConnectedSystem(features, head, saved["message_size"])
    system.load_state_dict(saved["state"])
    system.to(device).eval()
    for p in system.parameters():
        p.requires_grad_(False)
    return system, rows


def tensors(sessions, rows, device):
    base = tuple(t.to(device) for t in session_index(sessions, rows))
    answers = tuple(t.to(device) for t in ask_index(sessions, rows))
    return base, answers


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 2: learn when to ask and whom (v2-A)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cost", type=float, default=COST)
    parser.add_argument("--train-sessions", type=int, default=3000)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    device = torch.device("cuda" if args.device != "cpu" and torch.cuda.is_available() else "cpu")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    started = time.perf_counter()

    system, rows = load_stage1(args.seed, device)
    with torch.no_grad():
        table = system.messages()
    splits = {}
    for part, count in (("train", args.train_sessions), ("validation", None), ("test", None)):
        sessions = hedged.generate_part(part, args.seed, sessions=count)
        splits[part] = (sessions, *tensors(sessions, rows, device))
    print(f"tensors ready ({time.perf_counter() - started:.0f}s)", flush=True)

    x, y = [], []
    _, base, answers = splits["train"]
    for start in range(0, base[2].shape[0], 250):
        part_base = tuple(t[start:start + 250] for t in base)
        part_answers = tuple(t[start:start + 250] for t in answers)
        fx, fy = training_targets(system, table, part_base, part_answers)
        x.append(fx)
        y.append(fy)
    x, y = torch.cat(x), torch.cat(y)
    print(f"targets {len(y)} (mean gain {y.mean():.4f}, share > cost {(y > args.cost).float().mean():.3f}) "
          f"({time.perf_counter() - started:.0f}s)", flush=True)

    head = CuriosityHead().to(device)
    optimizer = torch.optim.Adam(head.parameters(), lr=args.learning_rate)
    generator = torch.Generator().manual_seed(args.seed)
    _, vbase, vanswers = splits["validation"]
    history, best = [], (-9.0, 0, None)
    for epoch in range(1, args.epochs + 1):
        head.train()
        order = torch.randperm(len(y), generator=generator).to(device)
        losses = []
        for start in range(0, len(order), 4096):
            batch = order[start:start + 4096]
            loss = nn.functional.mse_loss(head(x[batch]), y[batch])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        head.eval()
        result = run_policy(system, table, vbase, vanswers, "curiosity", head, args.cost)
        history.append({"epoch": epoch, "train_mse": round(statistics.mean(losses), 5),
                        "validation_net": result["net"], "validation_ask_rate": result["ask_rate"]})
        if epoch == 1 or epoch % 5 == 0:
            print(history[-1], flush=True)
        if result["net"] > best[0]:
            best = (result["net"], epoch, {k: v.detach().cpu().clone() for k, v in head.state_dict().items()})

    head.load_state_dict(best[2])
    head.eval()
    path = CHECKPOINT_DIR / f"world-v2a_curiosity_cost-{args.cost}_seed-{args.seed}.pt"
    torch.save({"state": best[2], "stage1": str(checkpoint_path(args.seed, 8).name), "cost": args.cost,
                "epoch": best[1]}, path)
    _, tbase, tanswers = splits["test"]
    test = {policy: run_policy(system, table, tbase, tanswers, policy, head, args.cost, seed=args.seed)
            for policy in ("never", "hand-rule", "curiosity", "hindsight-best")}
    ideal = asking.ideal_policies(splits["test"][0], args.cost)
    result = {"world": "v2-A", "experiment": "curiosity-head", "seed": args.seed, "cost": args.cost,
              "parameters": sum(p.numel() for p in head.parameters()), "selected_epoch": best[1],
              "history": history, "test": test, "ideal_bayes_with_cues": ideal,
              "training_seconds": round(time.perf_counter() - started, 1),
              "checkpoint": str(path.relative_to(PROJECT_ROOT))}
    out = RESULTS_DIR / f"world-v2a_curiosity_cost-{args.cost}_seed-{args.seed}.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    for policy, r in test.items():
        print(f"{policy:15s} net {r['net']:.4f}  ask {r['ask_rate']:.3f}  silent-share {r['asked_silent_share']}")
    print(f"ideal {ideal}\nSaved: {path}\nSaved: {out}")


if __name__ == "__main__":
    main()
