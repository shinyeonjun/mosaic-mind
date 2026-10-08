"""Stage 3 of the ladder: a thinking loop that decides how long to think (rule world v3).

Reading (frozen v1 reader body, cached features) -> two small connection heads:
    fact message  m_f[u, door, key]   for "door opens with key"
    link message  m_l[u, door pair]   for "two doors share a key"
Thinking: one state z[d, k] per (door, key), shared weights (door- and key-equivariant).
    z0[d, k] = max_u phi(m_f[u, d, k])                    what some sentence says directly
    a[d, e]  = [max_u sigmoid(w . psi(m_l[u, {d, e}])) > 0.5]   whether some sentence links d and e
               (0/1, straight-through gradient)
    belief:  p[e] = softmax_k out(z[e, k])            (the same readout that answers the question)
    step:    z[d, k] <- GRU([listen(max_e a[d, e] [p[e, k], max_k' p[e, k']]), z0[d, k]], z[d, k])
             doors pass beliefs, not hidden states (z0 is re-injected)
(Max, not sum: an earlier sum version built links never stated from two half-matching sentences
and could not follow chains beyond what it was trained on.)
Each step moves information one link further, so a chain of L links needs L steps.
Answer after step t: logits_t[k] = out(z_t[q, k]); answer if max prob > 0.5 else 모름.
Stopping (fixed rule, chosen 2026-10-08): stop when no door's answer moved by 0.01 or more in two
consecutive steps. A learned halting head is still trained (PonderNet loss) and reported for
comparison; it did not extrapolate in development (it stops at lengths seen in training).
Loss (PonderNet style): sum_t p_t * CE_t + cost * sum_t p_t * t, p_t = lambda_t prod_{j<t}(1 - lambda_j),
plus the mean CE over all steps, so the answer must stay right however long the loop runs,
plus a cost on link strength, so only links the answers need are kept. (Without it the
learned graph linked every pair of doors and answered from other cues; see the design note.)
Unanswerable episodes have a uniform target. Training sees chains of 1-3 links only.
Every door's answer (by the solver) is also a training target at every step, so the beliefs
doors pass to each other are answers that were checked, not unchecked internal values.

python -m cognitive_lab.world3.thinker --seed 42
"""

import argparse
import json
import random
import statistics
import time

import torch
from torch import nn

from cognitive_lab.world.templates import UNKNOWN
from cognitive_lab.world2.integrated import CHECKPOINT_DIR, PROJECT_ROOT, RESULTS_DIR
from cognitive_lab.world3 import chains
from cognitive_lab.world3.features import PAIRS, feature_table

MAX_UTTERANCES = 12
DOOR_COUNT, KEY_COUNT = len(chains.DOORS), len(chains.KEYS)


def door_targets(episodes: list[dict]) -> torch.Tensor:
    """The answer for every door [B,D] (-1 = 모름), by the solver: used as extra training targets."""
    return torch.tensor([[-1 if (a := chains.solve(e["gold"], d)) == UNKNOWN else chains.KEYS.index(a)
                          for d in chains.DOORS] for e in episodes])


def episode_tensors(episodes: list[dict], table: dict) -> tuple[torch.Tensor, ...]:
    """sentence index [B,U] (pad = S), query door [B], target key [B] (-1 = 모름), length [B]."""
    lookup = {s["text"]: i for i, s in enumerate(table["sentences"])}
    pad = len(table["sentences"])
    index = torch.full((len(episodes), MAX_UTTERANCES), pad, dtype=torch.long)
    for b, episode in enumerate(episodes):
        if len(episode["utterances"]) > MAX_UTTERANCES:
            raise ValueError(f"{episode['episode_id']} has {len(episode['utterances'])} utterances")
        index[b, :len(episode["utterances"])] = torch.tensor([lookup[u] for u in episode["utterances"]])
    query = torch.tensor([chains.DOORS.index(e["query"]) for e in episodes])
    target = torch.tensor([-1 if e["answer"] == UNKNOWN else chains.KEYS.index(e["answer"]) for e in episodes])
    length = torch.tensor([e["length"] for e in episodes])
    return index, query, target, length


class Thinker(nn.Module):
    def __init__(self, features: torch.Tensor, size: int = 8, width: int = 32, message: str = "belief"):
        super().__init__()
        self.message = message
        self.register_buffer("features", features, persistent=False)  # [S, 75, 768]
        hidden = features.shape[-1]
        self.fact_head = nn.Linear(hidden, size)
        self.link_head = nn.Linear(hidden, size)
        self.phi = nn.Sequential(nn.Linear(size, width), nn.Tanh(), nn.Linear(width, width))
        self.psi = nn.Sequential(nn.Linear(size, width), nn.Tanh(), nn.Linear(width, width))
        self.link_score = nn.Linear(width, 1)
        # Start with every link on (sigmoid(3) = 0.95) and let the link cost prune the useless
        # ones, like synapses: with 0/1 links that start off, nothing flows and nothing is learned.
        nn.init.constant_(self.link_score.bias, 3.0)
        self.listen = nn.Linear(2, width)  # what a door hears: [neighbor belief in key k, neighbor sureness]
        self.cell = nn.GRUCell(2 * width, width)
        self.out = nn.Sequential(nn.Linear(width, width), nn.Tanh(), nn.Linear(width, 1))
        # Halting sees only "did anything change in this step": stop when the thinking has
        # reached a fixed point. Given the state it learns to count steps; given the query's
        # sureness it learns "stop when sure", which stops too early on long chains.
        self.halt = nn.Sequential(nn.Linear(2, width), nn.Tanh(), nn.Linear(width, 1))
        rows, cols = zip(*[(chains.DOORS.index(a), chains.DOORS.index(b)) for a, b in PAIRS])
        self.register_buffer("pair_rows", torch.tensor(rows), persistent=False)
        self.register_buffer("pair_cols", torch.tensor(cols), persistent=False)

    def messages(self) -> tuple[torch.Tensor, torch.Tensor]:
        """fact [S+1, D, 3, M], link [S+1, D(D-1)/2, M]; the last row is the padding sentence."""
        facts = torch.tanh(self.fact_head(self.features[:, :DOOR_COUNT * KEY_COUNT]))
        links = torch.tanh(self.link_head(self.features[:, DOOR_COUNT * KEY_COUNT:]))
        facts = torch.cat([facts, facts.new_zeros(1, *facts.shape[1:])]).view(-1, DOOR_COUNT, KEY_COUNT, facts.shape[-1])
        links = torch.cat([links, links.new_zeros(1, *links.shape[1:])])
        return facts, links

    def think(self, index: torch.Tensor, query: torch.Tensor, steps: int, with_links: bool = False,
              extra_facts: torch.Tensor | None = None, extra_mask: torch.Tensor | None = None,
              extra_replace: bool = False):
        """Per-step logits [T,B,3] and halting probabilities [T,B] (and link strengths [B,45] and
        every door's per-step logits [T,B,D,3]). `extra_facts` [B,D,3,M] are fact messages from
        another part (rule world v4), used for the doors in `extra_mask` [B,D]: merged with what the board
        says (max), or, with `extra_replace`, instead of it (MK1 stage F: an arbiter has already weighed both)."""
        facts, links = self.messages()
        pad = index >= self.features.shape[0]
        # Max over sentences: "does some single sentence say this?" Summing would let two
        # half-matching sentences ("a = b", "b = c") add up to a link "a = c" never stated.
        z0 = self.phi(facts[index]).masked_fill(pad[:, :, None, None, None], -1e4).amax(1)  # [B,D,3,W]
        if extra_facts is not None:
            extra = self.phi(extra_facts).masked_fill(~extra_mask[:, :, None, None], -1e4)
            z0 = torch.where(extra_mask[:, :, None, None], extra, z0) if extra_replace else torch.maximum(z0, extra)
        strength = torch.sigmoid(self.link_score(self.psi(links[index]))).squeeze(-1)  # [B,U,45]
        strength = strength.masked_fill(pad[:, :, None], 0.0).amax(1)
        # A link exists or not: 0/1 in the forward pass, the soft value's gradient in training
        # (straight-through). Weak half-links would let information leak and fade.
        link = (strength > 0.5).float() + strength - strength.detach()
        self.last_strength = strength
        batch = index.shape[0]
        adjacency = strength.new_zeros(batch, DOOR_COUNT, DOOR_COUNT)
        adjacency[:, self.pair_rows, self.pair_cols] = link
        adjacency = adjacency + adjacency.transpose(1, 2)
        width = z0.shape[-1]
        z = z0
        rows = torch.arange(batch, device=index.device)
        beliefs = self.out(z).squeeze(-1).softmax(-1)  # every door's current answer [B,D,3]
        all_logits, all_halts, all_doors, all_changes = [], [], [], []
        for _ in range(steps):
            # Max over linked doors: what the strongest neighbor holds (does not fade with distance).
            if self.message == "belief":
                # Doors talk in beliefs: each sends its answer probabilities, not its hidden state.
                # A door is trained to become sure when a neighbor is sure; every door uses the same
                # rule, so sureness is passed on without fading, however long the chain.
                belief = self.out(z).squeeze(-1).softmax(-1)  # [B,D,3]
                heard = adjacency[:, :, :, None] * belief[:, None]  # [B,D(d),D(e),3]
                sure = heard.amax(-1, keepdim=True).expand_as(heard)
                incoming = self.listen(torch.stack([heard, sure], -1).amax(2))  # [B,D,3,W]
            else:  # "state": the strongest linked neighbor's hidden state
                incoming = (adjacency[:, :, :, None, None] * z[:, None]).amax(2)
            new_z = self.cell(torch.cat([incoming, z0], -1).reshape(-1, 2 * width), z.reshape(-1, width)).view_as(z)
            # Halting signals are changes in answers, not in hidden states: a hidden state keeps
            # settling slowly after the work is done, which lets halting learn to count steps.
            new_beliefs = self.out(new_z).squeeze(-1).softmax(-1)
            change_query = (new_beliefs[rows, query] - beliefs[rows, query]).abs().amax(-1)
            change_all = (new_beliefs - beliefs).abs().amax((1, 2))  # is any door's answer still moving?
            z, beliefs = new_z, new_beliefs
            all_changes.append(change_all)
            zq = z[rows, query]  # [B,3,W]
            logits = self.out(zq).squeeze(-1)
            halt_input = torch.stack([change_query, change_all], -1)
            all_logits.append(logits)
            all_doors.append(self.out(z).squeeze(-1))
            all_halts.append(torch.sigmoid(self.halt(halt_input)).squeeze(-1))
        self.last_changes = torch.stack(all_changes)  # [T,B] largest change of any door's answer
        if with_links:
            return torch.stack(all_logits), torch.stack(all_halts), strength, torch.stack(all_doors)
        return torch.stack(all_logits), torch.stack(all_halts)


def halting_distribution(halts: torch.Tensor) -> torch.Tensor:
    """p_t = lambda_t prod_{j<t}(1 - lambda_j); the remaining mass goes to the last step."""
    keep = torch.cumprod(torch.cat([torch.ones_like(halts[:1]), 1 - halts[:-1]]), 0)
    p = halts * keep
    p[-1] = keep[-1]
    return p


def step_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Cross-entropy per step [T,B]; 모름 targets are uniform over keys."""
    logp = logits.log_softmax(-1)
    onehot = nn.functional.one_hot(target.clamp(min=0), KEY_COUNT).float()
    goal = torch.where((target >= 0)[:, None], onehot, torch.full_like(onehot, 1 / KEY_COUNT))
    return -(goal * logp).sum(-1)


def scores(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """+1 right (a correct 모름 included), 0 for 모름 when an answer exists, -1 wrong."""
    q = logits.softmax(-1)
    best, choice = q.max(-1)
    answered = best > 0.5
    right = torch.where(target >= 0, choice == target, torch.zeros_like(answered))
    abstain_score = torch.where(target >= 0, 0.0, 1.0)
    return torch.where(answered, torch.where(right, 1.0, -1.0), abstain_score)


FIXED_POINT_EPSILON = 0.01  # stop when no door's answer moved by this much in 2 consecutive steps


def fixed_point_stop(changes: torch.Tensor, epsilon: float = FIXED_POINT_EPSILON) -> torch.Tensor:
    """First step index t >= 1 where the largest answer change was below epsilon at t-1 and t
    (long chains pause for one step between links), else the last step."""
    still = changes < epsilon
    settled = torch.zeros_like(still)
    settled[1:] = still[1:] & still[:-1]
    last = torch.full_like(changes[0], changes.shape[0] - 1, dtype=torch.long)
    return torch.where(settled.any(0), settled.float().argmax(0), last)


@torch.no_grad()
def evaluate(model: Thinker, data: tuple, steps: int, halting: str = "fixed-point") -> dict:
    """Adaptive scores (fixed-point rule, or the learned halting head) and forced-step scores."""
    index, query, target, length = data
    logits, halts = model.think(index, query, steps)
    if halting == "fixed-point":
        stop = fixed_point_stop(model.last_changes)
    else:
        stop = torch.where((halts > 0.5).any(0), (halts > 0.5).float().argmax(0), torch.full_like(query, steps - 1))
    rows = torch.arange(len(query), device=query.device)
    adaptive = scores(logits[stop, rows], target)
    forced = torch.stack([scores(logits[t], target) for t in range(steps)])  # [T,B]
    p_true = logits.softmax(-1)[:, rows, target.clamp(min=0)]  # [T,B]
    result = {"adaptive": round(adaptive.mean().item(), 4), "halting": halting, "by_length": {}}
    for value in sorted(set(length.tolist())):
        sel = length == value
        answerable = sel & (target >= 0)
        result["by_length"][value] = {
            "adaptive": round(adaptive[sel].mean().item(), 4),
            "mean_steps": round((stop[sel].float() + 1).mean().item(), 3),
            "mean_steps_answerable": round((stop[answerable].float() + 1).mean().item(), 3),
            "mean_steps_unanswerable": round((stop[sel & (target < 0)].float() + 1).mean().item(), 3),
            "forced": [round(forced[t, sel].mean().item(), 4) for t in range(steps)],
            # Leak audit: before step L the anchor's key cannot have reached the query through
            # links, so p(true key) must stay near 1/3.
            "pre_arrival_p_true": round(p_true[:value - 1, answerable].mean().item(), 4) if value > 1 else None,
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 3: a thinking loop that decides how long to think")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-steps", type=int, default=8, help="thinking steps available in training")
    parser.add_argument("--test-steps", type=int, default=16, help="thinking steps available at test")
    parser.add_argument("--think-cost", type=float, default=0.01, help="loss per expected thinking step")
    parser.add_argument("--message", choices=("belief", "state"), default="belief",
                        help="what doors pass to each other while thinking")
    parser.add_argument("--link-cost", type=float, default=0.1,
                        help="loss per unit of mean link strength (keep only links the answers need)")
    parser.add_argument("--all-doors", type=float, default=1.0,
                        help="weight of the mean cross-entropy of every door's answer at every step")
    parser.add_argument("--every-step", type=float, default=1.0,
                        help="weight of the mean cross-entropy over all steps (keeps long thinking stable)")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--init", help="checkpoint file to start from (stage 2 of growing)")
    parser.add_argument("--freeze-links", action="store_true",
                        help="keep the link parts (which doors are linked) fixed")
    parser.add_argument("--tag", default="", help="suffix for checkpoint and result names")
    parser.add_argument("--trace-development", action="store_true",
                        help="also log long-chain development scores each epoch (diagnostics only)")
    parser.add_argument("--test", action="store_true",
                        help="score the test set (chains 1-6); development runs look at validation only")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    device = torch.device("cuda" if args.device != "cpu" and torch.cuda.is_available() else "cpu")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    started = time.perf_counter()

    table = feature_table(device)
    data = {part: tuple(t.to(device) for t in episode_tensors(chains.generate(part, args.seed), table))
            for part in ("train", "validation", "development", "test")}
    model = Thinker(table["features"].to(device), message=args.message).to(device)
    if args.init:
        model.load_state_dict(torch.load(CHECKPOINT_DIR / args.init)["state"])
    if args.freeze_links:
        for part in (model.link_head, model.psi, model.link_score):
            for p in part.parameters():
                p.requires_grad_(False)
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.learning_rate)
    generator = torch.Generator().manual_seed(args.seed)
    history, best = [], ((-2.0, -2.0, -99.0), 0, None)
    index, query, target, _ = data["train"]
    all_targets = door_targets(chains.generate("train", args.seed)).to(device)
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = torch.randperm(len(query), generator=generator).to(device)
        losses = []
        # Warm-up: no link cost at first, while beliefs are still uninformative (otherwise every
        # link is switched off before passing beliefs can pay for it), full cost from epoch 10.
        link_cost = args.link_cost * min(1.0, (epoch - 1) / 9)
        for start in range(0, len(order), args.batch_size):
            batch = order[start:start + args.batch_size]
            logits, halts, strength, doors = model.think(index[batch], query[batch], args.train_steps, with_links=True)
            p = halting_distribution(halts)
            steps = torch.arange(1, args.train_steps + 1, device=device, dtype=torch.float)[:, None]
            per_step = step_loss(logits, target[batch])
            loss = ((p * per_step).sum(0).mean() + args.think_cost * (p * steps).sum(0).mean()
                    + args.every_step * per_step.mean() + link_cost * strength.mean()
                    + args.all_doors * step_loss(doors.flatten(1, 2), all_targets[batch].flatten()).mean())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())
        model.eval()
        validation = evaluate(model, data["validation"], args.train_steps)
        # Selection uses validation only (chains 1-3): the adaptive score, then how well the
        # answer holds when forced to keep thinking to the last training step, then fewer steps.
        stability = statistics.mean(r["forced"][-1] for r in validation["by_length"].values())
        history.append({"epoch": epoch, "train_loss": round(statistics.mean(losses), 4),
                        "validation": validation["adaptive"], "validation_last_step": round(stability, 4),
                        "validation_steps": {k: v["mean_steps"] for k, v in validation["by_length"].items()}})
        if args.trace_development:  # diagnostics for design decisions; never used for selection
            dev = evaluate(model, data["development"], args.test_steps)
            history[-1]["development"] = {k: (v["adaptive"], v["forced"][-1], v["mean_steps_answerable"])
                                          for k, v in dev["by_length"].items()}
        print(history[-1], flush=True)
        thinking = statistics.mean(r["mean_steps"] for r in validation["by_length"].values())
        if (validation["adaptive"], stability, -thinking) > best[0]:
            best = ((validation["adaptive"], stability, -thinking), epoch, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()})

    model.load_state_dict(best[2])
    model.eval()
    final = evaluate(model, data["validation"], args.train_steps)
    print(f"selected epoch {best[1]}; validation {final['adaptive']}")
    for length, r in final["by_length"].items():
        print(f"  validation L={length}: adaptive {r['adaptive']:.3f}, steps {r['mean_steps_answerable']:.2f} "
              f"(unanswerable {r['mean_steps_unanswerable']:.2f}), forced {r['forced']}")
    torch.save({"state": best[2], "epoch": best[1], "args": vars(args)},
               CHECKPOINT_DIR / f"world-v3_thinker{args.tag}_seed-{args.seed}.pt")
    longer = evaluate(model, data["validation"], args.test_steps)
    print("validation, forced up to the test step count:",
          {length: r["forced"][-4:] for length, r in longer["by_length"].items()})
    development = evaluate(model, data["development"], args.test_steps)
    for length, r in development["by_length"].items():
        print(f"  development L={length}: adaptive {r['adaptive']:.3f}, steps {r['mean_steps_answerable']:.2f} "
              f"(unanswerable {r['mean_steps_unanswerable']:.2f}), forced {r['forced']}")
    if not args.test:
        return
    path = CHECKPOINT_DIR / f"world-v3_thinker{args.tag}_seed-{args.seed}.pt"
    test = evaluate(model, data["test"], args.test_steps)
    test_learned_halting = evaluate(model, data["test"], args.test_steps, halting="learned")
    result = {"world": "v3", "experiment": "thinker", "seed": args.seed, "args": vars(args),
              "parameters": sum(p.numel() for p in model.parameters()), "selected_epoch": best[1],
              "history": history, "test": test, "test_learned_halting": test_learned_halting, "training_seconds": round(time.perf_counter() - started, 1),
              "checkpoint": str(path.relative_to(PROJECT_ROOT))}
    out = RESULTS_DIR / f"world-v3_thinker{args.tag}_seed-{args.seed}.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"selected epoch {best[1]}; test adaptive {test['adaptive']}")
    for length, r in test["by_length"].items():
        print(f"  L={length}: adaptive {r['adaptive']:.3f}, steps {r['mean_steps_answerable']:.2f} "
              f"(unanswerable {r['mean_steps_unanswerable']:.2f}), forced-3 {r['forced'][2]:.3f}, "
              f"forced-{args.test_steps} {r['forced'][-1]:.3f}, pre-arrival p(true) {r['pre_arrival_p_true']}, "
              f"learned halting {test_learned_halting['by_length'][length]['adaptive']:.3f}")
    print(f"Saved: {path}\nSaved: {out}")


if __name__ == "__main__":
    main()
