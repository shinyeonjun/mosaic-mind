"""Rule world v7: a router that calls parts for what is missing (design/world-v7-routing.md).

Episode types (not shown to the system): T (ask about a base door; its key only from speakers),
C (ask about a chain end; the board states the anchor's key), TC (chain end; anchor key only
from speakers). Train on T and C; test on T, C and the unseen TC.

Pipeline per episode (parts frozen: stage-1 trust part, stage-4 adapter, stage-3 thinker):
  1. the thinker reads the board (links + stated facts) and thinks to a fixed point;
  2. for each base door d the router sees only what is missing: [d linked to the queried door in
     the thinker's graph, how sure the thinker is of d's key, how sure of the answer], and predicts
     the score gain of asking the trust part about d. (A first version also gave "d is the queried
     door"; in training that identity coincides with "linked and unknown", the router keyed on it
     and never asked in the unseen chained case. Its result is archived.)
  3. it asks about every d whose predicted gain exceeds the cost (0.05 per question); the trust
     part's conclusions enter the thinker as facts (stage-4 adapter) and the thinker re-thinks.
Every subset of base doors is run for every episode, so any routing policy can be scored exactly.

python -m cognitive_lab.world7.routing --seed 42
"""

import argparse
import itertools
import json
import random
import time

import torch
from torch import nn

from cognitive_lab.world2 import hedged
from cognitive_lab.world3 import chains
from cognitive_lab.world3.thinker import fixed_point_stop, scores
from cognitive_lab.world4 import composite
from cognitive_lab.world4.system import CHECKPOINT_DIR, RESULTS_DIR, CompositeSystem, flat, load_parts, session_tensors

COST = 0.05
STEPS = 16
TYPES = ("T", "C", "TC")
MIX = {"train": ("T", "C"), "validation": ("T", "C"), "test": TYPES}
SESSIONS = {"train": 300, "validation": 40, "test": 100}
SUBSETS = [s for r in range(4) for s in itertools.combinations(range(3), r)]  # 8 subsets of base doors


def add_routing_chains(session: dict, part: str, seed_text: str) -> dict:
    rng = random.Random(f"world-v7|{seed_text}")
    for episode in session["episodes"]:
        anchor = hedged._queried(episode)
        kind = rng.choice(MIX[part])
        extras = rng.sample(composite.EXTRA_DOORS, len(composite.EXTRA_DOORS))
        length = composite.LENGTHS[episode["index"] % len(composite.LENGTHS)]
        links, chain_of = [], {}
        for base in composite.BASE_DOORS:
            size = length if base == anchor else rng.choice(composite.LENGTHS)
            chain = [base] + [extras.pop() for _ in range(size)]
            chain_of[base] = chain
            links += [(chain[i], chain[i + 1]) for i in range(size)]
        truths = dict(episode["hidden_other_truths"])
        truths[anchor] = episode["answer"]
        facts = [d for d in composite.BASE_DOORS if d != anchor and rng.random() < 0.5]
        if kind == "C":
            facts.append(anchor)
        board = [chains.same_text(rng, a, b) for a, b in links] + [chains.fact_text(rng, d, truths[d]) for d in facts]
        rng.shuffle(board)
        query = anchor if kind == "T" else chain_of[anchor][-1]
        episode["board"] = board
        episode["question"] = chains.QUESTION.format(D=query)
        episode["composite"] = {"anchor": anchor, "query": query, "length": 0 if kind == "T" else length,
                                "type": kind, "board_facts": facts}
    return session


def generate(part: str, seed: int) -> list[dict]:
    session_seed = seed + (400 if part == "test" else 0)  # fresh v2-H test sessions (+300 was seen by v1)
    return [add_routing_chains(s, part, s["session_id"]) for s in hedged.generate_part(part, session_seed, SESSIONS[part])]


@torch.no_grad()
def run_all_subsets(model: CompositeSystem, data: dict) -> dict:
    """Per episode: router features [N,3,4], and score of every subset of asked base doors [N,8]."""
    door_logits = flat(model.trust_logits(data))  # [N,3,3]
    board, query, target = flat(data["board"]), flat(data["query"]), flat(data["truths"])
    q = door_logits.softmax(-1)
    sure = q.amax(-1, keepdim=True) > 0.5
    crisp = torch.where(sure, nn.functional.one_hot(q.argmax(-1), 3).float(), torch.full_like(q, 1 / 3))
    messages = torch.tanh(model.adapter(torch.stack([crisp, crisp.amax(-1, keepdim=True).expand_as(crisp)], -1)))
    count = board.shape[0]
    rows = torch.arange(count, device=board.device)
    results = []
    for subset in SUBSETS:
        extra = messages.new_zeros(count, len(chains.DOORS), 3, messages.shape[-1])
        mask = torch.zeros(count, len(chains.DOORS), dtype=torch.bool, device=board.device)
        for d in subset:
            extra[:, d] = messages[:, d]
            mask[:, d] = True
        logits, _, strength, doors = model.thinker.think(board, query, STEPS, with_links=True,
                                                         extra_facts=extra if subset else None,
                                                         extra_mask=mask if subset else None)
        stop = fixed_point_stop(model.thinker.last_changes)
        answer = logits[stop, rows]
        results.append(scores(answer, target) - COST * len(subset))
        if not subset:
            first = (strength, doors[stop, rows].softmax(-1), answer.softmax(-1))
    strength, door_beliefs, answer_p = first
    # Which base doors are linked to the queried door in the thinker's own graph (0/1 links).
    linked = torch.zeros(count, 3, device=board.device)
    adjacency = torch.zeros(count, len(chains.DOORS), len(chains.DOORS), device=board.device)
    adjacency[:, model.thinker.pair_rows, model.thinker.pair_cols] = (strength > 0.5).float()
    adjacency = ((adjacency + adjacency.transpose(1, 2)) > 0).float() + torch.eye(len(chains.DOORS), device=board.device)
    reach = nn.functional.one_hot(query, len(chains.DOORS)).float()
    for _ in range(len(chains.DOORS)):
        reach = ((reach[:, None, :] @ adjacency).squeeze(1) > 0).float()
    linked = reach[:, :3]
    is_query = (query[:, None] == torch.arange(3, device=board.device)).float()  # used only by plans
    features = torch.stack([linked, door_beliefs[:, :3].amax(-1), answer_p.amax(-1, keepdim=True).expand(-1, 3)], -1)
    return {"features": features, "is_query": is_query, "subset_scores": torch.stack(results, 1), "types": flat(data["types"]),
            "anchor": flat(data["anchor"])}


class Router(nn.Module):
    """Per base door: features -> predicted gain of asking about it (shared over doors)."""

    def __init__(self, width: int = 32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(3, width), nn.Tanh(), nn.Linear(width, width), nn.Tanh(), nn.Linear(width, 1))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)


class WholeRouter(nn.Module):
    """Picks one plan from the same features laid side by side: ask nobody / the queried door /
    all three base doors."""

    def __init__(self, width: int = 32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(9, width), nn.Tanh(), nn.Linear(width, width), nn.Tanh(), nn.Linear(width, 3))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features.flatten(1))


def subset_index(asked: torch.Tensor) -> torch.Tensor:
    """asked [N,3] bool -> index into SUBSETS."""
    lookup = {s: i for i, s in enumerate(SUBSETS)}
    return torch.tensor([lookup[tuple(int(d) for d in torch.nonzero(row).flatten())] for row in asked.cpu()],
                        device=asked.device)


def plan_subsets(is_query: torch.Tensor) -> torch.Tensor:
    """Subset index of each whole-router plan per episode [N,3]: none, the queried door (if a base door), all."""
    is_query = is_query > 0.5
    query_plan = subset_index(is_query)
    return torch.stack([torch.zeros_like(query_plan), query_plan, torch.full_like(query_plan, len(SUBSETS) - 1)], 1)


def fit(model: nn.Module, x: torch.Tensor, y: torch.Tensor, valid_score, seed: int, epochs: int = 300) -> tuple[nn.Module, int]:
    torch.manual_seed(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    best = (-9.0, 0, None)
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(x), device=x.device)
        for start in range(0, len(order), 1024):
            batch = order[start:start + 1024]
            loss = nn.functional.mse_loss(model(x[batch]), y[batch])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            value = valid_score(model)
        if value > best[0]:
            best = (value, epoch, {k: v.clone() for k, v in model.state_dict().items()})
    model.load_state_dict(best[2])
    return model.eval(), best[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Router that calls parts for what is missing (v7)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    device = torch.device("cuda" if args.device != "cpu" and torch.cuda.is_available() else "cpu")
    started = time.perf_counter()
    trust, thinker, rows, chain_table = load_parts(args.seed, "grown", device)
    model = CompositeSystem(trust, thinker).to(device)
    model.load_state_dict(torch.load(CHECKPOINT_DIR / f"world-v4_grown_seed-{args.seed}.pt")["state"])
    model.eval()

    runs = {}
    for part in ("train", "validation", "test"):
        sessions = generate(part, args.seed)
        data = {k: v.to(device) for k, v in session_tensors(sessions, rows, chain_table).items()}
        data["types"] = torch.tensor([[TYPES.index(e["composite"]["type"]) for e in s["episodes"]] for s in sessions],
                                     device=device)
        runs[part] = run_all_subsets(model, data)
        print(f"{part}: {len(runs[part]['types'])} episodes ({time.perf_counter() - started:.0f}s)", flush=True)

    def gains(run):  # gain of asking about door d alone, without its cost
        return run["subset_scores"][:, 1:4] + COST - run["subset_scores"][:, :1]

    def router_score(router, run):
        asked = router(run["features"]) > COST
        return run["subset_scores"].gather(1, subset_index(asked)[:, None]).squeeze(1)

    def whole_score(whole, run):
        plans = plan_subsets(run["is_query"])
        pick = plans.gather(1, whole(run["features"]).argmax(-1, keepdim=True))
        return run["subset_scores"].gather(1, pick).squeeze(1)

    train, validation, test = runs["train"], runs["validation"], runs["test"]
    router, router_epoch = fit(Router().to(device), train["features"].flatten(0, 1), gains(train).flatten(),
                               lambda m: router_score(m, validation).mean().item(), args.seed)
    plan_values = train["subset_scores"].gather(1, plan_subsets(train["is_query"]))
    whole, whole_epoch = fit(WholeRouter().to(device), train["features"], plan_values,
                             lambda m: whole_score(m, validation).mean().item(), args.seed)

    with torch.no_grad():
        ours = router_score(router, test)
        asked = router(test["features"]) > COST
        theirs = whole_score(whole, test)
    anchor_only = torch.zeros_like(asked)
    anchor_only[torch.arange(len(asked), device=device), test["anchor"]] = True
    ideal_asked = anchor_only & (test["types"] != TYPES.index("C"))[:, None]
    ideal = test["subset_scores"].gather(1, subset_index(ideal_asked)[:, None]).squeeze(1)
    reference = {"never_ask": test["subset_scores"][:, 0], "ask_all": test["subset_scores"][:, -1],
                 "hindsight_best": test["subset_scores"].max(1).values}

    result = {"world": "v7", "experiment": "router", "seed": args.seed, "cost": COST,
              "router_parameters": sum(p.numel() for p in router.parameters()),
              "selected_epochs": {"router": router_epoch, "whole": whole_epoch}, "by_type": {}}
    for i, kind in enumerate(TYPES):
        sel = test["types"] == i
        not_anchor = asked[sel].clone()
        not_anchor[torch.arange(int(sel.sum()), device=device), test["anchor"][sel]] = False
        result["by_type"][kind] = {
            "router": round(ours[sel].mean().item(), 4), "whole_router": round(theirs[sel].mean().item(), 4),
            "ideal_routing": round(ideal[sel].mean().item(), 4),
            **{k: round(v[sel].mean().item(), 4) for k, v in reference.items()},
            "router_asks_per_episode": round(asked[sel].float().sum(1).mean().item(), 3),
            "router_asks_anchor": round(asked[sel][torch.arange(int(sel.sum()), device=device), test["anchor"][sel]].float().mean().item(), 3),
            "router_asks_other_doors": round(not_anchor.float().sum(1).mean().item(), 3),
            "episodes": int(sel.sum())}
    result["seconds"] = round(time.perf_counter() - started, 1)
    torch.save({"router": router.state_dict(), "whole": whole.state_dict()}, CHECKPOINT_DIR / f"world-v7_router_seed-{args.seed}.pt")
    out = RESULTS_DIR / f"world-v7_router_seed-{args.seed}.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
