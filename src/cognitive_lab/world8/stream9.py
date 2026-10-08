"""Rule world v9: two unannounced new relations; the system decides when to grow and finds the shape
(design/world-v8-growth.md, part v9).

50 blocks of 200 episodes, the answer revealed after each:
  1-10 same | 11-20 + new phrasing | 21-35 + next (50%) | 36-70 same, new phrasing, next, swap (25% each)
Every policy grows when the failure rate (wrong, or 모름 although there was an answer) exceeds 0.05 in
two blocks in a row. From the two triggering blocks on, it keeps answering and collecting until 600
failed episodes (at most 10 blocks), then studies a "wrong-answer notebook": every failed episode, as
many it got right, and 300 replayed from before the change (1/7 held out for validation), and installs
a new branch on the frozen thinker:
  never    never grows
  fixed    always the shape a person gave in v8 (one step along the link, either reading)
  search   (ours) tries all 6 shapes, and "study the newest branch more", and keeps the best
  oracle   the true shape of the relation that appeared (next before block 36, swap after)
(Version 1 gave fixed/oracle only one reading of "next" and had no competition between branches;
its partial results are in artifacts/results/archive/v9-first-run.)

python -m cognitive_lab.world8.stream9 --seed 45 --policy search
"""

import argparse
import json
import random
import time

import torch

from cognitive_lab.world2.integrated import CHECKPOINT_DIR, RESULTS_DIR
from cognitive_lab.world3.thinker import episode_tensors, fixed_point_stop
from cognitive_lab.world8.chains import episode
from cognitive_lab.world8.language import v9_tables
from cognitive_lab.world8.multibranch import MultiBranchThinker, restore, search_shape, snapshot, train_last_branch

BLOCKS, BLOCK_SIZE, THRESHOLD, RUN, REPLAY = 70, 200, 0.05, 2, 300
FAILED_NEEDED, MAX_STUDY_BLOCKS = 600, 10
NEXT_SHAPE, NEXT_BACK_SHAPE, SWAP_SHAPE = [2, 0, 1], [1, 2, 0], [2, 1, 0]


def family_mix(block: int) -> tuple[str, ...]:
    if block <= 10:
        return ("same",)
    if block <= 20:
        return ("same", "same-new")
    if block <= 35:
        return ("same", "next")
    return ("same", "same-new", "next", "swap")


def stream(seed: int) -> list[list[dict]]:
    rng = random.Random(f"world-v9|stream|{seed}")
    blocks = []
    for b in range(1, BLOCKS + 1):
        mix = family_mix(b)
        blocks.append([episode(rng, mix[n % len(mix)], rng.choice((1, 2, 3)), rng.random() >= 0.2, f"s{seed}-b{b}-{n}")
                       for n in range(BLOCK_SIZE)])
    return blocks


@torch.no_grad()
def answer(model, episodes, table, device) -> list[dict]:
    out = []
    for start in range(0, len(episodes), 50):
        chunk = episodes[start:start + 50]
        index, query, target, _ = (t.to(device) for t in episode_tensors(chunk, table))
        logits, _ = model.think(index, query, 16)
        stop = fixed_point_stop(model.last_changes)
        q = logits[stop, torch.arange(len(query), device=device)].softmax(-1)
        best, choice = q.max(-1)
        answered = best > 0.5
        right = torch.where(target >= 0, choice == target, torch.zeros_like(answered))
        score = torch.where(answered, torch.where(right, 1.0, -1.0), torch.where(target >= 0, 0.0, 1.0))
        failed = (answered & ~right) | (~answered & (target >= 0))
        out += [{"family": e["family"], "score": s, "failed": f} for e, s, f in zip(chunk, score.tolist(), failed.tolist())]
    return out


def fresh_model(seed: int, device: torch.device):
    table, relation = v9_tables(device)
    model = MultiBranchThinker(table["features"].to(device), relation.to(device), table["head"]).to(device)
    missing, unexpected = model.load_state_dict(torch.load(CHECKPOINT_DIR / f"world-v3_thinker-grown_seed-{seed}.pt")["state"],
                                                strict=False)
    if missing or unexpected:
        raise RuntimeError(f"unexpected load: {missing} {unexpected}")
    for p in model.parameters():
        p.requires_grad_(False)
    return model.eval(), table


def run(policy: str, blocks, seed: int, device: torch.device) -> dict:
    model, table = fresh_model(seed, device)
    tensors = lambda eps: tuple(t.to(device) for t in episode_tensors(eps, table))
    log, growths, signal_run, studying, buffer, decided_at = [], [], 0, False, [], None
    recent = []
    for b, episodes in enumerate(blocks, start=1):
        results = answer(model, episodes, table, device)
        marked = [(e, r["failed"]) for e, r in zip(episodes, results)]
        failure = sum(r["failed"] for r in results) / len(results)
        by_family = {}
        for r in results:
            by_family.setdefault(r["family"], []).append(r["score"])
        entry = {"block": b, "score": round(sum(r["score"] for r in results) / len(results), 4), "failure": round(failure, 4),
                 "by_family": {f: round(sum(v) / len(v), 4) for f, v in by_family.items()}, "branches": len(model.branches)}
        if studying:
            buffer += marked
        elif policy != "never":
            signal_run = signal_run + 1 if failure > THRESHOLD else 0
            if signal_run >= RUN:
                decided_at, signal_run, studying = b, 0, True
                buffer = recent + marked
                entry["decided_to_grow"] = True
        recent = marked
        collected = len(buffer) // BLOCK_SIZE
        if studying and (sum(f for _, f in buffer) >= FAILED_NEEDED or collected >= MAX_STUDY_BLOCKS):
            # Study notebook: every failed episode, as many episodes it got right (so the new branch
            # does not break what works), and replay from before the change.
            rng = random.Random(f"study|{seed}|{b}")
            failed = [e for e, f in buffer if f]
            fine = [e for e, f in buffer if not f]
            past = [e for blk in blocks[:decided_at - RUN] for e in blk]
            material = failed + rng.sample(fine, min(len(failed), len(fine))) + rng.sample(past, min(REPLAY, len(past)))
            rng.shuffle(material)
            cut = len(material) // 7
            data, validation = tensors(material[cut:]), tensors(material[:cut])
            started = time.perf_counter()
            growth = {"decided_at": decided_at, "installed_before": b + 1, "blocks_collected": collected,
                      "failed_studied": len(failed), "material": len(material),
                      "families_studied": {f: sum(e["family"] == f for e in material) for f in sorted({e["family"] for e in material})}}
            if policy == "search":
                found = search_shape(model, data, validation, seed)
                growth.update(chosen=found["chosen"], trials=found["trials"])
            else:
                # A person gives the shape. "Next" is the same rule whichever way a link is read
                # (one step forward, or one step back with the link reversed); the branch learns the
                # reversed reading far more reliably, so both readings are tried and the better kept.
                rule = "next" if policy == "fixed" or decided_at <= 35 else "swap"
                shapes = [NEXT_SHAPE, NEXT_BACK_SHAPE] if rule == "next" else [SWAP_SHAPE]
                best = None
                gates = [{k: v.clone() for k, v in old.relation_score.state_dict().items()} for old in model.branches]
                if len(model.branches) and model.branches[-1].shape.argmax(1).tolist() in shapes:
                    before = snapshot(model)
                    score, epoch = train_last_branch(model, data, validation, seed)
                    best = (score, epoch, "refine", snapshot(model))
                    restore(model, before)
                for shape in shapes:
                    for old, gate in zip(model.branches, gates):
                        old.relation_score.load_state_dict(gate)
                    model.add_branch(torch.eye(3)[shape])
                    score, epoch = train_last_branch(model, data, validation, seed)
                    state = snapshot(model)
                    del model.branches[-1]
                    if best is None or score > best[0]:
                        best = (score, epoch, shape, state)
                for old, gate in zip(model.branches, gates):
                    old.relation_score.load_state_dict(gate)
                if best[2] != "refine":
                    model.add_branch(torch.eye(3)[best[2]])
                restore(model, best[3])
                growth.update(chosen=best[2], validation=round(best[0], 4), epoch=best[1])
            growth["seconds"] = round(time.perf_counter() - started, 1)
            growths.append(growth)
            print(f"  {policy}: grew at block {decided_at} -> shape {growth['chosen']} ({growth['seconds']}s)", flush=True)
            studying, buffer = False, []
        log.append(entry)
    # Keep the grown branches (shape + weights), so the grown thinker can be re-measured and reused.
    path = CHECKPOINT_DIR / f"world-v9_branches_{policy}_seed-{seed}.pt"
    torch.save({"thinker_seed": seed, "branches": [{"shape": b.shape.argmax(1).tolist(), "state": b.state_dict()}
                                                   for b in model.branches]}, path)
    return {"blocks": log, "growths": growths, "branches_checkpoint": path.name}


def main() -> None:
    parser = argparse.ArgumentParser(description="Two new relations: when to grow and which shape (v9)")
    parser.add_argument("--seed", type=int, default=45, help="stage-3 thinker seed")
    parser.add_argument("--policy", choices=("never", "fixed", "search", "oracle"), required=True)
    parser.add_argument("--stream-offset", type=int, default=3000,
                        help="2000 was used while revising the design (results archived)")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    started = time.perf_counter()
    result = run(args.policy, stream(args.seed + args.stream_offset), args.seed, device)
    result.update(seed=args.seed, stream_seed=args.seed + args.stream_offset, policy=args.policy,
                  seconds=round(time.perf_counter() - started, 1))
    (RESULTS_DIR / f"world-v9_stream_{args.policy}_seed-{args.seed}.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    late = [e for e in result["blocks"] if e["block"] >= BLOCKS - 4]
    print(json.dumps({"policy": args.policy, "growths": [(g["decided_at"], g["chosen"]) for g in result["growths"]],
                      "late_score": round(sum(e["score"] for e in late) / len(late), 4),
                      "late_by_family": {f: round(sum(e["by_family"][f] for e in late) / len(late), 4)
                                         for f in late[0]["by_family"]}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
