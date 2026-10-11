"""Step 3: the concept learner beside the search satellite, answering only after a held-out demonstration check
(design/build-from-scratch-2026-10-10.md section 12).

  S  the search satellite alone (mk1v2/arc_search.py, 10 s)
  C  the search's answers when it has any; otherwise the learner, only when it passes the check:
     one demonstration is held out (never trained on, never in the context), the learner adapts on the others,
     and must reproduce the held-out output exactly (vote over rotations / flips / colour relabellings)

Data: ARC-AGI-1 training = development exam (even sorted index = tune, odd = check); ARC-AGI-1 evaluation (spent as
MK1 v2's ARC final) = pretraining data together with all 28 synthetic families.

python -m cognitive_lab.concepts.arc_learner search            # S on all 400 training tasks, cached
python -m cognitive_lab.concepts.arc_learner pretrain --steps 20000
python -m cognitive_lab.concepts.arc_learner run --half tune --ttt-steps 300
python -m cognitive_lab.concepts.arc_learner run --half check --final
"""

import argparse
import json
import math
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from cognitive_lab.concepts.generator import FAMILIES, task
from cognitive_lab.concepts.learner import (CKPT_DIR, DEVICE, OUTSIDE, RESULT_DIR, SIDE, UNKNOWN, Learner, exact,
                                            load, loss_of, pad)

ARC = Path("data/arc/ARC-AGI-master/data")
SEARCH_CACHE = RESULT_DIR / "arc-search-training.json"


def arc_tasks(folder: str) -> list[tuple[str, dict]]:
    return [(f.stem, json.loads(f.read_text(encoding="utf-8"))) for f in sorted((ARC / folder).glob("*.json"))]


def half(name: str) -> list[tuple[str, dict]]:
    items = arc_tasks("training")
    return items[0::2] if name == "tune" else items[1::2]


def fits(t: dict) -> bool:
    return all(len(g) <= SIDE and len(g[0]) <= SIDE
               for p in t["train"] + t["test"] for g in (p["input"], p.get("output", [[0]])))


# ----------------------------------------------------------------------------- transforms
def dihedral(g: np.ndarray, k: int) -> np.ndarray:
    g = np.rot90(g, k % 4)
    return np.fliplr(g) if k >= 4 else g


def undo_dihedral(g: np.ndarray, k: int) -> np.ndarray:
    if k >= 4:
        g = np.fliplr(g)
    return np.rot90(g, -(k % 4))


class View:
    """One rotation/flip plus one relabelling of colours 1-9, applied to every grid of a task."""

    def __init__(self, rng: random.Random, plain: bool = False):
        self.k = 0 if plain else rng.randrange(8)
        perm = list(range(1, 10))
        if not plain:
            rng.shuffle(perm)
        self.fwd = np.arange(10)
        self.fwd[1:] = perm
        self.back = np.argsort(self.fwd)

    def apply(self, g) -> np.ndarray:
        return self.fwd[dihedral(np.asarray(g), self.k)]

    def undo(self, g: np.ndarray) -> np.ndarray:
        return undo_dihedral(self.back[g], self.k)


def encode(context: list, query_in, query_out=None, view: View | None = None):
    """Up to 3 context pairs (missing slots are all OUTSIDE), the query input, the output slot; target if given."""
    f = view.apply if view else np.asarray
    grids = []
    for i in range(3):
        if i < len(context):
            grids += [pad(f(context[i][0])), pad(f(context[i][1]))]
        else:
            grids += [np.full((SIDE, SIDE), OUTSIDE), np.full((SIDE, SIDE), OUTSIDE)]
    grids += [pad(f(query_in)), np.full((SIDE, SIDE), UNKNOWN)]
    y = pad(f(query_out)) if query_out is not None else None
    return np.stack(grids).astype(np.int64), y


def decode(cells: np.ndarray):
    """The output grid in a predicted 14 x 14 slot: the top-left block of non-OUTSIDE cells; None if ragged."""
    h = int(np.argmax(cells[:, 0] == OUTSIDE)) if (cells[:, 0] == OUTSIDE).any() else SIDE
    w = int(np.argmax(cells[0] == OUTSIDE)) if (cells[0] == OUTSIDE).any() else SIDE
    if h == 0 or w == 0:
        return None
    g = cells[:h, :w]
    if (g == OUTSIDE).any() or (cells[h:, :] != OUTSIDE).any() or (cells[:, w:] != OUTSIDE).any():
        return None
    return g


def sample(pairs: list, rng: random.Random, drop: float = 0.0):
    """A training item from a list of (input, output) pairs: one target, up to 3 others as context."""
    order = list(range(len(pairs)))
    rng.shuffle(order)
    target, rest = pairs[order[0]], [pairs[i] for i in order[1:4]]
    if len(rest) == 3 and rng.random() < drop:
        rest = rest[:2]
    return encode(rest, target[0], target[1], View(rng))


# ----------------------------------------------------------------------------- pretraining v2
class Stream(torch.utils.data.IterableDataset):
    """70 % synthetic families (all 28), 30 % ARC evaluation tasks that fit; rotations, flips, colours, empty slots."""

    def __init__(self, bs: int):
        self.bs = bs

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        wid = info.id if info else 0
        rng = random.Random(f"v2-stream-{wid}")
        arc = [[(p["input"], p["output"]) for p in t["train"] + t["test"]]
               for _, t in arc_tasks("evaluation") if fits(t)]
        names = list(FAMILIES)
        i = 0
        while True:
            xs, ys = [], []
            while len(xs) < self.bs:
                if rng.random() < 0.3:
                    pairs = rng.choice(arc)
                else:
                    t = task(rng.choice(names), f"v2-{wid}-{i}")
                    i += 1
                    if not t:
                        continue
                    pairs = [(p["input"], p["output"]) for p in t["train"] + t["test"]]
                x, y = sample(pairs, rng, drop=0.2)
                xs.append(x)
                ys.append(y)
            yield torch.from_numpy(np.stack(xs)), torch.from_numpy(np.stack(ys))


def pretrain(steps: int, bs: int = 32, lr: float = 2e-4, start: str = "concept-learner", out: str = "concept-learner-v2"):
    model = load(start)  # continue from the step-2 learner
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    warm = 500
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps))))
    loader = torch.utils.data.DataLoader(Stream(bs), batch_size=None, num_workers=4, persistent_workers=True,
                                         prefetch_factor=4)
    val = {f: [t for t in (task(f, f"val-{i}") for i in range(30)) if t] for f in FAMILIES}
    log, t0, run = [], time.time(), 0.0
    for step, (x, y) in enumerate(loader, 1):
        loss = loss_of(model, x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        run = 0.98 * run + 0.02 * loss.item() if step > 1 else loss.item()
        if step % 1000 == 0 or step == steps:
            mean = float(np.mean([np.mean(exact(model, v)) for v in val.values()]))
            log.append({"step": step, "loss": run, "val_exact_families": mean, "minutes": (time.time() - t0) / 60})
            print(f"step {step:6d} loss {run:.4f} families exact {mean:.3f} {(time.time() - t0) / 60:.1f}m", flush=True)
            torch.save({"model": model.state_dict(), "config": {"d": 192, "layers": 6, "heads": 6}, "step": step},
                       CKPT_DIR / f"{out}.pt")
            (RESULT_DIR / f"{out}-log.json").write_text(json.dumps(log, indent=1))
        if step >= steps:
            break


# ----------------------------------------------------------------------------- per task: adapt, check, answer
def adapt(model: Learner, pairs: list, steps: int, rng: random.Random, lr: float = 1e-4, bs: int = 16):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    model.train()
    for _ in range(steps):
        xs, ys = zip(*(sample(pairs, rng) for _ in range(bs)))
        loss = loss_of(model, torch.from_numpy(np.stack(xs)), torch.from_numpy(np.stack(ys)))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()


@torch.no_grad()
def predict(model: Learner, context: list, query_in, rng: random.Random, views: int = 16) -> Counter:
    """Votes over views (the first is the plain view); each view also draws which 3 context pairs it sees."""
    model.eval()
    vs = [View(rng, plain=(i == 0)) for i in range(views)]
    xs = []
    for v in vs:
        ctx = rng.sample(context, min(3, len(context)))
        xs.append(encode(ctx, query_in, view=v)[0])
    with torch.autocast("cuda", dtype=torch.bfloat16):
        cells = model(torch.from_numpy(np.stack(xs)).to(DEVICE)).argmax(-1).cpu().numpy()
    votes = Counter()
    for v, c in zip(vs, cells):
        g = decode(c)
        if g is not None:
            votes[json.dumps(v.undo(g).tolist())] += 1
    model.train()
    return votes


def learner_answer(t: dict, ttt_steps: int, ckpt: str, seed: str) -> dict:
    """The check and, when it passes, up to two attempts per test input."""
    pairs = [(p["input"], p["output"]) for p in t["train"]]
    if len(pairs) < 3 or not fits(t):
        return {"tried": False}
    rng = random.Random(seed)
    torch.manual_seed(0)
    held, rest = pairs[-1], pairs[:-1]
    model = load(ckpt)
    adapt(model, rest, ttt_steps, rng)
    votes = predict(model, rest, held[0], rng)
    top = votes.most_common(1)
    passed = bool(top) and json.loads(top[0][0]) == held[1]
    share = top[0][1] / sum(votes.values()) if top else 0.0
    out = {"tried": True, "check_passed": passed, "check_share": share, "attempts": []}
    if passed:
        for p in t["test"]:
            tv = predict(model, pairs, p["input"], rng)
            out["attempts"].append([json.loads(g) for g, _ in tv.most_common(2)])
            out["test_share"] = out.get("test_share", []) + [tv.most_common(1)[0][1] / sum(tv.values()) if tv else 0.0]
    return out


# ----------------------------------------------------------------------------- systems and scoring
def search_all(budget: float = 10.0):
    from cognitive_lab.mk1v2.arc_search import solve
    cache = json.loads(SEARCH_CACHE.read_text()) if SEARCH_CACHE.exists() else {}
    for name, t in arc_tasks("training"):
        if name in cache:
            continue
        t0 = time.time()
        ans = solve(t, budget=budget)
        cache[name] = {"attempts": [[o[i] for _, o in ans] for i in range(len(t["test"]))],
                       "programs": [" > ".join(p) if isinstance(p, (list, tuple)) else str(p) for p, _ in ans],
                       "seconds": time.time() - t0}
        SEARCH_CACHE.parent.mkdir(parents=True, exist_ok=True)
        SEARCH_CACHE.write_text(json.dumps(cache))
        print(f"{name} answered {bool(ans)} {time.time() - t0:.1f}s", flush=True)
    return cache


def score(t: dict, attempts: list) -> float:
    """ARC: per test input, solved if one of the first two attempts is exact; mean over test inputs."""
    if not attempts:
        return 0.0
    return float(np.mean([any(a == p["output"] for a in (attempts[i] if i < len(attempts) else [])[:2])
                          for i, p in enumerate(t["test"])]))


def run(half_name: str, ttt_steps: int, ckpt: str = "concept-learner-v2", tag: str = ""):
    search = json.loads(SEARCH_CACHE.read_text())
    out_path = RESULT_DIR / f"step3-{half_name}-ttt{ttt_steps}{tag}.jsonl"
    done = {}
    if out_path.exists():
        done = {r["task"]: r for r in map(json.loads, out_path.read_text().splitlines())}
    rows = []
    for name, t in half(half_name):
        s = search[name]
        s_answered = any(s["attempts"][i] for i in range(len(t["test"])))
        if name in done:
            r = done[name]
        else:
            lr = {"tried": False} if s_answered else learner_answer(t, ttt_steps, ckpt, seed=name)
            r = {"task": name, "search_answered": s_answered, "search_score": score(t, s["attempts"]), "learner": lr,
                 "learner_score": score(t, lr.get("attempts", []))}
            with out_path.open("a") as fh:
                fh.write(json.dumps(r) + "\n")
            if lr.get("tried"):
                print(f"{name} check {lr['check_passed']} share {lr['check_share']:.2f} score {r['learner_score']:.2f}",
                      flush=True)
        rows.append(r)
    return rows


def report(rows: list, min_share: float = 0.0, reps: int = 10000):
    """S, C, answered precision, paired bootstrap of C - S over tasks."""
    s = np.array([r["search_score"] for r in rows])
    use = [bool(r["learner"].get("check_passed")) and r["learner"]["check_share"] >= min_share for r in rows]
    c = np.array([r["search_score"] if r["search_answered"] else (r["learner_score"] if u else 0.0)
                  for r, u in zip(rows, use)])
    answered_s = [r["search_answered"] for r in rows]
    answered_c = [a or u for a, u in zip(answered_s, use)]
    prec = lambda sc, ans: float(np.mean([x > 0 for x, a in zip(sc, ans) if a])) if any(ans) else float("nan")
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(rows), (reps, len(rows)))
    boots = (c - s)[idx].mean(1)
    tried = [r for r in rows if r["learner"].get("tried")]
    print(f"tasks {len(rows)}  learner tried {len(tried)}  check passed {sum(use)}  "
          f"right after pass {sum(r['learner_score'] > 0 for r, u in zip(rows, use) if u)}")
    print(f"S {s.mean():.3f} (answered {sum(answered_s)}, right {prec(s, answered_s):.3f})   "
          f"C {c.mean():.3f} (answered {sum(answered_c)}, right {prec(c, answered_c):.3f})")
    print(f"C - S {(c - s).mean():+.3f} [{np.percentile(boots, 2.5):+.3f}, {np.percentile(boots, 97.5):+.3f}]")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["search", "pretrain", "run", "report"])
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--half", default="tune", choices=["tune", "check"])
    p.add_argument("--ttt-steps", type=int, default=300)
    p.add_argument("--min-share", type=float, default=0.3)  # frozen on the tune half
    p.add_argument("--final", action="store_true")
    a = p.parse_args()
    if a.half == "check" and a.cmd in ("run", "report") and not a.final:
        raise SystemExit("the check half is the one step-3 verdict: pass --final once settings are frozen")
    if a.cmd == "search":
        search_all()
    elif a.cmd == "pretrain":
        pretrain(a.steps)
    else:
        rows = run(a.half, a.ttt_steps) if a.cmd == "run" else [
            json.loads(l) for l in (RESULT_DIR / f"step3-{a.half}-ttt{a.ttt_steps}.jsonl").read_text().splitlines()]
        report(rows, a.min_share)


if __name__ == "__main__":
    main()
