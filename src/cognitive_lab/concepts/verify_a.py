"""Path A: a stricter check before the learner may answer, judged once on a fresh development exam
(design/next-after-c-2026-10-11.md section 9).

  rotation check  hold out each of the last min(4, n) demonstrations in turn; for each, adapt a fresh copy of the
                  learner on the others and require the top vote to reproduce the held-out output exactly
  answer          only when every rotation passes: adapt a fresh copy on all demonstrations, vote on each test input,
                  and answer only if the smallest top-vote share over the test inputs is >= tau
  tau             chosen on ARC-AGI-1 training (all 400, already spent as step 3's development exam), then frozen

Fresh development exam: ARC-AGI-2 training minus every task that shares an id or an input/output pair with
ARC-AGI-1 (training or evaluation) or ConceptARC: 233 tasks (data/arc2/training-dedup.json).

python -m cognitive_lab.concepts.verify_a run --exam arc1        # learner on ARC-AGI-1 training (tuning data)
python -m cognitive_lab.concepts.verify_a tune                   # pick tau by the pre-registered rule
python -m cognitive_lab.concepts.verify_a search --exam arc2     # S on the fresh exam, cached
python -m cognitive_lab.concepts.verify_a run --exam arc2 --final
"""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from cognitive_lab.concepts.arc_learner import SEARCH_CACHE, adapt, arc_tasks, fits, predict, score
from cognitive_lab.concepts.learner import RESULT_DIR, load

ARC2 = Path("data/arc2/ARC-AGI-2-main/data/training")
ARC2_KEEP = Path("data/arc2/training-dedup.json")
CKPT = "concept-learner-v2"
TTT_STEPS = 300  # frozen in step 3
TAUS = [0.0, 0.3, 0.5, 0.7, 0.9]
MAX_ROTATIONS = 4


def exam(name: str) -> list[tuple[str, dict]]:
    if name == "arc1":
        return arc_tasks("training")
    keep = json.loads(ARC2_KEEP.read_text())["kept"]
    return [(k, json.loads((ARC2 / f"{k}.json").read_text(encoding="utf-8"))) for k in keep]


def search_cache(name: str) -> Path:
    return SEARCH_CACHE if name == "arc1" else RESULT_DIR / "arc-search-arc2-training.json"


def search_all(name: str, budget: float = 10.0):
    from cognitive_lab.mk1v2.arc_search import solve
    path = search_cache(name)
    cache = json.loads(path.read_text()) if path.exists() else {}
    for k, t in exam(name):
        if k in cache:
            continue
        t0 = time.time()
        ans = solve(t, budget=budget)
        cache[k] = {"attempts": [[o[i] for _, o in ans] for i in range(len(t["test"]))],
                    "programs": [" > ".join(p) if isinstance(p, (list, tuple)) else str(p) for p, _ in ans],
                    "seconds": time.time() - t0}
        path.write_text(json.dumps(cache))
        print(f"{k} answered {bool(ans)} {time.time() - t0:.1f}s", flush=True)


def learner_answer(t: dict, seed: str) -> dict:
    """Rotation check (stops at the first failure); when it passes, attempts and top-vote shares per test input."""
    pairs = [(p["input"], p["output"]) for p in t["train"]]
    if len(pairs) < 3 or not fits(t):
        return {"tried": False}
    rng = random.Random(seed)
    rotations = []
    for j in range(len(pairs) - 1, len(pairs) - 1 - min(MAX_ROTATIONS, len(pairs)), -1):
        torch.manual_seed(j)
        held, rest = pairs[j], pairs[:j] + pairs[j + 1:]
        model = load(CKPT)
        adapt(model, rest, TTT_STEPS, rng)
        votes = predict(model, rest, held[0], rng)
        top = votes.most_common(1)
        ok = bool(top) and json.loads(top[0][0]) == held[1]
        rotations.append({"held": j, "passed": ok, "share": top[0][1] / sum(votes.values()) if top else 0.0})
        if not ok:
            break
    out = {"tried": True, "rotations": rotations, "check_passed": all(r["passed"] for r in rotations),
           "attempts": [], "test_share": []}
    if out["check_passed"]:
        torch.manual_seed(len(pairs))
        model = load(CKPT)
        adapt(model, pairs, TTT_STEPS, rng)
        for p in t["test"]:
            tv = predict(model, pairs, p["input"], rng)
            out["attempts"].append([json.loads(g) for g, _ in tv.most_common(2)])
            out["test_share"].append(tv.most_common(1)[0][1] / sum(tv.values()) if tv else 0.0)
    return out


def run(name: str) -> list[dict]:
    search = json.loads(search_cache(name).read_text())
    out_path = RESULT_DIR / f"verify-a-{name}.jsonl"
    done = {}
    if out_path.exists():
        done = {r["task"]: r for r in map(json.loads, out_path.read_text().splitlines())}
    rows = []
    for k, t in exam(name):
        if k in done:
            rows.append(done[k])
            continue
        s = search[k]
        s_answered = any(s["attempts"][i] for i in range(len(t["test"])))
        lr = {"tried": False} if s_answered else learner_answer(t, seed=k)
        r = {"task": k, "search_answered": s_answered, "search_score": score(t, s["attempts"]), "learner": lr,
             "learner_score": score(t, lr.get("attempts", []))}
        with out_path.open("a") as fh:
            fh.write(json.dumps(r) + "\n")
        if lr.get("tried"):
            print(f"{k} rotations {[x['passed'] for x in lr['rotations']]} test {lr['test_share']} "
                  f"score {r['learner_score']:.2f}", flush=True)
        rows.append(r)
    return rows


def systems(rows: list, tau: float):
    s = np.array([r["search_score"] for r in rows])
    use = [bool(r["learner"].get("check_passed")) and min(r["learner"]["test_share"]) >= tau for r in rows]
    c = np.array([r["search_score"] if r["search_answered"] else (r["learner_score"] if u else 0.0)
                  for r, u in zip(rows, use)])
    answered = [r["search_answered"] or u for r, u in zip(rows, use)]
    right = [x > 0 for x, a in zip(c, answered) if a]
    return s, c, use, (float(np.mean(right)) if right else float("nan")), sum(answered)


def report(rows: list, tau: float, reps: int = 10000):
    s, c, use, prec, n_ans = systems(rows, tau)
    s_ans = [r["search_answered"] for r in rows]
    s_prec = float(np.mean([x > 0 for x, a in zip(s, s_ans) if a])) if any(s_ans) else float("nan")
    rng = np.random.default_rng(0)
    boots = (c - s)[rng.integers(0, len(rows), (reps, len(rows)))].mean(1)
    tried = sum(bool(r["learner"].get("tried")) for r in rows)
    passed = sum(bool(r["learner"].get("check_passed")) for r in rows)
    print(f"tasks {len(rows)}  learner tried {tried}  rotation check passed {passed}  answered (tau {tau}) {sum(use)}  "
          f"right {sum(r['learner_score'] > 0 for r, u in zip(rows, use) if u)}")
    print(f"S {s.mean():.3f} (answered {sum(s_ans)}, right {s_prec:.3f})   C {c.mean():.3f} (answered {n_ans}, "
          f"right {prec:.3f})")
    print(f"C - S {(c - s).mean():+.3f} [{np.percentile(boots, 2.5):+.3f}, {np.percentile(boots, 97.5):+.3f}]")
    return {"S": float(s.mean()), "C": float(c.mean()), "precision": prec, "answered": n_ans,
            "diff": float((c - s).mean()), "lower": float(np.percentile(boots, 2.5)),
            "upper": float(np.percentile(boots, 97.5))}


def tune(rows: list) -> float:
    """Pre-registered: the tau with the most learner-added solved tasks among those with C precision >= 0.93;
    ties go to the larger tau; if none reaches 0.93, the largest tau."""
    best = None
    for tau in TAUS:
        s, c, use, prec, _ = systems(rows, tau)
        added = float((c - s).sum())
        print(f"tau {tau:.1f}  answered by learner {sum(use)}  added {added:.2f}  C precision {prec:.3f}")
        if prec >= 0.93 and (best is None or added >= best[1]):
            best = (tau, added)
    tau = best[0] if best else TAUS[-1]
    (RESULT_DIR / "verify-a-tau.json").write_text(json.dumps({"tau": tau}))
    print(f"frozen tau {tau}")
    return tau


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["search", "run", "tune", "report"])
    p.add_argument("--exam", default="arc1", choices=["arc1", "arc2"])
    p.add_argument("--final", action="store_true")
    a = p.parse_args()
    if a.exam == "arc2" and a.cmd in ("run", "report") and not a.final:
        raise SystemExit("arc2 is the one path-A verdict: pass --final once tau is frozen")
    if a.cmd == "search":
        search_all(a.exam)
        return
    rows = run(a.exam) if a.cmd == "run" else [
        json.loads(l) for l in (RESULT_DIR / f"verify-a-{a.exam}.jsonl").read_text().splitlines()]
    if a.cmd == "tune":
        tune(rows)
        return
    tau = json.loads((RESULT_DIR / "verify-a-tau.json").read_text())["tau"]
    result = report(rows, tau)
    (RESULT_DIR / f"verify-a-{a.exam}-report.json").write_text(json.dumps({"tau": tau, **result}, indent=1))


if __name__ == "__main__":
    main()
