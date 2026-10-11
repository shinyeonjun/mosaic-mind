"""Family-count probe (design/next-after-c-2026-10-11.md section 6): does pretraining on more concept families make a
new family faster to learn?

Three learners of the same size and the same 20,000 pretraining steps on 7 / 14 / 28 of the original families
(nested, round-robin over the groups so every subset is balanced). Measured on the 8 probe families, which none of
them saw: exact match with the examples only (N=0) and after adapting on 8 tasks of the family (N=8, the frozen
step-2 recipe), plus a from-scratch reference at N=8.

python -m cognitive_lab.concepts.probe subsets
python -m cognitive_lab.concepts.probe all
"""

import argparse
import json
import random

import numpy as np
import torch

from cognitive_lab.concepts.generator import FAMILIES, task
from cognitive_lab.concepts.learner import CKPT_DIR, DEVICE, RECIPE, RESULT_DIR, Learner, adapt, exact, load, pretrain
from cognitive_lab.concepts.probe_families import PROBE

SIZES = (7, 14, 28)
RUNS = 2
N_EVAL = 200
N_ADAPT = 8


def subsets(seed: str = "probe-subsets") -> dict:
    rng = random.Random(seed)
    groups = {}
    for f, (g, _) in sorted(FAMILIES.items()):
        groups.setdefault(g, []).append(f)
    for g in groups.values():
        rng.shuffle(g)
    order, i = [], 0
    while len(order) < len(FAMILIES):
        for g in sorted(groups):
            if i < len(groups[g]):
                order.append(groups[g][i])
        i += 1
    return {n: order[:n] for n in SIZES}


def evaluate(ckpt: str | None, tag: str) -> dict:
    """Per probe family: zero-shot (if ckpt) and N=8 exact match per eval task, mean over adaptation runs."""
    out_path = RESULT_DIR / f"probe-{tag}.json"
    res = json.loads(out_path.read_text()) if out_path.exists() else {}
    for f in PROBE:
        if f in res:
            continue
        evals = [t for t in (task(f, f"eval-{i}") for i in range(N_EVAL * 2)) if t][:N_EVAL]
        r = {}
        if ckpt:
            r["zero_shot"] = exact(load(ckpt), evals)
        per_run = []
        for run in range(RUNS):
            if ckpt:
                m = load(ckpt)
            else:
                torch.manual_seed(1000 + run)
                m = Learner().to(DEVICE)
            adapt(m, f, N_ADAPT, run, **RECIPE)
            per_run.append(exact(m, evals))
        r["n8"] = np.mean(per_run, axis=0).tolist()
        res[f] = r
        out_path.write_text(json.dumps(res))
        print(f"{tag:10s} {f:24s} zero {np.mean(r.get('zero_shot', [0])):.3f}  N=8 {np.mean(r['n8']):.3f}", flush=True)
    return res


def boot(a: dict, b: dict, key: str, reps: int = 10000, seed: int = 0):
    """Difference of family-mean scores a - b, resampling eval tasks within each family."""
    rng = np.random.default_rng(seed)
    diffs = [np.array(a[f][key]) - np.array(b[f][key]) for f in PROBE]
    boots = np.mean([d[rng.integers(0, len(d), (reps, len(d)))].mean(1) for d in diffs], axis=0)
    return float(np.mean([d.mean() for d in diffs])), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def report() -> dict:
    res = {n: json.loads((RESULT_DIR / f"probe-f{n}.json").read_text()) for n in SIZES}
    scratch = json.loads((RESULT_DIR / "probe-scratch.json").read_text())
    table = {n: {"zero_shot": float(np.mean([np.mean(r[f]["zero_shot"]) for f in PROBE])),
                 "n8": float(np.mean([np.mean(r[f]["n8"]) for f in PROBE]))} for n, r in res.items()}
    table["scratch"] = {"n8": float(np.mean([np.mean(scratch[f]["n8"]) for f in PROBE]))}
    for k, v in table.items():
        print(f"{str(k):8s} " + "  ".join(f"{a} {b:.3f}" for a, b in v.items()))
    for f in PROBE:
        print(f"  {f:24s} " + "  ".join(f"f{n} {np.mean(res[n][f]['n8']):.3f}" for n in SIZES)
              + f"  scratch {np.mean(scratch[f]['n8']):.3f}")
    d = boot(res[28], res[14], "n8")
    d0 = boot(res[28], res[14], "zero_shot")
    print(f"decision: N=8 f28 - f14 {d[0]:+.3f} [{d[1]:+.3f}, {d[2]:+.3f}]  (B if >= +0.05 and lower > 0)")
    print(f"          N=0 f28 - f14 {d0[0]:+.3f} [{d0[1]:+.3f}, {d0[2]:+.3f}]")
    return {"table": table, "decision_n8": d, "n0": d0}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["subsets", "all", "report"])
    a = p.parse_args()
    if a.cmd == "subsets":
        for n, s in subsets().items():
            print(n, s)
    elif a.cmd == "all":
        for n, fams in subsets().items():
            if not (CKPT_DIR / f"probe-f{n}.pt").exists() or json.loads(
                    (RESULT_DIR / f"probe-f{n}-log.json").read_text())[-1]["step"] < 20000:
                pretrain(20000, families=fams, out=f"probe-f{n}")
            evaluate(f"probe-f{n}", f"f{n}")
        evaluate(None, "scratch")
        report()
    else:
        report()


if __name__ == "__main__":
    main()
