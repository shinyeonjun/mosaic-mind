"""Step 1a of the antiques path (design/antiques-rethink-2026-10-12.md, 7): is "which satellites to call" worth learning?

Full information on document questions: every satellite's signals are recorded for the calibration data (1000) and
the exam development questions (RGB 130, MuSiQue 100). For each of the 16 subsets of the optional satellites
{reader, selector, nli, gliner}, the two calibrator heads are learned on the calibration data with the speaker's own
signals plus that subset's, and the expected-score decision (answer iff 2 P(right) - 1 > 1 - P(answerable)) is
scored on development. Rows:
  speaker alone      answer whenever the speaker does not reject (no calibrator)
  subset S           calibrated decision with S (S = {reader, selector} is v3 lean as it runs now)
  oracle             per question, the best net over all 16 subsets (the most any dispatcher over these can get)
Cost per subset: the satellites' measured seconds per development question (reader and selector timed here in fp16,
NLI and GLiNER from their side files), plus each satellite's GPU memory.
Development is used to look, not to choose: whatever is chosen here is judged later on sealed data.

python -m cognitive_lab.mk1v3.route_table
"""

import itertools
import json
import random
import statistics
import time

import numpy as np

from cognitive_lab.mk1v3.calibrate import SIGNALS, Logistic, head_rows, net
from cognitive_lab.mk1v3.signals import ROOT

RESULTS = ROOT / "artifacts" / "results"
SIDES = [RESULTS / "mk1v3-nli-features.jsonl", RESULTS / "mk1v3-gliner-features.jsonl"]
GROUPS = {
    "speaker": {"right": ["lp_mean", "lp_min", "lp_first", "gap_mean", "length"], "answerable": ["rejected", "lp_mean", "lp_min"]},
    "reader": {"right": ["reader_margin", "reply_has_reader"], "answerable": ["reader_margin"]},
    "selector": {"right": ["chain1", "chain2"], "answerable": ["chain1", "chain2"]},
    "nli": {"right": ["nli_reply_entail", "nli_reply_contra", "nli_reader_entail"], "answerable": ["nli_reply_entail", "nli_reader_entail"]},
    "gliner": {"right": ["gl_typed", "gl_reply_type", "gl_reply_in_docs", "gl_reader_type"], "answerable": ["gl_typed", "gl_reader_type"]},
}
OPTIONAL = ["reader", "selector", "nli", "gliner"]
VRAM_MIB = {"reader": 560, "selector": 560, "nli": 560, "gliner": 580}  # fp16 weights (about 280-290M parameters each)


def load(path) -> list[dict]:
    side = {}
    for p in SIDES:
        for line in p.read_text(encoding="utf-8").splitlines():
            d = json.loads(line)
            side.setdefault(d.pop("id"), {}).update(d)
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        r.update(side.get(r["id"], {}))
        r["rejected"] = float(r["rejected"])
        rows.append(r)
    return rows


def feats(subset: tuple, head: str) -> list[str]:
    return [f for g in ("speaker",) + subset for f in GROUPS[g][head]]


def X(rows, subset, head):
    return np.array([[float(r[f]) for f in feats(subset, head)] for r in rows])


def fit(calib: list[dict], subset: tuple) -> dict:
    models = {}
    for head in ("right", "answerable"):
        hr = head_rows(calib, head)
        y = np.array([float(r["right"] if head == "right" else r["answerable"]) for r in hr])
        models[head] = Logistic().fit(X(hr, subset, head), y)
    return models


def decisions(rows: list[dict], subset: tuple, models: dict) -> list[str]:
    p_ans = models["answerable"].predict(X(rows, subset, "answerable"))
    p_right = models["right"].predict(X(rows, subset, "right"))
    return ["reject" if r["rejected"] else ("answer" if 2 * pr - 1 > 1 - pa else "reject")
            for r, pr, pa in zip(rows, p_right, p_ans)]


def timings(dev: list[dict]) -> dict:
    """Seconds per development question for the reader and the selector (fp16, as v3 runs them now)."""
    import torch

    from cognitive_lab.mk1v2.reader_en import CHECKPOINT as READER_EN
    from cognitive_lab.mk1v2.selector import Selector
    from cognitive_lab.mk1v3.exam_check import items
    from cognitive_lab.world5.reader_qa import Reader

    by_id = {it["id"]: it for it in items()}
    its = [by_id[r["id"]] for r in dev]
    dev_ = torch.device("cuda")
    out = {}
    reader = Reader(dev_, checkpoint=READER_EN, precision="fp16")
    t = time.perf_counter()
    for it in its:
        reader.read_many([(it["query"], d) for d in it["docs"]])
    out["reader"] = (time.perf_counter() - t) / len(its)
    del reader
    selector = Selector(dev_, dtype=torch.float16)
    t = time.perf_counter()
    for it in its:
        selector.chain(it["query"], it["docs"], hops=2)
    out["selector"] = (time.perf_counter() - t) / len(its)
    return out


def main() -> None:
    from cognitive_lab.mk1v3.exam_check import OUT as EXAM, items, rejudge

    calib = load(SIGNALS)
    by_id = {it["id"]: it for it in items()}
    dev = [rejudge(r, by_id[r["id"]]) for r in load(EXAM) if r["id"] in by_id]
    sources = sorted({r["source"] for r in dev})
    secs = timings(dev)
    secs["nli"] = statistics.mean(r["nli_seconds"] for r in dev)
    secs["gliner"] = statistics.mean(r["gl_seconds"] for r in dev)
    nets, table = {}, []
    alone = [net(r, "reject" if r["rejected"] else "answer") for r in dev]
    for k in range(len(OPTIONAL) + 1):
        for subset in itertools.combinations(OPTIONAL, k):
            d = decisions(dev, subset, fit(calib, subset))
            nets[subset] = [net(r, c) for r, c in zip(dev, d)]
    oracle = [max(nets[s][i] for s in nets) for i in range(len(dev))]
    rng = random.Random(0)
    v3 = ("reader", "selector")

    def by_src(v):
        return {s: round(statistics.mean(x for x, r in zip(v, dev) if r["source"] == s), 4) for s in sources}

    def ci(a, b, reps=2000):
        d = [x - y for x, y in zip(a, b)]
        boots = sorted(statistics.mean(rng.choice(d) for _ in d) for _ in range(reps))
        return [round(boots[int(0.025 * reps)], 4), round(boots[int(0.975 * reps) - 1], 4)]

    for subset, v in sorted(nets.items(), key=lambda kv: -statistics.mean(kv[1])):
        table.append({"subset": "+".join(subset) or "(speaker only)", "net": round(statistics.mean(v), 4), **by_src(v),
                      "vs_v3": round(statistics.mean(v) - statistics.mean(nets[v3]), 4), "ci_vs_v3": ci(v, nets[v3]),
                      "seconds": round(sum(secs[g] for g in subset), 3), "vram_mib": sum(VRAM_MIB[g] for g in subset)})
    out = {"seconds_per_question": {k: round(v, 3) for k, v in secs.items()},
           "speaker_alone": {"net": round(statistics.mean(alone), 4), **by_src(alone)},
           "oracle": {"net": round(statistics.mean(oracle), 4), **by_src(oracle)},
           "oracle_vs_v3": round(statistics.mean(oracle) - statistics.mean(nets[v3]), 4),
           "subsets": table}
    print(f"seconds per question: {out['seconds_per_question']}")
    print(f"speaker alone   net {out['speaker_alone']}")
    print(f"oracle          net {out['oracle']}  (v3 + {out['oracle_vs_v3']})")
    print(f"{'subset':32s} {'net':>6s} " + " ".join(f"{s:>8s}" for s in sources) + "   vs v3   95% CI            sec   MiB")
    for t in table:
        print(f"{t['subset']:32s} {t['net']:6.3f} " + " ".join(f"{t[s]:8.3f}" for s in sources)
              + f"  {t['vs_v3']:+.3f}  [{t['ci_vs_v3'][0]:+.3f},{t['ci_vs_v3'][1]:+.3f}]  {t['seconds']:5.2f}  {t['vram_mib']:4d}")
    (RESULTS / "route-table-docs.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
