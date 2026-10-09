"""Verdict for the task-stream exam (design/mk1-v2-spec.md, 10).

Gemma runs are rescored from their saved replies with taskstream.parse (one parser for every system). Difference =
MK1 minus the opponent in online accuracy (mean over the six tasks), per condition; 95% interval from a paired bootstrap
that resamples stream positions within each task (2,000 draws, seed 0).

python -m cognitive_lab.mk1v2.taskstream_compare --set dev
"""

import argparse
import json
import random
import statistics

from cognitive_lab.mk1v2.taskstream import RESULTS, parse, stream


def rights(system: str, part: str) -> dict:
    """{condition: {task: [0/1 per text]}}"""
    data = json.loads((RESULTS / f"exam-taskstream_{system}_{part}.json").read_text(encoding="utf-8"))["conditions"]
    out = {}
    for condition, v in data.items():
        truth = {t["task"]: t for t in stream(part, condition)}
        out[condition] = {}
        for t in v["tasks"]:
            if "replies" in t:
                task = truth[t["task"]]
                out[condition][t["task"]] = [int(parse(r, task["labels"], condition) == it["label"]) for r, it in zip(t["replies"], task["items"])]
            else:
                out[condition][t["task"]] = t["right"]
    return out


def score(r: dict) -> float:
    return statistics.mean(statistics.mean(x) for x in r.values())


def difference(a: dict, b: dict, draws: int = 2000) -> dict:
    rng = random.Random(0)
    values = []
    for _ in range(draws):
        per = []
        for task in a:
            idx = [rng.randrange(len(a[task])) for _ in a[task]]
            per.append(statistics.mean(a[task][i] - b[task][i] for i in idx))
        values.append(statistics.mean(per))
    values.sort()
    return {"difference": round(score(a) - score(b), 4), "ci95": [round(values[int(0.025 * draws)], 4), round(values[int(0.975 * draws) - 1], 4)]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--set", choices=("dev", "final"), default="dev")
    args = parser.parse_args()
    systems = {s: rights(s, args.set) for s in ("mk1", "e2b", "e4b") if (RESULTS / f"exam-taskstream_{s}_{args.set}.json").exists()}
    out = {"part": args.set, "accuracy": {s: {c: round(score(r[c]), 4) for c in r} for s, r in systems.items()}, "versus": {}}
    for opponent in ("e2b", "e4b"):
        if opponent in systems and "mk1" in systems:
            out["versus"][opponent] = {c: difference(systems["mk1"][c], systems[opponent][c]) for c in ("named", "symbolic")}
    print(json.dumps(out, indent=1))
    (RESULTS / f"exam-taskstream_verdict_{args.set}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
