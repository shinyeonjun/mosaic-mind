"""The pre-registered RGB verdict (design/mk1-v2-spec.md, 6): mean net score over the six testbeds, MK1 v2 against
each language model alone, paired bootstrap over questions (resampled within each testbed, equal testbed weights).

python -m cognitive_lab.mk1v2.rgb_compare --set final
"""

import argparse
import json
import random
import statistics

from cognitive_lab.mk1v2 import rgb


def nets(system: str, part: str) -> dict[str, dict]:
    r = json.loads((rgb.RESULTS / f"exam-rgb_{system}_{part}.json").read_text(encoding="utf-8"))
    return {s: {j["id"]: rgb.net(s, j) for j in v["replies"]} for s, v in r["settings"].items()}


def mean_net(table: dict) -> float:
    return statistics.mean(statistics.mean(v.values()) for v in table.values())


def paired_bootstrap(a: dict, b: dict, rounds: int = 5000, seed: int = 0) -> list[float]:
    rng = random.Random(seed)
    settings = list(a)
    ids = {s: sorted(a[s]) for s in settings}
    diffs = []
    for _ in range(rounds):
        per = []
        for s in settings:
            sample = [rng.choice(ids[s]) for _ in ids[s]]
            per.append(statistics.mean(a[s][i] - b[s][i] for i in sample))
        diffs.append(statistics.mean(per))
    diffs.sort()
    return [round(diffs[int(0.025 * rounds)], 4), round(diffs[int(0.975 * rounds)], 4)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--set", choices=("dev", "final"), default="dev")
    args = parser.parse_args()
    tables = {s: nets(s, args.set) for s in ("mk1", "e2b", "e4b")}
    out = {"set": args.set, "mean_net": {s: round(mean_net(t), 4) for s, t in tables.items()},
           "per_testbed": {s: {k: round(statistics.mean(v.values()), 4) for k, v in t.items()} for s, t in tables.items()},
           "mk1_minus_e4b_ci95": paired_bootstrap(tables["mk1"], tables["e4b"]),
           "mk1_minus_e2b_ci95": paired_bootstrap(tables["mk1"], tables["e2b"])}
    path = rgb.RESULTS / f"exam-rgb_verdict_{args.set}.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
