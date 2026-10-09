"""The pre-registered MuSiQue verdict (design/mk1-v2-spec.md, 9): mean net score, MK1 with the thinking satellite against
each language model alone, paired bootstrap over questions.

python -m cognitive_lab.mk1v2.musique_compare --set final
"""

import argparse
import json
import random
import statistics

from cognitive_lab.mk1v2.musique import RESULTS


def nets(system: str, part: str) -> dict:
    r = json.loads((RESULTS / f"exam-musique_{system}_{part}.json").read_text(encoding="utf-8"))
    return {j["id"]: j["net"] for j in r["replies"]}, r["summary"]


def bootstrap(a: dict, b: dict, rounds: int = 5000, seed: int = 0) -> list[float]:
    rng = random.Random(seed)
    ids = sorted(a)
    diffs = sorted(statistics.mean(a[i] - b[i] for i in (rng.choice(ids) for _ in ids)) for _ in range(rounds))
    return [round(diffs[int(0.025 * rounds)], 4), round(diffs[int(0.975 * rounds)], 4)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--set", choices=("dev", "final"), default="dev")
    args = parser.parse_args()
    tables = {s: nets(s, args.set) for s in ("mk1-think", "e2b", "e4b")}
    out = {"set": args.set, "summary": {s: t[1] for s, t in tables.items()},
           "mk1_minus_e4b_ci95": bootstrap(tables["mk1-think"][0], tables["e4b"][0]),
           "mk1_minus_e2b_ci95": bootstrap(tables["mk1-think"][0], tables["e2b"][0])}
    (RESULTS / f"exam-musique_verdict_{args.set}.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
