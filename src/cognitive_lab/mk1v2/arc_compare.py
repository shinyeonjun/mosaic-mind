"""Verdict for the ARC-AGI-1 exam (design/mk1-v2-spec.md, 12): MK1 minus each Gemma, paired bootstrap over tasks
(2,000 draws, seed 0), plus how the search satellite did when it answered (its precision).

python -m cognitive_lab.mk1v2.arc_compare --set dev --limit 100
"""

import argparse
import json
import random
import statistics

from cognitive_lab.mk1v2.arc import RESULTS


def load(system: str, part: str, tag: str) -> dict:
    return {r["id"]: r for r in json.loads((RESULTS / f"exam-arc_{system}_{part}{tag}.json").read_text(encoding="utf-8"))["records"]}


def difference(a: dict, b: dict, draws: int = 2000) -> dict:
    ids = sorted(a)
    rng = random.Random(0)
    values = sorted(statistics.mean(a[i]["score"] - b[i]["score"] for i in (rng.choice(ids) for _ in ids)) for _ in range(draws))
    return {"difference": round(statistics.mean(a[i]["score"] - b[i]["score"] for i in ids), 4),
            "ci95": [round(values[int(0.025 * draws)], 4), round(values[int(0.975 * draws) - 1], 4)]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--set", choices=("dev", "final"), default="dev")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    tag = f"_{args.limit}" if args.limit else ""
    systems = {s: load(s, args.set, tag) for s in ("e2b", "e4b", "mk1")}
    assert len({frozenset(v) for v in systems.values()}) == 1, "systems saw different tasks"
    mk1 = systems["mk1"]
    thought = [r for r in mk1.values() if r["thought"]]
    out = {"part": args.set, "tasks": len(mk1),
           "score": {s: round(statistics.mean(r["score"] for r in v.values()), 4) for s, v in systems.items()},
           "search": {"answered": len(thought),
                      "right_when_answered": round(statistics.mean(r["score"] for r in thought), 4) if thought else None,
                      "score_from_search_alone": round(sum(r["score"] for r in thought) / len(mk1), 4)},
           "versus": {f"mk1 - {o}": difference(mk1, systems[o]) for o in ("e2b", "e4b")}}
    print(json.dumps(out, indent=1))
    (RESULTS / f"exam-arc_verdict_{args.set}{tag}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
