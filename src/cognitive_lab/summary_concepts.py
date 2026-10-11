"""Collect the option C results (steps 1-3) for notebook 24 (design/build-from-scratch-2026-10-10.md, 9-13).

Reads the saved files in artifacts/concepts. The step-1 search check printed to the console only, so its numbers are
copied here from section 9. Example tasks are drawn from the generator (CPU only). No model runs here.
"""

import json

import numpy as np

from cognitive_lab.concepts.generator import FAMILIES, task
from cognitive_lab.concepts.split import SPLIT
from cognitive_lab.mk1v3.signals import ROOT

CONCEPTS = ROOT / "artifacts" / "concepts"

# section 9: search satellite solves out of 10 generated tasks (3 s each), per family.
SEARCH_SOLVES = {
    "recolour_largest": 0, "keep_extreme": 0, "remove_noise": 6, "fill_holes": 3, "outline": 0,
    "remove_border_touching": 0, "crop": 10, "recolour_by_colour": 10, "flip": 10, "rotate": 10,
    "mirror_complete": 10, "translate": 0, "scale_up": 10, "tile": 6, "inside_frame": 0, "centre_dot": 0,
    "connect_pairs": 0, "count_to_bar": 0, "recolour_by_size": 9, "size_parity": 8, "majority_colour": 0,
    "sort_bars": 0, "extend_to_wall": 8, "gravity": 10, "move_until_obstacle": 0, "ray_until_obstacle": 0,
    "keep_same_shape": 0, "odd_one_out": 0,
}

GROUP_NAMES = {"objects": "물체", "geometry": "기하", "number": "수", "agent": "목표 지향", "compare": "비교"}

# Where MK1 stands, trait by trait (README scoreboard, design docs). (trait, what, result, verdict)
TRAITS = [
    ("판단", "섞인 봉인 시험: 기호", "0.725 vs E4B 0.662", "이김"),
    ("판단", "섞인 봉인 시험: 이름", "0.750 vs E4B 0.796", "짐"),
    ("공부", "새 과제 흐름: 기호", "0.752 vs E4B 0.630", "이김"),
    ("성장", "자라기 봉인2 (과제 이름 없이)", "0.636 vs E4B 0.574", "이김"),
    ("생각 (검증 있음)", "ARC-AGI-1 eval", "0.074 vs E4B 0.023", "이김"),
    ("생각 (검증 없음)", "MuSiQue (섞인 봉인)", "0.588 vs E4B 0.583", "비김"),
    ("배우는 법 배우기", "라이브러리 학습", "효과 없음", "실패"),
    ("배우는 법 배우기", "개념 학습기: 새 합성 가족", "+0.300 vs 처음부터", "통과"),
    ("배우는 법 배우기", "개념 학습기 + 검증: ARC", "+0.030, 정답률 0.891", "실패"),
]


def generator_check() -> list[dict]:
    split_of = {f: k for k, v in SPLIT.items() for f in v}
    return [{"family": f, "group": g, "search": SEARCH_SOLVES[f], "split": split_of[f]} for f, (g, _) in FAMILIES.items()]


def examples(families=("outline", "translate", "count_to_bar", "move_until_obstacle", "odd_one_out", "gravity"),
             seed=1) -> list[dict]:
    out = []
    for f in families:
        t = task(f, seed)
        out.append({"family": f, "pairs": [(p["input"], p["output"]) for p in t["train"][:2]]})
    return out


def pretrain_logs() -> dict:
    return {k: json.loads((CONCEPTS / f"{name}-log.json").read_text())
            for k, name in (("step2", "concept-learner"), ("step3", "concept-learner-v2"))}


def _boot(results, a, b, reps=10000, seed=0):
    rng = np.random.default_rng(seed)
    diffs = [np.array(r[a]) - np.array(r[b]) for r in results.values()]
    boots = np.mean([d[rng.integers(0, len(d), (reps, len(d)))].mean(1) for d in diffs], axis=0)
    return float(np.mean([d.mean() for d in diffs])), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def step2() -> dict:
    """Recipe search on dev and the test verdict, as mean exact-match per family."""
    def load(name):
        return json.loads((CONCEPTS / f"{name}.json").read_text())["results"]

    recipes = []
    for label, name, runs in (("300스텝 lr 1e-4", "step2-dev", 3), ("1500스텝 lr 1e-4", "step2-dev-s1500-lr0.0001", 1),
                              ("1500스텝 lr 3e-4", "step2-dev-s1500-lr0.0003", 1),
                              ("3000스텝 lr 1e-4", "step2-dev-s3000-lr0.0001", 1)):
        r = load(name)
        recipes.append({"recipe": label, "runs": runs,
                        "scratch": float(np.mean([np.mean(v["scratch_n32"]) for v in r.values()])),
                        "pretrained": float(np.mean([np.mean(v["pretrained_n32"]) for v in r.values()])),
                        "diff": _boot(r, "pretrained_n32", "scratch_n32")})
    test = load("step2-test")
    per_family = {f: {k: float(np.mean(v)) for k, v in r.items()} for f, r in test.items()}
    return {"recipes": recipes, "test": per_family,
            "verdict": {n: _boot(test, f"pretrained_n{n}", f"scratch_n{n}") for n in (8, 32)}}


def step3(min_share: float = 0.3) -> dict:
    out = {}
    for half in ("tune", "check"):
        rows = [json.loads(l) for l in (CONCEPTS / f"step3-{half}-ttt300.jsonl").read_text().splitlines()]
        use = [bool(r["learner"].get("check_passed")) and r["learner"]["check_share"] >= min_share for r in rows]
        s = np.array([r["search_score"] for r in rows])
        c = np.array([r["search_score"] if r["search_answered"] else (r["learner_score"] if u else 0.0)
                      for r, u in zip(rows, use)])
        ans_s = [r["search_answered"] for r in rows]
        ans_c = [a or u for a, u in zip(ans_s, use)]
        rng = np.random.default_rng(0)
        boots = (c - s)[rng.integers(0, len(rows), (10000, len(rows)))].mean(1)
        out[half] = {
            "tasks": len(rows),
            "search_answered": sum(ans_s), "search_right": int(sum(x > 0 for x, a in zip(s, ans_s) if a)),
            "learner_tried": sum(bool(r["learner"].get("tried")) for r in rows),
            "learner_passed": sum(use),
            "learner_right": sum(r["learner_score"] > 0 for r, u in zip(rows, use) if u),
            "S": float(s.mean()), "C": float(c.mean()),
            "precision_S": float(np.mean([x > 0 for x, a in zip(s, ans_s) if a])),
            "precision_C": float(np.mean([x > 0 for x, a in zip(c, ans_c) if a])),
            "diff": (float((c - s).mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))),
            "passed_share": [(r["task"], r["learner"]["check_share"], r["learner"].get("test_share", [None])[0],
                              r["learner_score"]) for r, u in zip(rows, use) if u],
        }
    return out
