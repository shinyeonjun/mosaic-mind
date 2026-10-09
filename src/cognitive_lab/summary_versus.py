"""Collect the MK1-vs-one-language-model results (J1, J2) for notebook 19 (design/mk1-versus.md). No model runs here.

In-house worlds: per-problem scores (+1 right, 0 모름, -1 wrong) are saved in the result files, so confidence
intervals come from a bootstrap. Articles and doors are sessions of 40 questions (15 sessions), so they are
resampled by session; chains are one problem per session and are resampled by problem. All systems saw the same
problems in the same order, so MK1-minus-LFM differences are paired.
The zero-shot LFM run (J1) kept only the doors family in its result file (the chains and articles numbers stopped
before an out-of-memory crash); those two come from the design note's table and have no per-problem record.

KorQuAD: only the replies are saved, so they are rescored here against the exam's gold answers with the exam's
own scorer (cognitive_lab.mk1.korquad), and checked against the saved totals.
"""

import json

import numpy as np

from cognitive_lab.summary import RESULTS

FAMILIES = [("chains", "사슬\n(더 긴 추론, 모름)"), ("articles", "거짓 출처 섞인 기사\n(판단, 엄격 채점)"),
            ("doors", "기사로 여는 문\n(조합)")]
SESSION = {"chains": 1, "articles": 40, "doors": 40}
# Zero-shot LFM numbers that exist only in design/mk1-versus.md (J1 table; best way per family).
J1_FROM_NOTE = {"chains": 0.203, "articles": -0.625}
J1_ARTICLES_CONTAINS_FROM_NOTE = -0.378
SYSTEMS = [  # (key, plain label)
    ("mk1", "MK1"),
    ("llm", "LFM 학습 없음 (J1)"),
    ("llm-mk1-versus-lora", "LFM 같은 데이터로 학습 (J2)"),
]
KORQUAD = [  # (result name, plain label, size in billions)
    ("mk1", "MK1 읽기 전문가\n(KLUE 학습)", 0.28),
    ("llm", "LFM 학습 없음\n(J1)", 1.2),
    ("llm-mk1-klue-lora", "LFM KLUE로 학습\n(J2)", 1.2),
    ("llm-mk1-klue-lora-no-unknown", "LFM KLUE로 학습\n'모름' 금지 (J2)", 1.2),
]
UNKNOWN = "모름"


def _load(name: str) -> dict:
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def _blocks(scores: list[float], size: int) -> np.ndarray:
    return np.asarray(scores, dtype=float).reshape(-1, size)


def _boot_mean(scores: list[float], size: int, n: int, rng: np.random.Generator) -> np.ndarray:
    """Bootstrap distribution of the mean, resampling whole blocks (sessions) of `size` problems."""
    b = _blocks(scores, size)
    picks = rng.integers(0, len(b), size=(n, len(b)))
    return b[picks].mean(axis=(1, 2))


def versus_scores(n_boot: int = 4000, seed: int = 0) -> dict:
    """{family: {system: {score, right, wrong, way, lo, hi, scores or None}}}. The language model is credited with
    its best way per family (as in the note); lo/hi are the 2.5/97.5 bootstrap percentiles (None without records)."""
    rng = np.random.default_rng(seed)
    out = {f: {} for f, _ in FAMILIES}
    for key, _ in SYSTEMS:
        data = _load(f"mk1-versus_{key}.json")["families"]
        for family, _ in FAMILIES:
            if family not in data:
                if key == "llm" and family in J1_FROM_NOTE:
                    out[family][key] = {"score": J1_FROM_NOTE[family], "right": None, "wrong": None, "way": "설계 노트 표",
                                        "lo": None, "hi": None, "scores": None}
                continue
            entry = data[family]
            way = entry["ways"][entry["best_way"]]
            dist = _boot_mean(way["scores"], SESSION[family], n_boot, rng)
            out[family][key] = {"score": way["score"], "right": way["right"], "wrong": way["wrong"], "way": entry["best_way"],
                                "lo": float(np.percentile(dist, 2.5)), "hi": float(np.percentile(dist, 97.5)),
                                "scores": way["scores"], "contains_score": way.get("contains_score")}
    return out


def paired_gap(scores: dict, other: str, n_boot: int = 4000, seed: int = 1) -> dict:
    """{family: (MK1 - other, lo, hi)} with a session-paired bootstrap, for families where both have records."""
    rng = np.random.default_rng(seed)
    out = {}
    for family, _ in FAMILIES:
        a, b = scores[family].get("mk1"), scores[family].get(other)
        if not a or not b or a["scores"] is None or b["scores"] is None:
            continue
        diff = (np.asarray(a["scores"]) - np.asarray(b["scores"])).tolist()
        dist = _boot_mean(diff, SESSION[family], n_boot, rng)
        out[family] = (float(np.mean(diff)), float(np.percentile(dist, 2.5)), float(np.percentile(dist, 97.5)))
    return out


def all_ways(key: str) -> dict:
    """{family: {way: score}}: every answer way the language model was tried with (the note reports the best)."""
    return {f: {w: x["score"] for w, x in e["ways"].items()} for f, e in _load(f"mk1-versus_{key}.json")["families"].items()}


def lora_training() -> dict:
    """The two J2 adapter trainings (validation before and after one epoch, cost)."""
    return {"klue": _load("mk1-klue-lora-train.json"), "versus": _load("mk1-versus-lora-train.json")}


# -- KorQuAD 1.0 ----------------------------------------------------------------------------

def korquad_rows() -> dict:
    """{name: {"em": array, "f1": array, "answers": list, "summary": saved totals, "qps": float}} per system, every
    question rescored with the exam's scorer (0/1 EM, 0..1 F1). Raises if a rescore disagrees with the saved total."""
    from cognitive_lab.mk1.korquad import char_f1, exact, load_exam

    exam = load_exam()
    out = {}
    for name, _, _ in KORQUAD:
        r = _load(f"exam-korquad1_{name}.json")
        em = np.array([exact(a, q["answers"]) for q, a in zip(exam, r["answers"], strict=True)])
        f1 = np.array([char_f1(a, q["answers"]) for q, a in zip(exam, r["answers"], strict=True)])
        if abs(100 * em.mean() - r["all"]["exact_match"]) > 0.01 or abs(100 * f1.mean() - r["all"]["char_f1"]) > 0.01:
            raise ValueError(f"rescore of {name} does not match the saved totals")
        out[name] = {"em": em, "f1": f1, "answers": r["answers"], "summary": r["all"], "qps": r["questions_per_second"]}
    return out


def korquad_ci(rows: dict, n_boot: int = 2000, seed: int = 2) -> dict:
    """{name: {"em": (mean, lo, hi), "f1": (...)}} in percent, question-level bootstrap."""
    rng = np.random.default_rng(seed)
    n = len(next(iter(rows.values()))["em"])
    picks = rng.integers(0, n, size=(n_boot, n))
    out = {}
    for name, r in rows.items():
        out[name] = {}
        for metric in ("em", "f1"):
            dist = 100 * r[metric][picks].mean(axis=1)
            out[name][metric] = (100 * float(r[metric].mean()), float(np.percentile(dist, 2.5)), float(np.percentile(dist, 97.5)))
    return out


def korquad_unknown_split(rows: dict) -> dict:
    """Split by what the KLUE-trained LFM did: answered 모름 vs answered something. Per group, EM and F1 (percent)
    of MK1's reader, the trained LFM and the trained LFM with 모름 banned."""
    trained = rows["llm-mk1-klue-lora"]["answers"]
    unknown = np.array([a.strip() == UNKNOWN for a in trained])
    groups = {"모름이라고 한 문제": unknown, "답을 쓴 문제": ~unknown}
    out = {"n": {g: int(m.sum()) for g, m in groups.items()}, "share_unknown": float(unknown.mean()), "groups": {}}
    for g, mask in groups.items():
        out["groups"][g] = {name: {"em": 100 * float(rows[name]["em"][mask].mean()), "f1": 100 * float(rows[name]["f1"][mask].mean())}
                            for name in ("mk1", "llm-mk1-klue-lora", "llm-mk1-klue-lora-no-unknown")}
    return out
