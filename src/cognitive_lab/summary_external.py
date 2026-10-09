"""Collect the two sealed external exams of MK1 v2 (RGB, MuSiQue-Full) for the summary notebook (no models run here).

Sources: artifacts/results/exam-rgb_*, exam-musique_*, mk1v2-*-train.json, mk1v2-think-signals_*.json.
Meanings and the net scoring rule: design/mk1-v2-spec.md, sections 6-9. Final (sealed) is the headline, dev secondary.
"""

import json
from pathlib import Path

from cognitive_lab.mk1v2 import rgb, rgb_compare

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS = PROJECT_ROOT / "artifacts" / "results"

SYSTEM_LABELS = {"e2b": "Gemma E2B 혼자", "e4b": "Gemma E4B 혼자", "mk1": "MK1 v2", "mk1-think": "MK1 + 생각 위성"}
RGB_TESTBEDS = [  # (key, label)
    ("noise-0.0", "잡음 0%"), ("noise-0.4", "잡음 40%"), ("noise-0.8", "잡음 80%"),
    ("rejection", "거절"), ("counterfactual", "거짓 문서"), ("integration", "정보 합치기"),
]


def _load(name: str) -> dict:
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def _diff(ci: list[float], point: float) -> dict:
    return {"point": round(point, 4), "low": ci[0], "high": ci[1]}


# -- RGB ---------------------------------------------------------------------------------

def rgb_verdict() -> dict:
    """Final mean net, per-testbed net, and MK1 minus each Gemma with the paired bootstrap 95% interval."""
    v = _load("exam-rgb_verdict_final.json")
    m = v["mean_net"]
    return {
        "mean_net": m,
        "per_testbed": v["per_testbed"],
        "questions": {k: s["questions"] for k, s in _load("exam-rgb_mk1_final.json")["settings"].items()},
        "diffs": {"e2b": _diff(v["mk1_minus_e2b_ci95"], m["mk1"] - m["e2b"]),
                  "e4b": _diff(v["mk1_minus_e4b_ci95"], m["mk1"] - m["e4b"])},
    }


def rgb_official(part: str = "final") -> dict:
    """The official RGB metrics per system: accuracy per testbed, rejection rate, error detection rate."""
    out = {}
    for s in ("e2b", "e4b", "mk1"):
        settings = _load(f"exam-rgb_{s}_{part}.json")["settings"]
        out[s] = {k: {m: val for m, val in v.items() if m not in ("replies", "questions")} for k, v in settings.items()}
    return out


def rgb_mean_net(part: str) -> dict:
    """Mean net over the six testbeds (the dev files of the Gemmas carry no net, so it is recomputed from replies)."""
    return {s: round(rgb_compare.mean_net(rgb_compare.nets(s, part)), 4) for s in ("mk1", "e2b", "e4b")}


def rgb_gain_loss(base: str = "e2b", part: str = "final") -> dict:
    """Per testbed, how many questions MK1 scored higher / lower than `base` (same question, net score)."""
    a, b = rgb_compare.nets("mk1", part), rgb_compare.nets(base, part)
    out = {}
    for k, _ in RGB_TESTBEDS:
        diffs = [a[k][i] - b[k][i] for i in a[k]]
        out[k] = {"gained": sum(d > 0 for d in diffs), "lost": sum(d < 0 for d in diffs), "questions": len(diffs)}
    return out


# -- MuSiQue -----------------------------------------------------------------------------

def musique_verdict() -> dict:
    v = _load("exam-musique_verdict_final.json")
    s = v["summary"]
    return {"summary": s,
            "diffs": {"e2b": _diff(v["mk1_minus_e2b_ci95"], s["mk1-think"]["net"] - s["e2b"]["net"]),
                      "e4b": _diff(v["mk1_minus_e4b_ci95"], s["mk1-think"]["net"] - s["e4b"]["net"])}}


def musique_summary(part: str) -> dict:
    systems = ("e2b", "e4b", "mk1-think") + (("mk1",) if part == "dev" else ())
    return {s: _load(f"exam-musique_{s}_{part}.json")["summary"] for s in systems}


def musique_changes(base: str = "e2b", part: str = "final") -> dict:
    """How MK1 + thinking satellite changed each question relative to `base`, split by answerable / unanswerable.
    Keys: (base outcome, MK1 outcome) with outcomes right / rejected / wrong."""
    def outcome(r):
        return "rejected" if r["rejected"] else ("right" if r["right"] else "wrong")
    a = {r["id"]: r for r in _load(f"exam-musique_mk1-think_{part}.json")["replies"]}
    b = {r["id"]: r for r in _load(f"exam-musique_{base}_{part}.json")["replies"]}
    out = {"answerable": {}, "unanswerable": {}}
    for i, r in a.items():
        kind = i.split("|")[1]
        key = (outcome(b[i]), outcome(r))
        out[kind][key] = out[kind].get(key, 0) + 1
    return out


def chain_scores(part: str = "final") -> dict:
    """Weakest of the first two hop-selector scores per question (the evidence signal in mk1v2/think.py)."""
    out = {"answerable": [], "unanswerable": []}
    for s in _load(f"mk1v2-think-signals_{part}.json"):
        out[s["id"].split("|")[1]].append(min(score for _, score in s["chain"][:2]))
    return out


def think_thresholds() -> dict:
    return _load("mk1v2-think-thresholds.json")


# -- Trained satellites ------------------------------------------------------------------

def satellite_training() -> list[dict]:
    """Before / after on each satellite's own validation (HotpotQA for the thinking satellite parts,
    SQuAD 2.0 dev for the English reader), in percent."""
    rows = []
    r = _load("mk1v2-reader-en-train.json")
    rows.append({"part": "영어 읽기 위성 (RGB용)", "metric": "정답 일치 (SQuAD 2.0)", "before": r["history"][0]["exact_match"],
                 "after": r["history"][-1]["exact_match"], "questions": r["train_questions"], "minutes": r["minutes"]})
    r = _load("mk1v2-selector-train.json")
    rows.append({"part": "단계별 문서 선택기", "metric": "근거 2개 모두 찾기 (HotpotQA)",
                 "before": 100 * r["history"][0]["both_supporting_found"],
                 "after": 100 * r["history"][-1]["both_supporting_found"], "questions": r["questions"],
                 "minutes": r["minutes"]})
    r = _load("mk1v2-reader-hop-train.json")
    rows.append({"part": "다단계 독해기", "metric": "정답 (HotpotQA)", "before": 100 * r["history"][0]["answer_right"],
                 "after": 100 * r["history"][-1]["answer_right"], "questions": r["questions"], "minutes": r["minutes"]})
    return rows
