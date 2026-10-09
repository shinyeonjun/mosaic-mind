"""Collect MK1 stage F-K results (arbiters, verify, audit, outside experience, change) for notebook 18.

Read-only: every function loads artifacts/results JSON and returns plain Python structures.
Labels are plain Korean role names, matching design/mk1-integration.md and design/audit-2026-10-08.md.
"""

import json
import math
import statistics
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS = PROJECT_ROOT / "artifacts" / "results"
SEEDS = ("42", "43", "44")
T95 = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776}


def _load(name: str) -> dict | None:
    path = RESULTS / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def mean_ci(values: list[float]) -> tuple[float, float]:
    """Mean and half-width of a two-sided 95% t interval over seeds (0 for one value)."""
    if len(values) < 2:
        return (values[0] if values else math.nan), 0.0
    return statistics.mean(values), T95[len(values)] * statistics.stdev(values) / math.sqrt(len(values))


# -- F: source conflict (which channel to trust) ---------------------------------------

F_SYSTEMS = [  # (json key, label)
    ("think", "안내판만"),
    ("article-only", "기사만"),
    ("ask-available", "합치기 (E까지)"),
    ("arbitrate", "중재자"),
    ("guide", "길잡이 (정답 앎)"),
]


def f_conflict() -> dict:
    """Seed-42 stage F: C-only and overall score per board reliability, plus criteria."""
    r = _load("mk1-f_conflict.json")
    levels = ["1.0", "0.75", "0.5"]
    return {
        "levels": levels,
        "systems": [label for _, label in F_SYSTEMS],
        "c_only": {label: [r["by_accuracy"][lv]["C_only"][k] for lv in levels] for k, label in F_SYSTEMS},
        "total": {label: r["total"][k] for k, label in F_SYSTEMS},
        "arbiter_minus_merge_ci": r["arbiter_minus_merge_ci95"],
        "within_session": r["learning_within_session_r0.5_C"],
        "criteria": r["criteria"],
    }


def arbiter_training() -> list[dict]:
    """Validation door score per epoch for the learned arbiter (seeds 42-44) and the 3x-data v2."""
    runs = []
    for seed in SEEDS:
        r = _load(f"mk1-f_arbiter-train_seed-{seed}.json")
        if r:
            runs.append({"label": f"seed {seed} (600세션)", "seed": seed, "v2": False,
                         "epochs": [h["epoch"] for h in r["history"]],
                         "score": [h["validation_door_score"] for h in r["history"]],
                         "selected": r["selected_epoch"], "best": r["validation"]})
    r = _load("mk1-f_arbiter-v2-train_seed-42.json")
    if r:
        runs.append({"label": "seed 42, 데이터 3배+ (2,000세션)", "seed": "42", "v2": True,
                     "epochs": [h["epoch"] for h in r["history"]],
                     "score": [h["validation_door_score"] for h in r["history"]],
                     "selected": r["selected_epoch"], "best": r["validation"]})
    return runs


# -- G: should the router ask to verify? -----------------------------------------------

G_POLICIES = [
    ("verify-never", "확인 안 함"),
    ("verify-always", "항상 확인"),
    ("verify-router", "사령탑 그대로\n+ 중재자 확신"),
    ("verify-hindsight", "사후 선택\n(반칙)"),
]


def g_verify() -> dict:
    r = _load("mk1-g_verify.json")
    rows = [{"label": label, "net": r["policies"][k]["net"], "asks": r["policies"][k]["asks_in_C"],
             "cheat": k == "verify-hindsight"} for k, label in G_POLICIES]
    rows.insert(3, {"label": "최선의 길잡이\n(반반 세션만 확인)", "net": r["guides"][r["best_guide"]],
                    "asks": None, "cheat": False})
    return {"rows": rows, "best_guide_minus_router_ci": r["best_guide_minus_router_ci95"]}


# -- Audit -----------------------------------------------------------------------------

AUDIT_EFFECTS = [  # (json key, label, value key)
    ("C_curiosity_gain_on_T", "C 호기심 (T 문제)", "gain"),
    ("F_arbiter_minus_merge", "F 중재자 - 합치기", "diff"),
    ("H_memory_gain", "H 장기 기억 켬 - 끔", "gain"),
    ("H_transfer_gain", "H 다른 과제로 옮기기", "gain"),
]


def audit_effects() -> list[dict]:
    """Per stage and seed: effect, question-level CI, session-level CI."""
    r = _load("mk1-audit.json")
    rows = []
    for key, label, vkey in AUDIT_EFFECTS:
        for seed in SEEDS:
            d = r["D_seeds"][seed][key]
            rows.append({"stage": label, "seed": seed, "effect": d[vkey],
                         "ci_questions": d["ci_questions"], "ci_sessions": d["ci_sessions"]})
    return rows


def audit_leakage() -> dict:
    r = _load("mk1-audit.json")
    a = r["A_leakage"]
    return {
        "test_seen_share": a["test_questions_with_passage_seen_in_reader_training"],
        "reader": {"본 기사": a["reader_right_rate_seen_passage"], "처음 보는 기사": a["reader_right_rate_unseen_passage"]},
        "sessions": {seed: r["D_seeds"][seed]["D_articles_live"] for seed in SEEDS},
        "rule_world_sentences_seen": r["B_rule_world_sentences"],
    }


def audit_unseen_accuracy() -> dict:
    """Stage-F learned arbiter on board reliabilities it never trained on (0.9 inside, 0.25 outside)."""
    r = _load("mk1-audit.json")["E_arbiter_unseen_accuracies"]
    names = [("arbiter", "학습한 중재자"), ("bayes_with_training_levels", "베이즈 (학습 수준만 앎)"),
             ("board_only", "안내판만"), ("article_only", "기사만")]
    return {lv: {label: r[lv][k] for k, label in names} for lv in ("0.9", "0.25")}


# -- I: counting arbiter outside experience --------------------------------------------

I_SYSTEMS = [("learned arbiter", "학습한 중재자"), ("counting arbiter", "세는 중재자"),
             ("board only", "안내판만"), ("article only", "기사만")]
I_SEEN = ("1.0", "0.75", "0.5")


def i_by_accuracy() -> dict:
    """Score per board reliability (sorted high to low), per system: per-seed values and mean +- t CI."""
    r = _load("mk1-i_outside_experience.json")["seeds"]
    levels = sorted(r[SEEDS[0]]["by_accuracy"], key=float, reverse=True)
    out = {"levels": levels, "seen": [lv in I_SEEN for lv in levels], "systems": {}}
    for k, label in I_SYSTEMS:
        per_seed = {s: [r[s]["by_accuracy"][lv][k] for lv in levels] for s in SEEDS}
        stats = [mean_ci([per_seed[s][i] for s in SEEDS]) for i in range(len(levels))]
        out["systems"][label] = {"per_seed": per_seed, "mean": [m for m, _ in stats], "ci": [c for _, c in stats]}
    return out


def i_gaps() -> list[dict]:
    """Counting minus learned arbiter, outside (0.25/0.1) and inside (seen levels), session CIs."""
    r = _load("mk1-i_outside_experience.json")["seeds"]
    rows = []
    for key, label in [("outside_0.25_0.1", "경험 밖 (0.25, 0.1)"), ("inside_seen_levels", "경험 안 (1.0, 0.75, 0.5)")]:
        for s in SEEDS:
            rows.append({"group": label, "seed": s, "effect": r[s][key]["counting_minus_learned"], "ci": r[s][key]["ci_sessions"]})
    return rows


def i_priors() -> dict:
    r = _load("mk1-i_outside_experience.json")["seeds"]
    return {s: r[s]["prior_alpha_beta"] for s in SEEDS}


# -- K: noticing unexpected change -----------------------------------------------------

K_WORLDS = [("betrayal -> 0.25", "배신 → 0.25"), ("betrayal -> 0.1", "배신 → 0.1\n(학습 때 없던 수준)"),
            ("mild change", "약한 변화"), ("no change", "변화 없음")]


def k_gaps() -> list[dict]:
    """Change-detecting minus counting arbiter per world and seed, session CIs."""
    r = _load("mk1-k_change.json")["seeds"]
    return [{"world": label, "seed": s, "effect": r[s][k]["change_minus_counting"], "ci": r[s][k]["ci_sessions"]}
            for k, label in K_WORLDS for s in SEEDS]


def k_after_betrayal() -> dict:
    """C score from 10 problems after the change on, per seed, for both arbiters."""
    r = _load("mk1-k_change.json")["seeds"]
    worlds = [("betrayal -> 0.25", "배신 → 0.25"), ("betrayal -> 0.1", "배신 → 0.1"), ("mild change", "약한 변화")]
    return {label: {name: [r[s][k]["C_10_plus_after_change"][key] for s in SEEDS]
                    for key, name in [("counting-switch", "세는 중재자"), ("change-switch", "변화 감지 중재자")]}
            for k, label in worlds}


def k_hazards() -> dict:
    r = _load("mk1-k_change.json")["seeds"]
    return {s: r[s]["fitted"]["change-switch"]["hazard"] for s in SEEDS}


# -- One-line verdict table ------------------------------------------------------------

def verdicts() -> list[dict]:
    rows = []
    for name, label in [("mk1-f_conflict.json", "F 출처 충돌 (중재자)"), ("mk1-h_long_term_memory.json", "H 장기 기억"),
                        ("mk1-i_outside_experience.json", "I 경험 밖 (세는 중재자)"), ("mk1-k_change.json", "K 변화 알아채기")]:
        r = _load(name)
        if r:
            c = r["criteria"]
            rows.append({"stage": label, "passed": sum(bool(v) for v in c.values()), "total": len(c), "graduated": r["graduated"]})
    return rows
