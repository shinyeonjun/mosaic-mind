"""Collect headline results from artifacts/results for overview notebooks (no training here).

Every function returns plain Python structures and skips results that do not exist yet,
so the overview notebook can be re-run at any time while experiments are in progress.
"""

import json
import math
import statistics
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS = PROJECT_ROOT / "artifacts" / "results"
SEEDS = tuple(range(42, 48))
T95 = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571}  # two-sided t quantiles by sample size


def _load(name: str) -> dict | None:
    path = RESULTS / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def mean_ci(values: list[float]) -> tuple[float, float]:
    """Mean and half-width of a two-sided 95% t interval (0 for a single value)."""
    if len(values) < 2:
        return (values[0] if values else math.nan), 0.0
    return statistics.mean(values), T95[len(values)] * statistics.stdev(values) / math.sqrt(len(values))


# -- Rule world v1 ---------------------------------------------------------------------

V1_SYSTEMS = [  # (label, result agent name, role)
    ("LLM 혼자 · 프롬프트", "l1-generate", "llm"),
    ("LLM 혼자 · 미세조정", "l1-lora-generate", "llm"),
    ("규칙 파서 + 코드 판단", "p1-rule", "rule"),
    ("LLM 읽기 + 코드 판단 (P2)", "p2-lora", "parts"),
    ("작은 NLI 읽기 + 코드 판단 (P2-nli)", "p2-nli", "parts"),
]
V1_SPLITS = [("iid", "일반"), ("phrasing", "새 말투"), ("composition", "새 조합"), ("entity", "새 이름")]


def v1_scores() -> list[dict]:
    rows = []
    for label, agent, role in V1_SYSTEMS:
        for split, split_label in V1_SPLITS:
            result = _load(f"world-v1_{agent}_{split}-test_seed-42.json")
            if result is not None:
                rows.append({"system": label, "role": role, "split": split_label, "score": result["mean_score"]})
    return rows


# -- Rule world v2 ---------------------------------------------------------------------

def v2_learning_curves(agents: list[str]) -> dict[str, list[float]]:
    """Mean score per episode index (1..40), pooled over sessions and the seeds available."""
    curves = {}
    for agent in agents:
        totals, counts = [0.0] * 40, [0] * 40
        for seed in SEEDS:
            result = _load(f"world-v2_{agent}_test_seed-{seed}.json")
            if result is None:
                continue
            for record in result["predictions"]:
                totals[record["index"]] += record["score"]
                counts[record["index"]] += 1
        if any(counts):
            curves[agent] = [t / c for t, c in zip(totals, counts)]
    return curves


def paired_vs(agents: list[str], reference: str, suffix: str) -> list[dict]:
    """Seed-paired difference in overall score vs `reference` (mean, 95% CI half-width)."""
    rows = []
    for agent in agents:
        diffs = []
        for seed in SEEDS:
            a, b = _load(f"world-v2_{agent}_test_seed-{seed}{suffix}.json"), _load(f"world-v2_{reference}_test_seed-{seed}{suffix}.json")
            if a and b:
                diffs.append(a["overall"]["mean_score"] - b["overall"]["mean_score"])
        if diffs:
            mean, half = mean_ci(diffs)
            rows.append({"agent": agent, "diff": mean, "half": half, "seeds": len(diffs)})
    return rows


def overall(agents: list[str], suffix: str = "") -> list[dict]:
    rows = []
    for agent in agents:
        values = [r["overall"]["mean_score"] for seed in SEEDS
                  if (r := _load(f"world-v2_{agent}_test_seed-{seed}{suffix}.json"))]
        if values:
            mean, half = mean_ci(values)
            rows.append({"agent": agent, "mean": mean, "half": half, "seeds": len(values)})
    return rows


def scaling_timing() -> list[tuple[int, str, float]]:
    data = _load("world-v2_scaling-timing_seed-42.json") or []
    return [(int(s), name, float(seconds) * 1000) for s, name, seconds in data]


# -- Interface learning (rule world v1, composition split) ------------------------------

INTERFACE_SYSTEMS = [  # (label, file stem, role)
    ("일반 신경망 판단 + 설계 형식 (P2-N)", "interface-p2-neural", "flat"),
    ("대칭 판단 + 스스로 만든 형식", "interface-anchored-learned", "learned"),
    ("대칭 판단 + 스스로 만든 형식 + 커리큘럼", "interface-anchored-learned-curriculum", "curriculum"),
    ("대칭 판단 + 설계 형식", "interface-anchored-designed", "designed"),
    ("대칭·출처 구조 판단 + 설계 형식", "interface-anchored-designed-sourcefold", "sourcefold"),
    ("대칭·출처 구조 판단 + 스스로 만든 형식", "interface-anchored-learned-sourcefold", "sourcefold-learned"),
]


def interface_result(stem: str, split: str = "composition", seed: int = 42) -> dict | None:
    """One interface-learning run: test score, probe accuracy, and validation history."""
    result = _load(f"world-v1_{stem}_{split}_seed-{seed}.json")
    if result is None:
        return None
    return {"score": result["test"]["mean_score"], "by_case": result["test"]["by_case"],
            "probe": (result.get("probe") or {}).get("pair_relation_probe_accuracy"),
            "validation": [h["validation"] for h in result["history"]],
            "train_loss": [h["train_loss"] for h in result["history"]],
            "seconds": result.get("training_seconds")}


def interface_result_archived(name: str) -> dict | None:
    """A run kept under artifacts/results/archive (e.g. an 8-epoch result later overwritten)."""
    result = _load(f"archive/{name}")
    if result is None:
        return None
    return {"score": result["test"]["mean_score"], "by_case": result["test"]["by_case"],
            "probe": (result.get("probe") or {}).get("pair_relation_probe_accuracy"),
            "seconds": result.get("training_seconds")}


def interface_splits(stems: list[str], seed: int = 42) -> list[dict]:
    """Test score and message-probe accuracy per split for each interface system."""
    rows = []
    for stem in stems:
        for split, split_label in V1_SPLITS:
            result = _load(f"world-v1_{stem}_{split}_seed-{seed}.json")
            if result is not None:
                probe = (result.get("probe") or {}).get("pair_relation_probe_accuracy")
                rows.append({"stem": stem, "split": split_label, "score": result["test"]["mean_score"], "probe": probe})
    return rows


def interface_seeds(split: str = "composition") -> list[dict]:
    rows = []
    for label, stem, role in INTERFACE_SYSTEMS:
        scores = {}
        for seed in SEEDS:
            result = _load(f"world-v1_{stem}_{split}_seed-{seed}.json")
            if result is not None:
                scores[seed] = result["test"]["mean_score"]
        if scores:
            rows.append({"system": label, "role": role, "scores": scores})
    return rows
