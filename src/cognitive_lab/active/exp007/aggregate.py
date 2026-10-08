"""Aggregate the complete Experiment 007 fold/seed grid."""

import json
import math
from pathlib import Path

from cognitive_lab.active.exp007.train import (
    CONTROLLERS,
    MAX_PROBE_BUDGET,
    PROJECT_ROOT,
    T_CRITICAL_95_DF5,
)
from cognitive_lab.active.exp007.world import FOLD_COUNT, FOLD_TEST_RULES, RULES

SEEDS = tuple(range(42, 48))
BASELINES = ("random", "fixed_order", "information_gain")


def _counts(summary: dict) -> dict[str, int]:
    return {
        "episodes": int(summary["episodes"]),
        "correct": int(summary["correct_episodes"]),
        "probes": int(summary["total_probes"]),
        "identified": int(summary["identified_episodes"]),
    }


def _reward(counts: dict[str, int], probe_cost: float) -> float:
    return counts["correct"] / counts["episodes"] - probe_cost * (
        counts["probes"] / counts["episodes"]
    )


def _summary(counts: dict[str, int], probe_cost: float) -> dict:
    episodes = counts["episodes"]
    return {
        "episodes": episodes,
        "accuracy": round(counts["correct"] / episodes, 4),
        "mean_probes": round(counts["probes"] / episodes, 4),
        "mean_task_reward": round(_reward(counts, probe_cost), 4),
        "rule_identification_rate": round(counts["identified"] / episodes, 4),
    }


def _paired_change(before: dict[int, dict], after: dict[int, dict], probe_cost: float) -> dict:
    seed_changes = {
        seed: _reward(after[seed], probe_cost) - _reward(before[seed], probe_cost)
        for seed in sorted(set(before) & set(after))
    }
    mean = sum(seed_changes.values()) / len(seed_changes)
    interval = None
    if len(seed_changes) == 6:
        variance = sum((change - mean) ** 2 for change in seed_changes.values()) / 5
        margin = T_CRITICAL_95_DF5 * math.sqrt(variance / 6)
        interval = [round(mean - margin, 4), round(mean + margin, 4)]
    return {
        "seed_clusters": len(seed_changes),
        "mean_task_reward_change": round(mean, 4),
        "reward_change_95pct_seed_cluster_t_interval": interval,
        "per_seed_change": {str(seed): round(value, 4) for seed, value in seed_changes.items()},
    }


def run() -> dict:
    results_dir = PROJECT_ROOT / "artifacts" / "results"
    expected_runs = {
        (fold, seed): results_dir / f"experiment-007_fold-{fold}_seed-{seed}.json"
        for fold in range(FOLD_COUNT)
        for seed in SEEDS
    }
    missing = [path.name for path in expected_runs.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Experiment 007 aggregation needs all 18 fold/seed results; missing: "
            + ", ".join(missing)
        )

    runs = {}
    probe_costs = set()
    for (fold, seed), path in expected_runs.items():
        result = json.loads(path.read_text(encoding="utf-8"))
        expected_rules = [RULES[index].name for index in FOLD_TEST_RULES[fold]]
        if (
            result.get("experiment") != "007-compositional-rule-generalization"
            or result.get("fold") != fold
            or result.get("seed") != seed
            or result.get("data", {}).get("test_rules") != expected_rules
        ):
            raise ValueError(f"unexpected experiment metadata in {path.name}")
        probe_costs.add(float(result["training"]["probe_cost"]))
        runs[(fold, seed)] = result
    if len(probe_costs) != 1:
        raise ValueError("all runs must use the same probe cost")
    probe_cost = probe_costs.pop()

    policies = ("no_probe", *BASELINES, *CONTROLLERS)
    total: dict[tuple[str, int], dict[str, int]] = {}
    by_seed: dict[tuple[str, int, int], dict[str, int]] = {}
    per_rule: dict[tuple[str, int, int], dict[str, int]] = {}

    for (fold, seed), result in runs.items():
        for policy in policies:
            budgets = (0,) if policy == "no_probe" else range(1, MAX_PROBE_BUDGET + 1)
            for budget in budgets:
                if policy == "no_probe":
                    metrics = result["baselines"]["no_probe"]
                elif policy in BASELINES:
                    metrics = result["baselines"][policy]["by_budget"][str(budget)]
                else:
                    metrics = result["controllers"][policy]["test_by_budget"][str(budget)]
                counts = _counts(metrics)
                for target in (total.setdefault((policy, budget), {key: 0 for key in counts}),
                               by_seed.setdefault((policy, budget, seed), {key: 0 for key in counts})):
                    for key, value in counts.items():
                        target[key] += value
                for rule_id, rule in enumerate(RULES):
                    rule_metrics = metrics["per_rule_accuracy"].get(rule.name)
                    if not rule_metrics:
                        continue
                    key = (policy, budget, rule_id)
                    counts_by_rule = per_rule.setdefault(key, {"episodes": 0, "correct": 0})
                    counts_by_rule["episodes"] += rule_metrics["episodes"]
                    counts_by_rule["correct"] += rule_metrics["correct_episodes"]

    comparisons = {}
    compare_pairs = [
        *((controller, baseline) for controller in CONTROLLERS for baseline in ("random", "information_gain")),
        ("rims", "gru_parameter_matched"),
    ]
    for treatment, reference in compare_pairs:
        pair_name = f"{treatment}_minus_{reference}"
        comparisons[pair_name] = {}
        for budget in range(1, MAX_PROBE_BUDGET + 1):
            before = {
                seed: by_seed[(reference, budget, seed)] for seed in SEEDS
            }
            after = {
                seed: by_seed[(treatment, budget, seed)] for seed in SEEDS
            }
            comparisons[pair_name][str(budget)] = _paired_change(
                before, after, probe_cost
            )

    output = {
        "experiment": "007-compositional-rule-generalization-aggregate",
        "source_result_files": len(runs),
        "folds": FOLD_COUNT,
        "seeds": list(SEEDS),
        "probe_cost": probe_cost,
        "pooled_metrics": {
            policy: {
                str(budget): _summary(counts, probe_cost)
                for (name, budget), counts in total.items() if name == policy
            }
            for policy in policies
        },
        "paired_reward_comparisons": comparisons,
        "held_out_composition_accuracy": {
            policy: {
                str(budget): {
                    RULES[rule_id].name: {
                        "episodes": counts["episodes"],
                        "accuracy": round(counts["correct"] / counts["episodes"], 4),
                    }
                    for (name, current_budget, rule_id), counts in per_rule.items()
                    if name == policy and current_budget == budget and counts["episodes"]
                }
                for budget in ((0,) if policy == "no_probe" else range(1, MAX_PROBE_BUDGET + 1))
            }
            for policy in policies
        },
        "uncertainty_method": "For each seed, pool its three held-out folds, compute paired reward differences, then use a two-sided t interval over six seed clusters (df=5; critical value 2.571). Exploratory, not a definitive significance claim.",
        "interpretation_limit": "All 18 candidate rules and exact posterior updates are available at test time; this measures transfer to held-out compositions inside the fixed grammar.",
    }
    output_path = results_dir / "experiment-007_aggregate.json"
    output["output"] = str(output_path)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({
        "pooled_metrics": output["pooled_metrics"],
        "paired_reward_comparisons": output["paired_reward_comparisons"],
        "saved": str(output_path),
    }, indent=2))
    return output


if __name__ == "__main__":
    run()
