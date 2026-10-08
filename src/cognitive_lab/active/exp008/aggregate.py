"""Aggregate Experiment 008 and compare it with matched Experiment 007 runs."""

import json
from collections import defaultdict

from cognitive_lab.active.exp007.aggregate import (
    _counts,
    _paired_change,
    _summary,
)
from cognitive_lab.active.exp007.train import (
    CONTROLLERS,
    MAX_PROBE_BUDGET,
    PROJECT_ROOT,
)
from cognitive_lab.active.exp007.world import FOLD_COUNT, FOLD_TEST_RULES, RULES

SEEDS = tuple(range(42, 48))
COUNT_KEYS = ("episodes", "correct", "probes", "identified")


def run() -> dict:
    results_dir = PROJECT_ROOT / "artifacts" / "results"
    expected = {
        (fold, seed): results_dir / f"experiment-008_fold-{fold}_seed-{seed}.json"
        for fold in range(FOLD_COUNT)
        for seed in SEEDS
    }
    missing = [path.name for path in expected.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Experiment 008 aggregation needs all 18 fold/seed results; missing: "
            + ", ".join(missing)
        )

    runs = {}
    probe_costs = set()
    for (fold, seed), path in expected.items():
        result = json.loads(path.read_text(encoding="utf-8"))
        expected_rules = [RULES[index].name for index in FOLD_TEST_RULES[fold]]
        if (
            result.get("experiment") != "008-information-gain-teacher-imitation"
            or result.get("fold") != fold
            or result.get("seed") != seed
            or [RULES[index].name for index in result.get("data", {}).get("test_rule_ids", ())]
            != expected_rules
        ):
            raise ValueError(f"unexpected Experiment 008 metadata in {path.name}")
        reference = result["reference_test_metrics"]
        if set(reference["reinforce"]) != set(CONTROLLERS):
            raise ValueError(f"incomplete Experiment 007 controller reference in {path.name}")
        probe_costs.add(float(result["training"]["probe_cost_for_checkpoint_selection"]))
        runs[(fold, seed)] = result
    if len(probe_costs) != 1:
        raise ValueError("all Experiment 008 runs must use the same probe cost")
    probe_cost = probe_costs.pop()

    total = defaultdict(lambda: {key: 0 for key in COUNT_KEYS})
    by_seed = defaultdict(lambda: {key: 0 for key in COUNT_KEYS})
    per_rule = defaultdict(lambda: {"episodes": 0, "correct": 0})
    teacher_counts = defaultdict(lambda: {
        "policy_decisions": 0,
        "query_decisions": 0,
        "stop_decisions": 0,
        "teacher_action_matches": 0,
        "information_gain_optimal_queries": 0,
    })

    def add(policy: str, seed: int, budget: int, metrics: dict) -> None:
        counts = _counts(metrics)
        for target in (total[(policy, budget)], by_seed[(policy, budget, seed)]):
            for key, value in counts.items():
                target[key] += value
        for rule_id, rule in enumerate(RULES):
            row = metrics["per_rule_accuracy"].get(rule.name)
            if row:
                key = (policy, budget, rule_id)
                per_rule[key]["episodes"] += row["episodes"]
                per_rule[key]["correct"] += row["correct_episodes"]

    for (fold, seed), result in runs.items():
        for controller in CONTROLLERS:
            reference = result["reference_test_metrics"]
            for budget in range(1, MAX_PROBE_BUDGET + 1):
                budget_key = str(budget)
                imitation = result["controllers"][controller]["test_by_budget"][budget_key]
                reinforce = reference["reinforce"][controller][budget_key]
                information_gain = reference["information_gain"][budget_key]
                add(f"{controller}_imitation", seed, budget, imitation)
                add(f"{controller}_reinforce", seed, budget, reinforce)
                add(f"{controller}_information_gain", seed, budget, information_gain)
                for key, value in imitation["teacher_agreement"].items():
                    if key in teacher_counts[(controller, budget)]:
                        teacher_counts[(controller, budget)][key] += value

    pooled = {}
    comparisons = {}
    held_out_accuracy = {}
    teacher_agreement = {}
    for controller in CONTROLLERS:
        pooled[controller] = {}
        held_out_accuracy[controller] = {}
        teacher_agreement[controller] = {}
        for budget in range(1, MAX_PROBE_BUDGET + 1):
            pooled[controller][str(budget)] = {
                label: _summary(total[(f"{controller}_{label}", budget)], probe_cost)
                for label in ("imitation", "reinforce", "information_gain")
            }
            held_out_accuracy[controller][str(budget)] = {
                label: {
                    RULES[rule_id].name: {
                        "episodes": counts["episodes"],
                        "accuracy": round(counts["correct"] / counts["episodes"], 4),
                    }
                    for (policy, current_budget, rule_id), counts in per_rule.items()
                    if policy == f"{controller}_{label}"
                    and current_budget == budget
                    and counts["episodes"]
                }
                for label in ("imitation", "reinforce", "information_gain")
            }
            counts = teacher_counts[(controller, budget)]
            teacher_agreement[controller][str(budget)] = {
                **counts,
                "exact_teacher_action_rate": round(
                    counts["teacher_action_matches"] / counts["policy_decisions"], 4
                ) if counts["policy_decisions"] else None,
                "information_gain_optimal_rate": round(
                    counts["information_gain_optimal_queries"] / counts["query_decisions"], 4
                ) if counts["query_decisions"] else None,
            }

        comparisons[controller] = {}
        for reference in ("reinforce", "information_gain"):
            pair_name = f"imitation_minus_{reference}"
            comparisons[controller][pair_name] = {}
            for budget in range(1, MAX_PROBE_BUDGET + 1):
                before = {
                    seed: by_seed[(f"{controller}_{reference}", budget, seed)]
                    for seed in SEEDS
                }
                after = {
                    seed: by_seed[(f"{controller}_imitation", budget, seed)]
                    for seed in SEEDS
                }
                comparisons[controller][pair_name][str(budget)] = _paired_change(
                    before, after, probe_cost
                )

    output = {
        "experiment": "008-information-gain-teacher-imitation-aggregate",
        "source_result_files": len(runs),
        "folds": FOLD_COUNT,
        "seeds": list(SEEDS),
        "probe_cost": probe_cost,
        "pooled_metrics": pooled,
        "paired_reward_comparisons": comparisons,
        "teacher_action_agreement": teacher_agreement,
        "held_out_composition_accuracy": held_out_accuracy,
        "uncertainty_method": "For each seed, pool its three held-out folds, calculate paired task-reward changes, then use a two-sided t interval over six seed clusters (df=5; critical value 2.571). Exploratory, not a definitive significance claim.",
        "interpretation_limit": "Teacher labels are generated only from training-fold hidden rules. The exact information-gain teacher and posterior still use the fixed 18-rule catalog, including held-out compositions.",
    }
    output_path = results_dir / "experiment-008_aggregate.json"
    output["output"] = str(output_path)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({
        "pooled_metrics": pooled,
        "paired_reward_comparisons": comparisons,
        "teacher_action_agreement": teacher_agreement,
        "saved": str(output_path),
    }, indent=2))
    return output


if __name__ == "__main__":
    run()
