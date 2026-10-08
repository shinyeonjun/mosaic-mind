"""Aggregate Experiment 010 against matched oracle-decoder references."""

import json
from collections import defaultdict

from cognitive_lab.active.exp007.aggregate import _paired_change
from cognitive_lab.active.exp007.train import CONTROLLERS, MAX_PROBE_BUDGET, PROJECT_ROOT
from cognitive_lab.active.exp007.world import FOLD_COUNT, FOLD_TEST_RULES, RULES
from cognitive_lab.active.exp010.train import SEEDS

COUNT_KEYS = ("episodes", "correct", "probes")
REFERENCE_NAMES = (
    "history_only_query_policy_with_exact_answer_decoder",
    "belief_state_imitation_with_exact_answer_decoder",
    "reinforce_with_exact_answer_decoder",
    "information_gain_with_exact_answer_decoder",
)


def _counts(metrics: dict) -> dict[str, int]:
    return {
        "episodes": int(metrics["episodes"]),
        "correct": int(metrics["correct_episodes"]),
        "probes": int(metrics["total_probes"]),
    }


def _summary(counts: dict, probe_cost: float) -> dict:
    episodes = counts["episodes"]
    return {
        "episodes": episodes,
        "accuracy": round(counts["correct"] / episodes, 4),
        "mean_probes": round(counts["probes"] / episodes, 4),
        "mean_task_reward": round(
            counts["correct"] / episodes - probe_cost * counts["probes"] / episodes, 4
        ),
    }


def run() -> dict:
    results_dir = PROJECT_ROOT / "artifacts" / "results"
    expected = {
        (fold, seed): results_dir / f"experiment-010_fold-{fold}_seed-{seed}.json"
        for fold in range(FOLD_COUNT)
        for seed in SEEDS
    }
    missing = [path.name for path in expected.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Experiment 010 aggregation needs all 18 fold/seed results; missing: "
            + ", ".join(missing)
        )

    runs = {}
    probe_costs = set()
    for (fold, seed), path in expected.items():
        result = json.loads(path.read_text(encoding="utf-8"))
        expected_rules = [RULES[index].name for index in FOLD_TEST_RULES[fold]]
        actual_rules = [
            RULES[index].name
            for index in result.get("data", {}).get("test_rule_ids", ())
        ]
        if (
            result.get("experiment") != "010-history-only-end-to-end-rule-decoder"
            or result.get("fold") != fold
            or result.get("seed") != seed
            or actual_rules != expected_rules
            or result.get("input_contract", {}).get("controller_receives_exact_posterior_mask") is not False
            or result.get("input_contract", {}).get("final_answer_uses_exact_posterior") is not False
            or result.get("input_contract", {}).get("simulator_stops_on_exact_posterior_singleton") is not False
        ):
            raise ValueError(f"unexpected Experiment 010 metadata in {path.name}")
        if set(result["reference_test_metrics"]["history_only_query_policy_with_exact_answer_decoder"]) != set(CONTROLLERS):
            raise ValueError(f"incomplete matched references in {path.name}")
        probe_costs.add(float(result["training"]["probe_cost_for_checkpoint_selection"]))
        runs[(fold, seed)] = result
    if len(probe_costs) != 1:
        raise ValueError("all Experiment 010 runs must use the same probe cost")
    probe_cost = probe_costs.pop()

    total = defaultdict(lambda: {key: 0 for key in COUNT_KEYS})
    by_seed = defaultdict(lambda: {key: 0 for key in COUNT_KEYS})
    decoder_total = defaultdict(lambda: {
        "episodes": 0,
        "exact_rule_correct": 0,
        "scope_correct": 0,
        "reducer_correct": 0,
    })
    by_rule = defaultdict(lambda: {"episodes": 0, "correct": 0, "exact_rule_correct": 0})
    policy_diagnostics = defaultdict(lambda: {
        "policy_decisions": 0,
        "query_decisions": 0,
        "stop_decisions": 0,
        "teacher_action_matches": 0,
        "information_gain_optimal_queries": 0,
    })

    def add(
        policy: str,
        controller: str,
        seed: int,
        budget: int,
        metrics: dict,
        *,
        end_to_end: bool = False,
    ) -> None:
        counts = _counts(metrics)
        for target in (total[(controller, policy, budget)], by_seed[(controller, policy, budget, seed)]):
            for key, value in counts.items():
                target[key] += value
        for rule_id, rule in enumerate(RULES):
            row = metrics.get("per_rule_accuracy", {}).get(rule.name)
            if row:
                key = (controller, policy, budget, rule_id)
                by_rule[key]["episodes"] += int(row["episodes"])
                by_rule[key]["correct"] += int(row["correct_episodes"])
                if end_to_end:
                    by_rule[key]["exact_rule_correct"] += int(
                        row["exact_rule_prediction_episodes"]
                    )
        if end_to_end:
            decoder = decoder_total[(controller, budget)]
            decoder["episodes"] += counts["episodes"]
            decoder["exact_rule_correct"] += int(metrics["exact_rule_prediction_episodes"])
            decoder["scope_correct"] += int(metrics["scope_correct_episodes"])
            decoder["reducer_correct"] += int(metrics["reducer_correct_episodes"])
            for key, value in metrics["teacher_agreement"].items():
                if key in policy_diagnostics[(controller, budget)]:
                    policy_diagnostics[(controller, budget)][key] += int(value)

    for (_, seed), result in runs.items():
        references = result["reference_test_metrics"]
        for controller in CONTROLLERS:
            for budget in range(1, MAX_PROBE_BUDGET + 1):
                budget_key = str(budget)
                add(
                    "end_to_end", controller, seed, budget,
                    result["controllers"][controller]["test_by_budget"][budget_key],
                    end_to_end=True,
                )
                for reference in REFERENCE_NAMES:
                    if reference == "information_gain_with_exact_answer_decoder":
                        metrics = references[reference][budget_key]
                    else:
                        metrics = references[reference][controller][budget_key]
                    add(reference, controller, seed, budget, metrics)

    pooled = {}
    comparisons = {}
    decoder_metrics = {}
    held_out_accuracy = {}
    teacher_agreement = {}
    for controller in CONTROLLERS:
        pooled[controller] = {}
        comparisons[controller] = {}
        decoder_metrics[controller] = {}
        held_out_accuracy[controller] = {}
        teacher_agreement[controller] = {}
        for budget in range(1, MAX_PROBE_BUDGET + 1):
            pooled[controller][str(budget)] = {
                policy: _summary(total[(controller, policy, budget)], probe_cost)
                for policy in ("end_to_end", *REFERENCE_NAMES)
            }
            decoded = decoder_total[(controller, budget)]
            decoder_metrics[controller][str(budget)] = {
                "episodes": decoded["episodes"],
                "exact_rule_accuracy": round(decoded["exact_rule_correct"] / decoded["episodes"], 4),
                "scope_accuracy": round(decoded["scope_correct"] / decoded["episodes"], 4),
                "reducer_accuracy": round(decoded["reducer_correct"] / decoded["episodes"], 4),
            }
            held_out_accuracy[controller][str(budget)] = {
                RULES[rule_id].name: {
                    "episodes": counts["episodes"],
                    "answer_accuracy": round(counts["correct"] / counts["episodes"], 4),
                    "exact_rule_accuracy": round(
                        counts["exact_rule_correct"] / counts["episodes"], 4
                    ) if policy == "end_to_end" else None,
                }
                for (current_controller, policy, current_budget, rule_id), counts in by_rule.items()
                if current_controller == controller
                and policy == "end_to_end"
                and current_budget == budget
                and counts["episodes"]
            }
            counts = policy_diagnostics[(controller, budget)]
            teacher_agreement[controller][str(budget)] = {
                **counts,
                "exact_teacher_action_rate": round(
                    counts["teacher_action_matches"] / counts["policy_decisions"], 4
                ) if counts["policy_decisions"] else None,
                "information_gain_optimal_rate": round(
                    counts["information_gain_optimal_queries"] / counts["query_decisions"], 4
                ) if counts["query_decisions"] else None,
            }
        for reference in REFERENCE_NAMES:
            comparisons[controller][f"end_to_end_minus_{reference}"] = {
                str(budget): _paired_change(
                    {
                        seed: by_seed[(controller, reference, budget, seed)]
                        for seed in SEEDS
                    },
                    {
                        seed: by_seed[(controller, "end_to_end", budget, seed)]
                        for seed in SEEDS
                    },
                    probe_cost,
                )
                for budget in range(1, MAX_PROBE_BUDGET + 1)
            }

    output = {
        "experiment": "010-history-only-end-to-end-rule-decoder-aggregate",
        "source_result_files": len(runs),
        "folds": FOLD_COUNT,
        "seeds": list(SEEDS),
        "probe_cost": probe_cost,
        "pooled_task_metrics": pooled,
        "paired_reward_comparisons": comparisons,
        "decoded_rule_metrics": decoder_metrics,
        "held_out_composition_accuracy": held_out_accuracy,
        "policy_diagnostics": teacher_agreement,
        "uncertainty_method": "For each seed, pool its three held-out folds, calculate paired task-reward differences, then use a two-sided t interval across six seed clusters (df=5; critical value 2.571). Exploratory, not a definitive significance claim.",
        "interpretation_limit": "The agent has no exact posterior input, oracle stopping rule, or posterior-based answer decoder. It predicts from a fixed scope/reducer vocabulary; this tests held-out combinations of known primitives, not open-world invention of new primitives.",
    }
    output_path = results_dir / "experiment-010_aggregate.json"
    output["output"] = str(output_path)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({
        "source_result_files": output["source_result_files"],
        "pooled_task_metrics": pooled,
        "paired_reward_comparisons": comparisons,
        "decoded_rule_metrics": decoder_metrics,
        "saved": str(output_path),
    }, indent=2))
    return output


if __name__ == "__main__":
    run()
