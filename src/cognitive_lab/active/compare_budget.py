"""Compare learned budget growth with an appended information-gain probe."""

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
from types import SimpleNamespace

import torch

from cognitive_lab.active.baselines import generate_trials
from cognitive_lab.active.controller import FOLD_TEST_RULES, MAX_PROBE_BUDGET
from cognitive_lab.active.diagnose_rl import _load_runs, _rollout_model
from cognitive_lab.active.evaluate_extension import _extra_probe_result, _rounded_metrics
from cognitive_lab.active.train_controller_rl import (
    CONTROLLERS,
    PROJECT_ROOT,
    _build_controller,
)
from cognitive_lab.train import choose_device

T_CRITICAL_95_DF5 = 2.571


def _record(correct: bool, probes: int) -> dict:
    return {"correct": correct, "probes": probes}


def _add_pair(totals: Counter, before: dict, after: dict) -> None:
    was_correct = bool(before["correct"])
    is_correct = bool(after["correct"])
    transition = (
        "correct_to_correct" if was_correct and is_correct else
        "correct_to_wrong" if was_correct else
        "wrong_to_correct" if is_correct else "wrong_to_wrong"
    )
    totals["episodes"] += 1
    totals["before_correct"] += int(was_correct)
    totals["after_correct"] += int(is_correct)
    totals["before_probes"] += before["probes"]
    totals["after_probes"] += after["probes"]
    totals[transition] += 1


def _summarize_pair(
    totals: Counter, seed_totals: dict[int, Counter], probe_cost: float
) -> dict:
    episodes = totals["episodes"]
    before_accuracy = totals["before_correct"] / episodes
    after_accuracy = totals["after_correct"] / episodes
    before_probes = totals["before_probes"] / episodes
    after_probes = totals["after_probes"] / episodes
    seed_deltas = {
        seed: (
            (counts["after_correct"] - counts["before_correct"])
            - probe_cost * (counts["after_probes"] - counts["before_probes"])
        ) / counts["episodes"]
        for seed, counts in sorted(seed_totals.items())
    }
    mean_delta = sum(seed_deltas.values()) / len(seed_deltas)
    if len(seed_deltas) == 6:
        variance = sum((value - mean_delta) ** 2 for value in seed_deltas.values()) / 5
        margin = T_CRITICAL_95_DF5 * math.sqrt(variance / 6)
        interval = [round(mean_delta - margin, 4), round(mean_delta + margin, 4)]
    else:
        interval = None
    reward_change = (
        (after_accuracy - before_accuracy)
        - probe_cost * (after_probes - before_probes)
    )
    return {
        "episodes": episodes,
        "before": {
            "accuracy": round(before_accuracy, 4),
            "mean_probes": round(before_probes, 4),
            "mean_task_reward": round(before_accuracy - probe_cost * before_probes, 4),
        },
        "after": {
            "accuracy": round(after_accuracy, 4),
            "mean_probes": round(after_probes, 4),
            "mean_task_reward": round(after_accuracy - probe_cost * after_probes, 4),
        },
        "paired_change": {
            "accuracy_percentage_points": round((after_accuracy - before_accuracy) * 100, 3),
            "mean_probes": round(after_probes - before_probes, 4),
            "mean_task_reward": round(reward_change, 4),
            "reward_change_95pct_seed_cluster_t_interval": interval,
            "fixed_previously_wrong": totals["wrong_to_correct"],
            "made_previously_correct_wrong": totals["correct_to_wrong"],
            "unchanged_correct": totals["correct_to_correct"],
            "unchanged_wrong": totals["wrong_to_wrong"],
        },
        "seed_cluster_reward_changes": {
            str(seed): round(value, 4) for seed, value in seed_deltas.items()
        },
    }


def run(args: argparse.Namespace) -> dict:
    device = choose_device(args.device)
    runs = _load_runs(SimpleNamespace(fold=None, seed=None))
    costs = {float(result["training"]["probe_cost"]) for _, _, _, result in runs}
    if len(costs) != 1:
        raise ValueError("all 006B runs must use the same probe cost")
    probe_cost = costs.pop()
    totals: dict[tuple[str, int, str], Counter] = defaultdict(Counter)
    by_seed: dict[tuple[str, int, str, int], Counter] = defaultdict(Counter)
    replay_mismatches = []

    def add(controller: str, budget: int, comparison: str, seed: int, before: dict, after: dict):
        key = (controller, budget, comparison)
        _add_pair(totals[key], before, after)
        _add_pair(by_seed[(*key, seed)], before, after)

    for source_path, fold, seed, result in runs:
        test_rule_ids = tuple(result["data"]["test_rule_ids"])
        if (
            result["fold"] != fold
            or result["seed"] != seed
            or fold not in range(len(FOLD_TEST_RULES))
            or test_rule_ids != FOLD_TEST_RULES[fold]
        ):
            raise ValueError(f"fold metadata disagrees in {source_path.name}")
        trials = generate_trials(
            result["data"]["test_episodes"], seed + 2, test_rule_ids
        )
        for controller in CONTROLLERS:
            entry = result["controllers"][controller]
            checkpoint = PROJECT_ROOT / entry["checkpoint"]
            if not checkpoint.is_file():
                raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")
            model = _build_controller(controller).to(device)
            model.load_state_dict(
                torch.load(checkpoint, map_location=device, weights_only=True)
            )
            model.eval()
            records_by_budget = {}
            for budget in range(1, MAX_PROBE_BUDGET + 1):
                rows = _rollout_model(model, trials, budget, device)
                records_by_budget[budget] = rows
                saved = entry["test_by_budget"][str(budget)]
                replayed = _rounded_metrics(
                    sum(row["correct"] for row in rows),
                    sum(row["probes"] for row in rows),
                    len(rows),
                    probe_cost,
                )
                replayed["policy_stop_rate"] = round(
                    sum(row["stop_reason"] == "policy_stop" for row in rows) / len(rows), 4
                )
                if any(replayed[key] != saved[key] for key in replayed):
                    replay_mismatches.append({
                        "source_result": source_path.name,
                        "controller": controller,
                        "budget": budget,
                    })

            for budget in range(1, MAX_PROBE_BUDGET + 1):
                base_rows = records_by_budget[budget]
                for base_row in base_rows:
                    ig_correct, extra = _extra_probe_result(base_row)
                    ig_row = _record(ig_correct, base_row["probes"] + extra)
                    add(controller, budget, "fixed_information_gain_extension", seed,
                        base_row, ig_row)

                if budget == MAX_PROBE_BUDGET:
                    continue
                larger_rows = records_by_budget[budget + 1]
                for base_row, larger_row in zip(base_rows, larger_rows):
                    ig_correct, extra = _extra_probe_result(base_row)
                    ig_row = _record(ig_correct, base_row["probes"] + extra)
                    add(controller, budget, "learned_larger_budget", seed,
                        base_row, larger_row)
                    add(controller, budget, "learned_larger_budget_vs_ig", seed,
                        ig_row, larger_row)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    if replay_mismatches:
        raise RuntimeError(
            f"Checkpoint replay disagreed with saved 006B metrics: {len(replay_mismatches)} mismatches"
        )

    output = {
        "experiment": "006D-learned-budget-allocation-vs-information-gain",
        "source_result_files": len(runs),
        "evaluated_checkpoints": len(runs) * len(CONTROLLERS),
        "device": str(device),
        "probe_cost": probe_cost,
        "replay_mismatch_count": 0,
        "comparisons": {
            "fixed_information_gain_extension": "From each budget-b rollout, append one information-gain probe unless the controller STOPped or identified a rule.",
            "learned_larger_budget": "Run the same checkpoint with budget b+1 from the episode start and compare it with budget b. Its initial budget feature changes, so this is a budget-allocation comparison rather than a forced continuation of an identical prefix.",
            "learned_larger_budget_vs_ig": "Compare the budget b+1 learned rollout directly with the budget-b rollout followed by one information-gain probe.",
        },
        "by_controller_and_budget": {},
        "uncertainty_method": "Paired reward differences are averaged within each of six seeds across all three folds, then summarized with a two-sided t interval (df=5, critical value 2.571); exploratory with six seed clusters.",
        "interpretation_notes": [
            "Each comparison uses the same test episodes for both policies and counts correct-to-wrong regressions.",
            "The learned budget increase is available only for budget 1 to 2 and 2 to 3; no learned budget-4 checkpoint exists.",
            "Budget-3 information-gain extension goes beyond the trained evaluation cap.",
            "The confidence interval uses six seed-level clusters; treat it as an uncertainty summary, not a definitive significance claim.",
        ],
    }
    for controller in CONTROLLERS:
        output["by_controller_and_budget"][controller] = {}
        for budget in range(1, MAX_PROBE_BUDGET + 1):
            comparisons = ["fixed_information_gain_extension"]
            if budget < MAX_PROBE_BUDGET:
                comparisons += ["learned_larger_budget", "learned_larger_budget_vs_ig"]
            output["by_controller_and_budget"][controller][str(budget)] = {
                name: _summarize_pair(
                    totals[(controller, budget, name)],
                    {
                        seed: by_seed[(controller, budget, name, seed)]
                        for seed in sorted({
                            key[3] for key in by_seed if key[:3] == (controller, budget, name)
                        })
                    },
                    probe_cost,
                )
                for name in comparisons
            }

    output_path = args.output or (
        PROJECT_ROOT / "artifacts" / "results" / "experiment-006D_budget-vs-ig_all.json"
    )
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output["output"] = str(output_path)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output["by_controller_and_budget"], indent=2))
    print("Replay mismatches: 0")
    print(f"Saved: {output_path}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare larger learned query budgets with a one-step information-gain extension"
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output", type=Path)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
