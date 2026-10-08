"""Evaluate one extra information-gain probe on every 006B test episode."""

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import statistics
from types import SimpleNamespace

import torch

from cognitive_lab.active.baselines import (
    choose_answer,
    generate_trials,
    update_posterior,
)
from cognitive_lab.active.controller import FOLD_TEST_RULES, MAX_PROBE_BUDGET
from cognitive_lab.active.diagnose_rl import _best_information_gain, _load_runs, _rollout_model
from cognitive_lab.active.environment import PROBE_OUTCOMES
from cognitive_lab.active.train_controller_rl import (
    CONTROLLERS,
    PROJECT_ROOT,
    _build_controller,
)
from cognitive_lab.train import choose_device

T_CRITICAL_95_DF5 = 2.571


def _extra_probe_result(row: dict) -> tuple[bool, int]:
    """Return the answer after one query; STOP and identified rules are respected."""
    if row["stop_reason"] == "policy_stop":
        return row["correct"], 0
    posterior = tuple(row["remaining_rules"])
    if len(posterior) == 1:
        return row["correct"], 0
    asked = {
        step["probe_id"]
        for step in row["trajectory"]
        if step.get("action") == "probe"
    }
    probe_id = _best_information_gain(posterior, tuple(sorted(asked)))
    if probe_id is None:
        return row["correct"], 0
    outcome = PROBE_OUTCOMES[probe_id][row["rule_id"]]
    updated = update_posterior(posterior, probe_id, outcome)
    correct = choose_answer(row["case"], updated) == row["expected"]
    return correct, 1


def _rounded_metrics(correct: int, probes: int, episodes: int, probe_cost: float) -> dict:
    accuracy = correct / episodes
    mean_probes = probes / episodes
    return {
        "accuracy": round(accuracy, 4),
        "mean_probes": round(mean_probes, 4),
        "mean_task_reward": round(
            round(accuracy, 4) - probe_cost * round(mean_probes, 4), 4
        ),
    }


def _paired_summary(
    totals: Counter, seed_totals: dict[int, Counter], probe_cost: float
) -> dict:
    episodes = totals["episodes"]
    baseline_accuracy = totals["before_correct"] / episodes
    extended_accuracy = totals["after_correct"] / episodes
    baseline_probes = totals["before_probes"] / episodes
    extended_probes = (totals["before_probes"] + totals["extra_probes"]) / episodes
    seed_deltas = [
        (row["after_correct"] - row["before_correct"] - probe_cost * row["extra_probes"])
        / row["episodes"]
        for _, row in sorted(seed_totals.items())
    ]
    mean_delta = statistics.mean(seed_deltas)
    margin = (
        T_CRITICAL_95_DF5 * statistics.stdev(seed_deltas) / math.sqrt(len(seed_deltas))
        if len(seed_deltas) == 6 else None
    )
    return {
        "episodes": episodes,
        "before": {
            "accuracy": round(baseline_accuracy, 4),
            "mean_probes": round(baseline_probes, 4),
            "mean_task_reward": round(
                baseline_accuracy - probe_cost * baseline_probes, 4
            ),
        },
        "after_one_more_ig_probe": {
            "accuracy": round(extended_accuracy, 4),
            "mean_probes": round(extended_probes, 4),
            "extra_probes_per_episode": round(totals["extra_probes"] / episodes, 4),
            "mean_task_reward": round(
                extended_accuracy - probe_cost * extended_probes, 4
            ),
        },
        "paired_change": {
            "accuracy_percentage_points": round((extended_accuracy - baseline_accuracy) * 100, 3),
            "mean_probes": round(extended_probes - baseline_probes, 4),
            "mean_task_reward": round(
                (extended_accuracy - baseline_accuracy)
                - probe_cost * (extended_probes - baseline_probes),
                4,
            ),
            "reward_change_95pct_seed_cluster_t_interval": [
                round(mean_delta - margin, 4), round(mean_delta + margin, 4)
            ] if margin is not None else None,
            "fixed_previously_wrong": totals["wrong_to_correct"],
            "made_previously_correct_wrong": totals["correct_to_wrong"],
            "unchanged_correct": totals["correct_to_correct"],
            "unchanged_wrong": totals["wrong_to_wrong"],
        },
        "seed_cluster_reward_changes": {
            str(seed): round(delta, 4)
            for seed, delta in zip(sorted(seed_totals), seed_deltas)
        },
    }


def run(args: argparse.Namespace) -> dict:
    device = choose_device(args.device)
    runs = _load_runs(SimpleNamespace(fold=None, seed=None))
    probe_costs = {float(result["training"]["probe_cost"]) for _, _, _, result in runs}
    if len(probe_costs) != 1:
        raise ValueError("all 006B runs must use the same probe cost")
    probe_cost = probe_costs.pop()
    totals: dict[tuple[str, int], Counter] = defaultdict(Counter)
    by_seed: dict[tuple[str, int, int], Counter] = defaultdict(Counter)
    per_run = []
    replay_mismatches = []

    for source_path, fold, seed, result in runs:
        test_rule_ids = tuple(result["data"]["test_rule_ids"])
        if (
            result["fold"] != fold
            or result["seed"] != seed
            or fold not in range(len(FOLD_TEST_RULES))
            or test_rule_ids != FOLD_TEST_RULES[fold]
        ):
            raise ValueError(f"fold metadata disagrees in {source_path.name}")
        trials = generate_trials(result["data"]["test_episodes"], seed + 2, test_rule_ids)
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
            for budget in range(1, MAX_PROBE_BUDGET + 1):
                records = _rollout_model(model, trials, budget, device)
                before_correct = sum(row["correct"] for row in records)
                before_probes = sum(row["probes"] for row in records)
                before_stops = sum(row["stop_reason"] == "policy_stop" for row in records)
                saved = entry["test_by_budget"][str(budget)]
                replayed = _rounded_metrics(
                    before_correct, before_probes, len(records), probe_cost
                )
                replayed["policy_stop_rate"] = round(before_stops / len(records), 4)
                if any(replayed[key] != saved[key] for key in replayed):
                    replay_mismatches.append({
                        "source_result": source_path.name,
                        "controller": controller,
                        "budget": budget,
                        "replayed": replayed,
                        "saved": {key: saved[key] for key in replayed},
                    })

                run_counts = Counter()
                for row in records:
                    after_correct, extra_probe = _extra_probe_result(row)
                    before = bool(row["correct"])
                    after = bool(after_correct)
                    transition = (
                        "correct_to_correct" if before and after else
                        "correct_to_wrong" if before else
                        "wrong_to_correct" if after else "wrong_to_wrong"
                    )
                    run_counts[transition] += 1
                    run_counts["after_correct"] += int(after)
                    run_counts["extra_probes"] += extra_probe
                run_counts["episodes"] = len(records)
                run_counts["before_correct"] = before_correct
                run_counts["before_probes"] = before_probes
                key = (controller, budget)
                totals[key].update(run_counts)
                by_seed[(controller, budget, seed)].update(run_counts)
                per_run.append({
                    "source_result": source_path.name,
                    "fold": fold,
                    "seed": seed,
                    "controller": controller,
                    "budget": budget,
                    "paired": _paired_summary(
                        run_counts, {seed: run_counts}, probe_cost
                    ),
                })
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    if replay_mismatches:
        raise RuntimeError(
            f"Checkpoint replay disagreed with saved 006B metrics: {len(replay_mismatches)} mismatches"
        )

    summary = {
        "experiment": "006C-full-population-one-more-information-gain-probe",
        "source_result_files": len(runs),
        "evaluated_checkpoints": len(runs) * len(CONTROLLERS),
        "device": str(device),
        "probe_cost": probe_cost,
        "intervention": "After the saved controller rollout, ask one information-gain probe if the controller did not STOP and more than one candidate rule remains; use the existing majority-answer decoder afterward.",
        "query_choice_uses_test_case": False,
        "replay_mismatch_count": 0,
        "by_controller_and_budget": {
            controller: {
                str(budget): _paired_summary(
                    totals[(controller, budget)],
                    {
                        seed: by_seed[(controller, budget, seed)]
                        for seed in sorted({key[2] for key in by_seed if key[:2] == (controller, budget)})
                    },
                    probe_cost,
                )
                for budget in range(1, MAX_PROBE_BUDGET + 1)
            }
            for controller in CONTROLLERS
        },
        "per_fold_seed_runs": per_run,
        "uncertainty_method": "Paired reward differences are averaged within each of six seeds across all three folds, then summarized with a two-sided t interval (df=5, critical value 2.571); exploratory with six seed clusters.",
        "interpretation_notes": [
            "Unlike the failure-only analysis, this compares every test episode, so improvements and regressions both count.",
            "At budgets 1 and 2 the added query extends the probe budget by one; at budget 3 it is a fourth query beyond the trained evaluation cap.",
            "The intervention uses the information-gain heuristic for the additional query; it does not continue the learned controller's own policy.",
            "Test cases are used only to score the final answer, never to choose the extra query.",
            "The confidence interval uses six seed-level clusters; treat it as an uncertainty summary, not a definitive significance claim.",
        ],
    }
    output_path = args.output or (
        PROJECT_ROOT / "artifacts" / "results" / "experiment-006C_one-more-ig_all.json"
    )
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary["output"] = str(output_path)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary["by_controller_and_budget"], indent=2))
    print(f"Replay mismatches: 0")
    print(f"Saved: {output_path}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate one extra information-gain probe on every 006B test episode"
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output", type=Path)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
