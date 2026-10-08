"""Replay Experiment 006B checkpoints and explain policy mistakes."""

import argparse
from collections import Counter
from functools import lru_cache
import json
import math
from pathlib import Path
import random
import re

import torch

from cognitive_lab.active.baselines import choose_answer, generate_trials, select_probe, update_posterior
from cognitive_lab.active.controller import FOLD_TEST_RULES, MAX_PROBE_BUDGET, encode_state
from cognitive_lab.active.environment import ALL_PROBES, PROBE_OUTCOMES, RULES, RuleEnvironment, TaskCase
from cognitive_lab.active.train_controller_rl import (
    CONTROLLERS,
    PROJECT_ROOT,
    STOP_ACTION,
    _build_controller,
)
from cognitive_lab.train import choose_device


RESULT_PATTERN = re.compile(r"experiment-006B_fold-(\d+)_seed-(\d+)\.json$")


def _probe_json(probe_id: int) -> list[list[int | bool]]:
    return [[is_target, key] for is_target, key in ALL_PROBES[probe_id]]


@lru_cache(maxsize=None)
def _probe_information_gain(posterior: tuple[int, ...], probe_id: int) -> float:
    counts = Counter(PROBE_OUTCOMES[probe_id][rule_id] for rule_id in posterior)
    total = len(posterior)
    return math.log2(total) - sum(
        (count / total) * math.log2(count) for count in counts.values()
    )


@lru_cache(maxsize=None)
def _best_information_gain(posterior: tuple[int, ...], asked: tuple[int, ...]) -> int | None:
    return select_probe("information_gain", posterior, set(asked), random.Random(0))


def _answer_details(
    case: TaskCase, posterior: tuple[int, ...], expected: int, hidden_rule_id: int
) -> dict:
    answers = {rule_id: RuleEnvironment(RULES[rule_id]).answer(case) for rule_id in posterior}
    counts = Counter(answers.values())
    predicted = choose_answer(case, posterior)
    if hidden_rule_id not in posterior:
        error_type = "hidden_rule_eliminated"
    elif counts[expected] < max(counts.values()):
        error_type = "posterior_majority_favors_wrong_answer"
    elif predicted != expected:
        error_type = "wrong_tie_break"
    else:
        error_type = None
    return {
        "predicted": predicted,
        "correct": predicted == expected,
        "posterior_answer_counts": {str(answer): count for answer, count in sorted(counts.items())},
        "hidden_rule_survived": hidden_rule_id in posterior,
        "error_type": error_type,
    }


def _rollout_information_gain(rule_id: int, case: TaskCase, budget: int) -> dict:
    posterior = tuple(range(len(RULES)))
    asked: set[int] = set()
    trajectory = []
    environment = RuleEnvironment(RULES[rule_id])
    for step in range(budget):
        if len(posterior) == 1:
            break
        probe_id = _best_information_gain(posterior, tuple(sorted(asked)))
        if probe_id is None:
            break
        before = posterior
        outcome = environment.query(ALL_PROBES[probe_id])
        posterior = update_posterior(posterior, probe_id, outcome)
        asked.add(probe_id)
        trajectory.append({
            "step": step + 1,
            "probe_id": probe_id,
            "outcome": outcome,
            "posterior_before": list(before),
            "posterior_after": list(posterior),
        })
    expected = environment.answer(case)
    answer = _answer_details(case, posterior, expected, rule_id)
    return {
        **answer,
        "expected": expected,
        "probes": len(asked),
        "remaining_rules": list(posterior),
        "trajectory": trajectory,
    }


def _rollout_model(model: torch.nn.Module, trials: list[tuple[int, TaskCase]], budget: int, device: torch.device) -> list[dict]:
    records = [
        {
            "rule_id": rule_id,
            "case": case,
            "posterior": tuple(range(len(RULES))),
            "asked": set(),
            "history": [encode_state(tuple(range(len(RULES))), None, None, budget, budget)],
            "trajectory": [],
            "stopped": False,
        }
        for rule_id, case in trials
    ]
    model.eval()
    with torch.inference_mode():
        for step in range(budget):
            active = [
                index for index, record in enumerate(records)
                if not record["stopped"] and len(record["posterior"]) > 1
            ]
            if not active:
                break
            states = torch.tensor(
                [records[index]["history"] for index in active],
                dtype=torch.float32,
                device=device,
            )
            logits = model(states).clone()
            if logits.shape[-1] != STOP_ACTION + 1:
                raise ValueError(f"expected {STOP_ACTION + 1} actions, got {logits.shape[-1]}")
            best_probes = []
            for row, index in enumerate(active):
                record = records[index]
                posterior = record["posterior"]
                best_ig = _best_information_gain(posterior, tuple(sorted(record["asked"])))
                best_probes.append(best_ig)
                if record["asked"]:
                    logits[row, list(record["asked"])] = -torch.inf
            actions = logits.argmax(dim=-1).tolist()
            for row, index in enumerate(active):
                record = records[index]
                posterior = record["posterior"]
                best_ig = best_probes[row]
                action = actions[row]
                if action == STOP_ACTION:
                    record["stopped"] = True
                    record["trajectory"].append({
                        "step": step + 1,
                        "action": "STOP",
                        "posterior_before": list(posterior),
                        "best_information_gain_probe": best_ig,
                    })
                    continue

                environment = RuleEnvironment(RULES[record["rule_id"]])
                outcome = environment.query(ALL_PROBES[action])
                updated = update_posterior(posterior, action, outcome)
                selected_gain = _probe_information_gain(posterior, action)
                best_gain = _probe_information_gain(posterior, best_ig) if best_ig is not None else 0.0
                record["trajectory"].append({
                    "step": step + 1,
                    "action": "probe",
                    "probe_id": action,
                    "probe": _probe_json(action),
                    "outcome": outcome,
                    "posterior_before": list(posterior),
                    "posterior_after": list(updated),
                    "information_gain_bits": round(selected_gain, 6),
                    "best_information_gain_probe": best_ig,
                    "best_information_gain_bits": round(best_gain, 6),
                    "information_gain_gap_bits": round(max(0.0, best_gain - selected_gain), 6),
                    "matches_information_gain": action == best_ig,
                    "is_information_gain_optimal": best_gain - selected_gain <= 1e-6,
                })
                record["asked"].add(action)
                record["posterior"] = updated
                record["history"].append(
                    encode_state(updated, action, outcome, budget - len(record["asked"]), budget)
                )

    output = []
    for episode, record in enumerate(records):
        environment = RuleEnvironment(RULES[record["rule_id"]])
        expected = environment.answer(record["case"])
        answer = _answer_details(
            record["case"], record["posterior"], expected, record["rule_id"]
        )
        if len(record["posterior"]) == 1:
            stop_reason = "rule_identified"
        elif record["stopped"]:
            stop_reason = "policy_stop"
        else:
            stop_reason = "budget_exhausted"
        output.append({
            **answer,
            "rule_id": record["rule_id"],
            "episode": episode,
            "case": record["case"],
            "expected": expected,
            "probes": len(record["asked"]),
            "remaining_rules": list(record["posterior"]),
            "stop_reason": stop_reason,
            "trajectory": record["trajectory"],
        })
    return output


def _empty_totals() -> dict[str, float]:
    return {
        "episodes": 0,
        "correct": 0,
        "probes": 0,
        "policy_stops": 0,
        "query_decisions": 0,
        "ig_agreements": 0,
        "ig_optimal_probes": 0,
        "information_gain_gap_bits": 0.0,
        "zero_gain_probes": 0,
        "exact_sequence_matches": 0,
        "ig_correct": 0,
        "ig_probes": 0,
    }


def _metric_view(totals: dict, probe_cost: float) -> dict:
    episodes = totals["episodes"]
    decisions = totals["query_decisions"]
    probes = totals["probes"]
    accuracy = round(totals["correct"] / episodes, 4)
    mean_probes = round(probes / episodes, 4)
    return {
        "episodes": episodes,
        "accuracy": accuracy,
        "mean_task_reward": round(accuracy - probe_cost * mean_probes, 4),
        "mean_probes": mean_probes,
        "policy_stop_rate": round(totals["policy_stops"] / episodes, 4),
        "same_probe_id_as_information_gain_rate": round(totals["ig_agreements"] / decisions, 4) if decisions else None,
        "information_gain_optimal_probe_rate": round(totals["ig_optimal_probes"] / decisions, 4) if decisions else None,
        "mean_information_gain_gap_bits_per_probe": round(totals["information_gain_gap_bits"] / decisions, 6) if decisions else None,
        "zero_information_gain_probe_rate": round(totals["zero_gain_probes"] / decisions, 4) if decisions else None,
        "same_full_probe_sequence_as_information_gain_rate": round(totals["exact_sequence_matches"] / episodes, 4),
        "information_gain_accuracy": round(totals["ig_correct"] / episodes, 4),
        "information_gain_mean_probes": round(totals["ig_probes"] / episodes, 4),
    }


def _load_runs(args: argparse.Namespace) -> list[tuple[Path, int, int, dict]]:
    runs = []
    for path in sorted((PROJECT_ROOT / "artifacts" / "results").glob("experiment-006B_fold-*_seed-*.json")):
        match = RESULT_PATTERN.fullmatch(path.name)
        if not match:
            continue
        fold, seed = map(int, match.groups())
        if args.fold is not None and fold != args.fold:
            continue
        if args.seed is not None and seed != args.seed:
            continue
        result = json.loads(path.read_text(encoding="utf-8"))
        runs.append((path, fold, seed, result))
    if not runs:
        raise FileNotFoundError("No matching Experiment 006B result JSON files were found")
    return runs


def run(args: argparse.Namespace) -> dict:
    device = choose_device(args.device)
    runs = _load_runs(args)
    controllers = (args.controller,) if args.controller else CONTROLLERS
    budget_values = range(1, MAX_PROBE_BUDGET + 1)
    totals = {name: {str(budget): _empty_totals() for budget in budget_values} for name in controllers}
    replay_mismatches = []
    failure_types = {
        name: {str(budget): Counter() for budget in budget_values}
        for name in controllers
    }
    evaluated_runs = 0
    results_dir = PROJECT_ROOT / "artifacts" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    tag_parts = []
    if args.fold is not None:
        tag_parts.append(f"fold-{args.fold}")
    if args.seed is not None:
        tag_parts.append(f"seed-{args.seed}")
    if args.controller:
        tag_parts.append(args.controller)
    tag = "_" + "_".join(tag_parts) if tag_parts else "_all"
    summary_path = results_dir / f"experiment-006B_diagnostics{tag}.json"
    failures_path = results_dir / f"experiment-006B_failures{tag}.jsonl"
    failures = 0

    with failures_path.open("w", encoding="utf-8") as failure_file:
        for source_path, fold, seed, result in runs:
            test_rule_ids = tuple(result["data"]["test_rule_ids"])
            trials = generate_trials(result["data"]["test_episodes"], seed + 2, test_rule_ids)
            probe_cost = float(result["training"]["probe_cost"])
            ig_by_budget = {
                budget: [_rollout_information_gain(rule_id, case, budget) for rule_id, case in trials]
                for budget in budget_values
            }
            for budget in budget_values:
                saved_ig = result["baselines"]["baselines"]["information_gain"]["by_budget"][str(budget)]
                observed_ig = ig_by_budget[budget]
                ig_accuracy = sum(row["correct"] for row in observed_ig) / len(observed_ig)
                ig_probes = sum(row["probes"] for row in observed_ig) / len(observed_ig)
                if (round(ig_accuracy, 4), round(ig_probes, 4)) != (
                    saved_ig["accuracy"], saved_ig["mean_probes"]
                ):
                    replay_mismatches.append({
                        "result": source_path.name,
                        "policy": "information_gain",
                        "budget": budget,
                        "replayed": [round(ig_accuracy, 4), round(ig_probes, 4)],
                        "saved": [saved_ig["accuracy"], saved_ig["mean_probes"]],
                    })

            for controller_name in controllers:
                entry = result.get("controllers", {}).get(controller_name)
                if not entry:
                    continue
                checkpoint = PROJECT_ROOT / entry["checkpoint"]
                if not checkpoint.is_file():
                    raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")
                print(f"Evaluating fold={fold} seed={seed} controller={controller_name}")
                model = _build_controller(controller_name).to(device)
                model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
                evaluated_runs += 1
                for budget in budget_values:
                    model_records = _rollout_model(model, trials, budget, device)
                    ig_records = ig_by_budget[budget]
                    acc = sum(row["correct"] for row in model_records) / len(model_records)
                    mean_probes = sum(row["probes"] for row in model_records) / len(model_records)
                    policy_stop_rate = sum(
                        row["stop_reason"] == "policy_stop" for row in model_records
                    ) / len(model_records)
                    task_reward = round(round(acc, 4) - probe_cost * round(mean_probes, 4), 4)
                    saved = entry["test_by_budget"][str(budget)]
                    replayed_metrics = (
                        round(acc, 4), round(mean_probes, 4), round(policy_stop_rate, 4),
                        round(task_reward, 4),
                    )
                    saved_metrics = (
                        saved["accuracy"], saved["mean_probes"], saved["policy_stop_rate"],
                        saved["mean_task_reward"],
                    )
                    if replayed_metrics != saved_metrics:
                        replay_mismatches.append({
                            "result": source_path.name,
                            "policy": controller_name,
                            "budget": budget,
                            "replayed": replayed_metrics,
                            "saved": saved_metrics,
                        })
                    aggregate = totals[controller_name][str(budget)]
                    for row, ig in zip(model_records, ig_records):
                        aggregate["episodes"] += 1
                        aggregate["correct"] += int(row["correct"])
                        aggregate["probes"] += row["probes"]
                        aggregate["policy_stops"] += int(row["stop_reason"] == "policy_stop")
                        aggregate["query_decisions"] += len(row["trajectory"]) - int(row["stop_reason"] == "policy_stop")
                        aggregate["ig_agreements"] += sum(
                            step.get("matches_information_gain", False) for step in row["trajectory"]
                        )
                        aggregate["ig_optimal_probes"] += sum(
                            step.get("is_information_gain_optimal", False) for step in row["trajectory"]
                        )
                        aggregate["information_gain_gap_bits"] += sum(
                            step.get("information_gain_gap_bits", 0.0) for step in row["trajectory"]
                        )
                        aggregate["zero_gain_probes"] += sum(
                            step.get("information_gain_bits", 0.0) <= 1e-12 for step in row["trajectory"]
                            if step.get("action") == "probe"
                        )
                        model_sequence = [step["probe_id"] for step in row["trajectory"] if step.get("action") == "probe"]
                        ig_sequence = [step["probe_id"] for step in ig["trajectory"]]
                        aggregate["exact_sequence_matches"] += int(model_sequence == ig_sequence)
                        aggregate["ig_correct"] += int(ig["correct"])
                        aggregate["ig_probes"] += ig["probes"]
                        if not row["correct"]:
                            failure_types[controller_name][str(budget)][row["error_type"]] += 1
                            failure_file.write(json.dumps({
                                "source_result": source_path.name,
                                "fold": fold,
                                "seed": seed,
                                "controller": controller_name,
                                "budget": budget,
                                "episode": row["episode"],
                                "hidden_rule_id": row["rule_id"],
                                "hidden_rule": RULES[row["rule_id"]].name,
                                "case": {
                                    "target_door": row["case"].target_door,
                                    "events": [list(event) for event in row["case"].events],
                                },
                                "expected": row["expected"],
                                "predicted": row["predicted"],
                                "error_type": row["error_type"],
                                "hidden_rule_survived": row["hidden_rule_survived"],
                                "final_candidate_rule_ids": row["remaining_rules"],
                                "trajectory": row["trajectory"],
                                "information_gain_comparison": {
                                    "correct": ig["correct"],
                                    "predicted": ig["predicted"],
                                    "probes": ig["probes"],
                                    "trajectory": ig["trajectory"],
                                },
                            }, ensure_ascii=False) + "\n")
                            failures += 1
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    summary = {
        "experiment": "006B-action-diagnostics",
        "device": str(device),
        "source_result_files": len(runs),
        "evaluated_checkpoints": evaluated_runs,
        "test_episodes_per_source_run": runs[0][3]["data"]["test_episodes"],
        "probe_cost": runs[0][3]["training"]["probe_cost"],
        "controllers": {
            name: {budget: _metric_view(value, float(runs[0][3]["training"]["probe_cost"]))
                   for budget, value in by_budget.items()}
            for name, by_budget in totals.items()
        },
        "replay_mismatch_count": len(replay_mismatches),
        "replay_mismatches": replay_mismatches,
        "failure_case_count": failures,
        "failure_types": {
            name: {budget: dict(counts) for budget, counts in by_budget.items()}
            for name, by_budget in failure_types.items()
        },
        "files": {
            "summary": str(summary_path.relative_to(PROJECT_ROOT)),
            "failure_cases": str(failures_path.relative_to(PROJECT_ROOT)),
        },
        "interpretation_notes": [
            "Exact probe-ID agreement is separate from information-gain optimality so equally informative ties count as optimal.",
            "Information gain is a diagnostic reference, not a guarantee of best final task reward.",
            "The benchmark still uses the fixed six-rule candidate catalog; this does not test invention of unseen rules.",
        ],
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary["controllers"], indent=2))
    print(f"Replay mismatches: {len(replay_mismatches)}")
    print(f"Failure cases: {failures}")
    print(f"Saved: {summary_path}")
    print(f"Saved: {failures_path}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay Experiment 006B checkpoints and inspect their question choices"
    )
    parser.add_argument("--fold", type=int, choices=range(len(FOLD_TEST_RULES)))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--controller", choices=CONTROLLERS)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
