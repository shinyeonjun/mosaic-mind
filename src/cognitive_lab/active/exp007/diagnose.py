"""Replay Experiment 007 and locate the source of policy performance gaps."""

from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import re
import time

import torch

from cognitive_lab.active.environment import ALL_PROBES
from cognitive_lab.active.exp007.train import (
    CONTROLLERS,
    MAX_PROBE_BUDGET,
    PROJECT_ROOT,
    STOP_ACTION,
    _build_controller,
    _encode_state,
    _summarize,
    generate_trials,
)
from cognitive_lab.active.exp007.world import (
    FOLD_COUNT,
    FOLD_TEST_RULES,
    PROBE_OUTCOMES,
    RULES,
    _best_information_gain,
    apply_case,
    choose_answer,
    update_posterior,
    validate_world,
)
from cognitive_lab.train import choose_device

RESULT_PATTERN = re.compile(r"experiment-007_fold-(\d+)_seed-(\d+)\.json$")
COMPARE_KEYS = (
    "episodes", "correct_episodes", "total_probes", "identified_episodes",
    "accuracy", "mean_probes", "mean_task_reward",
    "rule_identification_rate", "mean_remaining_rules",
)


def _information_gain(posterior: tuple[int, ...], probe_id: int) -> float:
    outcomes = Counter(PROBE_OUTCOMES[probe_id][rule_id] for rule_id in posterior)
    return math.log2(len(posterior)) - sum(
        (count / len(posterior)) * math.log2(count)
        for count in outcomes.values()
    )


def _answer_error(case, posterior: tuple[int, ...], hidden_rule: int) -> tuple[int, int, str | None]:
    expected = apply_case(RULES[hidden_rule], case)
    predicted = choose_answer(case, posterior)
    answers = Counter(apply_case(RULES[rule_id], case) for rule_id in posterior)
    if hidden_rule not in posterior:
        error = "hidden_rule_eliminated"
    elif answers[expected] < max(answers.values()):
        error = "posterior_majority_favors_wrong_answer"
    elif predicted != expected:
        error = "wrong_tie_break"
    else:
        error = None
    return predicted, expected, error


def _rollout_policy(model, trials, budget: int, device: torch.device) -> list[dict]:
    all_rules = tuple(range(len(RULES)))
    records = [
        {
            "rule_id": rule_id,
            "case": case,
            "posterior": all_rules,
            "asked": set(),
            "history": [_encode_state(all_rules, None, None, budget, budget)],
            "stopped": False,
            "trajectory": [],
        }
        for rule_id, case in trials
    ]
    model.eval()
    with torch.inference_mode():
        for step in range(budget):
            active = [
                index for index, row in enumerate(records)
                if not row["stopped"] and len(row["posterior"]) > 1
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
            for row_index, record_index in enumerate(active):
                asked = records[record_index]["asked"]
                if asked:
                    logits[row_index, list(asked)] = -torch.inf
            actions = logits.argmax(dim=-1).tolist()
            for row_index, record_index in enumerate(active):
                row = records[record_index]
                posterior = row["posterior"]
                asked = tuple(sorted(row["asked"]))
                best_probe = _best_information_gain(posterior, asked)
                best_gain = _information_gain(posterior, best_probe) if best_probe is not None else 0.0
                action = actions[row_index]
                if action == STOP_ACTION:
                    row["stopped"] = True
                    row["trajectory"].append({
                        "step": step + 1,
                        "action": "STOP",
                        "candidate_count": len(posterior),
                        "best_ig_probe": best_probe,
                        "best_ig_bits": best_gain,
                        "unused_query_slots": budget - step,
                    })
                    continue

                observed = PROBE_OUTCOMES[action][row["rule_id"]]
                updated = update_posterior(posterior, action, observed)
                selected_gain = _information_gain(posterior, action)
                row["trajectory"].append({
                    "step": step + 1,
                    "action": "probe",
                    "probe_id": action,
                    "outcome": observed,
                    "candidate_count_before": len(posterior),
                    "candidate_count_after": len(updated),
                    "selected_ig_bits": selected_gain,
                    "best_ig_probe": best_probe,
                    "best_ig_bits": best_gain,
                    "ig_gap_bits": max(0.0, best_gain - selected_gain),
                    "matches_best_probe_id": action == best_probe,
                    "is_ig_optimal": best_gain - selected_gain <= 1e-6,
                })
                row["posterior"] = updated
                row["asked"].add(action)
                row["history"].append(_encode_state(
                    updated, action, observed,
                    budget - len(row["asked"]), budget,
                ))

    output = []
    for row in records:
        predicted, expected, error = _answer_error(
            row["case"], row["posterior"], row["rule_id"]
        )
        if len(row["posterior"]) == 1:
            stop_reason = "rule_identified"
        elif row["stopped"]:
            stop_reason = "policy_stop"
        else:
            stop_reason = "budget_exhausted"
        output.append({
            "rule_id": row["rule_id"],
            "correct": predicted == expected,
            "expected": expected,
            "predicted": predicted,
            "error_type": error,
            "probes": len(row["asked"]),
            "remaining_rules": len(row["posterior"]),
            "rule_identified": len(row["posterior"]) == 1,
            "stop_reason": stop_reason,
            "trajectory": row["trajectory"],
        })
    return output


def _rollout_information_gain(trials, budget: int) -> list[dict]:
    all_rules = tuple(range(len(RULES)))
    output = []
    for rule_id, case in trials:
        posterior = all_rules
        asked: set[int] = set()
        for _ in range(budget):
            if len(posterior) == 1:
                break
            probe_id = _best_information_gain(posterior, tuple(sorted(asked)))
            if probe_id is None:
                break
            posterior = update_posterior(
                posterior, probe_id, PROBE_OUTCOMES[probe_id][rule_id]
            )
            asked.add(probe_id)
        predicted, expected, error = _answer_error(case, posterior, rule_id)
        output.append({
            "rule_id": rule_id,
            "correct": predicted == expected,
            "predicted": predicted,
            "expected": expected,
            "error_type": error,
            "probes": len(asked),
            "remaining_rules": len(posterior),
            "rule_identified": len(posterior) == 1,
        })
    return output


def _add_episode(total: Counter, rule_total: Counter, policy: dict, ig: dict) -> None:
    total["episodes"] += 1
    total["correct"] += int(policy["correct"])
    total["probes"] += policy["probes"]
    total["identified"] += int(policy["rule_identified"])
    total["ig_correct"] += int(ig["correct"])
    total["ig_probes"] += ig["probes"]
    total["model_wrong_ig_correct"] += int(not policy["correct"] and ig["correct"])
    total["model_correct_ig_wrong"] += int(policy["correct"] and not ig["correct"])
    total["both_wrong"] += int(not policy["correct"] and not ig["correct"])
    total["both_correct"] += int(policy["correct"] and ig["correct"])
    total[f"error_{policy['error_type'] or 'none'}"] += 1
    total[f"stop_{policy['stop_reason']}"] += 1
    total["remaining_candidates"] += policy["remaining_rules"]
    total["ig_remaining_candidates"] += ig["remaining_rules"]
    if policy["stop_reason"] == "policy_stop":
        total["wrong_after_policy_stop"] += int(not policy["correct"])
        total["stop_remaining_candidates"] += policy["remaining_rules"]
        total["unused_query_slots_at_stop"] += sum(
            step["unused_query_slots"]
            for step in policy["trajectory"]
            if step["action"] == "STOP"
        )
    rule_total["episodes"] += 1
    rule_total["correct"] += int(policy["correct"])
    rule_total["ig_correct"] += int(ig["correct"])
    rule_total["model_wrong_ig_correct"] += int(not policy["correct"] and ig["correct"])
    rule_total["probes"] += policy["probes"]


def _decision_summary(total: Counter, probe_cost: float) -> dict:
    episodes = total["episodes"]
    decisions = total["query_decisions"]
    accuracy = total["correct"] / episodes
    mean_probes = total["probes"] / episodes
    stops = total["stop_policy_stop"]
    return {
        "episodes": episodes,
        "accuracy": round(accuracy, 4),
        "mean_probes": round(mean_probes, 4),
        "mean_task_reward": round(accuracy - probe_cost * mean_probes, 4),
        "information_gain_reference": {
            "accuracy": round(total["ig_correct"] / episodes, 4),
            "mean_probes": round(total["ig_probes"] / episodes, 4),
        },
        "paired_episode_outcomes": {
            "both_correct": total["both_correct"],
            "model_wrong_ig_correct": total["model_wrong_ig_correct"],
            "model_correct_ig_wrong": total["model_correct_ig_wrong"],
            "both_wrong": total["both_wrong"],
        },
        "model_errors": {
            key.removeprefix("error_"): value
            for key, value in total.items() if key.startswith("error_") and value
        },
        "episode_stop_reasons": {
            key.removeprefix("stop_"): value
            for key, value in total.items() if key.startswith("stop_") and value
        },
        "policy_stop_diagnosis": {
            "stopped_episodes": stops,
            "wrong_answers_after_stop": total["wrong_after_policy_stop"],
            "mean_candidates_at_stop": round(total["stop_remaining_candidates"] / stops, 4)
            if stops else None,
            "mean_unused_query_slots_at_stop": round(total["unused_query_slots_at_stop"] / stops, 4)
            if stops else None,
        },
        "query_choice_diagnosis": {
            "query_decisions": decisions,
            "same_probe_id_as_information_gain_rate": round(
                total["same_probe_id"] / decisions, 4
            ) if decisions else None,
            "information_gain_optimal_probe_rate": round(
                total["ig_optimal"] / decisions, 4
            ) if decisions else None,
            "mean_information_gain_gap_bits_per_query": round(
                total["ig_gap_bits"] / decisions, 6
            ) if decisions else None,
            "mean_selected_information_gain_bits": round(
                total["selected_ig_bits"] / decisions, 6
            ) if decisions else None,
            "mean_best_information_gain_bits": round(
                total["best_ig_bits"] / decisions, 6
            ) if decisions else None,
            "zero_information_gain_query_rate": round(
                total["zero_ig_queries"] / decisions, 4
            ) if decisions else None,
        },
        "mean_remaining_candidates": round(total["remaining_candidates"] / episodes, 4),
        "mean_information_gain_remaining_candidates": round(
            total["ig_remaining_candidates"] / episodes, 4
        ),
    }


def _load_runs() -> list[tuple[Path, int, int, dict]]:
    runs = []
    for path in sorted((PROJECT_ROOT / "artifacts" / "results").glob("experiment-007_fold-*_seed-*.json")):
        match = RESULT_PATTERN.fullmatch(path.name)
        if match:
            fold, seed = map(int, match.groups())
            if fold in range(FOLD_COUNT) and seed in range(42, 48):
                runs.append((path, fold, seed, json.loads(path.read_text(encoding="utf-8"))))
    expected = {(fold, seed) for fold in range(FOLD_COUNT) for seed in range(42, 48)}
    present = {(fold, seed) for _, fold, seed, _ in runs}
    if present != expected:
        raise FileNotFoundError(f"Expected all 18 Experiment 007 runs; missing {sorted(expected - present)}")
    return runs


def run(args) -> dict:
    validate_world()
    device = choose_device(args.device)
    runs = _load_runs()
    probe_costs = {float(result["training"]["probe_cost"]) for _, _, _, result in runs}
    if len(probe_costs) != 1:
        raise ValueError("all Experiment 007 runs must use the same probe cost")
    probe_cost = probe_costs.pop()
    totals: dict[tuple[str, int], Counter] = defaultdict(Counter)
    by_rule: dict[tuple[str, int, int], Counter] = defaultdict(Counter)
    replay_mismatches = []
    started = time.perf_counter()
    evaluated_checkpoints = 0

    for source_path, fold, seed, result in runs:
        test_rule_ids = tuple(result["data"]["test_rule_ids"])
        if (
            result["fold"] != fold
            or result["seed"] != seed
            or test_rule_ids != FOLD_TEST_RULES[fold]
        ):
            raise ValueError(f"fold metadata disagrees in {source_path.name}")
        trials = generate_trials(result["data"]["test_episodes"], seed + 2, test_rule_ids)
        ig_by_budget = {}
        for budget in range(1, MAX_PROBE_BUDGET + 1):
            ig_rows = _rollout_information_gain(trials, budget)
            ig_by_budget[budget] = ig_rows
            replayed = _summarize(ig_rows, probe_cost)
            saved = result["baselines"]["information_gain"]["by_budget"][str(budget)]
            if any(replayed[key] != saved[key] for key in COMPARE_KEYS):
                replay_mismatches.append({"result": source_path.name, "policy": "information_gain", "budget": budget})

        for name in CONTROLLERS:
            entry = result["controllers"][name]
            checkpoint = PROJECT_ROOT / entry["checkpoint"]
            if not checkpoint.is_file():
                raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")
            model = _build_controller(name).to(device)
            model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
            model.eval()
            evaluated_checkpoints += 1
            for budget in range(1, MAX_PROBE_BUDGET + 1):
                rows = _rollout_policy(model, trials, budget, device)
                replayed = _summarize(rows, probe_cost)
                saved = entry["test_by_budget"][str(budget)]
                if any(replayed[key] != saved[key] for key in COMPARE_KEYS):
                    replay_mismatches.append({"result": source_path.name, "policy": name, "budget": budget})
                total = totals[(name, budget)]
                for row, ig in zip(rows, ig_by_budget[budget]):
                    rule_total = by_rule[(name, budget, row["rule_id"])]
                    _add_episode(total, rule_total, row, ig)
                    for step in row["trajectory"]:
                        if step["action"] == "STOP":
                            total["stop_decisions"] += 1
                            continue
                        total["query_decisions"] += 1
                        total["same_probe_id"] += int(step["matches_best_probe_id"])
                        total["ig_optimal"] += int(step["is_ig_optimal"])
                        total["ig_gap_bits"] += step["ig_gap_bits"]
                        total["selected_ig_bits"] += step["selected_ig_bits"]
                        total["best_ig_bits"] += step["best_ig_bits"]
                        total["zero_ig_queries"] += int(step["selected_ig_bits"] <= 1e-12)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    if replay_mismatches:
        raise RuntimeError(
            f"Checkpoint/baseline replay disagreed with saved metrics: {len(replay_mismatches)} mismatches"
        )

    by_controller_and_budget = {
        name: {
            str(budget): _decision_summary(totals[(name, budget)], probe_cost)
            for budget in range(1, MAX_PROBE_BUDGET + 1)
        }
        for name in CONTROLLERS
    }
    by_controller_rule = {
        name: {
            str(budget): {
                RULES[rule_id].name: {
                    "episodes": counts["episodes"],
                    "accuracy": round(counts["correct"] / counts["episodes"], 4),
                    "information_gain_accuracy": round(counts["ig_correct"] / counts["episodes"], 4),
                    "model_errors_fixed_by_information_gain": counts["model_wrong_ig_correct"],
                }
                for (controller, current_budget, rule_id), counts in sorted(by_rule.items())
                if controller == name and current_budget == budget and counts["episodes"]
            }
            for budget in range(1, MAX_PROBE_BUDGET + 1)
        }
        for name in CONTROLLERS
    }
    output = {
        "experiment": "007-policy-gap-diagnosis",
        "source_result_files": len(runs),
        "evaluated_checkpoints": evaluated_checkpoints,
        "test_episodes_per_controller_and_budget": totals[(CONTROLLERS[0], 1)]["episodes"],
        "device": str(device),
        "probe_cost": probe_cost,
        "replay_mismatch_count": 0,
        "replay_mismatches": replay_mismatches,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "by_controller_and_budget": by_controller_and_budget,
        "by_controller_and_held_out_rule": by_controller_rule,
        "interpretation_notes": [
            "Information gain is computed under a uniform prior over the fixed 18-rule catalog.",
            "IG-optimal treats equal-information ties as optimal; same-probe-ID rate is reported separately.",
            "A wrong answer after policy STOP signals a potentially avoidable stop, but does not estimate the reward of a fully re-planned continuation.",
            "The diagnostic retains Experiment 007's fixed candidate catalog and exact posterior update; it does not test open-ended rule invention.",
        ],
    }
    output_path = PROJECT_ROOT / "artifacts" / "results" / "experiment-007_diagnostics.json"
    output["output"] = str(output_path)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({
        "replay_mismatches": 0,
        "by_controller_and_budget": {
            name: {
                budget: {
                    "reward": metrics["mean_task_reward"],
                    "accuracy": metrics["accuracy"],
                    "information_gain_accuracy": metrics["information_gain_reference"]["accuracy"],
                    "paired_episode_outcomes": metrics["paired_episode_outcomes"],
                    "model_errors": metrics["model_errors"],
                    "ig_optimal_probe_rate": metrics["query_choice_diagnosis"]["information_gain_optimal_probe_rate"],
                    "mean_ig_gap_bits": metrics["query_choice_diagnosis"]["mean_information_gain_gap_bits_per_query"],
                    "stop_rate": round(metrics["policy_stop_diagnosis"]["stopped_episodes"] / metrics["episodes"], 4),
                    "wrong_after_stop": metrics["policy_stop_diagnosis"]["wrong_answers_after_stop"],
                }
                for budget, metrics in by_controller_and_budget[name].items()
            }
            for name in CONTROLLERS
        },
        "weakest_rims_rules_at_budget_3": sorted(
            (
                {
                    "rule": rule,
                    "accuracy": values["accuracy"],
                    "information_gain_accuracy": values["information_gain_accuracy"],
                    "errors_fixed_by_information_gain": values["model_errors_fixed_by_information_gain"],
                }
                for rule, values in by_controller_rule["rims"]["3"].items()
            ),
            key=lambda row: row["accuracy"],
        )[:6],
        "saved": str(output_path),
    }, indent=2))
    return output


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Diagnose question choices and STOP decisions for Experiment 007"
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
