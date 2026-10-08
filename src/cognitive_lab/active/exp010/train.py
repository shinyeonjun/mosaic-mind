"""Train an end-to-end history-only rule learner for Experiment 010."""

import argparse
import json
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from cognitive_lab.active.exp007.diagnose import _information_gain
from cognitive_lab.active.environment import TaskCase
from cognitive_lab.active.exp007.train import (
    CONTROLLERS,
    MAX_PROBE_BUDGET,
    POLICY_ACTION_COUNT,
    PROJECT_ROOT,
    STATE_SIZE,
    STOP_ACTION,
    _balanced_rule_schedule,
    _build_controller,
    _encode_state,
    generate_trials,
)
from cognitive_lab.active.exp007.world import (
    FOLD_COUNT,
    FOLD_TEST_RULES,
    PROBE_OUTCOMES,
    REDUCERS,
    RULES,
    SCOPES,
    Rule,
    _best_information_gain,
    apply_case,
    update_posterior,
    validate_world,
)
from cognitive_lab.train import choose_device

SEEDS = tuple(range(42, 48))
FEATURE_SIZES = {"gru_256": 256, "gru_parameter_matched": 206, "rims": 256}


class HistoryOnlyAgent(nn.Module):
    """Share one recurrent state across query, scope, and reducer heads."""

    def __init__(self, core: nn.Module, feature_size: int):
        super().__init__()
        self.core = core
        self.scope_head = nn.Linear(feature_size, len(SCOPES))
        self.reducer_head = nn.Linear(feature_size, len(REDUCERS))

    def forward(self, observations: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        if observations.shape[-1] != STATE_SIZE:
            raise ValueError(f"expected state size {STATE_SIZE}, got {observations.shape[-1]}")
        history = observations.clone()
        history[..., :len(RULES)] = 0.0
        features = self.core.encode(history)
        return (
            self.core.policy(features),
            self.scope_head(features),
            self.reducer_head(features),
        )


def _build_agent(name: str) -> HistoryOnlyAgent:
    return HistoryOnlyAgent(_build_controller(name), FEATURE_SIZES[name])


def _history_frame(
    posterior: tuple[int, ...],
    previous_probe: int | None,
    previous_outcome: int | None,
    remaining_budget: int,
    budget_limit: int,
) -> list[float]:
    frame = _encode_state(
        posterior, previous_probe, previous_outcome, remaining_budget, budget_limit
    )
    frame[:len(RULES)] = [0.0] * len(RULES)
    return frame


def _reference_result(fold: int, seed: int) -> tuple[Path, dict]:
    path = PROJECT_ROOT / "artifacts" / "results" / f"experiment-009_fold-{fold}_seed-{seed}.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"Run matching Experiment 009 first; missing {path.name}"
        )
    result = json.loads(path.read_text(encoding="utf-8"))
    expected_test_ids = FOLD_TEST_RULES[fold]
    if (
        result.get("experiment") != "009-history-only-query-policy"
        or result.get("fold") != fold
        or result.get("seed") != seed
        or tuple(result.get("data", {}).get("test_rule_ids", ())) != expected_test_ids
        or result.get("input_contract", {}).get("controller_receives_exact_posterior_mask") is not False
        or result.get("input_contract", {}).get("history_sequence_is_recurrent_input") is not True
    ):
        raise ValueError(f"unexpected Experiment 009 metadata in {path.name}")
    return path, result


def _teacher_examples(
    train_rule_ids: tuple[int, ...], batch_size: int, seed: int
) -> list[list[tuple[list[list[float]], int, tuple[int, ...], int, int]]]:
    rng = random.Random(seed)
    hidden_rules = _balanced_rule_schedule(batch_size, train_rule_ids, rng)
    budgets = [rng.randint(1, MAX_PROBE_BUDGET) for _ in range(batch_size)]
    by_step = [[] for _ in range(MAX_PROBE_BUDGET)]
    all_rules = tuple(range(len(RULES)))

    for hidden_rule, budget in zip(hidden_rules, budgets):
        rule = RULES[hidden_rule]
        scope_label = SCOPES.index(rule.scope)
        reducer_label = REDUCERS.index(rule.reducer)
        posterior = all_rules
        asked: set[int] = set()
        history = [_history_frame(all_rules, None, None, budget, budget)]

        for step in range(budget):
            action = (
                _best_information_gain(posterior, tuple(sorted(asked)))
                if len(posterior) > 1
                else STOP_ACTION
            )
            if action is None:
                raise RuntimeError("teacher found no informative probe")
            by_step[step].append((
                history.copy(), action, tuple(sorted(asked)), scope_label, reducer_label
            ))
            if action == STOP_ACTION:
                break
            observed = PROBE_OUTCOMES[action][hidden_rule]
            posterior = update_posterior(posterior, action, observed)
            asked.add(action)
            history.append(_history_frame(
                posterior, action, observed, budget - len(asked), budget
            ))
    return by_step


def _train_batch(
    model: HistoryOnlyAgent,
    optimizer: torch.optim.Optimizer,
    train_rule_ids: tuple[int, ...],
    batch_size: int,
    seed: int,
    device: torch.device,
) -> tuple[float, int, int]:
    examples_by_step = _teacher_examples(train_rule_ids, batch_size, seed)
    optimizer.zero_grad(set_to_none=True)
    total_loss: Tensor | None = None
    action_matches = 0
    decisions = 0

    for examples in examples_by_step:
        if not examples:
            continue
        states = torch.tensor(
            [history for history, _, _, _, _ in examples],
            dtype=torch.float32,
            device=device,
        )
        actions = torch.tensor(
            [action for _, action, _, _, _ in examples], dtype=torch.long, device=device
        )
        scope_labels = torch.tensor(
            [scope for _, _, _, scope, _ in examples], dtype=torch.long, device=device
        )
        reducer_labels = torch.tensor(
            [reducer for _, _, _, _, reducer in examples], dtype=torch.long, device=device
        )
        action_logits, scope_logits, reducer_logits = model(states)
        action_logits = action_logits.clone()
        for row, (_, _, asked, _, _) in enumerate(examples):
            if asked:
                action_logits[row, list(asked)] = -torch.inf

        query_loss = F.cross_entropy(action_logits, actions)
        scope_loss = F.cross_entropy(scope_logits, scope_labels)
        reducer_loss = F.cross_entropy(reducer_logits, reducer_labels)
        loss = query_loss + 0.5 * (scope_loss + reducer_loss)
        total_loss = loss * len(examples) if total_loss is None else total_loss + loss * len(examples)
        action_matches += int((action_logits.argmax(dim=-1) == actions).sum().item())
        decisions += len(examples)

    if total_loss is None or decisions == 0:
        raise RuntimeError("teacher batch contained no decisions")
    mean_loss = total_loss / decisions
    mean_loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    return float(mean_loss.detach().item()), action_matches, decisions


def _rollout(
    model: HistoryOnlyAgent,
    trials: list[tuple[int, TaskCase]],
    budget: int,
    device: torch.device,
) -> list[dict]:
    all_rules = tuple(range(len(RULES)))
    records = [{
        "hidden_rule": rule_id,
        "case": case,
        "posterior_for_diagnostics_only": all_rules,
        "asked": set(),
        "history": [_history_frame(all_rules, None, None, budget, budget)],
        "stopped": False,
        "policy_decisions": 0,
        "query_decisions": 0,
        "stop_decisions": 0,
        "teacher_action_matches": 0,
        "information_gain_optimal_queries": 0,
    } for rule_id, case in trials]

    model.eval()
    with torch.inference_mode():
        for _ in range(budget):
            active = [index for index, row in enumerate(records) if not row["stopped"]]
            if not active:
                break
            states = torch.tensor(
                [records[index]["history"] for index in active],
                dtype=torch.float32,
                device=device,
            )
            action_logits, _, _ = model(states)
            action_logits = action_logits.clone()
            for row_index, record_index in enumerate(active):
                asked = records[record_index]["asked"]
                if asked:
                    action_logits[row_index, list(asked)] = -torch.inf
            actions = action_logits.argmax(dim=-1).tolist()

            for row_index, record_index in enumerate(active):
                record = records[record_index]
                action = actions[row_index]
                posterior = record["posterior_for_diagnostics_only"]
                teacher_action = (
                    STOP_ACTION
                    if len(posterior) == 1
                    else _best_information_gain(posterior, tuple(sorted(record["asked"])))
                )
                record["policy_decisions"] += 1
                record["teacher_action_matches"] += int(action == teacher_action)
                if action == STOP_ACTION:
                    record["stop_decisions"] += 1
                    record["stopped"] = True
                    continue

                record["query_decisions"] += 1
                best_probe = _best_information_gain(
                    posterior, tuple(sorted(record["asked"]))
                ) if len(posterior) > 1 else None
                if best_probe is not None and abs(
                    _information_gain(posterior, action)
                    - _information_gain(posterior, best_probe)
                ) <= 1e-12:
                    record["information_gain_optimal_queries"] += 1

                hidden_rule = record["hidden_rule"]
                observed = PROBE_OUTCOMES[action][hidden_rule]
                record["posterior_for_diagnostics_only"] = update_posterior(
                    posterior, action, observed
                )
                record["asked"].add(action)
                record["history"].append(_history_frame(
                    record["posterior_for_diagnostics_only"], action, observed,
                    budget - len(record["asked"]), budget,
                ))

        predicted_scopes = [0] * len(records)
        predicted_reducers = [0] * len(records)
        lengths = {len(row["history"]) for row in records}
        for length in lengths:
            indices = [
                index for index, row in enumerate(records)
                if len(row["history"]) == length
            ]
            final_states = torch.tensor(
                [records[index]["history"] for index in indices],
                dtype=torch.float32,
                device=device,
            )
            _, scope_logits, reducer_logits = model(final_states)
            for local_index, record_index in enumerate(indices):
                predicted_scopes[record_index] = int(scope_logits[local_index].argmax().item())
                predicted_reducers[record_index] = int(reducer_logits[local_index].argmax().item())

    output = []
    for row, scope_id, reducer_id in zip(records, predicted_scopes, predicted_reducers):
        hidden_rule = row["hidden_rule"]
        predicted_rule = Rule(SCOPES[scope_id], REDUCERS[reducer_id])
        predicted_rule_id = RULES.index(predicted_rule)
        expected = apply_case(RULES[hidden_rule], row["case"])
        predicted = apply_case(predicted_rule, row["case"])
        output.append({
            "rule_id": hidden_rule,
            "correct": predicted == expected,
            "probes": len(row["asked"]),
            "exact_rule_prediction": predicted_rule_id == hidden_rule,
            "scope_prediction_correct": scope_id == SCOPES.index(RULES[hidden_rule].scope),
            "reducer_prediction_correct": reducer_id == REDUCERS.index(RULES[hidden_rule].reducer),
            "policy_stopped": row["stopped"],
            "policy_decisions": row["policy_decisions"],
            "query_decisions": row["query_decisions"],
            "stop_decisions": row["stop_decisions"],
            "teacher_action_matches": row["teacher_action_matches"],
            "information_gain_optimal_queries": row["information_gain_optimal_queries"],
        })
    return output


def _summarize(records: list[dict], probe_cost: float) -> dict:
    count = len(records)
    correct = sum(row["correct"] for row in records)
    probes = sum(row["probes"] for row in records)
    policy_decisions = sum(row["policy_decisions"] for row in records)
    query_decisions = sum(row["query_decisions"] for row in records)
    per_rule = {}
    for rule_id, rule in enumerate(RULES):
        group = [row for row in records if row["rule_id"] == rule_id]
        if group:
            per_rule[rule.name] = {
                "episodes": len(group),
                "correct_episodes": sum(row["correct"] for row in group),
                "accuracy": round(sum(row["correct"] for row in group) / len(group), 4),
                "exact_rule_decoding_rate": round(
                    sum(row["exact_rule_prediction"] for row in group) / len(group), 4
                ),
            "exact_rule_prediction_episodes": sum(
                row["exact_rule_prediction"] for row in group
            ),
            }
    return {
        "episodes": count,
        "correct_episodes": correct,
        "total_probes": probes,
        "accuracy": round(correct / count, 4),
        "mean_probes": round(probes / count, 4),
        "mean_task_reward": round(correct / count - probe_cost * probes / count, 4),
        "exact_rule_decoding_rate": round(
            sum(row["exact_rule_prediction"] for row in records) / count, 4
        ),
        "exact_rule_prediction_episodes": sum(
            row["exact_rule_prediction"] for row in records
        ),
        "scope_prediction_accuracy": round(
            sum(row["scope_prediction_correct"] for row in records) / count, 4
        ),
        "scope_correct_episodes": sum(row["scope_prediction_correct"] for row in records),
        "reducer_prediction_accuracy": round(
            sum(row["reducer_prediction_correct"] for row in records) / count, 4
        ),
        "reducer_correct_episodes": sum(row["reducer_prediction_correct"] for row in records),
        "policy_stop_rate": round(sum(row["policy_stopped"] for row in records) / count, 4),
        "teacher_agreement": {
            "policy_decisions": policy_decisions,
            "query_decisions": query_decisions,
            "stop_decisions": sum(row["stop_decisions"] for row in records),
            "teacher_action_matches": sum(row["teacher_action_matches"] for row in records),
            "information_gain_optimal_queries": sum(
                row["information_gain_optimal_queries"] for row in records
            ),
            "exact_teacher_action_rate": round(
                sum(row["teacher_action_matches"] for row in records) / policy_decisions, 4
            ) if policy_decisions else None,
            "information_gain_optimal_rate": round(
                sum(row["information_gain_optimal_queries"] for row in records) / query_decisions,
                4,
            ) if query_decisions else None,
        },
        "per_rule_accuracy": per_rule,
    }


def _evaluate(
    model: HistoryOnlyAgent,
    trials: list[tuple[int, TaskCase]],
    device: torch.device,
    probe_cost: float,
) -> dict[str, dict]:
    return {
        str(budget): _summarize(_rollout(model, trials, budget, device), probe_cost)
        for budget in range(1, MAX_PROBE_BUDGET + 1)
    }


def _train_controller(
    model: HistoryOnlyAgent,
    train_rule_ids: tuple[int, ...],
    validation_trials: list[tuple[int, TaskCase]],
    device: torch.device,
    train_size: int,
    batch_size: int,
    epochs: int,
    seed: int,
    learning_rate: float,
    probe_cost: float,
) -> dict:
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    best_score = -float("inf")
    best_epoch = 0
    best_state = None
    epoch_losses = []
    epoch_action_accuracies = []
    started = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        batch_sizes = [
            min(batch_size, train_size - start)
            for start in range(0, train_size, batch_size)
        ]
        if len(batch_sizes) > 1 and batch_sizes[-1] == 1:
            batch_sizes[-2] += 1
            batch_sizes.pop()
        start = 0
        loss_total = 0.0
        action_matches = 0
        decisions = 0
        for current_size in batch_sizes:
            loss, matches, current_decisions = _train_batch(
                model,
                optimizer,
                train_rule_ids,
                current_size,
                seed + epoch * 1_000_003 + start,
                device,
            )
            loss_total += loss * current_decisions
            action_matches += matches
            decisions += current_decisions
            start += current_size
        epoch_losses.append(round(loss_total / decisions, 6))
        epoch_action_accuracies.append(round(action_matches / decisions, 4))

        validation = _evaluate(model, validation_trials, device, probe_cost)
        score = sum(
            validation[str(budget)]["mean_task_reward"]
            for budget in range(1, MAX_PROBE_BUDGET + 1)
        ) / MAX_PROBE_BUDGET
        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }

    if best_state is None:
        raise RuntimeError("joint imitation training produced no checkpoint")
    model.load_state_dict(best_state)
    return {
        "best_epoch": best_epoch,
        "best_validation_mean_task_reward": round(best_score, 4),
        "best_validation_by_budget": _evaluate(model, validation_trials, device, probe_cost),
        "training_seconds": round(time.perf_counter() - started, 3),
        "mean_joint_loss_first_epoch": epoch_losses[0],
        "mean_joint_loss_last_epoch": epoch_losses[-1],
        "teacher_action_accuracy_first_epoch": epoch_action_accuracies[0],
        "teacher_action_accuracy_last_epoch": epoch_action_accuracies[-1],
    }


def run(args: argparse.Namespace) -> dict:
    validate_world()
    if args.fold not in range(FOLD_COUNT):
        raise ValueError(f"fold must be 0..{FOLD_COUNT - 1}")
    if args.seed not in SEEDS:
        raise ValueError(f"seed must be one of {SEEDS}")

    train_rule_ids = tuple(
        rule_id for rule_id in range(len(RULES))
        if rule_id not in FOLD_TEST_RULES[args.fold]
    )
    test_rule_ids = FOLD_TEST_RULES[args.fold]
    reference_path, reference = _reference_result(args.fold, args.seed)
    data = reference["data"]
    config = reference["training"]
    if tuple(data["train_rule_ids"]) != train_rule_ids:
        raise ValueError(f"training-rule split disagrees in {reference_path.name}")

    train_size = int(data["train_episodes_per_epoch"])
    validation_size = int(data["validation_episodes"])
    test_size = int(data["test_episodes"])
    batch_size = int(config["batch_size_episodes"])
    epochs = int(config["max_epochs"])
    learning_rate = float(config["learning_rate"])
    probe_cost = float(config["probe_cost_for_checkpoint_selection"])
    device = choose_device(args.device)
    validation_trials = generate_trials(validation_size, args.seed + 1, train_rule_ids)
    test_trials = generate_trials(test_size, args.seed + 2, test_rule_ids)

    result = {
        "experiment": "010-history-only-end-to-end-rule-decoder",
        "fold": args.fold,
        "seed": args.seed,
        "device": str(device),
        "torch_version": torch.__version__,
        "data": {
            "train_episodes_per_epoch": train_size,
            "total_training_episodes_per_controller": train_size * epochs,
            "validation_episodes": validation_size,
            "test_episodes": test_size,
            "train_rule_ids": list(train_rule_ids),
            "test_rule_ids": list(test_rule_ids),
            "teacher_trajectories_use_training_rules_only": True,
            "test_cases_replayed_from_experiment_009": True,
            "test_case_hidden_until_answer_decoding": True,
        },
        "input_contract": {
            "controller_receives_exact_posterior_mask": False,
            "controller_receives": [
                "previous probe identity",
                "observed probe outcome",
                "remaining query budget",
                "initial query budget",
            ],
            "candidate_rule_catalog_available_to_agent": False,
            "scope_and_reducer_vocabularies_available_to_decoder": True,
            "query_policy_can_stop": True,
            "simulator_stops_on_exact_posterior_singleton": False,
            "final_answer_uses_exact_posterior": False,
            "final_answer": "compose predicted scope and reducer, then apply them to the task case",
            "posterior_used_only_inside_simulator_and_for_diagnostics": True,
        },
        "training": {
            "method": "joint supervised information-gain query imitation and factorized scope/reducer prediction",
            "matched_reference_experiment_009": str(reference_path.relative_to(PROJECT_ROOT)),
            "query_teacher": "maximum expected entropy reduction; STOP when exact training-teacher posterior is a singleton",
            "answer_labels": "hidden training-fold rule scope and reducer, at each observed history",
            "teacher_hidden_rules": "training-fold rules only",
            "loss": "query cross-entropy + 0.5 * (scope cross-entropy + reducer cross-entropy)",
            "optimizer": config["optimizer"],
            "learning_rate": learning_rate,
            "max_epochs": epochs,
            "batch_size_episodes": batch_size,
            "checkpoint_selection_metric": "mean validation end-to-end task reward across budgets 1-3",
            "probe_cost_for_checkpoint_selection": probe_cost,
        },
        "architecture": {
            "recurrent_state_size": STATE_SIZE,
            "policy_action_count_including_stop": POLICY_ACTION_COUNT,
            "shared_heads": {
                "query_policy": POLICY_ACTION_COUNT,
                "scope_decoder": len(SCOPES),
                "reducer_decoder": len(REDUCERS),
            },
            "controllers": {},
        },
        "reference_test_metrics": {
            "history_only_query_policy_with_exact_answer_decoder": {
                name: reference["controllers"][name]["test_by_budget"]
                for name in CONTROLLERS
            },
            "belief_state_imitation_with_exact_answer_decoder": reference["reference_test_metrics"]["belief_state_imitation"],
            "reinforce_with_exact_answer_decoder": reference["reference_test_metrics"]["reinforce"],
            "information_gain_with_exact_answer_decoder": reference["reference_test_metrics"]["information_gain"],
        },
        "controllers": {},
        "interpretation_limit": "The agent must infer a rule composition from history and apply it without an exact posterior. The teacher, simulator, and evaluation diagnostics still use the fixed 18-rule grammar; scope and reducer primitives are known, so this tests held-out composition transfer, not open-world primitive discovery.",
    }

    checkpoint_dir = PROJECT_ROOT / "artifacts" / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    for name in CONTROLLERS:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        model = _build_agent(name)
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        training = _train_controller(
            model,
            train_rule_ids,
            validation_trials,
            device,
            train_size,
            batch_size,
            epochs,
            args.seed,
            learning_rate,
            probe_cost,
        )
        checkpoint = checkpoint_dir / (
            f"experiment-010_fold-{args.fold}_{name}_seed-{args.seed}.pt"
        )
        torch.save(model.state_dict(), checkpoint)
        result["architecture"]["controllers"][name] = {
            "parameters": parameter_count,
            "feature_size": FEATURE_SIZES[name],
        }
        result["controllers"][name] = {
            **training,
            "test_by_budget": _evaluate(model, test_trials, device, probe_cost),
            "checkpoint": str(checkpoint.relative_to(PROJECT_ROOT)),
        }
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    result_dir = PROJECT_ROOT / "artifacts" / "results"
    result_dir.mkdir(parents=True, exist_ok=True)
    result_path = result_dir / f"experiment-010_fold-{args.fold}_seed-{args.seed}.json"
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({
        "experiment": result["experiment"],
        "fold": args.fold,
        "seed": args.seed,
        "held_out_rules": [RULES[index].name for index in test_rule_ids],
        "test_by_budget": {
            name: {
                budget: {
                    "accuracy": metrics["accuracy"],
                    "task_reward": metrics["mean_task_reward"],
                    "exact_rule_decoding_rate": metrics["exact_rule_decoding_rate"],
                    "scope_accuracy": metrics["scope_prediction_accuracy"],
                    "reducer_accuracy": metrics["reducer_prediction_accuracy"],
                    "mean_probes": metrics["mean_probes"],
                    "policy_stop_rate": metrics["policy_stop_rate"],
                }
                for budget, metrics in result["controllers"][name]["test_by_budget"].items()
            }
            for name in CONTROLLERS
        },
        "saved": str(result_path),
    }, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train an end-to-end rule learner from history-only interaction"
    )
    parser.add_argument("--fold", type=int, choices=range(FOLD_COUNT), default=0)
    parser.add_argument("--seed", type=int, choices=SEEDS, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
