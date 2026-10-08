"""Train Experiment 007 controllers to imitate its exact information-gain policy."""

import argparse
import json
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from cognitive_lab.active.environment import TaskCase
from cognitive_lab.active.exp007.diagnose import (
    _rollout_information_gain,
    _rollout_policy,
)
from cognitive_lab.active.exp007.train import (
    CONTROLLERS,
    MAX_PROBE_BUDGET,
    PROJECT_ROOT,
    _balanced_rule_schedule,
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
    update_posterior,
    validate_world,
)
from cognitive_lab.train import choose_device

SEEDS = tuple(range(42, 48))
REPLAY_KEYS = (
    "episodes", "correct_episodes", "total_probes", "identified_episodes",
    "accuracy", "mean_probes", "mean_task_reward",
    "rule_identification_rate", "mean_remaining_rules",
)


def _reference_result(fold: int, seed: int) -> tuple[Path, dict]:
    path = PROJECT_ROOT / "artifacts" / "results" / f"experiment-007_fold-{fold}_seed-{seed}.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"Run Experiment 007 first; missing matching reference: {path.name}"
        )
    result = json.loads(path.read_text(encoding="utf-8"))
    if (
        result.get("experiment") != "007-compositional-rule-generalization"
        or result.get("fold") != fold
        or result.get("seed") != seed
        or tuple(result.get("data", {}).get("test_rule_ids", ())) != FOLD_TEST_RULES[fold]
    ):
        raise ValueError(f"unexpected Experiment 007 metadata in {path.name}")
    return path, result


def _teacher_examples(
    train_rule_ids: tuple[int, ...],
    batch_size: int,
    seed: int,
) -> list[list[tuple[list[list[float]], int, tuple[int, ...]]]]:
    rng = random.Random(seed)
    hidden_rules = _balanced_rule_schedule(batch_size, train_rule_ids, rng)
    budgets = [rng.randint(1, MAX_PROBE_BUDGET) for _ in range(batch_size)]
    by_step: list[list[tuple[list[list[float]], int, tuple[int, ...]]]] = [
        [] for _ in range(MAX_PROBE_BUDGET)
    ]
    all_rules = tuple(range(len(RULES)))

    for hidden_rule, budget in zip(hidden_rules, budgets):
        posterior = all_rules
        asked: set[int] = set()
        history = [_encode_state(all_rules, None, None, budget, budget)]
        for step in range(budget):
            if len(posterior) <= 1:
                break
            probe_id = _best_information_gain(posterior, tuple(sorted(asked)))
            if probe_id is None:
                raise RuntimeError("teacher found no informative probe for a non-singleton posterior")
            by_step[step].append((history, probe_id, tuple(sorted(asked))))
            observed = PROBE_OUTCOMES[probe_id][hidden_rule]
            posterior = update_posterior(posterior, probe_id, observed)
            asked.add(probe_id)
            history = history + [_encode_state(
                posterior,
                probe_id,
                observed,
                budget - len(asked),
                budget,
            )]
    return by_step


def _train_batch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_rule_ids: tuple[int, ...],
    batch_size: int,
    seed: int,
    device: torch.device,
) -> tuple[float, int, int]:
    examples_by_step = _teacher_examples(train_rule_ids, batch_size, seed)
    optimizer.zero_grad(set_to_none=True)
    total_loss: Tensor | None = None
    decisions = 0
    matches = 0

    for examples in examples_by_step:
        if not examples:
            continue
        states = torch.tensor(
            [history for history, _, _ in examples],
            dtype=torch.float32,
            device=device,
        )
        labels = torch.tensor(
            [action for _, action, _ in examples], dtype=torch.long, device=device
        )
        logits = model(states).clone()
        for row, (_, _, asked) in enumerate(examples):
            if asked:
                logits[row, list(asked)] = -torch.inf
        loss = F.cross_entropy(logits, labels, reduction="sum")
        total_loss = loss if total_loss is None else total_loss + loss
        decisions += len(examples)
        matches += int((logits.argmax(dim=-1) == labels).sum().item())

    if total_loss is None or decisions == 0:
        raise RuntimeError("teacher batch contained no query decisions")
    mean_loss = total_loss / decisions
    mean_loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    return float(mean_loss.detach().item()), matches, decisions


def _evaluate(
    model: nn.Module,
    trials: list[tuple[int, TaskCase]],
    device: torch.device,
    probe_cost: float,
) -> dict[str, dict]:
    result = {}
    for budget in range(1, MAX_PROBE_BUDGET + 1):
        rows = _rollout_policy(model, trials, budget, device)
        metrics = _summarize(rows, probe_cost)
        decisions = [
            step
            for row in rows
            for step in row["trajectory"]
        ]
        query_steps = [step for step in decisions if step["action"] == "probe"]
        query_count = len(query_steps)
        teacher_matches = sum(step["matches_best_probe_id"] for step in query_steps)
        ig_optimal = sum(step["is_ig_optimal"] for step in query_steps)
        metrics["teacher_agreement"] = {
            "policy_decisions": len(decisions),
            "query_decisions": query_count,
            "stop_decisions": len(decisions) - query_count,
            "teacher_action_matches": teacher_matches,
            "information_gain_optimal_queries": ig_optimal,
            "exact_teacher_action_rate": round(teacher_matches / len(decisions), 4)
            if decisions else None,
            "information_gain_optimal_rate": round(ig_optimal / query_count, 4)
            if query_count else None,
        }
        result[str(budget)] = metrics
    return result


def _train_controller(
    model: nn.Module,
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
    best_validation = None
    epoch_losses = []
    epoch_teacher_accuracy = []
    epoch_decisions = []
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
        match_total = 0
        decision_total = 0
        for current_size in batch_sizes:
            loss, matches, decisions = _train_batch(
                model,
                optimizer,
                train_rule_ids,
                current_size,
                seed + epoch * 1_000_003 + start,
                device,
            )
            loss_total += loss * decisions
            match_total += matches
            decision_total += decisions
            start += current_size
        epoch_losses.append(round(loss_total / decision_total, 6))
        epoch_teacher_accuracy.append(round(match_total / decision_total, 4))
        epoch_decisions.append(decision_total)

        validation = _evaluate(model, validation_trials, device, probe_cost)
        score = sum(
            validation[str(budget)]["mean_task_reward"]
            for budget in range(1, MAX_PROBE_BUDGET + 1)
        ) / MAX_PROBE_BUDGET
        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_validation = validation
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }

    if best_state is None or best_validation is None:
        raise RuntimeError("imitation training produced no checkpoint")
    model.load_state_dict(best_state)
    return {
        "best_epoch": best_epoch,
        "best_validation_mean_task_reward": round(best_score, 4),
        "best_validation_by_budget": best_validation,
        "training_seconds": round(time.perf_counter() - started, 3),
        "mean_imitation_cross_entropy_first_epoch": epoch_losses[0],
        "mean_imitation_cross_entropy_last_epoch": epoch_losses[-1],
        "teacher_action_accuracy_first_epoch": epoch_teacher_accuracy[0],
        "teacher_action_accuracy_last_epoch": epoch_teacher_accuracy[-1],
        "mean_teacher_decisions_per_epoch": round(sum(epoch_decisions) / len(epoch_decisions), 2),
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
    config = reference["training"]
    data = reference["data"]
    if tuple(data["train_rule_ids"]) != train_rule_ids:
        raise ValueError(f"training-rule split disagrees in {reference_path.name}")

    train_size = int(data["train_episodes_per_epoch"])
    validation_size = int(data["validation_episodes"])
    test_size = int(data["test_episodes"])
    batch_size = int(config["batch_size"])
    epochs = int(config["max_epochs"])
    learning_rate = float(config["learning_rate"])
    probe_cost = float(config["probe_cost"])
    device = choose_device(args.device)
    validation_trials = generate_trials(validation_size, args.seed + 1, train_rule_ids)
    test_trials = generate_trials(test_size, args.seed + 2, test_rule_ids)

    for budget in range(1, MAX_PROBE_BUDGET + 1):
        replayed = _summarize(
            _rollout_information_gain(test_trials, budget), probe_cost
        )
        saved = reference["baselines"]["information_gain"]["by_budget"][str(budget)]
        if any(replayed[key] != saved[key] for key in REPLAY_KEYS):
            raise ValueError(
                f"test episode replay differs from Experiment 007 at budget {budget}"
            )

    result = {
        "experiment": "008-information-gain-teacher-imitation",
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
            "teacher_rule_catalog_size": len(RULES),
            "probe_budget_range": [1, MAX_PROBE_BUDGET],
            "test_cases_replayed_from_experiment_007": True,
        },
        "training": {
            "method": "supervised cross-entropy imitation of exact information gain",
            "teacher_action": "maximum expected entropy reduction; deterministic lowest probe ID for exact ties",
            "teacher_observes_held_out_hidden_rules": False,
            "optimizer": "AdamW",
            "learning_rate": learning_rate,
            "max_epochs": epochs,
            "batch_size_episodes": batch_size,
            "probe_cost_for_checkpoint_selection": probe_cost,
            "checkpoint_selection_metric": "mean validation task reward across budgets 1-3, matching Experiment 007",
            "reference_experiment_007": str(reference_path.relative_to(PROJECT_ROOT)),
        },
        "architecture": reference["architecture"],
        "reference_test_metrics": {
            "reinforce": {
                name: entry["test_by_budget"]
                for name, entry in reference["controllers"].items()
            },
            "information_gain": reference["baselines"]["information_gain"]["by_budget"],
        },
        "controllers": {},
        "interpretation_limit": "The teacher is an exact information-gain algorithm over the fixed 18-rule catalog. Training uses hidden rules from train folds only; test rules and exact posterior filtering remain in the fixed catalog, so this measures imitation transfer within this simulator, not open-world discovery.",
    }

    checkpoint_dir = PROJECT_ROOT / "artifacts" / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    for name in CONTROLLERS:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        model = _build_controller(name)
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
            f"experiment-008_fold-{args.fold}_{name}_seed-{args.seed}.pt"
        )
        torch.save(model.state_dict(), checkpoint)
        result["controllers"][name] = {
            "parameters": parameter_count,
            **training,
            "test_by_budget": _evaluate(model, test_trials, device, probe_cost),
            "checkpoint": str(checkpoint.relative_to(PROJECT_ROOT)),
        }
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    result_dir = PROJECT_ROOT / "artifacts" / "results"
    result_dir.mkdir(parents=True, exist_ok=True)
    result_path = result_dir / f"experiment-008_fold-{args.fold}_seed-{args.seed}.json"
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({
        "experiment": result["experiment"],
        "fold": args.fold,
        "seed": args.seed,
        "held_out_rules": [RULES[index].name for index in test_rule_ids],
        "best_epoch": {
            name: result["controllers"][name]["best_epoch"] for name in CONTROLLERS
        },
        "test_by_budget": {
            name: {
                budget: {
                    "accuracy": metrics["accuracy"],
                    "task_reward": metrics["mean_task_reward"],
                    "teacher_action_agreement": metrics["teacher_agreement"]["exact_teacher_action_rate"],
                    "information_gain_optimal_rate": metrics["teacher_agreement"]["information_gain_optimal_rate"],
                }
                for budget, metrics in result["controllers"][name]["test_by_budget"].items()
            }
            for name in CONTROLLERS
        },
        "reference_007_test_reward": {
            name: {
                budget: metrics["mean_task_reward"]
                for budget, metrics in result["reference_test_metrics"]["reinforce"][name].items()
            }
            for name in CONTROLLERS
        },
        "saved": str(result_path),
    }, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train Experiment 008 policies to imitate information-gain queries"
    )
    parser.add_argument("--fold", type=int, choices=range(FOLD_COUNT), default=0)
    parser.add_argument("--seed", type=int, choices=SEEDS, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
