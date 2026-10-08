"""Train information-gain imitators without exposing the exact posterior mask."""

import argparse
import json
import random
from pathlib import Path

import torch
from torch import Tensor, nn

from cognitive_lab.active.exp007.train import (
    CONTROLLERS,
    MAX_PROBE_BUDGET,
    PROJECT_ROOT,
    STATE_SIZE,
    _build_controller,
    _summarize,
    generate_trials,
)
from cognitive_lab.active.exp007.world import (
    FOLD_COUNT,
    FOLD_TEST_RULES,
    RULES,
    validate_world,
)
from cognitive_lab.active.exp008.train import (
    REPLAY_KEYS,
    _evaluate,
    _train_controller,
)
from cognitive_lab.active.exp007.diagnose import _rollout_information_gain
from cognitive_lab.train import choose_device

SEEDS = tuple(range(42, 48))


class HistoryOnlyController(nn.Module):
    """Keep the Experiment 007 core but zero its exact candidate-mask inputs."""

    def __init__(self, core: nn.Module):
        super().__init__()
        self.core = core

    def forward(self, observations: Tensor) -> Tensor:
        if observations.shape[-1] != STATE_SIZE:
            raise ValueError(f"expected state size {STATE_SIZE}, got {observations.shape[-1]}")
        # The first 18 values are the exact posterior mask; the remaining values
        # encode only previous probes, feedback, and the query budget.
        actor_input = torch.cat((
            torch.zeros_like(observations[..., :len(RULES)]),
            observations[..., len(RULES):],
        ), dim=-1)
        return self.core(actor_input)


def _reference_result(fold: int, seed: int) -> tuple[Path, dict]:
    path = PROJECT_ROOT / "artifacts" / "results" / f"experiment-008_fold-{fold}_seed-{seed}.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"Run matching Experiment 008 first; missing {path.name}"
        )
    result = json.loads(path.read_text(encoding="utf-8"))
    if (
        result.get("experiment") != "008-information-gain-teacher-imitation"
        or result.get("fold") != fold
        or result.get("seed") != seed
        or tuple(result.get("data", {}).get("test_rule_ids", ())) != FOLD_TEST_RULES[fold]
        or result.get("data", {}).get("teacher_trajectories_use_training_rules_only") is not True
    ):
        raise ValueError(f"unexpected Experiment 008 metadata in {path.name}")
    return path, result


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

    for budget in range(1, MAX_PROBE_BUDGET + 1):
        replayed = _summarize(
            _rollout_information_gain(test_trials, budget), probe_cost
        )
        saved = reference["reference_test_metrics"]["information_gain"][str(budget)]
        if any(replayed[key] != saved[key] for key in REPLAY_KEYS):
            raise ValueError(
                f"test episodes disagree with Experiment 008 at budget {budget}"
            )

    result = {
        "experiment": "009-history-only-query-policy",
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
            "test_cases_replayed_from_experiment_008": True,
        },
        "input_contract": {
            "controller_receives_exact_posterior_mask": False,
            "controller_receives": [
                "previous probe identity",
                "observed probe outcome",
                "remaining query budget",
                "initial query budget",
            ],
            "history_sequence_is_recurrent_input": True,
            "candidate_catalog_available_to_controller": False,
            "posterior_remains_inside_simulator_and_teacher": True,
            "posterior_remains_in_final_answer_decoder": True,
            "simulator_ends_querying_when_exact_posterior_is_singleton": True,
        },
        "training": {
            "method": "same supervised information-gain imitation recipe as Experiment 008",
            "matched_reference_experiment_008": str(reference_path.relative_to(PROJECT_ROOT)),
            "teacher_action": "maximum expected entropy reduction under the fixed 18-rule candidate catalog",
            "teacher_hidden_rules": "training-fold rules only",
            "optimizer": config["optimizer"],
            "learning_rate": learning_rate,
            "max_epochs": epochs,
            "batch_size_episodes": batch_size,
            "checkpoint_selection_metric": config["checkpoint_selection_metric"],
            "probe_cost_for_checkpoint_selection": probe_cost,
        },
        "architecture": reference["architecture"],
        "reference_test_metrics": {
            "reinforce": reference["reference_test_metrics"]["reinforce"],
            "information_gain": reference["reference_test_metrics"]["information_gain"],
            "belief_state_imitation": {
                name: reference["controllers"][name]["test_by_budget"]
                for name in CONTROLLERS
            },
        },
        "controllers": {},
        "interpretation_limit": "This ablation tests whether the query policy can use recurrent interaction history without receiving the exact posterior mask. The simulator and final answer decoder still use the exact posterior, and the teacher still searches the fixed 18-rule catalog; this is not a fully end-to-end or open-world learner.",
    }

    checkpoint_dir = PROJECT_ROOT / "artifacts" / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    for name in CONTROLLERS:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        model = HistoryOnlyController(_build_controller(name))
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
            f"experiment-009_fold-{args.fold}_{name}_seed-{args.seed}.pt"
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
    result_path = result_dir / f"experiment-009_fold-{args.fold}_seed-{args.seed}.json"
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({
        "experiment": result["experiment"],
        "fold": args.fold,
        "seed": args.seed,
        "held_out_rules": [RULES[index].name for index in test_rule_ids],
        "test_by_budget": {
            name: {
                budget: {
                    "accuracy_with_exact_answer_decoder": metrics["accuracy"],
                    "task_reward_with_exact_answer_decoder": metrics["mean_task_reward"],
                    "teacher_action_agreement": metrics["teacher_agreement"]["exact_teacher_action_rate"],
                    "information_gain_optimal_rate": metrics["teacher_agreement"]["information_gain_optimal_rate"],
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
        description="Train a history-only query policy for Experiment 009"
    )
    parser.add_argument("--fold", type=int, choices=range(FOLD_COUNT), default=0)
    parser.add_argument("--seed", type=int, choices=SEEDS, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
