"""Train and compare recurrent controllers for Experiment 006A."""

import argparse
import json
from pathlib import Path
import random

import torch

from cognitive_lab.active.baselines import evaluate_baselines, generate_trials
from cognitive_lab.active.controller import (
    FOLD_TEST_RULES,
    MAX_PROBE_BUDGET,
    POLICY_ACTION_COUNT,
    STATE_SIZE,
    build_controller,
    evaluate_controller,
    generate_imitation_data,
    train_controller,
)
from cognitive_lab.active.environment import ALL_PROBES, RULES
from cognitive_lab.train import choose_device
from cognitive_lab.task import ACTION_COUNT

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONTROLLERS = ("gru_256", "gru_parameter_matched", "rims")


def run(args: argparse.Namespace) -> dict:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = choose_device(args.device)
    test_rule_ids = FOLD_TEST_RULES[args.fold]
    train_rule_ids = tuple(
        rule_id for rule_id in range(len(RULES)) if rule_id not in test_rule_ids
    )
    train_data = generate_imitation_data(args.train_size, train_rule_ids, args.seed)
    validation_data = generate_imitation_data(
        args.validation_size, train_rule_ids, args.seed + 1
    )
    test_trials = generate_trials(args.test_size, args.seed + 2, test_rule_ids)
    result = {
        "experiment": "006A-active-controller-imitation",
        "seed": args.seed,
        "fold": args.fold,
        "device": str(device),
        "torch_version": torch.__version__,
        "data": {
            "train_episodes": args.train_size,
            "validation_episodes": args.validation_size,
            "test_episodes": args.test_size,
            "train_rule_ids": list(train_rule_ids),
            "test_rule_ids": list(test_rule_ids),
            "test_rule_names": [RULES[index].name for index in test_rule_ids],
            "hypothesis_catalog": "all six candidates remain visible; hidden-rule episodes are held out",
            "test_cases_hidden_until_probing_finishes": True,
            "probe_budget": args.probe_budget,
        },
        "training": {
            "method": "behavioral cloning",
            "teacher": "exact information gain over the six-rule catalog",
            "random_probe_probability": 0.5,
            "behavior_policy": "follow the teacher unless a random unasked probe is sampled",
            "optimizer": "AdamW",
            "learning_rate": 1e-3,
            "weight_decay": 0.01,
            "max_epochs": args.epochs,
            "batch_size": args.batch_size,
            "checkpoint_selection_metric": "validation teacher-action accuracy",
            "loss": "inverse-square-root-frequency weighted cross entropy",
        },
        "architecture": {
            "input_size": STATE_SIZE,
            "input_features": {
                "posterior_rule_mask": len(RULES),
                "previous_probe_one_hot": len(ALL_PROBES),
                "previous_outcome_one_hot": ACTION_COUNT,
                "remaining_and_total_budget": 2,
            },
            "output_size": POLICY_ACTION_COUNT,
            "output_actions": "one logit per probe candidate",
            "maximum_interaction_steps": MAX_PROBE_BUDGET,
            "maximum_history_frames": MAX_PROBE_BUDGET + 1,
            "answer_policy": "posterior majority; lowest action id breaks ties",
            "controllers": {
                "gru_256": {"core": "GRUCore", "hidden_size": 256},
                "gru_parameter_matched": {"core": "GRUCore", "hidden_size": 205},
                "rims": {"core": "RIMsCore", "modules": 4, "module_size": 64, "top_k": 1},
            },
        },
        "baselines": evaluate_baselines(
            args.test_size,
            args.probe_budget,
            args.seed + 2,
            rule_ids=test_rule_ids,
            trials=test_trials,
        ),
        "controllers": {},
    }

    checkpoint_dir = PROJECT_ROOT / "artifacts" / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    for name in CONTROLLERS:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        model = build_controller(name)
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        training_metrics = train_controller(
            model,
            train_data,
            validation_data,
            device,
            args.epochs,
            args.batch_size,
            args.seed,
        )
        checkpoint_path = (
            checkpoint_dir
            / f"experiment-006A_fold-{args.fold}_{name}_seed-{args.seed}.pt"
        )
        torch.save(model.state_dict(), checkpoint_path)
        result["architecture"]["controllers"][name]["parameters"] = parameter_count
        result["controllers"][name] = {
            **training_metrics,
            "test_by_budget": evaluate_controller(
                model, test_trials, args.probe_budget, device
            ),
            "checkpoint": str(checkpoint_path.relative_to(PROJECT_ROOT)),
        }

    results_dir = PROJECT_ROOT / "artifacts" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_dir / f"experiment-006A_fold-{args.fold}_seed-{args.seed}.json"
    results_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Saved: {results_path}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train GRU and RIMs controllers for active probing"
    )
    parser.add_argument("--fold", type=int, choices=range(len(FOLD_TEST_RULES)), default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--train-size", type=int, default=4096)
    parser.add_argument("--validation-size", type=int, default=512)
    parser.add_argument("--test-size", type=int, default=1024)
    parser.add_argument("--probe-budget", type=int, choices=range(1, 4), default=3)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.train_size, args.validation_size, args.test_size) <= 0:
        parser.error("epochs, batch size, and episode counts must be positive")
    run(args)


if __name__ == "__main__":
    main()
