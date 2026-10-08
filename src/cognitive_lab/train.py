import argparse
import json
import random
import time
from pathlib import Path
from typing import Callable

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

from cognitive_lab.core.gru import GRUCore
from cognitive_lab.core.rims import RIMsCore
from cognitive_lab.task import (
    ACTION_COUNT,
    FOLD_COUNT,
    INPUT_SIZE,
    KEY_COUNT,
    SEQUENCE_LENGTH,
    composition_fold_templates,
    composition_split_templates,
    generate_composition_episodes,
    generate_episodes,
    generate_fold_episodes,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but PyTorch cannot access a CUDA device")
    return device


def evaluate(
    model: nn.Module,
    data: tuple[Tensor, ...],
    device: torch.device,
    per_composition: bool = False,
    include_balanced_accuracy: bool = False,
) -> dict:
    observations, labels, corrected, missing = (
        tensor.to(device) for tensor in data[:4]
    )
    model.eval()
    with torch.no_grad():
        predictions = model(observations).argmax(dim=-1)
        matches = predictions.eq(labels)

    def subset_accuracy(mask: Tensor) -> float | None:
        return round(matches[mask].float().mean().item(), 4) if mask.any() else None

    metrics = {
        "accuracy": subset_accuracy(torch.ones_like(matches, dtype=torch.bool)),
        "corrected_accuracy": subset_accuracy(corrected),
        "missing_rule_accuracy": subset_accuracy(missing),
    }
    if include_balanced_accuracy:
        corrected_accuracy = metrics["corrected_accuracy"]
        missing_rule_accuracy = metrics["missing_rule_accuracy"]
        metrics["balanced_accuracy"] = (
            round((corrected_accuracy + missing_rule_accuracy) / 2, 4)
            if corrected_accuracy is not None and missing_rule_accuracy is not None
            else None
        )
    if per_composition:
        if len(data) < 5:
            raise ValueError("per-composition metrics require template IDs")
        template_ids = data[4].to(device)
        composition_metrics = {}
        for template_id in torch.unique(template_ids).tolist():
            if template_id < 0:
                continue
            target_door, keys = divmod(template_id, KEY_COUNT * KEY_COUNT)
            previous_key, final_key = divmod(keys, KEY_COUNT)
            mask = template_ids == template_id
            template_metrics = {
                "examples": int(mask.sum().item()),
                "accuracy": subset_accuracy(mask),
            }
            if previous_key != final_key:
                template_metrics["corrected_accuracy"] = subset_accuracy(mask & corrected)
            composition_metrics[f"{target_door},{previous_key},{final_key}"] = (
                template_metrics
            )
        metrics["composition_metrics"] = composition_metrics
    return metrics


def train_one(
    model: nn.Module,
    train_data: tuple[Tensor, ...],
    validation_data: tuple[Tensor, ...],
    device: torch.device,
    epochs: int,
    batch_size: int,
    seed: int,
    selection_metric: str = "accuracy",
    collect_epoch_metrics: bool = False,
) -> tuple[int, dict, list[dict]]:
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    loader = DataLoader(
        TensorDataset(train_data[0], train_data[1]),
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    best_validation_score = -1.0
    best_epoch = 0
    best_state: dict[str, Tensor] | None = None
    epoch_metrics = []

    for epoch in range(1, epochs + 1):
        model.train()
        for observations, labels in loader:
            observations, labels = observations.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(observations), labels)
            loss.backward()
            optimizer.step()

        include_balanced_accuracy = selection_metric == "balanced_accuracy"
        training_metrics = (
            evaluate(
                model,
                train_data,
                device,
                include_balanced_accuracy=include_balanced_accuracy,
            )
            if collect_epoch_metrics
            else None
        )
        metrics = evaluate(
            model,
            validation_data,
            device,
            include_balanced_accuracy=include_balanced_accuracy,
        )
        if collect_epoch_metrics:
            epoch_metrics.append(
                {"epoch": epoch, "train": training_metrics, "validation": metrics}
            )
        current_score = metrics[selection_metric]
        if current_score is None:
            raise RuntimeError(f"validation metric {selection_metric!r} has no examples")
        if current_score > best_validation_score:
            best_validation_score = current_score
            best_epoch = epoch
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("training produced no checkpoint")
    model.load_state_dict(best_state)
    for record in epoch_metrics:
        record["selected_checkpoint"] = record["epoch"] == best_epoch
    return best_epoch, evaluate(
        model,
        validation_data,
        device,
        include_balanced_accuracy=selection_metric == "balanced_accuracy",
    ), epoch_metrics


def build_models() -> dict[str, Callable[[], nn.Module]]:
    return {
        "gru_state_matched": lambda: GRUCore(INPUT_SIZE, ACTION_COUNT, hidden_size=256),
        "gru_parameter_matched": lambda: GRUCore(INPUT_SIZE, ACTION_COUNT, hidden_size=153),
        "rims": lambda: RIMsCore(
            INPUT_SIZE, ACTION_COUNT, module_count=4, module_size=64, top_k=1
        ),
    }


def run(args: argparse.Namespace) -> dict:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = choose_device(args.device)

    is_fold_experiment = args.experiment in ("004", "005")
    fold = args.fold if is_fold_experiment else 0
    if args.experiment == "005":
        selection_metric = "balanced_accuracy"
    elif args.experiment == "004":
        selection_metric = "corrected_accuracy"
    else:
        selection_metric = "accuracy"
    if args.experiment == "003":
        experiment_name = "003-heldout-composition"
        train_data = generate_composition_episodes(
            args.train_size, args.seed, "train"
        )
        validation_data = generate_composition_episodes(
            args.validation_size, args.seed + 1, "validation"
        )
        test_data = generate_composition_episodes(
            args.test_size, args.seed + 2, "test"
        )
    elif is_fold_experiment:
        experiment_name = (
            "005-balanced-checkpoint-selection"
            if args.experiment == "005"
            else "004-composition-folds"
        )
        train_data = generate_fold_episodes(
            args.train_size, args.seed, "train", fold
        )
        validation_data = generate_fold_episodes(
            args.validation_size, args.seed + 1, "validation", fold
        )
        test_data = generate_fold_episodes(
            args.test_size, args.seed + 2, "test", fold
        )
    else:
        experiment_name = "002-parameter-matched"
        train_data = generate_episodes(args.train_size, args.seed)
        validation_data = generate_episodes(args.validation_size, args.seed + 1)
        test_data = generate_episodes(args.test_size, args.seed + 2)
    result = {
        "experiment": experiment_name,
        "seed": args.seed,
        "device": str(device),
        "torch_version": torch.__version__,
        "data": {
            "train_episodes": args.train_size,
            "validation_episodes": args.validation_size,
            "test_episodes": args.test_size,
            "sequence_length": SEQUENCE_LENGTH,
        },
        "training": {
            "max_epochs": args.epochs,
            "batch_size": args.batch_size,
            "optimizer": "AdamW",
            "learning_rate": 1e-3,
            "checkpoint_selection_metric": selection_metric,
            "epoch_metrics_recorded": args.epoch_metrics,
        },
        "architecture": {
            "gru_state_matched": {"layers": 1, "hidden_size": 256},
            "gru_parameter_matched": {"layers": 1, "hidden_size": 153},
            "rims": {"modules": 4, "module_size": 64, "top_k": 1},
        },
        "models": {},
    }
    if args.experiment == "003":
        result["data_split"] = {
            "strategy": "heldout_target_rule_composition",
            "composition_unit": ["target_door", "previous_key", "final_key"],
            "templates": composition_split_templates(),
        }
    elif is_fold_experiment:
        result["fold"] = fold
        result["data_split"] = {
            "strategy": "three_fold_heldout_composition",
            "fold_count": FOLD_COUNT,
            "composition_unit": ["target_door", "previous_key", "final_key"],
            "templates": composition_fold_templates(fold),
            "test_coverage": "each of the 27 templates is held out once across folds",
        }
    if args.experiment == "005":
        result["training"]["checkpoint_selection_definition"] = (
            "mean(validation corrected_accuracy, validation missing_rule_accuracy)"
        )
    checkpoint_dir = PROJECT_ROOT / "artifacts" / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    run_prefix = f"experiment-{args.experiment}"
    if is_fold_experiment:
        run_prefix += f"_fold-{fold}"
    if args.epoch_metrics:
        run_prefix += "_epoch-metrics"

    for name, factory in build_models().items():
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        model = factory()
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        started = time.perf_counter()
        best_epoch, validation_metrics, epoch_metrics = train_one(
            model,
            train_data,
            validation_data,
            device,
            args.epochs,
            args.batch_size,
            args.seed,
            selection_metric,
            collect_epoch_metrics=args.epoch_metrics,
        )
        elapsed = round(time.perf_counter() - started, 3)
        test_metrics = evaluate(
            model,
            test_data,
            device,
            per_composition=is_fold_experiment,
            include_balanced_accuracy=args.experiment == "005",
        )
        checkpoint_path = (
            checkpoint_dir
            / f"{run_prefix}_{name}_seed-{args.seed}.pt"
        )
        torch.save(model.state_dict(), checkpoint_path)
        result["models"][name] = {
            "parameters": parameter_count,
            "best_epoch": best_epoch,
            "training_seconds": elapsed,
            "validation": validation_metrics,
            "test": test_metrics,
            "checkpoint": str(checkpoint_path.relative_to(PROJECT_ROOT)),
        }
        if args.epoch_metrics:
            result["models"][name]["epoch_metrics"] = epoch_metrics

    results_dir = PROJECT_ROOT / "artifacts" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_dir / f"{run_prefix}_seed-{args.seed}.json"
    results_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Saved: {results_path}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train and compare GRU baselines with the RIMs-style core"
    )
    parser.add_argument(
        "--experiment",
        choices=("002", "003", "004", "005"),
        default="002",
        help="002 random split; 003 one held-out split; 004 corrected selection; 005 balanced selection",
    )
    parser.add_argument(
        "--fold", type=int, choices=tuple(range(FOLD_COUNT)), default=0,
        help="held-out composition fold for experiment 004 or 005",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--train-size", type=int, default=4096)
    parser.add_argument("--validation-size", type=int, default=512)
    parser.add_argument("--test-size", type=int, default=1024)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--epoch-metrics",
        action="store_true",
        help="record train and validation metrics after every epoch in a separate result",
    )
    args = parser.parse_args()
    if args.epochs <= 0 or args.batch_size <= 0:
        parser.error("--epochs and --batch-size must be positive")
    if args.experiment not in ("004", "005") and args.fold != 0:
        parser.error("--fold can only be set for experiment 004 or 005")
    if args.epoch_metrics and args.experiment not in ("004", "005"):
        parser.error("--epoch-metrics is only supported for experiment 004 or 005")
    run(args)


if __name__ == "__main__":
    main()

