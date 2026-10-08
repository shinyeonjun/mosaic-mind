"""Train active-probing controllers from task reward, without teacher actions."""

import argparse
import json
import math
from pathlib import Path
import random
import time

import torch
from torch import Tensor, nn
from torch.distributions import Categorical

from cognitive_lab.active.baselines import (
    choose_answer,
    evaluate_baselines,
    generate_trials,
)
from cognitive_lab.active.controller import (
    FOLD_TEST_RULES,
    MAX_PROBE_BUDGET,
    STATE_SIZE,
    evaluate_controller,
)
from cognitive_lab.active.environment import (
    ALL_PROBES,
    PROBE_OUTCOMES,
    RULES,
    RuleEnvironment,
    TaskCase,
    generate_case,
)
from cognitive_lab.core.gru import GRUCore
from cognitive_lab.core.rims import RIMsCore
from cognitive_lab.train import choose_device

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONTROLLERS = ("gru_256", "gru_parameter_matched", "rims")
STOP_ACTION = len(ALL_PROBES)
POLICY_ACTION_COUNT = STOP_ACTION + 1


def _build_controller(name: str) -> nn.Module:
    if name == "gru_256":
        return GRUCore(STATE_SIZE, POLICY_ACTION_COUNT, hidden_size=256)
    if name == "gru_parameter_matched":
        return GRUCore(STATE_SIZE, POLICY_ACTION_COUNT, hidden_size=205)
    if name == "rims":
        return RIMsCore(
            STATE_SIZE, POLICY_ACTION_COUNT, module_count=4, module_size=64, top_k=1
        )
    raise ValueError(f"unknown controller: {name!r}")


def _balanced_rule_schedule(
    episodes: int, rule_ids: tuple[int, ...], rng: random.Random
) -> list[int]:
    schedule = []
    while len(schedule) < episodes:
        batch = list(rule_ids)
        rng.shuffle(batch)
        schedule.extend(batch[: episodes - len(schedule)])
    return schedule


def _rollout_batch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    rule_ids: tuple[int, ...],
    batch_size: int,
    seed: int,
    device: torch.device,
    outcome_table: Tensor,
    probe_cost: float,
) -> dict[str, float]:
    """Sample complete episodes, then apply a terminal-reward policy gradient."""
    rng = random.Random(seed)
    sampled_rules = _balanced_rule_schedule(batch_size, rule_ids, rng)
    budgets = [rng.randint(1, MAX_PROBE_BUDGET) for _ in range(batch_size)]
    hidden_rules = torch.tensor(sampled_rules, dtype=torch.long, device=device)
    budget_limits = torch.tensor(budgets, dtype=torch.long, device=device)
    posterior = torch.ones((batch_size, len(RULES)), dtype=torch.bool, device=device)
    asked = torch.zeros((batch_size, len(ALL_PROBES)), dtype=torch.bool, device=device)
    stopped = torch.zeros(batch_size, dtype=torch.bool, device=device)
    histories = torch.zeros(
        (batch_size, MAX_PROBE_BUDGET + 1, STATE_SIZE),
        dtype=torch.float32,
        device=device,
    )
    histories[:, 0, : len(RULES)] = 1.0
    normalized_budgets = budget_limits.float() / MAX_PROBE_BUDGET
    histories[:, 0, -2] = normalized_budgets
    histories[:, 0, -1] = normalized_budgets
    log_prob_terms: list[tuple[Tensor, Tensor]] = []
    for step in range(MAX_PROBE_BUDGET):
        active = (
            (budget_limits > step) & (posterior.sum(dim=1) > 1) & ~stopped
        ).nonzero().flatten()
        if not len(active):
            break

        logits = model(histories[active, : step + 1])
        probe_logits = logits[:, :STOP_ACTION].masked_fill(asked[active], -torch.inf)
        logits = torch.cat((probe_logits, logits[:, STOP_ACTION:]), dim=1)
        distribution = Categorical(logits=logits)
        actions = distribution.sample()
        log_prob_terms.append((active, distribution.log_prob(actions)))

        stop_selected = actions == STOP_ACTION
        stopped[active[stop_selected]] = True
        query_rows = active[~stop_selected]
        query_actions = actions[~stop_selected]
        if len(query_rows):
            possible_outcomes = outcome_table[query_actions]
            observed = possible_outcomes.gather(
                1, hidden_rules[query_rows, None]
            ).squeeze(1)
            posterior[query_rows] = posterior[query_rows] & possible_outcomes.eq(
                observed[:, None]
            )
            asked[query_rows, query_actions] = True

            next_states = torch.zeros(
                (len(query_rows), STATE_SIZE), dtype=torch.float32, device=device
            )
            next_states[:, : len(RULES)] = posterior[query_rows].float()
            probe_start = len(RULES)
            outcome_start = probe_start + len(ALL_PROBES)
            row_ids = torch.arange(len(query_rows), device=device)
            next_states[row_ids, probe_start + query_actions] = 1.0
            next_states[row_ids, outcome_start + observed] = 1.0
            next_states[:, -2] = (
                budget_limits[query_rows] - step - 1
            ).float() / MAX_PROBE_BUDGET
            next_states[:, -1] = budget_limits[query_rows].float() / MAX_PROBE_BUDGET
            histories[query_rows, step + 1] = next_states

    probe_counts = asked.sum(dim=1).tolist()
    remaining_rules = posterior.tolist()
    rewards = []
    correct_count = 0
    case_rng = random.Random(seed ^ 0x5DEECE66D)
    for index, rule_id in enumerate(sampled_rules):
        # The final case is generated only after the policy has finished probing.
        case = generate_case(case_rng)
        expected = RuleEnvironment(RULES[rule_id]).answer(case)
        surviving_rules = tuple(
            rule_index
            for rule_index, survives in enumerate(remaining_rules[index])
            if survives
        )
        correct = choose_answer(case, surviving_rules) == expected
        correct_count += int(correct)
        rewards.append(float(correct) - probe_cost * probe_counts[index])

    returns = torch.tensor(rewards, dtype=torch.float32, device=device)
    advantages = torch.zeros_like(returns)
    for budget in range(1, MAX_PROBE_BUDGET + 1):
        indices = (budget_limits == budget).nonzero().flatten()
        group = returns[indices]
        if len(group) > 1:
            # Leave-one-out baselines are action-independent for each episode.
            advantages[indices] = group - (group.sum() - group) / (len(group) - 1)
        elif len(returns) > 1:
            advantages[indices] = group - (returns.sum() - group) / (len(returns) - 1)

    advantages /= returns.std(unbiased=False).clamp_min(0.05)
    policy_loss = torch.zeros((), dtype=torch.float32, device=device)
    for indices, log_probs in log_prob_terms:
        policy_loss -= (log_probs * advantages[indices]).sum() / batch_size

    optimizer.zero_grad(set_to_none=True)
    policy_loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    return {
        "mean_task_reward": sum(rewards) / batch_size,
        "accuracy": correct_count / batch_size,
        "mean_probes": sum(probe_counts) / batch_size,
        "policy_stop_rate": stopped.float().mean().item(),
    }


def _add_task_reward(metrics: dict, probe_cost: float) -> dict:
    return {
        **metrics,
        "mean_task_reward": round(
            metrics["accuracy"] - probe_cost * metrics["mean_probes"], 4
        ),
    }


def _evaluate(
    model: nn.Module,
    trials: list[tuple[int, TaskCase]],
    device: torch.device,
    probe_cost: float,
) -> dict[str, dict]:
    return {
        budget: _add_task_reward(metrics, probe_cost)
        for budget, metrics in evaluate_controller(
            model, trials, MAX_PROBE_BUDGET, device, stop_action=STOP_ACTION
        ).items()
    }


def _evaluate_baselines(
    trials: list[tuple[int, TaskCase]], seed: int, probe_cost: float
) -> dict:
    results = evaluate_baselines(
        len(trials), MAX_PROBE_BUDGET, seed, trials=trials
    )
    for name, value in results["baselines"].items():
        if name == "no_probe":
            results["baselines"][name] = _add_task_reward(value, probe_cost)
        else:
            value["by_budget"] = {
                budget: _add_task_reward(metrics, probe_cost)
                for budget, metrics in value["by_budget"].items()
            }
    return results


def _train_controller(
    model: nn.Module,
    train_rule_ids: tuple[int, ...],
    validation_trials: list[tuple[int, TaskCase]],
    device: torch.device,
    episodes_per_epoch: int,
    batch_size: int,
    epochs: int,
    seed: int,
    learning_rate: float,
    probe_cost: float,
) -> dict:
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    outcome_table = torch.tensor(PROBE_OUTCOMES, dtype=torch.long, device=device)
    best_score = -float("inf")
    best_epoch = 0
    best_state = None
    last_epoch_metrics = {}
    started = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        totals = {
            key: 0.0
            for key in ("mean_task_reward", "accuracy", "mean_probes", "policy_stop_rate")
        }
        batch_sizes = [
            min(batch_size, episodes_per_epoch - start)
            for start in range(0, episodes_per_epoch, batch_size)
        ]
        if len(batch_sizes) > 1 and batch_sizes[-1] == 1:
            batch_sizes[-2] += 1
            batch_sizes.pop()
        start = 0
        for current_batch_size in batch_sizes:
            batch_metrics = _rollout_batch(
                model,
                optimizer,
                train_rule_ids,
                current_batch_size,
                seed + epoch * 1_000_003 + start,
                device,
                outcome_table,
                probe_cost,
            )
            for key in totals:
                totals[key] += batch_metrics[key] * current_batch_size
            start += current_batch_size
        last_epoch_metrics = {
            key: round(value / episodes_per_epoch, 4) for key, value in totals.items()
        }

        validation_by_budget = _evaluate(model, validation_trials, device, probe_cost)
        validation_score = sum(
            validation_by_budget[str(budget)]["mean_task_reward"]
            for budget in range(1, MAX_PROBE_BUDGET + 1)
        ) / MAX_PROBE_BUDGET
        if validation_score > best_score:
            best_score = validation_score
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }

    if best_state is None:
        raise RuntimeError("training produced no controller checkpoint")
    model.load_state_dict(best_state)
    validation_by_budget = _evaluate(model, validation_trials, device, probe_cost)
    return {
        "best_epoch": best_epoch,
        "best_validation_mean_task_reward": round(best_score, 4),
        "last_epoch_training": last_epoch_metrics,
        "validation_by_budget": validation_by_budget,
        "training_seconds": round(time.perf_counter() - started, 3),
    }


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
    validation_trials = generate_trials(
        args.validation_size, args.seed + 1, train_rule_ids
    )
    test_trials = generate_trials(args.test_size, args.seed + 2, test_rule_ids)
    result = {
        "experiment": "006B-reward-only-active-controller",
        "seed": args.seed,
        "fold": args.fold,
        "device": str(device),
        "torch_version": torch.__version__,
        "data": {
            "train_episodes_per_epoch": args.train_size,
            "total_training_episodes_per_controller": args.train_size * args.epochs,
            "validation_episodes": args.validation_size,
            "test_episodes": args.test_size,
            "train_rule_ids": list(train_rule_ids),
            "test_rule_ids": list(test_rule_ids),
            "hypothesis_catalog": "all six candidate rules; held-out rule cases are excluded from training",
            "test_cases_hidden_until_probing_finishes": True,
            "probe_budget_range": [1, MAX_PROBE_BUDGET],
        },
        "training": {
            "method": "episodic REINFORCE with leave-one-out reward baseline",
            "teacher": "none",
            "action_labels": "none",
            "reward": "1 for a correct final answer, 0 otherwise, minus probe_cost per question",
            "probe_cost": args.probe_cost,
            "belief_update": "exact filtering over the shared six-rule candidate catalog",
            "training_action_selection": "categorical sampling over unasked probes and STOP",
            "evaluation_action_selection": "greedy argmax over unasked probes and STOP",
            "stopping": "the policy may emit STOP at any step; otherwise the episode ends at its budget",
            "optimizer": "AdamW",
            "learning_rate": args.learning_rate,
            "max_epochs": args.epochs,
            "episodes_per_epoch": args.train_size,
            "batch_size": args.batch_size,
            "checkpoint_selection_metric": "mean validation task reward across budgets 1–3",
        },
        "architecture": {
            "output_size": POLICY_ACTION_COUNT,
            "probe_actions": len(ALL_PROBES),
            "stop_action_id": STOP_ACTION,
            "controllers": {},
        },
        "baselines": _evaluate_baselines(test_trials, args.seed + 2, args.probe_cost),
        "controllers": {},
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
        training_metrics = _train_controller(
            model,
            train_rule_ids,
            validation_trials,
            device,
            args.train_size,
            args.batch_size,
            args.epochs,
            args.seed,
            args.learning_rate,
            args.probe_cost,
        )
        checkpoint_path = checkpoint_dir / f"experiment-006B_fold-{args.fold}_{name}_seed-{args.seed}.pt"
        torch.save(model.state_dict(), checkpoint_path)
        result["architecture"]["controllers"][name] = {"parameters": parameter_count}
        result["controllers"][name] = {
            **training_metrics,
            "test_by_budget": _evaluate(model, test_trials, device, args.probe_cost),
            "checkpoint": str(checkpoint_path.relative_to(PROJECT_ROOT)),
        }

    results_dir = PROJECT_ROOT / "artifacts" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_dir / f"experiment-006B_fold-{args.fold}_seed-{args.seed}.json"
    results_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Saved: {results_path}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train active-probing policies from task reward without teacher actions"
    )
    parser.add_argument("--fold", type=int, choices=range(len(FOLD_TEST_RULES)), default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--train-size", type=int, default=4096)
    parser.add_argument("--validation-size", type=int, default=512)
    parser.add_argument("--test-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--probe-cost", type=float, default=0.05)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if min(args.epochs, args.validation_size, args.test_size) <= 0:
        parser.error("epochs and validation/test episode counts must be positive")
    if args.batch_size < 2 or args.train_size < 2:
        parser.error("train size and batch size must be at least 2")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("learning rate must be positive")
    if not math.isfinite(args.probe_cost) or args.probe_cost < 0:
        parser.error("probe cost must be a finite non-negative number")
    run(args)


if __name__ == "__main__":
    main()
