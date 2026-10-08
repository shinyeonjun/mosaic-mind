"""Train and evaluate active-query policies in the Experiment 007 world."""

import argparse
import json
import math
from pathlib import Path
import random
import time

import torch
from torch import Tensor, nn
from torch.distributions import Categorical

from cognitive_lab.active.environment import ALL_PROBES, TaskCase, generate_case
from cognitive_lab.active.exp007.world import (
    FOLD_COUNT,
    FOLD_TEST_RULES,
    PROBE_OUTCOMES,
    REDUCERS,
    RULES,
    SCOPES,
    apply_case,
    choose_answer,
    select_probe,
    update_posterior,
    validate_world,
)
from cognitive_lab.core.gru import GRUCore
from cognitive_lab.core.rims import RIMsCore
from cognitive_lab.task import ACTION_COUNT
from cognitive_lab.train import choose_device

PROJECT_ROOT = Path(__file__).resolve().parents[4]
MAX_PROBE_BUDGET = 3
STOP_ACTION = len(ALL_PROBES)
POLICY_ACTION_COUNT = STOP_ACTION + 1
STATE_SIZE = len(RULES) + len(ALL_PROBES) + ACTION_COUNT + 2
PROBE_COST = 0.05
CONTROLLERS = ("gru_256", "gru_parameter_matched", "rims")
T_CRITICAL_95_DF5 = 2.571


def _build_controller(name: str) -> nn.Module:
    if name == "gru_256":
        return GRUCore(STATE_SIZE, POLICY_ACTION_COUNT, hidden_size=256)
    if name == "gru_parameter_matched":
        return GRUCore(STATE_SIZE, POLICY_ACTION_COUNT, hidden_size=206)
    if name == "rims":
        return RIMsCore(
            STATE_SIZE, POLICY_ACTION_COUNT, module_count=4, module_size=64, top_k=1
        )
    raise ValueError(f"unknown controller: {name!r}")


def _encode_state(
    posterior: tuple[int, ...],
    previous_probe: int | None,
    previous_outcome: int | None,
    remaining_budget: int,
    budget_limit: int,
) -> list[float]:
    state = [float(rule_id in posterior) for rule_id in range(len(RULES))]
    state.extend(float(probe_id == previous_probe) for probe_id in range(len(ALL_PROBES)))
    state.extend(float(outcome == previous_outcome) for outcome in range(ACTION_COUNT))
    state.extend((remaining_budget / MAX_PROBE_BUDGET, budget_limit / MAX_PROBE_BUDGET))
    return state


def _balanced_rule_schedule(
    episodes: int, rule_ids: tuple[int, ...], rng: random.Random
) -> list[int]:
    schedule = []
    while len(schedule) < episodes:
        batch = list(rule_ids)
        rng.shuffle(batch)
        schedule.extend(batch[: episodes - len(schedule)])
    return schedule


def generate_trials(
    episodes: int, seed: int, rule_ids: tuple[int, ...]
) -> list[tuple[int, TaskCase]]:
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    if not rule_ids or len(set(rule_ids)) != len(rule_ids) or any(
        type(rule_id) is not int or rule_id not in range(len(RULES))
        for rule_id in rule_ids
    ):
        raise ValueError("rule_ids must contain unique Experiment 007 rules")
    rng = random.Random(seed)
    schedule = _balanced_rule_schedule(episodes, rule_ids, rng)
    return [(rule_id, generate_case(rng)) for rule_id in schedule]


def _summarize(records: list[dict], probe_cost: float) -> dict:
    count = len(records)
    correct = sum(row["correct"] for row in records)
    probes = sum(row["probes"] for row in records)
    per_rule = {}
    for rule_id, rule in enumerate(RULES):
        group = [row for row in records if row["rule_id"] == rule_id]
        if not group:
            continue
        per_rule[rule.name] = {
            "episodes": len(group),
            "correct_episodes": sum(row["correct"] for row in group),
            "accuracy": round(sum(row["correct"] for row in group) / len(group), 4),
        }
    accuracy = correct / count
    mean_probes = probes / count
    return {
        "episodes": count,
        "correct_episodes": correct,
        "total_probes": probes,
        "identified_episodes": sum(row["rule_identified"] for row in records),
        "accuracy": round(accuracy, 4),
        "mean_probes": round(mean_probes, 4),
        "mean_task_reward": round(accuracy - probe_cost * mean_probes, 4),
        "rule_identification_rate": round(
            sum(row["rule_identified"] for row in records) / count, 4
        ),
        "mean_remaining_rules": round(
            sum(row["remaining_rules"] for row in records) / count, 4
        ),
        "per_rule_accuracy": per_rule,
    }


def _rollout_baseline(
    strategy: str,
    budget: int,
    trials: list[tuple[int, TaskCase]],
    seed: int,
) -> list[dict]:
    records = []
    for index, (hidden_rule, case) in enumerate(trials):
        posterior = tuple(range(len(RULES)))
        asked: set[int] = set()
        rng = random.Random(seed + index * 1_000_003)
        for _ in range(budget):
            if len(posterior) == 1:
                break
            probe_id = select_probe(strategy, posterior, asked, rng)
            if probe_id is None:
                break
            posterior = update_posterior(
                posterior, probe_id, PROBE_OUTCOMES[probe_id][hidden_rule]
            )
            asked.add(probe_id)
        predicted = choose_answer(case, posterior)
        expected = apply_case(RULES[hidden_rule], case)
        records.append({
            "rule_id": hidden_rule,
            "correct": predicted == expected,
            "probes": len(asked),
            "remaining_rules": len(posterior),
            "rule_identified": len(posterior) == 1,
        })
    return records


def _evaluate_baselines(
    trials: list[tuple[int, TaskCase]], seed: int, probe_cost: float
) -> dict:
    output = {
        "no_probe": _summarize(_rollout_baseline("no_probe", 0, trials, seed), probe_cost)
    }
    for strategy in ("random", "fixed_order", "information_gain"):
        output[strategy] = {
            "by_budget": {
                str(budget): _summarize(
                    _rollout_baseline(strategy, budget, trials, seed), probe_cost
                )
                for budget in range(1, MAX_PROBE_BUDGET + 1)
            }
        }
    return output


def _rollout_model(
    model: nn.Module,
    trials: list[tuple[int, TaskCase]],
    budget: int,
    device: torch.device,
) -> list[dict]:
    all_rules = tuple(range(len(RULES)))
    records = [
        {
            "rule_id": rule_id,
            "case": case,
            "posterior": all_rules,
            "asked": set(),
            "history": [_encode_state(all_rules, None, None, budget, budget)],
            "stopped": False,
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
            if logits.shape[-1] != POLICY_ACTION_COUNT:
                raise ValueError(f"expected {POLICY_ACTION_COUNT} actions, got {logits.shape[-1]}")
            for row_index, record_index in enumerate(active):
                asked = records[record_index]["asked"]
                if asked:
                    logits[row_index, list(asked)] = -torch.inf
            actions = logits.argmax(dim=-1).tolist()
            for row_index, record_index in enumerate(active):
                record = records[record_index]
                action = actions[row_index]
                if action == STOP_ACTION:
                    record["stopped"] = True
                    continue
                observed = PROBE_OUTCOMES[action][record["rule_id"]]
                record["posterior"] = update_posterior(
                    record["posterior"], action, observed
                )
                record["asked"].add(action)
                record["history"].append(_encode_state(
                    record["posterior"], action, observed,
                    budget - len(record["asked"]), budget,
                ))

    output = []
    for row in records:
        expected = apply_case(RULES[row["rule_id"]], row["case"])
        predicted = choose_answer(row["case"], row["posterior"])
        output.append({
            "rule_id": row["rule_id"],
            "correct": predicted == expected,
            "probes": len(row["asked"]),
            "remaining_rules": len(row["posterior"]),
            "rule_identified": len(row["posterior"]) == 1,
            "policy_stopped": row["stopped"],
        })
    return output


def _evaluate_model(
    model: nn.Module,
    trials: list[tuple[int, TaskCase]],
    device: torch.device,
    probe_cost: float,
) -> dict:
    return {
        str(budget): _summarize(
            _rollout_model(model, trials, budget, device), probe_cost
        )
        for budget in range(1, MAX_PROBE_BUDGET + 1)
    }


def _rollout_batch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_rule_ids: tuple[int, ...],
    batch_size: int,
    seed: int,
    device: torch.device,
    outcome_table: Tensor,
    probe_cost: float,
) -> dict[str, float]:
    rng = random.Random(seed)
    hidden_rules = _balanced_rule_schedule(batch_size, train_rule_ids, rng)
    budgets = [rng.randint(1, MAX_PROBE_BUDGET) for _ in range(batch_size)]
    hidden = torch.tensor(hidden_rules, dtype=torch.long, device=device)
    budget_limits = torch.tensor(budgets, dtype=torch.long, device=device)
    posterior = torch.ones((batch_size, len(RULES)), dtype=torch.bool, device=device)
    asked = torch.zeros((batch_size, len(ALL_PROBES)), dtype=torch.bool, device=device)
    stopped = torch.zeros(batch_size, dtype=torch.bool, device=device)
    histories = torch.zeros(
        batch_size, MAX_PROBE_BUDGET + 1, STATE_SIZE,
        dtype=torch.float32, device=device,
    )
    normalized_budget = budget_limits.float() / MAX_PROBE_BUDGET
    histories[:, 0, :len(RULES)] = 1.0
    histories[:, 0, -2] = normalized_budget
    histories[:, 0, -1] = normalized_budget
    log_prob_terms: list[tuple[Tensor, Tensor]] = []

    for step in range(MAX_PROBE_BUDGET):
        active = (
            (budget_limits > step) & (posterior.sum(dim=1) > 1) & ~stopped
        ).nonzero().flatten()
        if not len(active):
            break
        logits = model(histories[active, :step + 1])
        probe_logits = logits[:, :STOP_ACTION].masked_fill(asked[active], -torch.inf)
        distribution = Categorical(logits=torch.cat((probe_logits, logits[:, STOP_ACTION:]), dim=1))
        actions = distribution.sample()
        log_prob_terms.append((active, distribution.log_prob(actions)))
        stop_selected = actions == STOP_ACTION
        stopped[active[stop_selected]] = True
        query_rows = active[~stop_selected]
        query_actions = actions[~stop_selected]
        if len(query_rows):
            possible_outcomes = outcome_table[query_actions]
            observed = possible_outcomes.gather(1, hidden[query_rows, None]).squeeze(1)
            posterior[query_rows] &= possible_outcomes.eq(observed[:, None])
            asked[query_rows, query_actions] = True
            states = torch.zeros(len(query_rows), STATE_SIZE, dtype=torch.float32, device=device)
            states[:, :len(RULES)] = posterior[query_rows].float()
            probe_start = len(RULES)
            outcome_start = probe_start + len(ALL_PROBES)
            row_ids = torch.arange(len(query_rows), device=device)
            states[row_ids, probe_start + query_actions] = 1.0
            states[row_ids, outcome_start + observed] = 1.0
            states[:, -2] = (budget_limits[query_rows] - step - 1).float() / MAX_PROBE_BUDGET
            states[:, -1] = budget_limits[query_rows].float() / MAX_PROBE_BUDGET
            histories[query_rows, step + 1] = states

    probe_counts = asked.sum(dim=1).tolist()
    candidate_masks = posterior.tolist()
    case_rng = random.Random(seed ^ 0x5DEECE66D)
    rewards = []
    correct_count = 0
    for index, rule_id in enumerate(hidden_rules):
        case = generate_case(case_rng)
        expected = apply_case(RULES[rule_id], case)
        candidates = tuple(
            candidate for candidate, remains in enumerate(candidate_masks[index]) if remains
        )
        correct = choose_answer(case, candidates) == expected
        correct_count += int(correct)
        rewards.append(float(correct) - probe_cost * probe_counts[index])

    returns = torch.tensor(rewards, dtype=torch.float32, device=device)
    advantages = torch.zeros_like(returns)
    for budget in range(1, MAX_PROBE_BUDGET + 1):
        indices = (budget_limits == budget).nonzero().flatten()
        group = returns[indices]
        if len(group) > 1:
            advantages[indices] = group - (group.sum() - group) / (len(group) - 1)
        elif len(returns) > 1:
            advantages[indices] = group - (returns.sum() - group) / (len(returns) - 1)
    advantages /= returns.std(unbiased=False).clamp_min(0.05)
    loss = torch.zeros((), dtype=torch.float32, device=device)
    for indices, log_probs in log_prob_terms:
        loss -= (log_probs * advantages[indices]).sum() / batch_size
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    return {
        "mean_task_reward": sum(rewards) / batch_size,
        "accuracy": correct_count / batch_size,
        "mean_probes": sum(probe_counts) / batch_size,
        "policy_stop_rate": stopped.float().mean().item(),
    }


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
    outcome_table = torch.tensor(PROBE_OUTCOMES, dtype=torch.long, device=device)
    best_score = -float("inf")
    best_epoch = 0
    best_state = None
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
        for current_size in batch_sizes:
            _rollout_batch(
                model, optimizer, train_rule_ids, current_size,
                seed + epoch * 1_000_003 + start, device,
                outcome_table, probe_cost,
            )
            start += current_size
        validation = _evaluate_model(model, validation_trials, device, probe_cost)
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
        raise RuntimeError("training produced no checkpoint")
    model.load_state_dict(best_state)
    return {
        "best_epoch": best_epoch,
        "best_validation_mean_task_reward": round(best_score, 4),
        "training_seconds": round(time.perf_counter() - started, 3),
        "validation_by_budget": _evaluate_model(model, validation_trials, device, probe_cost),
    }


def run(args: argparse.Namespace) -> dict:
    validate_world()
    if args.fold not in range(FOLD_COUNT):
        raise ValueError(f"fold must be 0..{FOLD_COUNT - 1}")
    train_rule_ids = tuple(
        rule_id for rule_id in range(len(RULES))
        if rule_id not in FOLD_TEST_RULES[args.fold]
    )
    test_rule_ids = FOLD_TEST_RULES[args.fold]
    validation_trials = generate_trials(args.validation_size, args.seed + 1, train_rule_ids)
    test_trials = generate_trials(args.test_size, args.seed + 2, test_rule_ids)
    device = choose_device(args.device)
    probe_cost = args.probe_cost
    result = {
        "experiment": "007-compositional-rule-generalization",
        "seed": args.seed,
        "fold": args.fold,
        "device": str(device),
        "torch_version": torch.__version__,
        "data": {
            "train_episodes_per_epoch": args.train_size,
            "total_training_episodes_per_controller": args.train_size * args.epochs,
            "validation_episodes": args.validation_size,
            "test_episodes": args.test_size,
            "candidate_rule_count": len(RULES),
            "train_rule_ids": list(train_rule_ids),
            "train_rules": [RULES[index].name for index in train_rule_ids],
            "test_rule_ids": list(test_rule_ids),
            "test_rules": [RULES[index].name for index in test_rule_ids],
            "fold_assignment": "(scope_index + reducer_index) mod 3",
            "test_case_hidden_until_probing_finishes": True,
            "probe_budget_range": [1, MAX_PROBE_BUDGET],
            "probe_candidates": len(ALL_PROBES),
        },
        "training": {
            "method": "episodic REINFORCE with leave-one-out reward baseline",
            "teacher": "none",
            "reward": "1 for a correct answer, minus probe_cost per question",
            "probe_cost": probe_cost,
            "belief_update": "exact filtering over all 18 grammar-generated candidate rules",
            "optimizer": "AdamW",
            "learning_rate": args.learning_rate,
            "max_epochs": args.epochs,
            "batch_size": args.batch_size,
            "checkpoint_selection_metric": "mean validation task reward across budgets 1–3",
        },
        "architecture": {
            "state_size": STATE_SIZE,
            "action_count": POLICY_ACTION_COUNT,
            "controllers": {},
        },
        "baselines": _evaluate_baselines(test_trials, args.seed + 2, probe_cost),
        "controllers": {},
        "world_validation": "all rule signatures distinct across all 216 probes; every fold retains every scope and reducer in training",
        "interpretation_limit": "The test rules are held out from policy training but remain in the fixed 18-rule candidate catalog and exact belief update.",
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
            model, train_rule_ids, validation_trials, device,
            args.train_size, args.batch_size, args.epochs,
            args.seed, args.learning_rate, probe_cost,
        )
        checkpoint = checkpoint_dir / (
            f"experiment-007_fold-{args.fold}_{name}_seed-{args.seed}.pt"
        )
        torch.save(model.state_dict(), checkpoint)
        result["architecture"]["controllers"][name] = {
            "parameters": parameter_count,
            "parameter_match_difference_vs_rims": None,
        }
        result["controllers"][name] = {
            **training,
            "test_by_budget": _evaluate_model(model, test_trials, device, probe_cost),
            "checkpoint": str(checkpoint.relative_to(PROJECT_ROOT)),
        }
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    rims_parameters = result["architecture"]["controllers"]["rims"]["parameters"]
    for name in CONTROLLERS:
        result["architecture"]["controllers"][name]["parameter_match_difference_vs_rims"] = round(
            (result["architecture"]["controllers"][name]["parameters"] - rims_parameters)
            / rims_parameters,
            4,
        )

    result_dir = PROJECT_ROOT / "artifacts" / "results"
    result_dir.mkdir(parents=True, exist_ok=True)
    result_path = result_dir / f"experiment-007_fold-{args.fold}_seed-{args.seed}.json"
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    def brief(metrics: dict) -> dict:
        return {
            key: metrics[key]
            for key in ("accuracy", "mean_probes", "mean_task_reward", "rule_identification_rate")
        }

    baseline_brief = {
        name: (
            brief(metrics)
            if name == "no_probe"
            else {budget: brief(values) for budget, values in metrics["by_budget"].items()}
        )
        for name, metrics in result["baselines"].items()
    }
    print(json.dumps({
        "fold": args.fold,
        "seed": args.seed,
        "held_out_rules": result["data"]["test_rules"],
        "baseline_test_by_budget": baseline_brief,
        "controller_test_by_budget": {
            name: {
                budget: brief(metrics)
                for budget, metrics in value["test_by_budget"].items()
            }
            for name, value in result["controllers"].items()
        },
        "parameter_counts": result["architecture"]["controllers"],
        "saved": str(result_path),
    }, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train active-query policies on compositional rule generalization"
    )
    parser.add_argument("--fold", type=int, choices=range(FOLD_COUNT), default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--train-size", type=int, default=4096)
    parser.add_argument("--validation-size", type=int, default=512)
    parser.add_argument("--test-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--probe-cost", type=float, default=PROBE_COST)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if min(args.epochs, args.train_size, args.validation_size, args.test_size) <= 0:
        parser.error("epochs and all episode counts must be positive")
    if args.batch_size < 2 or args.train_size < 2:
        parser.error("train size and batch size must be at least 2")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("learning rate must be positive")
    if not math.isfinite(args.probe_cost) or args.probe_cost < 0:
        parser.error("probe cost must be a finite non-negative number")
    run(args)


if __name__ == "__main__":
    main()
