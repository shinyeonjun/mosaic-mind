"""Non-neural policies and paired evaluation for active probing."""

from collections import Counter
import math
import random

from cognitive_lab.task import ABSTAIN
from cognitive_lab.active.environment import (
    ALL_PROBES,
    PROBE_OUTCOMES,
    RULES,
    RuleEnvironment,
    TaskCase,
    apply_case,
    generate_case,
)

STRATEGIES = ("random", "fixed_order", "information_gain")


def update_posterior(
    posterior: tuple[int, ...], probe_id: int, observed: int
) -> tuple[int, ...]:
    updated = tuple(
        rule_id
        for rule_id in posterior
        if PROBE_OUTCOMES[probe_id][rule_id] == observed
    )
    if not updated:
        raise RuntimeError("probe feedback contradicts every supported rule")
    return updated


def select_probe(
    strategy: str,
    posterior: tuple[int, ...],
    asked: set[int],
    rng: random.Random,
) -> int | None:
    available = [probe_id for probe_id in range(len(ALL_PROBES)) if probe_id not in asked]
    if not available or strategy == "no_probe":
        return None
    if strategy == "random":
        return rng.choice(available)
    if strategy == "fixed_order":
        return available[0]
    if strategy != "information_gain":
        raise ValueError(f"unknown strategy: {strategy!r}")

    before = math.log2(len(posterior))
    best_probe = None
    best_gain = 0.0
    for probe_id in available:
        outcomes = Counter(PROBE_OUTCOMES[probe_id][rule_id] for rule_id in posterior)
        expected_entropy = sum(
            (count / len(posterior)) * math.log2(count)
            for count in outcomes.values()
        )
        gain = before - expected_entropy
        if gain > best_gain + 1e-12:
            best_gain = gain
            best_probe = probe_id
    return best_probe


def choose_answer(case: TaskCase, posterior: tuple[int, ...]) -> int:
    predictions = Counter(apply_case(RULES[rule_id], case) for rule_id in posterior)
    highest_count = max(predictions.values())
    return min(action for action, count in predictions.items() if count == highest_count)


def _run_trial(
    strategy: str,
    budget: int,
    rule_id: int,
    case: TaskCase,
    rng: random.Random,
) -> dict:
    environment = RuleEnvironment(RULES[rule_id])
    posterior = tuple(range(len(RULES)))
    asked: set[int] = set()
    for _ in range(budget):
        if len(posterior) == 1:
            break
        probe_id = select_probe(strategy, posterior, asked, rng)
        if probe_id is None:
            break
        asked.add(probe_id)
        posterior = update_posterior(
            posterior, probe_id, environment.query(ALL_PROBES[probe_id])
        )

    expected = environment.answer(case)
    predicted = choose_answer(case, posterior)
    return {
        "rule_id": rule_id,
        "correct": predicted == expected,
        "expected": expected,
        "predicted": predicted,
        "probes": len(asked),
        "remaining_rules": len(posterior),
        "rule_identified": len(posterior) == 1,
    }


def summarize_records(records: list[dict]) -> dict:
    total = len(records)
    correct = sum(record["correct"] for record in records)
    abstain_records = [record for record in records if record["expected"] == ABSTAIN]
    per_rule = {}
    for rule_id, rule in enumerate(RULES):
        subset = [record for record in records if record["rule_id"] == rule_id]
        per_rule[rule.name] = {
            "episodes": len(subset),
            "accuracy": round(sum(record["correct"] for record in subset) / len(subset), 4)
            if subset else None,
        }
    return {
        "episodes": total,
        "accuracy": round(correct / total, 4) if total else None,
        "abstain_case_accuracy": round(
            sum(record["correct"] for record in abstain_records) / len(abstain_records), 4
        ) if abstain_records else None,
        "abstain_cases": len(abstain_records),
        "mean_probes": round(sum(record["probes"] for record in records) / total, 4)
        if total else None,
        "rule_identification_rate": round(
            sum(record["rule_identified"] for record in records) / total, 4
        ) if total else None,
        "mean_remaining_rules": round(
            sum(record["remaining_rules"] for record in records) / total, 4
        ) if total else None,
        "per_rule": per_rule,
    }


def generate_trials(
    episodes: int, seed: int, rule_ids: tuple[int, ...] | None = None
) -> list[tuple[int, TaskCase]]:
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    selected_rules = tuple(range(len(RULES))) if rule_ids is None else tuple(rule_ids)
    if (
        not selected_rules
        or len(set(selected_rules)) != len(selected_rules)
        or any(type(rule_id) is not int or rule_id not in range(len(RULES)) for rule_id in selected_rules)
    ):
        raise ValueError("rule_ids must contain unique supported rule indices")

    data_rng = random.Random(seed)
    trials = []
    while len(trials) < episodes:
        batch_rule_ids = list(selected_rules)
        data_rng.shuffle(batch_rule_ids)
        for rule_id in batch_rule_ids:
            if len(trials) == episodes:
                break
            trials.append((rule_id, generate_case(data_rng)))
    return trials


def evaluate_baselines(
    episodes: int,
    max_probe_budget: int,
    seed: int,
    rule_ids: tuple[int, ...] | None = None,
    trials: list[tuple[int, TaskCase]] | None = None,
) -> dict:
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    if max_probe_budget < 0:
        raise ValueError("max_probe_budget must be non-negative")
    if trials is None:
        trials = generate_trials(episodes, seed, rule_ids)
    elif len(trials) != episodes:
        raise ValueError("episodes must match the supplied trial count")
    if any(
        type(rule_id) is not int or rule_id not in range(len(RULES))
        or not isinstance(case, TaskCase)
        for rule_id, case in trials
    ):
        raise ValueError("trials must contain supported rule indices and TaskCase values")
    sampled_rule_ids = sorted({rule_id for rule_id, _ in trials})
    if rule_ids is not None and not set(sampled_rule_ids) <= set(rule_ids):
        raise ValueError("trials contain rules outside rule_ids")
    rule_counts = Counter(rule_id for rule_id, _ in trials)
    balanced = max(rule_counts.values()) - min(rule_counts.values()) <= 1

    no_probe = [
        _run_trial("no_probe", 0, rule_id, case, random.Random(seed + index))
        for index, (rule_id, case) in enumerate(trials)
    ]
    results = {"no_probe": summarize_records(no_probe)}
    for strategy_index, strategy in enumerate(STRATEGIES):
        by_budget = {}
        for budget in range(max_probe_budget + 1):
            records = [
                _run_trial(
                    strategy,
                    budget,
                    rule_id,
                    case,
                    random.Random(seed + (index + 1) * 1_000_003 + strategy_index),
                )
                for index, (rule_id, case) in enumerate(trials)
            ]
            by_budget[str(budget)] = summarize_records(records)
        results[strategy] = {"by_budget": by_budget}
    return {
        "experiment": "006-active-probing",
        "seed": seed,
        "data": {
            "episodes": episodes,
            "rule_sampling": (
                "balanced across sampled rule IDs" if balanced else "provided trial corpus"
            ),
            "sampled_rule_ids": sampled_rule_ids,
            "rule_count": len(RULES),
            "evaluated_rule_count": len(sampled_rule_ids),
            "probe_length": 3,
            "probe_candidates": len(ALL_PROBES),
            "task_event_range": [4, 8],
            "max_probe_budget": max_probe_budget,
        },
        "baselines": results,
    }
