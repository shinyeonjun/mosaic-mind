"""A finite compositional rule world for Experiment 007."""

from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
import math
import random
from typing import Literal

from cognitive_lab.active.environment import ALL_PROBES, TaskCase
from cognitive_lab.task import ABSTAIN

Scope = Literal["all", "target_only", "distractors_only"]
Reducer = Literal["first", "second", "last", "minimum", "maximum", "mode"]
SCOPES: tuple[Scope, ...] = ("all", "target_only", "distractors_only")
REDUCERS: tuple[Reducer, ...] = (
    "first", "second", "last", "minimum", "maximum", "mode"
)
FOLD_COUNT = 3


@dataclass(frozen=True)
class Rule:
    scope: Scope
    reducer: Reducer

    @property
    def name(self) -> str:
        return f"{self.scope}:{self.reducer}"


RULES = tuple(Rule(scope, reducer) for scope in SCOPES for reducer in REDUCERS)
RULE_INDEX = {rule: index for index, rule in enumerate(RULES)}
FOLD_TEST_RULES = tuple(
    tuple(
        rule_id
        for rule_id, rule in enumerate(RULES)
        if (SCOPES.index(rule.scope) + REDUCERS.index(rule.reducer)) % FOLD_COUNT == fold
    )
    for fold in range(FOLD_COUNT)
)


def apply_rule(rule: Rule, events: tuple[tuple[bool, int], ...]) -> int:
    """Filter ordered (is_target, key) events, then apply one reducer."""
    if rule not in RULE_INDEX:
        raise ValueError(f"unsupported rule: {rule!r}")
    if rule.scope == "all":
        keys = [key for _, key in events]
    elif rule.scope == "target_only":
        keys = [key for is_target, key in events if is_target]
    else:
        keys = [key for is_target, key in events if not is_target]

    if not keys:
        return ABSTAIN
    if rule.reducer == "first":
        return keys[0]
    if rule.reducer == "second":
        return keys[1] if len(keys) >= 2 else ABSTAIN
    if rule.reducer == "last":
        return keys[-1]
    if rule.reducer == "minimum":
        return min(keys)
    if rule.reducer == "maximum":
        return max(keys)
    counts = Counter(keys)
    highest_count = max(counts.values())
    return min(key for key, count in counts.items() if count == highest_count)


PROBE_OUTCOMES = tuple(
    tuple(apply_rule(rule, probe) for rule in RULES)
    for probe in ALL_PROBES
)


def apply_case(rule: Rule, case: TaskCase) -> int:
    events = tuple(
        (door == case.target_door, key) for door, key in case.events
    )
    return apply_rule(rule, events)


def update_posterior(
    posterior: tuple[int, ...], probe_id: int, observed: int
) -> tuple[int, ...]:
    updated = tuple(
        rule_id
        for rule_id in posterior
        if PROBE_OUTCOMES[probe_id][rule_id] == observed
    )
    if not updated:
        raise RuntimeError("probe feedback contradicts every candidate rule")
    return updated


@lru_cache(maxsize=None)
def _best_information_gain(
    posterior: tuple[int, ...], asked: tuple[int, ...]
) -> int | None:
    if len(posterior) <= 1:
        return None
    before = math.log2(len(posterior))
    asked_set = set(asked)
    best_probe = None
    best_gain = 0.0
    for probe_id in range(len(ALL_PROBES)):
        if probe_id in asked_set:
            continue
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


def select_probe(
    strategy: str,
    posterior: tuple[int, ...],
    asked: set[int],
    rng: random.Random,
) -> int | None:
    available = [probe_id for probe_id in range(len(ALL_PROBES)) if probe_id not in asked]
    if not available or strategy == "no_probe" or len(posterior) <= 1:
        return None
    if strategy == "random":
        return rng.choice(available)
    if strategy == "fixed_order":
        return available[0]
    if strategy == "information_gain":
        return _best_information_gain(posterior, tuple(sorted(asked)))
    raise ValueError(f"unknown strategy: {strategy!r}")


def choose_answer(case: TaskCase, posterior: tuple[int, ...]) -> int:
    predictions = Counter(apply_case(RULES[rule_id], case) for rule_id in posterior)
    highest_count = max(predictions.values())
    return min(action for action, count in predictions.items() if count == highest_count)


def validate_world() -> None:
    """Reject ambiguous rules or folds that hide an unseen primitive."""
    signatures = {
        tuple(outcomes[rule_id] for outcomes in PROBE_OUTCOMES)
        for rule_id in range(len(RULES))
    }
    if len(signatures) != len(RULES):
        raise RuntimeError("candidate rules are not distinguishable by the probe set")
    if any(len(fold) != len(RULES) // FOLD_COUNT for fold in FOLD_TEST_RULES):
        raise RuntimeError("each fold must hold out six compositions")
    if set.union(*(set(fold) for fold in FOLD_TEST_RULES)) != set(range(len(RULES))):
        raise RuntimeError("test folds must cover each composition exactly once")
    for fold, test_rules in enumerate(FOLD_TEST_RULES):
        train_rules = tuple(rule_id for rule_id in range(len(RULES)) if rule_id not in test_rules)
        if {RULES[rule_id].scope for rule_id in train_rules} != set(SCOPES):
            raise RuntimeError(f"fold {fold} hides a scope primitive from training")
        if {RULES[rule_id].reducer for rule_id in train_rules} != set(REDUCERS):
            raise RuntimeError(f"fold {fold} hides a reducer primitive from training")
