"""Small deterministic worlds for active rule discovery."""

from dataclasses import dataclass
from itertools import product
import random
from typing import Literal

from cognitive_lab.task import ABSTAIN, DOOR_COUNT, KEY_COUNT, SEQUENCE_LENGTH

Scope = Literal["target_only", "all"]
Resolution = Literal["first", "last", "priority"]
ProbeEvent = tuple[bool, int]
Probe = tuple[ProbeEvent, ...]
TaskEvent = tuple[int, int]

SCOPES: tuple[Scope, ...] = ("target_only", "all")
RESOLUTIONS: tuple[Resolution, ...] = ("first", "last", "priority")
PROBE_LENGTH = 3
MIN_TASK_EVENTS = 4


@dataclass(frozen=True)
class Rule:
    scope: Scope
    resolution: Resolution

    def __post_init__(self) -> None:
        if self.scope not in SCOPES:
            raise ValueError(f"unknown rule scope: {self.scope!r}")
        if self.resolution not in RESOLUTIONS:
            raise ValueError(f"unknown conflict resolution: {self.resolution!r}")

    @property
    def name(self) -> str:
        return f"{self.scope}:{self.resolution}"


RULES = tuple(Rule(scope, resolution) for scope in SCOPES for resolution in RESOLUTIONS)
RULE_INDEX = {rule: index for index, rule in enumerate(RULES)}
PROBE_EVENT_TYPES = tuple(
    (is_target, key)
    for is_target in (False, True)
    for key in range(KEY_COUNT)
)
ALL_PROBES: tuple[Probe, ...] = tuple(product(PROBE_EVENT_TYPES, repeat=PROBE_LENGTH))


@dataclass(frozen=True)
class TaskCase:
    target_door: int
    events: tuple[TaskEvent, ...]

    def __post_init__(self) -> None:
        if type(self.target_door) is not int or self.target_door not in range(DOOR_COUNT):
            raise ValueError(f"target_door must be from 0 to {DOOR_COUNT - 1}")
        if (
            not isinstance(self.events, tuple)
            or not MIN_TASK_EVENTS <= len(self.events) <= SEQUENCE_LENGTH
        ):
            raise ValueError(
                f"events must be a tuple of {MIN_TASK_EVENTS} to {SEQUENCE_LENGTH} entries"
            )
        for event in self.events:
            if not isinstance(event, tuple) or len(event) != 2:
                raise ValueError("each event must be a (door, key) pair")
            door, key = event
            if type(door) is not int or door not in range(DOOR_COUNT):
                raise ValueError(f"event door must be from 0 to {DOOR_COUNT - 1}")
            if type(key) is not int or key not in range(KEY_COUNT):
                raise ValueError(f"event key must be from 0 to {KEY_COUNT - 1}")


def _validate_probe(probe: Probe) -> None:
    if not isinstance(probe, tuple) or len(probe) != PROBE_LENGTH:
        raise ValueError(f"a probe must contain exactly {PROBE_LENGTH} events")
    for event in probe:
        if not isinstance(event, tuple) or len(event) != 2:
            raise ValueError("each probe event must be a (is_target, key) pair")
        is_target, key = event
        if type(is_target) is not bool:
            raise ValueError("probe is_target values must be booleans")
        if type(key) is not int or key not in range(KEY_COUNT):
            raise ValueError(f"probe key must be from 0 to {KEY_COUNT - 1}")


def apply_rule(rule: Rule, events: tuple[ProbeEvent, ...]) -> int:
    """Apply one rule to ordered (is_target, key) events; return a key or ABSTAIN."""
    keys = [
        key
        for is_target, key in events
        if rule.scope == "all" or is_target
    ]
    if not keys:
        return ABSTAIN
    if rule.resolution == "first":
        return keys[0]
    if rule.resolution == "last":
        return keys[-1]
    return max(keys)


def apply_case(rule: Rule, case: TaskCase) -> int:
    events = tuple((door == case.target_door, key) for door, key in case.events)
    return apply_rule(rule, events)


class RuleEnvironment:
    """A hidden-rule oracle exposing only probe and final-case outcomes."""

    def __init__(self, hidden_rule: Rule) -> None:
        if hidden_rule not in RULES:
            raise ValueError("hidden_rule must be one of the six supported rules")
        self._hidden_rule = hidden_rule

    def query(self, probe: Probe) -> int:
        _validate_probe(probe)
        return apply_rule(self._hidden_rule, probe)

    def answer(self, case: TaskCase) -> int:
        if not isinstance(case, TaskCase):
            raise TypeError("case must be a TaskCase")
        return apply_case(self._hidden_rule, case)


def generate_case(rng: random.Random) -> TaskCase:
    """Create a 4–8-event held-out case, separate from three-event probes."""
    target_door = rng.randrange(DOOR_COUNT)
    event_count = rng.randint(4, SEQUENCE_LENGTH)
    events: list[TaskEvent] = []
    distractor_doors = [door for door in range(DOOR_COUNT) if door != target_door]
    force_distractors = rng.random() < 0.2
    for _ in range(event_count):
        door = rng.choice(distractor_doors) if force_distractors else rng.randrange(DOOR_COUNT)
        events.append((door, rng.randrange(KEY_COUNT)))
    return TaskCase(target_door, tuple(events))


def probe_outcome_table() -> tuple[tuple[int, ...], ...]:
    """Return outcomes indexed by probe ID, then by RULES index."""
    return tuple(
        tuple(apply_rule(rule, probe) for rule in RULES)
        for probe in ALL_PROBES
    )


PROBE_OUTCOMES = probe_outcome_table()
