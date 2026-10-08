"""GRU and RIMs policies trained to imitate the information-gain teacher."""

from collections import defaultdict
import random
import time

import torch
from torch import Tensor, nn

from cognitive_lab.active.baselines import (
    select_probe,
    summarize_records,
    update_posterior,
    choose_answer,
)
from cognitive_lab.active.environment import (
    ALL_PROBES,
    RULES,
    RuleEnvironment,
    TaskCase,
)
from cognitive_lab.core.gru import GRUCore
from cognitive_lab.core.rims import RIMsCore
from cognitive_lab.task import ACTION_COUNT

MAX_PROBE_BUDGET = 3
POLICY_ACTION_COUNT = len(ALL_PROBES)
STATE_SIZE = len(RULES) + len(ALL_PROBES) + ACTION_COUNT + 2
FOLD_TEST_RULES = ((0, 3), (1, 4), (2, 5))


def build_controller(name: str) -> nn.Module:
    if name == "gru_256":
        return GRUCore(STATE_SIZE, POLICY_ACTION_COUNT, hidden_size=256)
    if name == "gru_parameter_matched":
        return GRUCore(STATE_SIZE, POLICY_ACTION_COUNT, hidden_size=205)
    if name == "rims":
        return RIMsCore(
            STATE_SIZE, POLICY_ACTION_COUNT, module_count=4, module_size=64, top_k=1
        )
    raise ValueError(f"unknown controller: {name!r}")


def encode_state(
    posterior: tuple[int, ...],
    previous_probe: int | None,
    previous_outcome: int | None,
    remaining_budget: int,
    budget_limit: int,
) -> list[float]:
    frame = [float(rule_id in posterior) for rule_id in range(len(RULES))]
    frame.extend(float(probe_id == previous_probe) for probe_id in range(len(ALL_PROBES)))
    frame.extend(float(action == previous_outcome) for action in range(ACTION_COUNT))
    frame.extend((remaining_budget / MAX_PROBE_BUDGET, budget_limit / MAX_PROBE_BUDGET))
    return frame


def _balanced_rule_schedule(
    episodes: int, rule_ids: tuple[int, ...], rng: random.Random
) -> list[int]:
    schedule = []
    while len(schedule) < episodes:
        batch = list(rule_ids)
        rng.shuffle(batch)
        schedule.extend(batch[: episodes - len(schedule)])
    return schedule


def generate_imitation_data(
    episodes: int,
    rule_ids: tuple[int, ...],
    seed: int,
    random_probe_probability: float = 0.5,
) -> dict[int, tuple[Tensor, Tensor]]:
    """Create interaction histories labelled by exact information gain."""
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    if not rule_ids or len(set(rule_ids)) != len(rule_ids) or any(
        type(rule_id) is not int or rule_id not in range(len(RULES))
        for rule_id in rule_ids
    ):
        raise ValueError("rule_ids must contain unique supported rule indices")
    if not 0 <= random_probe_probability <= 1:
        raise ValueError("random_probe_probability must be between 0 and 1")

    rng = random.Random(seed)
    rules = _balanced_rule_schedule(episodes, rule_ids, rng)
    grouped: dict[int, list[tuple[list[list[float]], int]]] = defaultdict(list)

    for rule_id in rules:
        budget_limit = rng.randint(1, MAX_PROBE_BUDGET)
        environment = RuleEnvironment(RULES[rule_id])
        posterior = tuple(range(len(RULES)))
        asked: set[int] = set()
        previous_probe = None
        previous_outcome = None
        history = [
            encode_state(posterior, previous_probe, previous_outcome, budget_limit, budget_limit)
        ]

        for _ in range(budget_limit):
            if len(posterior) == 1:
                break
            teacher_probe = select_probe("information_gain", posterior, asked, rng)
            if teacher_probe is None:
                break
            target = teacher_probe
            grouped[len(history)].append((history.copy(), target))

            available = [probe_id for probe_id in range(len(ALL_PROBES)) if probe_id not in asked]
            selected_probe = (
                rng.choice(available)
                if rng.random() < random_probe_probability
                else target
            )
            outcome = environment.query(ALL_PROBES[selected_probe])
            asked.add(selected_probe)
            posterior = update_posterior(posterior, selected_probe, outcome)
            previous_probe, previous_outcome = selected_probe, outcome
            history.append(
                encode_state(
                    posterior,
                    previous_probe,
                    previous_outcome,
                    budget_limit - len(asked),
                    budget_limit,
                )
            )

    return {
        length: (
            torch.tensor([sequence for sequence, _ in examples], dtype=torch.float32),
            torch.tensor([target for _, target in examples], dtype=torch.long),
        )
        for length, examples in grouped.items()
    }


def _action_accuracy(model: nn.Module, data: dict[int, tuple[Tensor, Tensor]], device: torch.device) -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for states, targets in data.values():
            for start in range(0, len(targets), 512):
                batch = states[start : start + 512].to(device)
                labels = targets[start : start + 512].to(device)
                logits = _mask_asked_probes(model(batch), batch)
                correct += int(logits.argmax(dim=-1).eq(labels).sum().item())
                total += len(labels)
    if total == 0:
        raise ValueError("the dataset contains no controller decisions")
    return correct / total


def _mask_asked_probes(logits: Tensor, states: Tensor) -> Tensor:
    start = len(RULES)
    end = start + len(ALL_PROBES)
    asked = states[:, :, start:end].amax(dim=1).bool()
    return logits.masked_fill(asked, -torch.inf)


def train_controller(
    model: nn.Module,
    train_data: dict[int, tuple[Tensor, Tensor]],
    validation_data: dict[int, tuple[Tensor, Tensor]],
    device: torch.device,
    epochs: int,
    batch_size: int,
    seed: int,
) -> dict:
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    all_targets = torch.cat([targets for _, targets in train_data.values()])
    counts = torch.bincount(all_targets, minlength=POLICY_ACTION_COUNT).float()
    class_weights = torch.zeros_like(counts)
    observed_classes = counts > 0
    class_weights[observed_classes] = counts[observed_classes].rsqrt()
    class_weights[observed_classes] /= class_weights[observed_classes].mean()
    loss_fn = nn.CrossEntropyLoss(weight=class_weights.to(device))
    best_score = -1.0
    best_epoch = 0
    best_state = None
    rng = random.Random(seed)
    started = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        batches = []
        for states, targets in train_data.values():
            order = torch.randperm(len(targets))
            batches.extend(
                (states[index], targets[index])
                for index in order.split(batch_size)
            )
        rng.shuffle(batches)
        for states, targets in batches:
            states, targets = states.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = _mask_asked_probes(model(states), states)
            loss = loss_fn(logits, targets)
            loss.backward()
            optimizer.step()

        validation_accuracy = _action_accuracy(model, validation_data, device)
        if validation_accuracy > best_score:
            best_score = validation_accuracy
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }

    if best_state is None:
        raise RuntimeError("training produced no controller checkpoint")
    model.load_state_dict(best_state)
    return {
        "best_epoch": best_epoch,
        "validation_action_accuracy": round(best_score, 4),
        "training_seconds": round(time.perf_counter() - started, 3),
    }


def _initial_record(rule_id: int, case: TaskCase, budget: int) -> dict:
    return {
        "rule_id": rule_id,
        "case": case,
        "posterior": tuple(range(len(RULES))),
        "asked": set(),
        "stopped": False,
        "history": [encode_state(tuple(range(len(RULES))), None, None, budget, budget)],
    }


def evaluate_controller(
    model: nn.Module,
    trials: list[tuple[int, TaskCase]],
    max_probe_budget: int,
    device: torch.device,
    stop_action: int | None = None,
) -> dict:
    """Evaluate a probe policy; optionally allow a dedicated learned stop action."""
    if not 0 <= max_probe_budget <= MAX_PROBE_BUDGET:
        raise ValueError(f"max_probe_budget must be from 0 to {MAX_PROBE_BUDGET}")
    if not trials:
        raise ValueError("trials must contain at least one episode")
    if stop_action is not None and (
        type(stop_action) is not int or stop_action != len(ALL_PROBES)
    ):
        raise ValueError("stop_action must be the first action after the probe range")
    model.eval()
    by_budget = {}
    for budget in range(max_probe_budget + 1):
        records = [_initial_record(rule_id, case, budget) for rule_id, case in trials]
        with torch.no_grad():
            for _ in range(budget):
                active_indices = [
                    index
                    for index, record in enumerate(records)
                    if not record["stopped"] and len(record["posterior"]) > 1
                ]
                if not active_indices:
                    break
                histories = [records[index]["history"] for index in active_indices]
                logits = model(torch.tensor(histories, dtype=torch.float32, device=device))
                if stop_action is not None and logits.shape[-1] <= stop_action:
                    raise ValueError("model output does not contain the configured stop action")
                for row, index in enumerate(active_indices):
                    record = records[index]
                    if record["asked"]:
                        logits[row, list(record["asked"])] = -torch.inf
                    action = int(logits[row].argmax().item())
                    if action == stop_action:
                        record["stopped"] = True
                        continue
                    environment = RuleEnvironment(RULES[record["rule_id"]])
                    outcome = environment.query(ALL_PROBES[action])
                    record["asked"].add(action)
                    record["posterior"] = update_posterior(
                        record["posterior"], action, outcome
                    )
                    record["history"].append(
                        encode_state(
                            record["posterior"],
                            action,
                            outcome,
                            budget - len(record["asked"]),
                            budget,
                        )
                    )

        summaries = []
        for record in records:
            environment = RuleEnvironment(RULES[record["rule_id"]])
            expected = environment.answer(record["case"])
            predicted = choose_answer(record["case"], record["posterior"])
            summaries.append(
                {
                    "rule_id": record["rule_id"],
                    "correct": predicted == expected,
                    "expected": expected,
                    "predicted": predicted,
                    "probes": len(record["asked"]),
                    "remaining_rules": len(record["posterior"]),
                    "rule_identified": len(record["posterior"]) == 1,
                }
            )
        metrics = summarize_records(summaries)
        if stop_action is not None:
            metrics["policy_stop_rate"] = round(
                sum(record["stopped"] for record in records) / len(records), 4
            )
        by_budget[str(budget)] = metrics
    return by_budget
