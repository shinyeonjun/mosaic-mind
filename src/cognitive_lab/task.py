import random

import torch
from torch import Tensor

DOOR_COUNT = 3
KEY_COUNT = 3
ABSTAIN = KEY_COUNT
ACTION_COUNT = KEY_COUNT + 1
SEQUENCE_LENGTH = 8
INPUT_SIZE = DOOR_COUNT + KEY_COUNT + DOOR_COUNT
ACTION_NAMES = ("USE_RED_KEY", "USE_BLUE_KEY", "USE_GREEN_KEY", "ABSTAIN")
SPLIT_NAMES = ("train", "validation", "test")
FOLD_COUNT = 3


def composition_split_templates() -> dict[str, dict[str, list[tuple[int, int, int]]]]:
    """Return fixed, disjoint (target door, previous key, final key) templates."""
    splits = {name: {"ordinary": [], "corrected": []} for name in SPLIT_NAMES}
    for final_key in range(KEY_COUNT):
        for offset, split in enumerate(SPLIT_NAMES):
            target_door = (final_key + offset) % DOOR_COUNT
            splits[split]["ordinary"].append((target_door, final_key, final_key))

        corrected = [
            (door, previous_key, final_key)
            for door in range(DOOR_COUNT)
            for previous_key in range(KEY_COUNT)
            if previous_key != final_key
        ]
        validation = (final_key, (final_key + 1) % KEY_COUNT, final_key)
        test = (
            (final_key + 1) % DOOR_COUNT,
            (final_key + 2) % KEY_COUNT,
            final_key,
        )
        splits["validation"]["corrected"].append(validation)
        splits["test"]["corrected"].append(test)
        splits["train"]["corrected"].extend(
            template for template in corrected if template not in (validation, test)
        )
    return splits


def composition_fold_templates(
    fold: int,
) -> dict[str, dict[str, list[tuple[int, int, int]]]]:
    """Return a balanced split; each target-rule template is tested in one fold."""
    if fold not in range(FOLD_COUNT):
        raise ValueError(f"fold must be from 0 to {FOLD_COUNT - 1}, got {fold}")

    splits = _build_composition_fold_templates(fold)
    all_templates = {
        (door, previous_key, final_key)
        for door in range(DOOR_COUNT)
        for previous_key in range(KEY_COUNT)
        for final_key in range(KEY_COUNT)
    }
    test_coverage = []
    expected_counts = {
        "train": {"ordinary": KEY_COUNT, "corrected": 3 * KEY_COUNT},
        "validation": {"ordinary": KEY_COUNT, "corrected": KEY_COUNT},
        "test": {"ordinary": KEY_COUNT, "corrected": 2 * KEY_COUNT},
    }
    for candidate_fold in range(FOLD_COUNT):
        candidate = (
            splits
            if candidate_fold == fold
            else _build_composition_fold_templates(candidate_fold)
        )
        groups = {
            split: set(candidate[split]["ordinary"] + candidate[split]["corrected"])
            for split in SPLIT_NAMES
        }
        split_pairs = (
            ("train", "validation"),
            ("train", "test"),
            ("validation", "test"),
        )
        if any(groups[left] & groups[right] for left, right in split_pairs):
            raise RuntimeError("composition templates overlap within a fold")
        if set.union(*groups.values()) != all_templates:
            raise RuntimeError("a fold does not partition all target-rule templates")
        for split, counts in expected_counts.items():
            unexpected_counts = any(
                len(candidate[split][group]) != count
                for group, count in counts.items()
            )
            if unexpected_counts:
                raise RuntimeError(f"unexpected {split} template counts")
        test_coverage.extend(groups["test"])

    if len(test_coverage) != len(all_templates) or set(test_coverage) != all_templates:
        raise RuntimeError("the three test folds must cover each template exactly once")
    return splits


def _build_composition_fold_templates(
    fold: int,
) -> dict[str, dict[str, list[tuple[int, int, int]]]]:
    splits = {name: {"ordinary": [], "corrected": []} for name in SPLIT_NAMES}
    for final_key in range(KEY_COUNT):
        for split, offset in (("test", 0), ("validation", 1), ("train", 2)):
            target_door = (final_key + fold + offset) % DOOR_COUNT
            splits[split]["ordinary"].append((target_door, final_key, final_key))

        corrected = [
            (door, previous_key, final_key)
            for door in range(DOOR_COUNT)
            for previous_key in range(KEY_COUNT)
            if previous_key != final_key
        ]
        test_templates = [
            template
            for template in corrected
            if sum(template) % FOLD_COUNT == fold
        ]
        remaining = [template for template in corrected if template not in test_templates]
        validation_index = (fold + final_key) % len(remaining)
        validation_template = remaining.pop(validation_index)

        splits["test"]["corrected"].extend(test_templates)
        splits["validation"]["corrected"].append(validation_template)
        splits["train"]["corrected"].extend(remaining)
    return splits


def generate_episodes(
    episode_count: int, seed: int
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Return observations, labels, correction flags, and missing-rule flags."""
    if episode_count <= 0:
        raise ValueError("episode_count must be positive")

    rng = random.Random(seed)
    observations = torch.zeros(episode_count, SEQUENCE_LENGTH, INPUT_SIZE)
    labels = torch.empty(episode_count, dtype=torch.long)
    corrected = torch.zeros(episode_count, dtype=torch.bool)
    missing = torch.zeros(episode_count, dtype=torch.bool)

    for episode in range(episode_count):
        target_door = rng.randrange(DOOR_COUNT)
        missing_rule = rng.random() < 0.15
        has_correction = not missing_rule and rng.random() < 0.4
        missing[episode] = missing_rule
        corrected[episode] = has_correction
        events: list[tuple[int, int] | None] = [None] * SEQUENCE_LENGTH

        if missing_rule:
            labels[episode] = ABSTAIN
        else:
            first_position = rng.randrange(
                SEQUENCE_LENGTH - 1 if has_correction else SEQUENCE_LENGTH
            )
            first_key = rng.randrange(KEY_COUNT)
            events[first_position] = (target_door, first_key)
            labels[episode] = first_key

            if has_correction:
                correction_position = rng.randrange(first_position + 1, SEQUENCE_LENGTH)
                corrected_key = rng.choice([key for key in range(KEY_COUNT) if key != first_key])
                events[correction_position] = (target_door, corrected_key)
                labels[episode] = corrected_key

        distractor_doors = [door for door in range(DOOR_COUNT) if door != target_door]
        for step, event in enumerate(events):
            if event is None:
                event = (rng.choice(distractor_doors), rng.randrange(KEY_COUNT))
            observed_door, observed_key = event
            observations[episode, step, observed_door] = 1.0
            observations[episode, step, DOOR_COUNT + observed_key] = 1.0
            observations[episode, step, DOOR_COUNT + KEY_COUNT + target_door] = 1.0

    return observations, labels, corrected, missing


def generate_composition_episodes(
    episode_count: int, seed: int, split: str
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Generate episodes whose target-rule compositions are split-disjoint."""
    if split not in SPLIT_NAMES:
        raise ValueError(f"split must be one of {SPLIT_NAMES}, got {split!r}")
    observations, labels, corrected, missing, _ = _generate_template_episodes(
        episode_count, seed, composition_split_templates()[split]
    )
    return observations, labels, corrected, missing


def generate_fold_episodes(
    episode_count: int, seed: int, split: str, fold: int
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Generate a fold split and return IDs for per-composition evaluation."""
    if split not in SPLIT_NAMES:
        raise ValueError(f"split must be one of {SPLIT_NAMES}, got {split!r}")
    return _generate_template_episodes(
        episode_count, seed, composition_fold_templates(fold)[split]
    )


def _generate_template_episodes(
    episode_count: int,
    seed: int,
    templates: dict[str, list[tuple[int, int, int]]],
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    if episode_count <= 0:
        raise ValueError("episode_count must be positive")

    rng = random.Random(seed)
    observations = torch.zeros(episode_count, SEQUENCE_LENGTH, INPUT_SIZE)
    labels = torch.empty(episode_count, dtype=torch.long)
    corrected = torch.zeros(episode_count, dtype=torch.bool)
    missing = torch.zeros(episode_count, dtype=torch.bool)
    template_ids = torch.full((episode_count,), -1, dtype=torch.long)

    for episode in range(episode_count):
        missing_rule = rng.random() < 0.15
        has_correction = not missing_rule and rng.random() < 0.4
        missing[episode] = missing_rule
        corrected[episode] = has_correction
        events: list[tuple[int, int] | None] = [None] * SEQUENCE_LENGTH

        if missing_rule:
            target_door = rng.randrange(DOOR_COUNT)
            labels[episode] = ABSTAIN
        else:
            template_group = "corrected" if has_correction else "ordinary"
            target_door, first_key, final_key = rng.choice(templates[template_group])
            template_ids[episode] = (
                target_door * KEY_COUNT + first_key
            ) * KEY_COUNT + final_key
            first_position = rng.randrange(
                SEQUENCE_LENGTH - 1 if has_correction else SEQUENCE_LENGTH
            )
            events[first_position] = (target_door, first_key)
            labels[episode] = final_key

            if has_correction:
                correction_position = rng.randrange(first_position + 1, SEQUENCE_LENGTH)
                events[correction_position] = (target_door, final_key)

        distractor_doors = [door for door in range(DOOR_COUNT) if door != target_door]
        for step, event in enumerate(events):
            if event is None:
                event = (rng.choice(distractor_doors), rng.randrange(KEY_COUNT))
            observed_door, observed_key = event
            observations[episode, step, observed_door] = 1.0
            observations[episode, step, DOOR_COUNT + observed_key] = 1.0
            observations[episode, step, DOOR_COUNT + KEY_COUNT + target_door] = 1.0

    return observations, labels, corrected, missing, template_ids

