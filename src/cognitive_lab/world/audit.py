"""Shortcut audit: surface-feature baselines must stay near chance on held-out parts."""

import math
import random
from collections import Counter

from cognitive_lab.world.templates import key_name

UNKNOWN_LABEL = 3
KEY_CHANCE = 1 / 3
MIN_MARGIN = 0.03


def _margin(rate: float, count: int) -> float:
    """Allow three standard errors of sampling noise, but never less than MIN_MARGIN."""
    return max(MIN_MARGIN, 3 * math.sqrt(rate * (1 - rate) / max(1, count)))


def _key_mentions(episode: dict) -> list[int]:
    keys = episode["entities"]["keys"]
    return [
        keys.index(key_name(event["key"]))
        for event in episode["gold_events"]
        if event["type"] != "chatter"
    ]


def heuristic_hits(episode: dict) -> dict[str, float]:
    """Expected hit rate of each surface key heuristic on one key-answer episode (ties split evenly)."""
    answer = _label(episode)
    mentions = _key_mentions(episode)
    counts = Counter(mentions)
    full = {key: counts.get(key, 0) for key in range(3)}

    def share(candidates: list[int]) -> float:
        return 1 / len(candidates) if answer in candidates else 0.0

    return {
        "first_key_mentioned": float(bool(mentions) and mentions[0] == answer),
        "last_key_mentioned": float(bool(mentions) and mentions[-1] == answer),
        "most_frequent_key": share([k for k, c in full.items() if c == max(full.values())]),
        "least_frequent_key": share([k for k, c in full.items() if c == min(full.values())]),
    }


def _pick(counter: Counter, rng: random.Random, highest: bool) -> int:
    counts = {key: counter.get(key, 0) for key in range(3)}
    target = max(counts.values()) if highest else min(counts.values())
    return rng.choice([key for key, count in counts.items() if count == target])


def _label(episode: dict) -> int:
    return episode["split_keys"]["answer_index"]


def _surface_cell(episode: dict) -> tuple:
    sources = tuple(sorted(Counter(u["source"] for u in episode["utterances"]).items()))
    return len(episode["utterances"]), sources


def _accuracy(predictions: list[int], labels: list[int]) -> float | None:
    if not labels:
        return None
    return round(sum(p == y for p, y in zip(predictions, labels)) / len(labels), 4)


def audit_split(
    train: list[dict], test: list[dict], seed: int, case_weights: dict[str, float] | None = None
) -> dict:
    """Fit surface baselines on `train`, evaluate on `test`, and flag leaks."""
    rng = random.Random(seed)
    keyed = [episode for episode in test if _label(episode) != UNKNOWN_LABEL]
    key_labels = [_label(episode) for episode in keyed]

    door_majority: dict[int, int] = {}
    by_door: dict[int, Counter] = {}
    for episode in train:
        if _label(episode) != UNKNOWN_LABEL:
            door = episode["split_keys"]["queried_door_index"]
            by_door.setdefault(door, Counter())[_label(episode)] += 1
    for door, counter in by_door.items():
        door_majority[door] = counter.most_common(1)[0][0]

    key_baselines = {
        "queried_door_only": [
            door_majority.get(e["split_keys"]["queried_door_index"], 0) for e in keyed
        ],
        "first_key_mentioned": [(_key_mentions(e) or [0])[0] for e in keyed],
        "last_key_mentioned": [(_key_mentions(e) or [0])[-1] for e in keyed],
        "most_frequent_key": [_pick(Counter(_key_mentions(e)), rng, True) for e in keyed],
        "least_frequent_key": [_pick(Counter(_key_mentions(e)), rng, False) for e in keyed],
    }

    results = {}
    rejected = []
    for name, predictions in key_baselines.items():
        accuracy = _accuracy(predictions, key_labels)
        margin = _margin(KEY_CHANCE, len(keyed))
        ok = abs(accuracy - KEY_CHANCE) <= margin
        results[name] = {
            "accuracy_on_key_answers": accuracy,
            "chance": round(KEY_CHANCE, 4),
            "allowed_margin": round(margin, 4),
            "ok": ok,
        }
        if not ok:
            rejected.append(name)

    by_case: dict[str, list[dict]] = {}
    for episode in keyed:
        by_case.setdefault(episode["split_keys"]["case"], []).append(episode)
    per_case = {}
    for case, episodes in sorted(by_case.items()):
        rates = {}
        for name in ("first_key_mentioned", "last_key_mentioned", "most_frequent_key", "least_frequent_key"):
            rate = sum(heuristic_hits(e)[name] for e in episodes) / len(episodes)
            rates[name] = round(rate, 4)
        margin = _margin(KEY_CHANCE, len(episodes))
        leaks = [name for name, rate in rates.items() if abs(rate - KEY_CHANCE) > margin]
        per_case[case] = {"episodes": len(episodes), "hit_rates": rates, "allowed_margin": round(margin, 4), "leaks": leaks}
        rejected.extend(f"{case}:{name}" for name in leaks)
    results["per_case_key_heuristics"] = per_case

    cells: dict[tuple, Counter] = {}
    for episode in train:
        cells.setdefault(_surface_cell(episode), Counter())[_label(episode) == UNKNOWN_LABEL] += 1
    unknown_truth = [_label(e) == UNKNOWN_LABEL for e in test]
    majority_unknown = Counter(unknown_truth).most_common(1)[0][0]
    unknown_predictions = [
        cells[_surface_cell(e)].most_common(1)[0][0] if _surface_cell(e) in cells else majority_unknown
        for e in test
    ]
    detector_accuracy = _accuracy(unknown_predictions, unknown_truth)
    majority_rate = round(unknown_truth.count(majority_unknown) / len(unknown_truth), 4)
    detector_ok = detector_accuracy <= majority_rate + _margin(majority_rate, len(test))
    results["unknown_from_length_and_sources"] = {
        "binary_accuracy": detector_accuracy,
        "majority_rate": majority_rate,
        "ok": detector_ok,
    }
    if not detector_ok:
        rejected.append("unknown_from_length_and_sources")

    label_counts = Counter(_label(e) for e in test)
    key_shares = {k: round(label_counts[k] / max(1, len(keyed)), 4) for k in range(3)}
    balance_margin = _margin(KEY_CHANCE, len(keyed))
    balance_ok = all(abs(share - KEY_CHANCE) <= balance_margin for share in key_shares.values())
    results["key_label_balance"] = {"shares_among_key_answers": key_shares, "ok": balance_ok}
    if not balance_ok:
        rejected.append("key_label_balance")

    if case_weights is not None:
        case_counts = Counter(e["split_keys"]["case"] for e in test)
        present = {case: w for case, w in case_weights.items() if case_counts[case] or w == 0}
        total_weight = sum(present.values()) or 1.0
        mix = {}
        mix_ok = True
        for case, weight in present.items():
            expected = weight / total_weight
            observed = case_counts[case] / len(test)
            ok = abs(observed - expected) <= _margin(expected, len(test))
            mix[case] = {"expected": round(expected, 4), "observed": round(observed, 4), "ok": ok}
            mix_ok &= ok
        missing_cases = sorted(set(case_weights) - set(present))
        results["case_mix"] = {"cases": mix, "missing_cases": missing_cases, "ok": mix_ok}
        if not mix_ok:
            rejected.append("case_mix")

    return {
        "test_episodes": len(test),
        "key_answer_episodes": len(keyed),
        "unknown_share": round(label_counts[UNKNOWN_LABEL] / len(test), 4),
        "baselines": results,
        "rejected_by": rejected,
        "accepted": not rejected,
    }
