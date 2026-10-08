"""Deterministic episode generator and split definitions for rule world v1 (Phase 1)."""

import random
from collections import Counter
from itertools import permutations

from cognitive_lab.world.audit import KEY_CHANCE, heuristic_hits
from cognitive_lab.world.solver import solve
from cognitive_lab.world.templates import (
    ENTITY_SETS,
    SOURCES,
    TEMPLATES,
    UNKNOWN,
    WORLD_RULES,
    door_name,
    key_name,
    phrasing_part,
)

PARTS = ("train", "validation", "test")
SPLITS = ("iid", "phrasing", "composition", "entity")
MIN_UTTERANCES = 4
MAX_UTTERANCES = 10
FILTER_TOLERANCE = 0.01
FILTER_WARMUP = 20
FILTER_MAX_ATTEMPTS = 500

# Weights sum to 1; UNKNOWN answers come from "none" and "one_negative".
CASE_WEIGHTS = {
    "simple": 0.15,
    "self_correction": 0.15,
    "elimination": 0.15,
    "none": 0.10,
    "one_negative": 0.10,
    "conflict": 0.15,
    "conflict_correction": 0.20,
}

SOURCE_PATTERNS = {
    "simple": [(s,) for s in range(len(SOURCES))],
    "self_correction": [(s,) for s in range(len(SOURCES))],
    "elimination": [(a, b) for a in range(len(SOURCES)) for b in range(len(SOURCES))],
    "none": [()],
    "one_negative": [(s,) for s in range(len(SOURCES))],
    "conflict": list(permutations(range(len(SOURCES)), 2)),
    "conflict_correction": list(permutations(range(len(SOURCES)), 2)),
}


def composition_key(case: str, pattern: tuple[int, ...]) -> str:
    return f"{case}:{'-'.join(map(str, pattern))}"


def composition_parts() -> dict[str, dict[str, list[tuple[int, ...]]]]:
    """Hold out source patterns per case type; cases with < 3 patterns stay in all parts."""
    parts = {part: {} for part in PARTS}
    for index, (case, patterns) in enumerate(SOURCE_PATTERNS.items()):
        if len(patterns) < 3:
            for part in PARTS:
                parts[part][case] = list(patterns)
            continue
        shuffled = list(patterns)
        random.Random(1000 + index).shuffle(shuffled)
        held = max(1, round(len(shuffled) * 0.2))
        parts["test"][case] = shuffled[:held]
        parts["validation"][case] = shuffled[held : 2 * held]
        parts["train"][case] = shuffled[2 * held :]
    return parts


def split_config(split: str, part: str) -> dict:
    """Return what a split part may use: templates, source patterns, and entity names."""
    if split not in SPLITS or part not in PARTS:
        raise ValueError(f"unknown split/part: {split}/{part}")
    templates = {category: list(items) for category, items in TEMPLATES.items()}
    patterns = {case: list(p) for case, p in SOURCE_PATTERNS.items()}
    entities = ENTITY_SETS["base"]
    if split == "phrasing":
        templates = {
            category: [t for i, t in enumerate(items) if phrasing_part(i) == part]
            for category, items in TEMPLATES.items()
        }
    elif split == "composition":
        patterns = composition_parts()[part]
    elif split == "entity":
        entities = ENTITY_SETS["base"] if part == "train" else ENTITY_SETS[part]
    return {"templates": templates, "patterns": patterns, "entities": entities}


def _event(kind: str, source: int, door: int, key: int) -> dict:
    return {"kind": kind, "source": source, "door": door, "key": key}


def _core_events(rng: random.Random, case: str, pattern: tuple[int, ...], answer: int) -> list[dict]:
    """Events about the queried door (door index filled in later as -1 placeholder)."""
    others = [k for k in range(3) if k != answer]
    if case == "simple":
        return [_event("assert", pattern[0], -1, answer)]
    if case == "self_correction":
        return [
            _event("assert", pattern[0], -1, rng.choice(others)),
            _event("correct", pattern[0], -1, answer),
        ]
    if case == "elimination":
        first, second = rng.sample(others, 2)
        return [_event("negate", pattern[0], -1, first), _event("negate", pattern[1], -1, second)]
    if case == "none":
        return []
    if case == "one_negative":
        return [_event("negate", pattern[0], -1, rng.randrange(3))]
    if case in ("conflict", "conflict_correction"):
        first, second = pattern
        winner_key, loser_key = answer, rng.choice(others)
        first_key = winner_key if first > second else loser_key
        second_key = winner_key if second > first else loser_key
        second_kind = "correct" if case == "conflict_correction" else "assert"
        return [_event("assert", first, -1, first_key), _event(second_kind, second, -1, second_key)]
    raise ValueError(f"unknown case: {case}")


def _distractor_chain(
    rng: random.Random,
    truth: dict[int, int],
    room: int,
    speakers: dict[tuple[int, int], str],
) -> list[dict]:
    """One distractor unit about another door, a self-corrected pair, or chatter.

    Facts about other doors agree with that door's hidden true key. A correction pair
    states a wrong key and then corrects it to the truth, and only a speaker who says
    nothing else about that door may use one, so no speaker contradicts themself
    without a correction marker. `speakers` maps (door, source) to "plain" or "pair".
    """
    slots = [(door, source) for door in truth for source in range(len(SOURCES))]
    plain = [slot for slot in slots if speakers.get(slot) != "pair"]
    free = [slot for slot in slots if slot not in speakers]
    choices = ["chatter"] + (["assert", "negate"] if plain else []) + (["pair"] if room >= 2 and free else [])
    kind = rng.choice(choices)
    if kind == "chatter":
        return [{"kind": "chatter"}]
    if kind == "pair":
        door, source = rng.choice(free)
        speakers[(door, source)] = "pair"
        wrong = rng.choice([k for k in range(3) if k != truth[door]])
        return [_event("assert", source, door, wrong), _event("correct", source, door, truth[door])]
    door, source = rng.choice(plain)
    speakers[(door, source)] = "plain"
    if kind == "assert":
        return [_event("assert", source, door, truth[door])]
    return [_event("negate", source, door, rng.choice([k for k in range(3) if k != truth[door]]))]


def _interleave(rng: random.Random, chains: list[list[dict]]) -> list[dict]:
    """Randomly merge chains while keeping each chain's internal order."""
    chains = [list(chain) for chain in chains if chain]
    merged = []
    while chains:
        weights = [len(chain) for chain in chains]
        index = rng.choices(range(len(chains)), weights=weights)[0]
        merged.append(chains[index].pop(0))
        if not chains[index]:
            chains.pop(index)
    return merged


def choose_plan(
    rng: random.Random, config: dict, decks: dict[str, list]
) -> tuple[str, tuple[int, ...], int, int]:
    """Pick what the episode tests: case type, source pattern, queried door, answer key.

    Case types and (door, answer) pairs are dealt from shuffled decks that are refilled
    when empty. The case deck holds each case in proportion to CASE_WEIGHTS (per 20),
    and the pair deck holds all nine pairs, so case mix, answer labels, and the
    door-to-answer mapping cannot drift by chance even in small parts.
    """
    if not decks["cases"]:
        decks["cases"].extend(
            case for case, weight in CASE_WEIGHTS.items() for _ in range(round(weight * 20))
        )
        rng.shuffle(decks["cases"])
    if not decks["pairs"]:
        decks["pairs"].extend((door, key) for door in range(3) for key in range(3))
        rng.shuffle(decks["pairs"])
    case = decks["cases"].pop()
    door, answer = decks["pairs"].pop()
    return case, rng.choice(config["patterns"][case]), door, answer


def generate_episode(
    rng: random.Random, config: dict, episode_id: str, plan: tuple[str, tuple[int, ...], int, int]
) -> dict:
    """Realize a plan as surface text: distractors, ordering, and phrasing are sampled here."""
    doors, keys = config["entities"]["doors"], config["entities"]["keys"]
    case, pattern, door, answer_key = plan

    core = _core_events(rng, case, pattern, answer_key)
    for event in core:
        event["door"] = door
    total = rng.randint(MIN_UTTERANCES, MAX_UTTERANCES)
    truth = {d: rng.randrange(3) for d in range(3) if d != door}
    speakers: dict[tuple[int, int], str] = {}
    chains, used = [core], len(core)
    while used < total:
        chain = _distractor_chain(rng, truth, total - used, speakers)
        chains.append(chain)
        used += len(chain)
    events = _interleave(rng, chains)

    utterances, gold_events, template_ids = [], [], []
    for step, event in enumerate(events):
        template_id, text = rng.choice(config["templates"][event["kind"]])
        template_ids.append(template_id)
        if event["kind"] == "chatter":
            source = rng.randrange(len(SOURCES))
            gold_events.append({"step": step, "type": "chatter", "source": SOURCES[source]})
        else:
            source = event["source"]
            text = text.format(D=door_name(doors[event["door"]]), K=key_name(keys[event["key"]]))
            gold_events.append({
                "step": step,
                "type": event["kind"],
                "door": doors[event["door"]],
                "key": keys[event["key"]],
                "polarity": event["kind"] != "negate",
                "source": SOURCES[source],
            })
        utterances.append({"source": SOURCES[source], "text": text})

    question_id, question = rng.choice(config["templates"]["question"])
    template_ids.append(question_id)
    has_answer = case not in ("none", "one_negative")
    answer = key_name(keys[answer_key]) if has_answer else UNKNOWN

    solved = solve(gold_events, doors[door], list(keys))
    solved_answer = UNKNOWN if solved == UNKNOWN else key_name(solved)
    if solved_answer != answer:
        raise RuntimeError(f"{episode_id}: generator says {answer}, solver says {solved_answer}")

    return {
        "episode_id": episode_id,
        "phase": 1,
        "world_rules": WORLD_RULES,
        "entities": {"doors": [door_name(d) for d in doors], "keys": [key_name(k) for k in keys]},
        "utterances": utterances,
        "question": question.format(D=door_name(doors[door])),
        "answer": answer,
        "gold_events": gold_events,
        "gold_question": {"door": doors[door]},
        "tags": [case, composition_key(case, pattern)],
        "split_keys": {
            "case": case,
            "composition": composition_key(case, pattern),
            "template_ids": template_ids,
            "entity_set": doors[0],
            "queried_door_index": door,
            "answer_index": answer_key if has_answer else 3,
        },
    }


def _closer_to_chance(total: float, count: int, hit: float) -> bool:
    current = abs(total / count - KEY_CHANCE) if count else 0.0
    updated = abs((total + hit) / (count + 1) - KEY_CHANCE)
    return updated <= FILTER_TOLERANCE or updated <= current


def generate_part(split: str, part: str, size: int, seed: int) -> tuple[list[dict], dict]:
    """Generate a part with adversarial filtering against surface key heuristics.

    The plan (case, sources, door, answer) is fixed first so filtering cannot change
    the case mix or label balance; only the surface realization is resampled when it would push any heuristic's
    running hit rate further from chance (1/3) outside a small tolerance.
    """
    part_offset = PARTS.index(part)
    rng = random.Random(f"world-v1|{split}|{part}|{seed + part_offset}")
    config = split_config(split, part)
    episodes, resampled = [], 0
    decks: dict[str, list] = {"cases": [], "pairs": []}
    hits: dict[str, Counter] = {}
    keyed: Counter = Counter()
    for index in range(size):
        episode_id = f"v1-{split}-{part}-s{seed}-{index:05d}"
        plan = choose_plan(rng, config, decks)
        case = plan[0]
        case_hits = hits.setdefault(case, Counter())
        for _ in range(FILTER_MAX_ATTEMPTS):
            episode = generate_episode(rng, config, episode_id, plan)
            if episode["split_keys"]["answer_index"] == 3:
                break
            candidate = heuristic_hits(episode)
            if keyed[case] < FILTER_WARMUP or all(
                _closer_to_chance(case_hits[name], keyed[case], hit)
                for name, hit in candidate.items()
            ):
                break
            resampled += 1
        if episode["split_keys"]["answer_index"] != 3:
            keyed[case] += 1
            case_hits.update(heuristic_hits(episode))
        episodes.append(episode)
    stats = {
        "episodes": size,
        "resampled_candidates": resampled,
        "case_counts": dict(Counter(e["split_keys"]["case"] for e in episodes)),
        "heuristic_hit_rates_by_case": {
            case: {name: round(total / keyed[case], 4) for name, total in case_hits.items()}
            for case, case_hits in hits.items()
            if keyed[case]
        },
    }
    return episodes, stats
