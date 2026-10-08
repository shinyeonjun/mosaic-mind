"""Rule world v2-H: the standard v2 world, plus tone cues that hint whether a statement is right.

Every session is the standard v2 session (same speakers, accuracies, claims, phrasing). A
separate random stream then adds a tone cue to each statement:
    sure   ("확실한데, ...")         more likely when the statement is right
    plain  (no prefix)
    hedge  ("잘은 모르겠는데, ...")  more likely when the statement is wrong
The cue depends only on whether the statement is right, not on who says it, so
    P(statement, cue | T, level) = P(statement | T, level) * P(cue | right/wrong).
Retracted statements (the first half of a correction) get a cue from the even mixture, so
their cue carries no information; chatter gets no cue.

The designed fact format (door, key, polarity) cannot hold the cue. A learned message can.

python -m cognitive_lab.world2.hedged    # ceilings: what the cue is worth to an ideal judge
"""

import random

from cognitive_lab.world2 import generator as base
from cognitive_lab.world2.bayes_learner import PRIOR, claim_likelihood
from cognitive_lab.world2.generator import ACCURACY_LEVELS, accuracies_at

CUES = ("sure", "plain", "hedge")
CUE_PREFIXES = {
    "sure": ("확실한데, ", "내가 직접 봤어. ", "틀림없어. "),
    "plain": ("",),
    "hedge": ("잘은 모르겠는데, ", "들은 얘기라 확실하진 않은데, ", "기억이 가물가물한데, "),
}
# P(cue | right) and P(cue | wrong). Likelihood ratios: sure 3.0, plain 1.0, hedge 1/3.
CUE_GIVEN_RIGHT = {"sure": 0.45, "plain": 0.40, "hedge": 0.15}
CUE_GIVEN_WRONG = {"sure": 0.15, "plain": 0.40, "hedge": 0.45}


def cue_factor(cue: str, right: bool) -> float:
    return (CUE_GIVEN_RIGHT if right else CUE_GIVEN_WRONG)[cue]


def _draw(rng: random.Random, table: dict[str, float]) -> str:
    return rng.choices(CUES, weights=[table[c] for c in CUES])[0]


def add_cues(session: dict, seed_text: str) -> dict:
    """Add tone cues in place, from a random stream separate from the session's own."""
    rng = random.Random(f"world-v2-H|cues|{seed_text}")
    mixture = {c: (CUE_GIVEN_RIGHT[c] + CUE_GIVEN_WRONG[c]) / 2 for c in CUES}
    for episode in session["episodes"]:
        truths = dict(episode["hidden_other_truths"])
        truths[_queried(episode)] = episode["answer"]
        events = episode["gold_events"]
        for event, utterance in zip(events, episode["utterances"]):
            if event["type"] == "chatter":
                event["cue"] = None
                continue
            retracted = event["type"] == "assert" and _is_retracted(event, events)
            if retracted:
                cue = _draw(rng, mixture)
            else:
                truth = truths[_door_label(episode, event["door"])]
                key = _key_label(episode, event["key"])
                right = (key == truth) if event["polarity"] else (key != truth)
                cue = _draw(rng, CUE_GIVEN_RIGHT if right else CUE_GIVEN_WRONG)
            event["cue"] = cue
            utterance["text"] = rng.choice(CUE_PREFIXES[cue]) + utterance["text"]
    session["world"] = "v2-H"
    return session


def _door_label(episode: dict, door: str) -> str:
    return next(d for d in episode["entities"]["doors"] if d.startswith(door))


def _key_label(episode: dict, key: str) -> str:
    return next(k for k in episode["entities"]["keys"] if k.startswith(key))


def _queried(episode: dict) -> str:
    return _door_label(episode, episode["gold_question"]["door"])


def _is_retracted(event: dict, events: list[dict]) -> bool:
    """An assert later corrected by the same speaker about the same door."""
    return any(e["type"] == "correct" and e["source"] == event["source"] and e["door"] == event["door"]
               and e["step"] > event["step"] for e in events)


def generate_part(part: str, seed: int, sessions: int | None = None) -> list[dict]:
    return [add_cues(s, s["session_id"]) for s in base.generate_part(part, seed, sessions)]


def final_claims(episode: dict) -> list[tuple[str, int, bool, str]]:
    """(speaker, key index, polarity, cue) of each speaker's final statement about the queried door."""
    door = episode["gold_question"]["door"]
    keys = episode["entities"]["keys"]
    latest = {}
    for event in episode["gold_events"]:
        if event["type"] != "chatter" and event["door"] == door:
            latest[event["source"]] = (event["source"], keys.index(_key_label(episode, event["key"])),
                                       event["polarity"], event["cue"])
    return list(latest.values())


def _posterior(claims, level_of, use_cues: bool) -> list[float]:
    weights = [1.0, 1.0, 1.0]
    for speaker, key, polarity, cue in claims:
        for truth in range(3):
            right = (key == truth) if polarity else (key != truth)
            factor = cue_factor(cue, right) if use_cues else 1.0
            weights[truth] *= factor * level_of(speaker, key, polarity, truth)
    total = sum(weights)
    return [w / total for w in weights]


def _score(q: list[float], truth: int) -> int:
    best = max(range(3), key=q.__getitem__)
    return 0 if q[best] <= 0.5 else (1 if best == truth else -1)


def ceilings(sessions: list[dict], hazard: float = 0.01) -> dict[str, float]:
    """Mean score of ideal judges with and without cues: knowing accuracies, and learning them (B3)."""
    totals = {"oracle": 0, "oracle+cue": 0, "learner": 0, "learner+cue": 0}
    count = 0
    for session in sessions:
        belief = {s: list(PRIOR) for s in session["speakers"]}
        for episode in session["episodes"]:
            accuracy = dict(zip(session["speakers"], accuracies_at(session, episode["index"])))
            for s, b in belief.items():
                belief[s] = [(1 - hazard) * x + hazard * p for x, p in zip(b, PRIOR)]
            claims = final_claims(episode)
            truth = episode["entities"]["keys"].index(episode["answer"])
            known = lambda s, k, p, t: claim_likelihood(k, p, t, accuracy[s])
            learned = lambda s, k, p, t: sum(b * claim_likelihood(k, p, t, l) for b, l in zip(belief[s], ACCURACY_LEVELS))
            for name, level_of, cue in (("oracle", known, False), ("oracle+cue", known, True),
                                        ("learner", learned, False), ("learner+cue", learned, True)):
                totals[name] += _score(_posterior(claims, level_of, cue), truth) if claims else 0
            # Cues are independent of the level given right/wrong, so the level update is unchanged.
            for speaker, key, polarity, _ in claims:
                right = (key == truth) if polarity else (key != truth)
                updated = [b * (l if right else 1 - l) for b, l in zip(belief[speaker], ACCURACY_LEVELS)]
                belief[speaker] = [u / sum(updated) for u in updated]
            count += 1
    return {name: round(total / count, 4) for name, total in totals.items()}


def main() -> None:
    for seed in (42, 43):
        sessions = generate_part("test", seed)
        print(f"seed {seed}: {ceilings(sessions)}", flush=True)


if __name__ == "__main__":
    main()
