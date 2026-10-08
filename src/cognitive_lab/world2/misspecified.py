"""Rule world v2-M: speakers that break the standard statement model, and its oracle.

The code judges (B3/B4/B5) were derived from the *standard* model: each speaker has one
accuracy p, errors pick a wrong key uniformly, speakers are independent, and accuracy
changes at most once (episode 21). Here every speaker secretly has one of four types:

- standard:  as before.
- polarity:  separate accuracies for positive and negative statements, at least 0.44 apart.
- drifting:  accuracy level moves one step up/down the ordered levels with probability
             DRIFT_STEP before every episode (a random walk, no single change point).
- copier:    when its leader (another fixed speaker) also talks about the queried door,
             it repeats the leader's final claim with probability COPY_RATE; otherwise it
             behaves like a standard speaker. Errors are therefore correlated.

There is no session-level change at episode 21. The oracle O2-M knows every type and
parameter and computes the exact posterior under this process, including the copier
mixture P(x | T, leader claim c) = COPY_RATE * 1[x = c] + (1 - COPY_RATE) * P_own(x | T),
where P_own includes the polarity choice probability (it does not cancel in a mixture).
"""

import random

from cognitive_lab.world.generator import _interleave
from cognitive_lab.world.templates import ENTITY_SETS, TEMPLATES, UNKNOWN, door_name, key_name
from cognitive_lab.world2.agents import NonLearningJudge, decide
from cognitive_lab.world2.generator import (
    ACCURACY_LEVELS, MAX_UTTERANCES, MIN_UTTERANCES, NAME_POOLS, PARTS, SESSION_COUNTS, SESSION_LENGTH,
    SPEAKER_COUNT_WEIGHTS, SPEAKERS_PER_SESSION, STATEMENT_WEIGHTS, WORLD_RULES, _statement,
)

TYPES = ("standard", "polarity", "drifting", "copier")
ORDERED_LEVELS = tuple(sorted(ACCURACY_LEVELS))  # 0.2, 0.5, 0.75, 0.95
DRIFT_STEP = 0.15
COPY_RATE = 0.7
MIN_POLARITY_GAP = 0.44
NEGATIVE_SHARE = STATEMENT_WEIGHTS["negate"]  # probability a talker's final claim is negative


def _profile(rng: random.Random, speakers: int) -> list[dict]:
    """Leaders are never copiers, so copy chains cannot form cycles and every leader's
    claim exists before its copier speaks (the oracle's mixture assumes exactly that)."""
    kinds = [rng.choice(TYPES) for _ in range(speakers)]
    for s in range(speakers):
        if kinds[s] == "copier" and all(kinds[o] == "copier" for o in range(speakers) if o != s):
            kinds[s] = "standard"
    profiles = []
    for s, kind in enumerate(kinds):
        if kind == "polarity":
            while True:
                pos, neg = rng.choice(ACCURACY_LEVELS), rng.choice(ACCURACY_LEVELS)
                if abs(pos - neg) >= MIN_POLARITY_GAP:
                    break
            profiles.append({"type": kind, "positive": pos, "negative": neg})
        elif kind == "drifting":
            profiles.append({"type": kind, "start": rng.randrange(len(ORDERED_LEVELS))})
        elif kind == "copier":
            profiles.append({"type": kind, "accuracy": rng.choice(ACCURACY_LEVELS),
                             "leader": rng.choice([o for o in range(speakers) if o != s and kinds[o] != "copier"])})
        else:
            profiles.append({"type": kind, "accuracy": rng.choice(ACCURACY_LEVELS)})
    return profiles


def _trajectory(rng: random.Random, profiles: list[dict]) -> list[list[list[float]]]:
    """Per episode, per speaker: [positive accuracy, negative accuracy]."""
    levels = {s: p["start"] for s, p in enumerate(profiles) if p["type"] == "drifting"}
    out = []
    for _ in range(SESSION_LENGTH):
        for s in levels:
            if rng.random() < DRIFT_STEP:
                levels[s] = min(max(levels[s] + rng.choice((-1, 1)), 0), len(ORDERED_LEVELS) - 1)
        row = []
        for s, p in enumerate(profiles):
            if p["type"] == "polarity":
                row.append([p["positive"], p["negative"]])
            elif p["type"] == "drifting":
                row.append([ORDERED_LEVELS[levels[s]]] * 2)
            else:
                row.append([p["accuracy"]] * 2)
        out.append(row)
    return out


def _talker_chain(rng, speaker, accuracy_pair, door, truth, leader_claim):
    """Chain of events for one talker; copiers may repeat the leader's final claim."""
    if leader_claim is not None and rng.random() < COPY_RATE:
        kind, key = leader_claim
        return [{"kind": kind, "speaker": speaker, "door": door, "key": key}], (kind, key)
    kind = rng.choices(list(STATEMENT_WEIGHTS), weights=list(STATEMENT_WEIGHTS.values()))[0]
    if kind == "negate":
        key = _statement(rng, accuracy_pair[1], truth, negative=True)
        return [{"kind": "negate", "speaker": speaker, "door": door, "key": key}], ("negate", key)
    final = _statement(rng, accuracy_pair[0], truth, negative=False)
    if kind == "assert":
        return [{"kind": "assert", "speaker": speaker, "door": door, "key": final}], ("assert", final)
    retracted = rng.choice([k for k in range(3) if k != final])
    return ([{"kind": "assert", "speaker": speaker, "door": door, "key": retracted},
             {"kind": "correct", "speaker": speaker, "door": door, "key": final}], ("assert", final))


def _episode(rng, session, index, door, truth) -> dict:
    doors, keys = ENTITY_SETS["base"]["doors"], ENTITY_SETS["base"]["keys"]
    profiles, accuracy = session["hidden"]["profiles"], session["hidden"]["trajectory"][index]
    speakers = session["speakers"]
    count = rng.choices(range(4), weights=SPEAKER_COUNT_WEIGHTS)[0]
    talkers = rng.sample(range(len(speakers)), count)
    # Leaders speak before the copiers that follow them.
    talkers.sort(key=lambda s: profiles[s]["type"] == "copier" and profiles[s]["leader"] in talkers)
    finals, chains = {}, []
    for s in talkers:
        p = profiles[s]
        leader_claim = finals.get(p["leader"]) if p["type"] == "copier" else None
        chain, finals[s] = _talker_chain(rng, s, accuracy[s], door, truth, leader_claim)
        chains.append(chain)
    used = sum(len(c) for c in chains)
    total = max(used, rng.randint(MIN_UTTERANCES, MAX_UTTERANCES))
    other_truth = {d: rng.randrange(3) for d in range(3) if d != door}
    while used < total:
        if rng.random() < 0.3:
            chains.append([{"kind": "chatter", "speaker": rng.randrange(len(speakers))}])
        else:
            other, speaker = rng.choice(list(other_truth)), rng.randrange(len(speakers))
            negative = rng.random() < 0.25
            chains.append([{"kind": "negate" if negative else "assert", "speaker": speaker, "door": other,
                            "key": _statement(rng, accuracy[speaker][1 if negative else 0], other_truth[other], negative)}])
        used += 1
    # Copier/leader order is kept within the interleave only through each chain; the
    # final-claim semantics (latest statement per speaker) do not depend on that order.
    events = _interleave(rng, chains)
    utterances, gold_events = [], []
    for step, event in enumerate(events):
        _, text = rng.choice(TEMPLATES[event["kind"]])
        name = speakers[event["speaker"]]
        if event["kind"] == "chatter":
            gold_events.append({"step": step, "type": "chatter", "source": name})
        else:
            text = text.format(D=door_name(doors[event["door"]]), K=key_name(keys[event["key"]]))
            gold_events.append({"step": step, "type": event["kind"], "door": doors[event["door"]],
                                "key": keys[event["key"]], "polarity": event["kind"] != "negate", "source": name})
        utterances.append({"source": name, "text": text})
    _, question = rng.choice(TEMPLATES["question"])
    return {
        "episode_id": f"{session['session_id']}-e{index:02d}", "session_id": session["session_id"],
        "index": index, "world_rules": WORLD_RULES,
        "entities": {"doors": [door_name(d) for d in doors], "keys": [key_name(k) for k in keys]},
        "speakers": list(speakers), "utterances": utterances,
        "question": question.format(D=door_name(doors[door])), "answer": key_name(keys[truth]),
        "gold_events": gold_events, "gold_question": {"door": doors[door]},
        "split_keys": {"case": f"speakers-{count}", "answer_index": truth, "queried_door_index": door},
    }


def generate_session(rng: random.Random, part: str, session_id: str) -> dict:
    speakers = rng.sample(NAME_POOLS[part], SPEAKERS_PER_SESSION)
    profiles = _profile(rng, SPEAKERS_PER_SESSION)
    session = {"session_id": session_id, "speakers": speakers,
               "hidden": {"variant": "misspecified", "profiles": profiles,
                          "trajectory": _trajectory(rng, profiles), "drift": None}}
    deck: list[tuple[int, int]] = []
    episodes = []
    for index in range(SESSION_LENGTH):
        if not deck:
            deck.extend((d, k) for d in range(3) for k in range(3))
            rng.shuffle(deck)
        door, truth = deck.pop()
        episodes.append(_episode(rng, session, index, door, truth))
    session["episodes"] = episodes
    return session


def generate_part(part: str, seed: int, sessions: int | None = None) -> list[dict]:
    rng = random.Random(f"world-v2m|{part}|{seed + PARTS.index(part)}")
    count = sessions or SESSION_COUNTS[part]
    return [generate_session(rng, part, f"v2m-{part}-s{seed}-{n:03d}") for n in range(count)]


def _own(key: int, polarity: bool, truth: int, accuracy_pair: list[float]) -> float:
    """P(final claim | T) for a non-copied claim, including the polarity choice."""
    if polarity:
        p = accuracy_pair[0]
        return (1 - NEGATIVE_SHARE) * (p if key == truth else (1 - p) / 2)
    p = accuracy_pair[1]
    return NEGATIVE_SHARE * ((1 - p) if key == truth else p / 2)


class OracleMJudge(NonLearningJudge):
    """O2-M: exact posterior under the misspecified process with every type known."""

    gold_session = True

    def start_session(self, meta: dict) -> None:
        super().start_session(meta)
        self.hidden = meta["hidden"]
        self.index = {name: s for s, name in enumerate(self.speakers)}

    def act(self, episode: dict) -> str:
        claims, names = self.claims(episode)
        if not claims:
            return UNKNOWN
        accuracy = self.hidden["trajectory"][episode["index"]]
        profiles = self.hidden["profiles"]
        final = {self.index[s]: (key, polarity) for s, key, polarity in claims if s in self.index}
        weights = [1.0, 1.0, 1.0]
        for s, (key, polarity) in final.items():
            leader = profiles[s].get("leader") if profiles[s]["type"] == "copier" else None
            for truth in range(3):
                own = _own(key, polarity, truth, accuracy[s])
                if leader is not None and leader in final:
                    weights[truth] *= COPY_RATE * (final[leader] == (key, polarity)) + (1 - COPY_RATE) * own
                else:
                    weights[truth] *= own
        total = sum(weights)
        return decide([w / total for w in weights], names)
