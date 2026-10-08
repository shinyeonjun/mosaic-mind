"""Session generator for rule world v2.

A session is 40 episodes with 4 speakers whose hidden accuracies persist (and, in drift
sessions, one accuracy changes from episode 21). Every statement is generated from the
speaker's accuracy and the door's hidden true key, so the Bayes posterior with true
accuracies (oracle O2) is the exact ideal judgment. Phrasing reuses v1 templates.
"""

import random

from cognitive_lab.world.generator import _interleave
from cognitive_lab.world.templates import ENTITY_SETS, TEMPLATES, door_name, key_name

PARTS = ("train", "validation", "test")
SESSION_LENGTH = 40
SPEAKERS_PER_SESSION = 4
ACCURACY_LEVELS = (0.95, 0.75, 0.5, 0.2)
DRIFT_SHARE = 0.3
DRIFT_AT = 20  # zero-based episode index where the drifting speaker's accuracy changes
MIN_DRIFT = 0.44  # 0.5 <-> 0.95 counts (float 0.95 - 0.5 < 0.45)
MIN_UTTERANCES, MAX_UTTERANCES = 4, 10
# Probability of 0, 1, 2, 3 speakers talking about the queried door.
SPEAKER_COUNT_WEIGHTS = (0.10, 0.30, 0.35, 0.25)
STATEMENT_WEIGHTS = {"assert": 0.65, "negate": 0.20, "corrected": 0.15}

# Disjoint name pools so no name can carry a learned trust prior across parts.
NAME_POOLS = {
    "train": ("민수", "서연", "도윤", "하린", "지호", "예린", "시우", "수아", "준서", "채원",
              "현우", "지안", "유준", "다은", "건우", "소율", "우진", "하은", "선우", "윤서"),
    "validation": ("태윤", "아린", "민재", "서아", "은호", "나연", "재원", "가은", "승우", "유나"),
    "test": ("시현", "다인", "로운", "세아", "이준", "채아", "주원", "라희", "도하", "예서",
             "하람", "연우", "서진", "보민", "율희", "지환", "소윤", "태오", "리아", "온유"),
}
SESSION_COUNTS = {"train": 300, "validation": 40, "test": 100}

# Scaling experiments (speakers != 4). The 4-speaker world keeps the distributions above
# bit-for-bit. With S speakers, each one talks about the queried door with probability
# TALK_RATE (the 4-speaker mean, 1.75 / 4), so evidence per speaker stays comparable.
TALK_RATE = 0.4375
# "sparse" scaling keeps the expected number of claims per episode at 1.75 instead, so
# each speaker appears rarely and must be judged from little evidence.
CLAIMS_PER_EPISODE = 1.75
EXTRA_NAMES = {"train": (), "test": (),
               "validation": ("하율", "도현", "서윤", "지우", "예준", "수빈", "민서", "준혁", "채윤", "시온")}


def name_pool(part: str, speakers: int) -> tuple[str, ...]:
    pool = NAME_POOLS[part] if speakers <= len(NAME_POOLS[part]) else NAME_POOLS[part] + EXTRA_NAMES[part]
    if speakers > len(pool):
        raise ValueError(f"{part} name pool has {len(pool)} names; {speakers} speakers requested")
    return pool

WORLD_RULES = (
    "모든 문은 정확히 한 종류의 열쇠로 열린다. 사람마다 말이 맞을 확률이 다르고, 누가 믿을 만한지는 "
    "알려주지 않는다. 같은 사람이 정정하면 정정한 말이 그 사람의 말이다. 매 문제 뒤에 진짜 열쇠를 알려준다. "
    "확실하지 않으면 '모름'이라고 답할 수 있다."
)


def _statement(rng: random.Random, accuracy: float, truth: int, negative: bool) -> int:
    """Key named by a statement: true with probability `accuracy`."""
    wrong = [k for k in range(3) if k != truth]
    if negative:
        return rng.choice(wrong) if rng.random() < accuracy else truth
    return truth if rng.random() < accuracy else rng.choice(wrong)


def _speaker_chain(rng: random.Random, speaker: int, accuracy: float, door: int, truth: int) -> list[dict]:
    kind = rng.choices(list(STATEMENT_WEIGHTS), weights=list(STATEMENT_WEIGHTS.values()))[0]
    if kind == "negate":
        return [{"kind": "negate", "speaker": speaker, "door": door,
                 "key": _statement(rng, accuracy, truth, negative=True)}]
    final = _statement(rng, accuracy, truth, negative=False)
    if kind == "assert":
        return [{"kind": "assert", "speaker": speaker, "door": door, "key": final}]
    retracted = rng.choice([k for k in range(3) if k != final])
    return [{"kind": "assert", "speaker": speaker, "door": door, "key": retracted},
            {"kind": "correct", "speaker": speaker, "door": door, "key": final}]


def accuracies_at(session: dict, index: int) -> list[float]:
    accuracies = list(session["hidden"]["accuracies"])
    drift = session["hidden"]["drift"]
    if drift and index >= drift["at"]:
        accuracies[drift["speaker"]] = drift["to"]
    return accuracies


def _episode(rng: random.Random, session: dict, index: int, door: int, truth: int,
             talk_rate: float = TALK_RATE) -> dict:
    entities = ENTITY_SETS["base"]
    doors, keys = entities["doors"], entities["keys"]
    accuracies = accuracies_at(session, index)
    speakers = session["speakers"]
    size = len(speakers)

    if size == SPEAKERS_PER_SESSION:
        count = rng.choices(range(4), weights=SPEAKER_COUNT_WEIGHTS)[0]
        talkers = rng.sample(range(size), count)
    else:
        talkers = [s for s in range(size) if rng.random() < talk_rate]
        talkers = rng.sample(talkers, len(talkers))
        count = len(talkers)
    chains = [_speaker_chain(rng, s, accuracies[s], door, truth) for s in talkers]
    used = sum(len(chain) for chain in chains)
    total = max(used, rng.randint(MIN_UTTERANCES, MAX_UTTERANCES))
    other_truth = {d: rng.randrange(3) for d in range(3) if d != door}
    while used < total:
        if rng.random() < 0.3:
            chains.append([{"kind": "chatter", "speaker": rng.randrange(size)}])
        else:
            other = rng.choice(list(other_truth))
            speaker = rng.randrange(size)
            negative = rng.random() < 0.25
            chains.append([{"kind": "negate" if negative else "assert", "speaker": speaker, "door": other,
                            "key": _statement(rng, accuracies[speaker], other_truth[other], negative)}])
        used += 1
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
                                "key": keys[event["key"]], "polarity": event["kind"] != "negate",
                                "source": name})
        utterances.append({"source": name, "text": text})

    _, question = rng.choice(TEMPLATES["question"])
    return {
        "episode_id": f"{session['session_id']}-e{index:02d}",
        "session_id": session["session_id"],
        "index": index,
        "world_rules": WORLD_RULES,
        "entities": {"doors": [door_name(d) for d in doors], "keys": [key_name(k) for k in keys]},
        "speakers": list(speakers),
        "utterances": utterances,
        "question": question.format(D=door_name(doors[door])),
        "answer": key_name(keys[truth]),
        "gold_events": gold_events,
        "gold_question": {"door": doors[door]},
        "split_keys": {"case": f"speakers-{count}", "answer_index": truth, "queried_door_index": door},
        # Truth of the other doors (never shown to agents); lets derived worlds label distractor claims.
        "hidden_other_truths": {door_name(doors[d]): key_name(keys[k]) for d, k in other_truth.items()},
    }


def generate_session(rng: random.Random, part: str, session_id: str,
                     speakers_per_session: int = SPEAKERS_PER_SESSION, sparse: bool = False) -> dict:
    speakers = rng.sample(name_pool(part, speakers_per_session), speakers_per_session)
    while True:
        accuracies = [rng.choice(ACCURACY_LEVELS) for _ in range(speakers_per_session)]
        if len(set(accuracies)) > 1:
            break
    drift = None
    if rng.random() < DRIFT_SHARE:
        speaker = rng.randrange(speakers_per_session)
        options = [level for level in ACCURACY_LEVELS if abs(level - accuracies[speaker]) >= MIN_DRIFT]
        drift = {"speaker": speaker, "from": accuracies[speaker], "to": rng.choice(options), "at": DRIFT_AT}
    session = {"session_id": session_id, "speakers": speakers,
               "hidden": {"accuracies": accuracies, "drift": drift}}
    deck: list[tuple[int, int]] = []
    episodes = []
    for index in range(SESSION_LENGTH):
        if not deck:
            deck.extend((d, k) for d in range(3) for k in range(3))
            rng.shuffle(deck)
        door, truth = deck.pop()
        rate = CLAIMS_PER_EPISODE / speakers_per_session if sparse else TALK_RATE
        episodes.append(_episode(rng, session, index, door, truth, rate))
    session["episodes"] = episodes
    return session


def generate_part(part: str, seed: int, sessions: int | None = None,
                  speakers: int = SPEAKERS_PER_SESSION, sparse: bool = False) -> list[dict]:
    if sparse and speakers == SPEAKERS_PER_SESSION:
        raise ValueError("the 4-speaker world already has 1.75 claims per episode; sparse applies to scaling runs")
    scale = "" if speakers == SPEAKERS_PER_SESSION else f"|speakers-{speakers}" + ("|sparse" if sparse else "")
    rng = random.Random(f"world-v2|{part}|{seed + PARTS.index(part)}{scale}")
    count = sessions or SESSION_COUNTS[part]
    prefix = "v2" if speakers == SPEAKERS_PER_SESSION else f"v2s{speakers}" + ("sp" if sparse else "")
    return [generate_session(rng, part, f"{prefix}-{part}-s{seed}-{n:03d}", speakers, sparse) for n in range(count)]
