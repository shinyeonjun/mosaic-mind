"""P1-rule: a hand-written Korean parser feeding the exact solver.

The parser turns each utterance into at most one event (door, key, polarity) and
the shared solver makes the judgment, so P1-rule differs from P1-gold (S1) only
in how it reads text.

Marker lists were written only from the phrasing split's *train* templates, so
the phrasing split measures how a hand-written parser handles unseen wording.
Unseen templates in iid/composition/entity test sets can still break it; that
error is part of what P1-rule's score reports.
"""

from cognitive_lab.world.solver import solve
from cognitive_lab.world.templates import UNKNOWN

# From train negate templates: 안 열려 / 열리지 않는다 / 못 열어요 / 안 열려요 / 여는 게 아니다
NEGATION_MARKERS = ("안 ", "않", "못 ", "아니다")


def _strip_suffix(name: str, suffix: str) -> str:
    return name.removesuffix(suffix)


def _fact_clause(text: str, doors: list[str], keys: list[str]) -> str:
    """Keep the last sentence naming a door and a key; correction prefixes come earlier."""
    sentences = [part.strip() for part in text.split(".") if part.strip()]
    factual = [s for s in sentences if any(d in s for d in doors) and any(k in s for k in keys)]
    return factual[-1] if factual else text


def parse_utterance(text: str, doors: list[str], keys: list[str]) -> tuple[str, str, bool] | None:
    """Return (door label, key label, polarity) or None when no single fact is stated."""
    clause = _fact_clause(text, doors, keys)
    door_hits = [door for door in doors if door in clause]
    key_hits = [key for key in keys if key in clause]
    if len(door_hits) != 1 or len(key_hits) != 1:
        return None
    polarity = not any(marker in clause for marker in NEGATION_MARKERS)
    return _strip_suffix(door_hits[0], " 문"), _strip_suffix(key_hits[0], " 열쇠"), polarity


def parse_episode(episode: dict) -> tuple[list[dict], str | None]:
    doors, keys = episode["entities"]["doors"], episode["entities"]["keys"]
    events = []
    for step, utterance in enumerate(episode["utterances"]):
        parsed = parse_utterance(utterance["text"], doors, keys)
        if parsed is None:
            events.append({"step": step, "type": "chatter", "source": utterance["source"]})
            continue
        door, key, polarity = parsed
        events.append({
            "step": step,
            "type": "assert" if polarity else "negate",
            "door": door,
            "key": key,
            "polarity": polarity,
            "source": utterance["source"],
        })
    asked = [door for door in doors if door in episode["question"]]
    question_door = _strip_suffix(asked[0], " 문") if len(asked) == 1 else None
    return events, question_door


def rule_pipeline_agent(episode: dict) -> str:
    events, door = parse_episode(episode)
    if door is None:
        return UNKNOWN
    keys = [_strip_suffix(key, " 열쇠") for key in episode["entities"]["keys"]]
    answer = solve(events, door, keys)
    return UNKNOWN if answer == UNKNOWN else f"{answer} 열쇠"
