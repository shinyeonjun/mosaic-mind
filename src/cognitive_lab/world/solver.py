"""Exact reference solver (S1) over gold events; it defines the world's semantics."""

from cognitive_lab.world.templates import SOURCES, UNKNOWN


def solve(gold_events: list[dict], door: str, keys: list[str]) -> str:
    """Return the key label for `door`, or UNKNOWN.

    Each source's latest positive claim about the door replaces its earlier one.
    Sources are read from most to least trusted: a source's negations exclude keys,
    its positive claim wins unless a more trusted source already excluded that key,
    and two exclusions leave the remaining key by the world axiom.
    """
    positive: dict[str, str] = {}
    negative: dict[str, set[str]] = {}
    for event in gold_events:
        if event["type"] == "chatter" or event["door"] != door:
            continue
        if event["polarity"]:
            positive[event["source"]] = event["key"]
        else:
            negative.setdefault(event["source"], set()).add(event["key"])

    excluded: set[str] = set()
    for source in reversed(SOURCES):
        excluded |= negative.get(source, set())
        claimed = positive.get(source)
        if claimed is not None and claimed not in excluded:
            return claimed
        if len(excluded) == len(keys) - 1:
            break
    remaining = [key for key in keys if key not in excluded]
    return remaining[0] if len(remaining) == 1 else UNKNOWN
