"""Rule world v8 episodes: v3 chains whose links come in one of three families.

  same      v3 "same key" templates
  same-new  "same key" in new phrasings (every link of the episode)
  next      main-chain links are "next key" links (each with probability 0.5, at least one) going
            from the anchor towards the queried door; other links are v3 "same" links
The answer is the anchor's key moved forward once per "next" link on the way (해 -> 달 -> 별 -> 해).
Like v3: 12 doors, three facts naming the three keys once each, two distractor chains of 2-3
doors, shuffled sentences, 20% unanswerable (one main link removed).

python -m cognitive_lab.world8.chains
"""

import random

from cognitive_lab.world.templates import UNKNOWN
from cognitive_lab.world3.chains import DOORS, KEYS, QUESTION, SAME_TEMPLATES, fact_text
from cognitive_lab.world8.language import NEXT_TEMPLATES, SAME_NEW_TEMPLATES, SWAP_TEMPLATES, next_key, swap_key

FAMILIES = ("same", "same-new", "next")
V9_FAMILIES = ("same", "same-new", "next", "swap")


def link_text(rng: random.Random, a: str, b: str, relation: str) -> str:
    if relation == "next":  # b's key is the one after a's
        return rng.choice(NEXT_TEMPLATES).format(A=a, B=b)
    if relation == "swap":  # b's key is a's with 해 and 별 swapped
        return rng.choice(SWAP_TEMPLATES).format(A=a, B=b)
    if rng.random() < 0.5:
        a, b = b, a
    return rng.choice(SAME_NEW_TEMPLATES if relation == "same-new" else SAME_TEMPLATES).format(A=a, B=b)


def episode(rng: random.Random, family: str, length: int, answerable: bool, episode_id: str) -> dict:
    doors = rng.sample(DOORS, len(DOORS))
    chain, rest = doors[:length + 1], doors[length + 1:]
    key = rng.choice(KEYS)
    base_relation = "same-new" if family == "same-new" else "same"
    relations = [base_relation] * length
    if family in ("next", "swap"):
        relations = [family if rng.random() < 0.5 else "same" for _ in range(length)]
        if family not in relations:
            relations[rng.randrange(length)] = family
    links = [(chain[i], chain[i + 1], relations[i]) for i in range(length)]
    answer = key
    for relation in relations:
        answer = next_key(answer) if relation == "next" else (swap_key(answer) if relation == "swap" else answer)
    if not answerable:
        links.pop(rng.randrange(len(links)))
    facts = [(chain[0], key)]
    sizes = rng.choice(((2, 2), (2, 3), (3, 2)))
    groups = [rest[:sizes[0]], rest[sizes[0]:sum(sizes)]]
    for group, group_key in zip(groups, rng.sample([k for k in KEYS if k != key], 2)):
        facts.append((group[0], group_key))
        links += [(group[i], group[i + 1], base_relation) for i in range(len(group) - 1)]
    sentences = [{"kind": "fact", "doors": (d,), "key": k, "text": fact_text(rng, d, k)} for d, k in facts]
    sentences += [{"kind": r, "doors": (a, b), "text": link_text(rng, a, b, r)} for a, b, r in links]
    rng.shuffle(sentences)
    return {"episode_id": episode_id, "family": family, "utterances": [s["text"] for s in sentences],
            "question": QUESTION.format(D=chain[-1]), "query": chain[-1],
            "answer": answer if answerable else UNKNOWN, "length": length, "answerable": answerable,
            "gold": sentences, "relations": relations}


def generate(family: str, part: str, seed: int, count: int) -> list[dict]:
    rng = random.Random(f"world-v8|{family}|{part}|{seed}")
    return [episode(rng, family, (1, 2, 3)[n % 3], rng.random() >= 0.2, f"v8-{family}-{part}-s{seed}-{n:05d}")
            for n in range(count)]


if __name__ == "__main__":
    for family in FAMILIES:
        e = generate(family, "test", 0, 3)[2]
        print(family, e["relations"], e["answer"], "|", " / ".join(e["utterances"]), "|", e["question"])


def door_keys(gold: list[dict]) -> dict[str, str]:
    """Every door's key implied by the sentences (same: equal, next: one after), or 모름."""
    known = {s["doors"][0]: s["key"] for s in gold if s["kind"] == "fact"}
    links = [s for s in gold if s["kind"] in ("same", "same-new", "next", "swap")]
    changed = True
    while changed:
        changed = False
        for s in links:
            a, b = s["doors"]
            if s["kind"] == "next":
                pairs = ((a, b, next_key), (b, a, lambda k: KEYS[(KEYS.index(k) - 1) % len(KEYS)]))
            elif s["kind"] == "swap":
                pairs = ((a, b, swap_key), (b, a, swap_key))
            else:
                pairs = ((a, b, lambda k: k), (b, a, lambda k: k))
            for src, dst, f in pairs:
                if src in known and dst not in known:
                    known[dst] = f(known[src])
                    changed = True
    return {d: known.get(d, UNKNOWN) for d in DOORS}
