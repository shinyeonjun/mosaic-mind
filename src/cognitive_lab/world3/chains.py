"""Rule world v3: which key opens the queried door, when the clue is a chain of "same key" links.

An episode has one fact chain ending at the queried door q:
    anchor --same-- d1 --same-- ... --same-- q          (L links, L = chain length)
plus a fact "anchor opens with key K", and exactly 2 distractor chains of 2-3 doors, each with
its own fact. The three facts name the three keys once each, and every fact door also has a
link, so neither how often a key is mentioned nor "the fact door that has a link" says
anything about the answer. (Both were shortcuts in earlier versions; fixed 2026-10-08.)
Sentences are shuffled, so reading them once in order does not follow the chain.
Unanswerable episodes (20%) remove one link of the main chain: the answer is then 모름.

Train/validation: L in 1-3. Development: L in 4-6 (design decisions only, separate stream).
Test: L in 1-6 (4-6 never trained on). With L links, information needs
L steps to move from the anchor to q, so longer chains need more thinking.

python -m cognitive_lab.world3.chains      # sample episodes + solver check
"""

import random

from cognitive_lab.world.templates import TEMPLATES, UNKNOWN

DOORS = tuple(f"{c} 문" for c in ("빨간", "파란", "초록", "노란", "보라", "하얀", "검은", "주황", "분홍", "회색",
                                   "갈색", "남색"))
KEYS = tuple(f"{k} 열쇠" for k in ("해", "달", "별"))
SAME_TEMPLATES = (
    "{A}은 {B}과 같은 열쇠로 열려.",
    "{A}과 {B}은 열쇠가 같아.",
    "{B}을 여는 열쇠로 {A}도 열 수 있어.",
    "{A} 열쇠는 {B} 열쇠랑 똑같아요.",
    "{A}이랑 {B}은 같은 열쇠를 써.",
)
FACT_TEMPLATES = tuple(t for _, t in TEMPLATES["assert"])
QUESTION = "{D}은 무슨 열쇠로 열어?"
UNANSWERABLE_SHARE = 0.2
LENGTHS = {"train": (1, 2, 3), "validation": (1, 2, 3), "development": (4, 5, 6), "test": (1, 2, 3, 4, 5, 6)}
COUNTS = {"train": 20000, "validation": 2000, "development": 1500, "test": 3000}


def fact_text(rng: random.Random, door: str, key: str) -> str:
    return rng.choice(FACT_TEMPLATES).format(D=door, K=key)


def same_text(rng: random.Random, a: str, b: str) -> str:
    if rng.random() < 0.5:
        a, b = b, a
    return rng.choice(SAME_TEMPLATES).format(A=a, B=b)


def episode(rng: random.Random, length: int, answerable: bool, episode_id: str) -> dict:
    doors = rng.sample(DOORS, len(DOORS))
    chain = doors[:length + 1]  # chain[0] = anchor, chain[-1] = queried door
    rest = doors[length + 1:]
    key = rng.choice(KEYS)
    links = [(chain[i], chain[i + 1]) for i in range(length)]
    if not answerable:
        links.pop(rng.randrange(len(links)))
    facts = [(chain[0], key)]
    sizes = rng.choice(((2, 2), (2, 3), (3, 2)))
    groups = [rest[:sizes[0]], rest[sizes[0]:sum(sizes)]]
    assert len(groups[1]) == sizes[1], "not enough doors for the distractor chains"
    other_keys = rng.sample([k for k in KEYS if k != key], 2)
    for group, group_key in zip(groups, other_keys):
        facts.append((group[0], group_key))
        links += [(group[i], group[i + 1]) for i in range(len(group) - 1)]
    sentences = [{"kind": "fact", "doors": (d,), "key": k, "text": fact_text(rng, d, k)} for d, k in facts]
    sentences += [{"kind": "same", "doors": (a, b), "text": same_text(rng, a, b)} for a, b in links]
    rng.shuffle(sentences)
    query = chain[-1]
    return {"episode_id": episode_id, "doors": list(DOORS), "keys": list(KEYS),
            "utterances": [s["text"] for s in sentences], "question": QUESTION.format(D=query),
            "query": query, "answer": key if answerable else UNKNOWN,
            "length": length, "answerable": answerable, "gold": sentences}


def generate(part: str, seed: int, count: int | None = None) -> list[dict]:
    rng = random.Random(f"world-v3|{part}|{seed}")
    lengths = LENGTHS[part]
    total = count or COUNTS[part]
    return [episode(rng, lengths[n % len(lengths)], rng.random() >= UNANSWERABLE_SHARE, f"v3-{part}-s{seed}-{n:05d}")
            for n in range(total)]


def solve(gold: list[dict], query: str) -> str:
    """Union-find over "same" links; the key of the query's group if exactly one is stated."""
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for s in gold:
        if s["kind"] == "same":
            parent[find(s["doors"][0])] = find(s["doors"][1])
    keys = {s["key"] for s in gold if s["kind"] == "fact" and find(s["doors"][0]) == find(query)}
    return keys.pop() if len(keys) == 1 else UNKNOWN


def main() -> None:
    for part in ("train", "development", "test"):
        episodes = generate(part, 42)
        wrong = sum(solve(e["gold"], e["query"]) != e["answer"] for e in episodes)
        print(f"{part}: {len(episodes)} episodes, solver mismatches {wrong}, "
              f"max utterances {max(len(e['utterances']) for e in episodes)}")
    for e in generate("test", 42, 6)[3:6]:
        print(e["length"], e["answerable"], e["answer"], "|", " / ".join(e["utterances"]), "|", e["question"])


if __name__ == "__main__":
    main()
