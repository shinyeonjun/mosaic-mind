"""Rule world v4: trust and chains in one problem.

Each session is a v2-H session (4 speakers with hidden accuracies, tone cues, feedback after
every episode). In each episode the speakers talk about the three base doors (빨간/파란/초록 문),
exactly as in v2-H. A board (안내판, always right) adds "same key" links that attach a short
chain of new doors to every base door; the question asks about the door at the end of the chain
that starts at the v2-H queried door (the anchor), L = 1-3 links away.
So: who to trust decides the anchor's key (stage-1 part), and following the links carries it
to the queried door (stage-3 part). The answer is the anchor's true key; after each episode
the world reveals it and which base door was the anchor (as v2 reveals the asked door's key).

Every base door gets a chain of 1-3 new doors, so chain shape alone does not single out the anchor.

python -m cognitive_lab.world4.composite     # sample + checks
"""

import random

from cognitive_lab.world2 import hedged
from cognitive_lab.world3.chains import DOORS, QUESTION, same_text

BASE_DOORS = DOORS[:3]
EXTRA_DOORS = DOORS[3:]
LENGTHS = (1, 2, 3)


def add_chains(session: dict, seed_text: str) -> dict:
    rng = random.Random(f"world-v4|chains|{seed_text}")
    for episode in session["episodes"]:
        anchor = hedged._queried(episode)
        if anchor not in BASE_DOORS:
            raise ValueError(f"unexpected anchor {anchor}")
        extras = rng.sample(EXTRA_DOORS, len(EXTRA_DOORS))
        length = LENGTHS[episode["index"] % len(LENGTHS)]
        links, chains = [], {}
        for base in BASE_DOORS:
            size = length if base == anchor else rng.choice(LENGTHS)
            chain = [base] + [extras.pop() for _ in range(size)]
            chains[base] = chain
            links += [(chain[i], chain[i + 1]) for i in range(size)]
        query = chains[anchor][-1]
        board = [same_text(rng, a, b) for a, b in links]
        rng.shuffle(board)
        episode["board"] = board
        episode["question"] = QUESTION.format(D=query)
        episode["composite"] = {"anchor": anchor, "query": query, "length": length,
                                "links": [list(link) for link in links]}
    session["world"] = "v4"
    return session


def generate_part(part: str, seed: int, sessions: int | None = None) -> list[dict]:
    return [add_chains(s, s["session_id"]) for s in hedged.generate_part(part, seed, sessions)]


def main() -> None:
    sessions = generate_part("train", 42, sessions=50)
    episodes = [e for s in sessions for e in s["episodes"]]
    lengths = {}
    for e in episodes:
        c = e["composite"]
        lengths[c["length"]] = lengths.get(c["length"], 0) + 1
        assert c["query"] in e["question"] and c["anchor"] in BASE_DOORS
        assert e["answer"] in ("해 열쇠", "달 열쇠", "별 열쇠")
    print("episodes", len(episodes), "lengths", lengths, "max board", max(len(e["board"]) for e in episodes))
    e = episodes[7]
    print(e["composite"], "|", e["answer"])
    print(" / ".join(f"{u['source']}: {u['text']}" for u in e["utterances"]))
    print("안내판:", " / ".join(e["board"]), "|", e["question"])


if __name__ == "__main__":
    main()
