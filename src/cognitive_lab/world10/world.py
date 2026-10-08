"""Rule world v10: a real article tells which key opens a door; a board of chains carries it on.

Built on v6 article sessions (4 sources with hidden accuracies, 40 KLUE questions, each reported by
1-3 sources as the true or the covered passage). Per question:
- board: every base door starts a "same key" chain of 1-3 new doors (as v7); half of the other base
  doors have their key stated;
- key labels (given by structure): 해/달/별 열쇠 are labelled with the true answer, the planted fake
  answer and another question's answer from the session, in random order;
- the article door: one base door; its key is the one labelled with the question's true answer.
Types (hidden from the system): C (the board states the article door's key; query: chain end),
A (query: the article door), AC (query: chain end; the article door's key only from the article).

World v11 (`board_accuracies`): each session draws a hidden board accuracy r; in C the board's statement
about the article door is right with probability r, else it names another key (design/mk1-integration.md, F).
With `switch`, the board's accuracy changes once, at a hidden question between the 10th and the 30th, to another
level (MK1 stage K: unexpected uncertainty, design/neuroscience-review-2026-10-09.md, 4); with `switch_to`, the
new level is drawn from those (e.g. a trusted board that starts misleading: 1.0/0.75 -> 0.25/0.1).

python -m cognitive_lab.world10.world      # sample
"""

import random

from cognitive_lab.world3 import chains
from cognitive_lab.world4 import composite
from cognitive_lab.world5 import klue
from cognitive_lab.world6 import world as v6
from cognitive_lab.world6.reports import covered, load

TYPES = ("C", "A", "AC")


def generate(part: str, seed: int, count: int | None = None, board_accuracies: tuple | None = None,
             switch: bool = False, switch_to: tuple | None = None) -> list[dict]:
    """MK1 sessions: {"sources", "linked", "episodes": [...]}; `answer` is the query door's key."""
    questions = {q["guid"]: q for q in klue.load(part)}
    records = load()
    sessions = []
    for s in v6.generate(part, seed, count):
        rng = random.Random(f"world-v10|{s['session_id']}")
        golds = [questions[e["guid"]]["answers"][0] for e in s["episodes"]]
        board_accuracy = rng.choice(board_accuracies) if board_accuracies else 1.0
        switch_at, accuracy_after = None, board_accuracy
        if switch:
            switch_at = rng.randint(10, 30)
            accuracy_after = rng.choice([a for a in (switch_to or board_accuracies) if a != board_accuracy])
        episodes = []
        for t, e in enumerate(s["episodes"]):
            q, record = questions[e["guid"]], records[e["guid"]]
            kind = rng.choice(TYPES)
            others = [g for g in golds if klue.normalize(g) not in (klue.normalize(q["answers"][0]), klue.normalize(record["fake"]))]
            labels = [q["answers"][0], record["fake"], rng.choice(others)]
            rng.shuffle(labels)
            legend = dict(zip(chains.KEYS, labels))
            truths = {d: rng.choice(chains.KEYS) for d in composite.BASE_DOORS}
            article_door = rng.choice(composite.BASE_DOORS)
            truths[article_door] = chains.KEYS[labels.index(q["answers"][0])]
            extras = rng.sample(composite.EXTRA_DOORS, len(composite.EXTRA_DOORS))
            links, chain_of = [], {}
            for base in composite.BASE_DOORS:
                size = rng.choice(composite.LENGTHS)
                chain_of[base] = [base] + [extras.pop() for _ in range(size)]
                links += [(chain_of[base][i], chain_of[base][i + 1]) for i in range(size)]
            stated = [d for d in composite.BASE_DOORS if d != article_door and rng.random() < 0.5]
            said = dict(truths)
            board_right = None
            if kind == "C":
                stated.append(article_door)
                if board_accuracies:
                    board_right = rng.random() < (accuracy_after if switch_at is not None and t >= switch_at else board_accuracy)
                    if not board_right:
                        said[article_door] = rng.choice([k for k in chains.KEYS if k != truths[article_door]])
            board = [chains.same_text(rng, a, b) for a, b in links] + [chains.fact_text(rng, d, said[d]) for d in stated]
            rng.shuffle(board)
            query = article_door if kind == "A" else chain_of[article_door][-1]
            false_passage = covered(q["context"], q["answers"][0], record["fake"])
            episodes.append({
                "tag": kind, "board_right": board_right, "utterances": [], "board": board, "query": query, "keys": list(chains.KEYS),
                "answer": truths[article_door], "anchor": article_door, "answers": {},
                "legend": legend, "article": {"door": article_door, "question": q["question"], "answers": q["answers"],
                                              "reports": [{"source": r["speaker"],
                                                           "passage": q["context"] if r["truthful"] else false_passage}
                                                          for r in e["reports"]]}})
        sessions.append({"speakers": [], "sources": v6.SPEAKERS, "linked": True, "board_accuracy": board_accuracy,
                         "switch_at": switch_at, "board_accuracy_after": accuracy_after,
                         "episodes": episodes})
    return sessions


def main() -> None:
    sessions = generate("test", 42, count=2)
    e = sessions[0]["episodes"][0]
    print(e["tag"], "query:", e["query"], "| article door:", e["article"]["door"], "| answer:", e["answer"])
    print("legend:", e["legend"], "| question:", e["article"]["question"], "| gold:", e["article"]["answers"])
    print("board:", " / ".join(e["board"]))


if __name__ == "__main__":
    main()
