"""Reading less: which thinker hypotheses a board sentence must be read against (MK1 reading optimisation).

Every rule-world sentence names one or two doors. Reading it against all 102 thinker hypotheses (36
"door opens with key" + 66 "two doors share a key") is the slowest step of MK1. Measured (2026-10-08,
design/mk1-integration.md, reading optimisation):
- "same key" hypotheses: only the pair a sentence names is needed. Every other pair gets a filler
  feature (the mean feature of sentences not naming that pair, from training sentences only) and the
  graduation scores do not move.
- "opens with key" hypotheses: all 36 are needed. With fillers for doors a sentence does not name, the
  thinker dropped from 0.975 to 0.956: it uses the noise level of "nobody said anything about this door"
  (a max over several sentences' readings), which one mean filler cannot imitate.
- speech (trust part): all 9 (3 doors x 3 keys) are needed; reading only the named door's 3 cost the
  curiosity stage 0.003.
So a board sentence is read 37 times instead of 102 (36 + the 1 pair it names, if any).

Finding the doors is string matching on door names: a hand rule that only works in the rule worlds.
Real text (stage D) would need a learned notion of what a sentence is about.
"""

import torch

from cognitive_lab.mk1.reader import ReaderService
from cognitive_lab.world2.integrated import CACHE_DIR
from cognitive_lab.world3 import chains
from cognitive_lab.world3.features import hypotheses, hypothesis_text

FILLERS = CACHE_DIR / "mk1-fillers.pt"
CALIBRATION_SENTENCES = 300


def doors_in(text: str) -> set[str]:
    return {d for d in chains.DOORS if d in text}


def board_relevant(text: str) -> list[int]:
    """Indices into the 102 thinker hypotheses to read: every fact hypothesis, and the pair the sentence names."""
    named = doors_in(text)
    return [j for j, h in enumerate(hypotheses()) if h[0] == "fact" or {h[1], h[2]} <= named]


def fillers(reader: ReaderService) -> torch.Tensor:
    """[102, 768]: per hypothesis, the mean feature of training sentences it is not read against."""
    if FILLERS.exists():
        return torch.load(FILLERS)["board"]
    texts = list(dict.fromkeys(x for e in chains.generate("train", 45) for x in e["utterances"]))[:CALIBRATION_SENTENCES]
    board = torch.zeros(len(hypotheses()), 768)
    for j, h in enumerate(hypotheses()):
        rest = [t for t in texts if j not in board_relevant(t)]
        if rest:
            board[j] = reader.features([(t, hypothesis_text(h)) for t in rest]).mean(0)
    torch.save({"board": board, "sentences": len(texts)}, FILLERS)
    return board
