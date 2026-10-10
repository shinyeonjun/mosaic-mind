"""Family split for step 2 (design/build-from-scratch-2026-10-10.md section 10), fixed before any training.

train: the concept learner is pretrained on these families only.
dev:   held-out families used to choose the adaptation recipe (steps, learning rate). Never the verdict.
test:  held-out families for the one verdict. Not generated, trained on or looked at before the recipe is frozen.

Seeds keep the streams apart: pretraining draws task(f, "train-i"), adaptation task(f, "adapt-<run>-i"),
evaluation task(f, "eval-i").
"""

import random

from cognitive_lab.concepts.generator import FAMILIES

PER_GROUP = {"objects": (2, 1), "geometry": (2, 1), "number": (1, 1), "agent": (1, 1), "compare": (1, 0)}  # (test, dev)


def split(seed: str = "mk1-c-step2") -> dict:
    rng = random.Random(seed)
    out = {"train": [], "dev": [], "test": []}
    for group, (n_test, n_dev) in PER_GROUP.items():
        names = sorted(n for n, (g, _) in FAMILIES.items() if g == group)
        rng.shuffle(names)
        out["test"] += names[:n_test]
        out["dev"] += names[n_test:n_test + n_dev]
        out["train"] += names[n_test + n_dev:]
    return out


SPLIT = split()

if __name__ == "__main__":
    for k, v in SPLIT.items():
        print(f"{k:5s} {len(v):2d} {v}")
