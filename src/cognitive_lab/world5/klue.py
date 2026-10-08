"""KLUE-MRC splits and scoring for ladder stage 5 (design/world-v5-real-korean.md).

Splits (fixed): train 3,000 and validation 500 from KLUE train (disjoint); test 2,000 from
KLUE validation. About 31% of questions have no answer in the passage (is_impossible).

Scoring per question: answerable -> +1 right, -1 wrong, 0 for 모름; unanswerable -> +1 for
모름, -1 for any answer. "Right" (after removing spaces and punctuation), against any gold text:
exact match; or the gold inside the prediction with at most 3 extra characters (particles such as
"에", "이다"); or the prediction inside the gold covering at least 60% of it. (Character F1 was
rejected: it scores "2013년" right for "2014년".)
"""

import random
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data" / "klue-mrc" / "mrc"
UNKNOWN = "모름"
SIZES = {"train": 3000, "validation": 500, "test": 2000}
EXTRA_CHARACTERS = 3
MIN_COVERAGE = 0.6


def _read(name: str) -> list[dict]:
    import pyarrow.parquet as pq

    return pq.read_table(DATA_DIR / f"{name}-00000-of-00001.parquet").to_pylist()


def load(part: str) -> list[dict]:
    if part == "test":
        rows = _read("validation")
        random.Random("world-v5|test").shuffle(rows)
        rows = rows[:SIZES["test"]]
    else:
        rows = _read("train")
        random.Random("world-v5|train").shuffle(rows)
        rows = rows[:SIZES["train"]] if part == "train" else rows[SIZES["train"]:SIZES["train"] + SIZES["validation"]]
    return [{"guid": r["guid"], "context": r["context"], "question": r["question"],
             "answers": list(r["answers"]["text"]), "impossible": bool(r["is_impossible"])} for r in rows]


def normalize(text: str) -> str:
    return re.sub(r"[\s\W_]+", "", text)


def is_right(prediction: str, golds: list[str]) -> bool:
    p = normalize(prediction)
    if not p:
        return False
    for gold in golds:
        g = normalize(gold)
        if p == g or (g in p and len(p) - len(g) <= EXTRA_CHARACTERS) or (p in g and len(p) >= MIN_COVERAGE * len(g)):
            return True
    return False


def score(prediction: str, question: dict) -> int:
    if question["impossible"]:
        return 1 if prediction == UNKNOWN else -1
    if prediction == UNKNOWN:
        return 0
    return 1 if is_right(prediction, question["answers"]) else -1


if __name__ == "__main__":
    for part in ("train", "validation", "test"):
        rows = load(part)
        print(part, len(rows), "impossible", sum(r["impossible"] for r in rows))
    train, test = {r["guid"] for r in load("train")}, {r["guid"] for r in load("test")}
    validation = {r["guid"] for r in load("validation")}
    assert not (train & validation) and not (train & test) and not (validation & test)
    assert is_right("2014년에", ["2014년"]) and is_right("2014", ["2014년"]) and is_right("2014년 ", ["2014년"])
    assert not is_right("2013년", ["2014년"]) and not is_right("년", ["2014년"])
    assert not is_right("BMW 740Li 25주년 에디션을 출시한 회사", ["뉴 740Li 25주년"])
    print("splits disjoint; scoring checks pass")
