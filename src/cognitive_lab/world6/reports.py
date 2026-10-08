"""Report versions for rule world v6 (design/world-v6-real-trust.md), cached.

For every answerable question in our KLUE splits: the planted false answer is the reading
specialist's best alternative span from the same passage (not overlapping the gold answer,
0.4-2.5x its length, present verbatim), and the false report is the passage with every mention
of the gold answer covered by that span. Checked on 300 validation questions: the specialist
returns the planted answer 81% of the time, the gold 8%, and its margin barely separates the
versions (AUC 0.58). (Swapping the two spans instead left the gold elsewhere in the passage:
the specialist still read it 35% of the time, which squeezed the sources' effective accuracies
to 0.42-0.69. Kept as world-v6_reports-swap.jsonl.)
The specialist's reading of both versions is cached, so sessions can be composed freely.

python -m cognitive_lab.world6.reports
"""

import json

import torch

from cognitive_lab.world5 import klue
from cognitive_lab.world5.reader_qa import Reader

CACHE = klue.PROJECT_ROOT / "artifacts" / "cache" / "world-v6_reports-cover.jsonl"
OLD_CACHE = klue.PROJECT_ROOT / "artifacts" / "cache" / "world-v6_reports-swap.jsonl"


def planted(reader: Reader, question: dict) -> str | None:
    gold = question["answers"][0]
    g = klue.normalize(gold)
    candidates, _ = reader.spans(question["question"], question["context"], k=20, max_tokens=12)
    for _, text in candidates:
        t = klue.normalize(text)
        if t and g not in t and t not in g and 0.4 <= len(text) / len(gold) <= 2.5 and text in question["context"]:
            return text
    return None


def covered(context: str, gold: str, fake: str) -> str:
    return context.replace(gold, fake)


def load(path=CACHE) -> dict[str, dict]:
    return {r["guid"]: r for r in map(json.loads, path.read_text(encoding="utf-8").splitlines()) if r}


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reader = Reader(device)
    done = load() if CACHE.exists() else {}
    earlier = load(OLD_CACHE) if OLD_CACHE.exists() else {}  # reuse planted answers and true reports
    with CACHE.open("a", encoding="utf-8") as out:
        for part in ("train", "validation", "test"):
            kept = skipped = 0
            for q in klue.load(part):
                if q["impossible"]:
                    continue
                if q["guid"] in done:
                    kept += 1
                    continue
                old = earlier.get(q["guid"])
                fake = old["fake"] if old else planted(reader, q)
                if fake is None:
                    skipped += 1
                    continue
                record = {"guid": q["guid"], "part": part, "question": q["question"], "answers": q["answers"],
                          "fake": fake,
                          "true_report": old["true_report"] if old else reader.read(q["question"], q["context"]),
                          "false_report": reader.read(q["question"], covered(q["context"], q["answers"][0], fake))}
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                kept += 1
            out.flush()
            print(f"{part}: {kept} questions with both versions, {skipped} without a plausible alternative", flush=True)


if __name__ == "__main__":
    main()
