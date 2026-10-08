"""Real Korean articles on the MK1 board (stage D, design/mk1-integration.md).

World v6 as MK1 meets it: per question, 1-3 of 4 sources each hand over a passage, either the true
article or a version where every mention of the answer is covered by a planted span. MK1 reads every
passage itself with the reading specialist (world5/reader_qa.py), groups equal answers into candidates,
and the real-text trust part (world6/trust.py) picks one or says 모름. After each question the world
reveals which candidates were right (feedback, as in v6).

Readings are remembered (question, passage) -> (answer, span score, no-answer score). The memory can be
seeded from the v6 cache, which was read in bf16: then MK1 sees exactly what the v6 trust part was
trained and tested on (a wiring check). Without seeding, MK1 reads live in fp32.
"""

import json

import torch

from cognitive_lab.world2.integrated import CACHE_DIR
from cognitive_lab.world5 import klue
from cognitive_lab.world6.reports import covered, load

QA_MEMORY = CACHE_DIR / "mk1-qa-memory.jsonl"


def article_sessions(sessions: list[dict], part: str) -> list[dict]:
    """World v6 sessions with passages instead of readings. `answers` is used only for feedback/scoring."""
    questions = {q["guid"]: q for q in klue.load(part)}
    records = load()
    out = []
    for s in sessions:
        episodes = []
        for e in s["episodes"]:
            q, record = questions[e["guid"]], records[e["guid"]]
            false_passage = covered(q["context"], q["answers"][0], record["fake"])
            episodes.append({"question": q["question"], "answers": q["answers"], "tag": "article",
                             "reports": [{"source": r["speaker"], "passage": q["context"] if r["truthful"] else false_passage}
                                         for r in e["reports"]]})
        out.append({"sources": 4, "episodes": episodes})
    return out


class ArticleReading:
    """The reading specialist with a reading memory (live, optionally seeded from the v6 bf16 cache).
    New passages are read in batches (`Reader.read_many`). `precision`: "fp32" (default) or "fp16" (fast
    mode: 2.3x, 11 of 2,313 test answers differ from fp32, no batch dependence; design/mk1-integration.md)."""

    def __init__(self, device: torch.device, seed_from_cache: bool = False, store=QA_MEMORY, precision: str = "fp32"):
        self.device, self.precision = device, precision
        if store is not None and precision != "fp32":
            store = store.with_name(f"{store.stem}-{precision}{store.suffix}")
        self.store = None if seed_from_cache else store
        self.memo: dict[tuple[str, str], dict] = {}
        self.reader = None
        self.new = 0
        if seed_from_cache:
            for part in ("train", "validation", "test"):
                questions = {q["guid"]: q for q in klue.load(part)}
                for record in load().values():
                    q = questions.get(record["guid"])
                    if q is None:
                        continue
                    self.memo[(q["question"], q["context"])] = record["true_report"]
                    self.memo[(q["question"], covered(q["context"], q["answers"][0], record["fake"]))] = record["false_report"]
        elif store is not None and store.exists():
            for line in store.read_text(encoding="utf-8").splitlines():
                r = json.loads(line)
                self.memo[(r["question"], r["passage"])] = r["reading"]

    def read(self, pairs: list[tuple[str, str]]) -> list[dict]:
        missing = [p for p in dict.fromkeys(pairs) if p not in self.memo]
        if missing:
            if self.reader is None:
                from cognitive_lab.world5.reader_qa import Reader

                self.reader = Reader(self.device, precision=self.precision)
            lines = []
            for start in range(0, len(missing), 64):
                chunk = missing[start:start + 64]
                for (question, passage), reading in zip(chunk, self.reader.read_many(chunk)):
                    self.memo[(question, passage)] = reading
                    lines.append(json.dumps({"question": question, "passage": passage, "reading": reading}, ensure_ascii=False))
            self.new += len(missing)
            if self.store is not None:
                with self.store.open("a", encoding="utf-8") as out:
                    out.write("\n".join(lines) + "\n")
        return [self.memo[p] for p in pairs]


def trust_inputs(episodes: list[list[dict]], readings: list[list[list[dict]]], slots: int,
                 speakers: int) -> tuple[dict, list]:
    """World-v6 tensors (world6/trust.py `tensors`) built from MK1's own readings.
    episodes [B][E], readings [B][E][report]; candidates = distinct normalized answers in report order."""
    batch, length = len(episodes), len(episodes[0])
    says = torch.full((batch, length, speakers), -1, dtype=torch.long)
    scores = torch.zeros(batch, length, speakers, 3)
    right = torch.zeros(batch, length, slots, dtype=torch.bool)
    valid = torch.zeros(batch, length, slots, dtype=torch.bool)
    candidates: list[list] = [[None] * length for _ in range(batch)]
    for b in range(batch):
        for t, (episode, read) in enumerate(zip(episodes[b], readings[b])):
            found = []
            for r in read:
                if not any(klue.normalize(r["answer"]) == klue.normalize(c) for c in found):
                    found.append(r["answer"])
            if len(found) > slots:
                raise ValueError("more than two candidates")
            candidates[b][t] = found
            valid[b, t, :len(found)] = True
            right[b, t, :len(found)] = torch.tensor([klue.is_right(c, episode["answers"]) for c in found])  # feedback
            for report, r in zip(episode["reports"], read):
                k = next(i for i, c in enumerate(found) if klue.normalize(c) == klue.normalize(r["answer"]))
                says[b, t, report["source"]] = k
                scores[b, t, report["source"]] = torch.tensor([r["span_score"] / 10, r["null_score"] / 10,
                                                               (r["span_score"] - r["null_score"]) / 10])
    return {"says": says, "scores": scores, "right": right, "valid": valid}, candidates
