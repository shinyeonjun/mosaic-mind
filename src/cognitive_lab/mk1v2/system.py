"""MK1 v2 on RGB: the speaking satellite answers, other satellites decide whether it may (design/agi-blueprint-2026-10-09.md, 1).

Per question (5 documents):
  speak    Gemma 4 E2B answers from the documents with the exam's own prompt (exactly what "E2B alone" does)
  read     the English reading satellite (KLUE reader continued on SQuAD 2.0, mk1v2/reader_en.py): best span per
           document and its no-answer score -> evidence margin = best span score - no-answer score
  verify   mDeBERTa NLI: does a document entail "the answer to <question> is <short answer>"?
  decide   no evidence (margin and support both under their thresholds) -> reject; the speaker rejected but the
           evidence is strong -> answer with the short answer; otherwise keep the speaker's reply
A first version re-assembled the answer from the satellites' short answers and lost Gemma's multi-fact answers
(integration 0.8 -> 0.4); it is kept in git history. Error detection (counterfactual) needs a knowledge satellite
(next step B) and is not attempted here. Thresholds are set on the development 10% only, then frozen.
"""

import json
import statistics

import torch

from cognitive_lab.mk1v2 import rgb
from cognitive_lab.mk1v2.reader_en import CHECKPOINT as READER_EN

REJECT = "I can not answer the question because of the insufficient information in documents."
ERRORS = "There are factual errors in the provided documents."
NLI_DIR = rgb.ROOT / "model" / "mDeBERTa-v3-base-mnli-xnli"


class Satellites:
    def __init__(self, device: torch.device, llm):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        from cognitive_lab.world5.reader_qa import Reader

        self.device, self.llm = device, llm
        self.reader = Reader(device, checkpoint=READER_EN)
        self.nli_tok = AutoTokenizer.from_pretrained(NLI_DIR)
        self.nli = AutoModelForSequenceClassification.from_pretrained(NLI_DIR).to(device).eval()
        labels = {v.lower(): int(k) for k, v in self.nli.config.id2label.items()}
        self.entail, self.contra = labels["entailment"], labels["contradiction"]

    def doc_answer(self, query: str, docs: list[str]) -> str:
        return self.llm.chat([{"role": "system", "content": "Answer the question with a short phrase taken from the documents only."},
                              {"role": "user", "content": "Documents:\n" + "\n".join(docs) + f"\n\nQuestion: {query}"}],
                             max_tokens=30).strip()

    def recall(self, query: str) -> str:
        return self.llm.chat([{"role": "user", "content": f"Answer with a short phrase only. {query}"}], max_tokens=30).strip()

    @torch.no_grad()
    def support(self, docs: list[str], query: str, answer: str) -> tuple[float, float]:
        """Max entailment and max contradiction over documents for "the answer to <query> is <answer>"."""
        if not answer:
            return 0.0, 0.0
        hypothesis = f"The answer to the question \"{query}\" is {answer}."
        x = self.nli_tok(docs, [hypothesis] * len(docs), truncation=True, max_length=512, padding=True, return_tensors="pt").to(self.device)
        p = self.nli(**x).logits.float().softmax(-1)
        return p[:, self.entail].max().item(), p[:, self.contra].max().item()

    def signals(self, item: dict) -> dict:
        reads = self.reader.read_many([(item["query"], d) for d in item["docs"]])
        best = max(reads, key=lambda r: r["span_score"] - r["null_score"])
        base = self.llm.chat([{"role": "system", "content": rgb.SYSTEM},
                              {"role": "user", "content": rgb.INSTRUCTION.format(DOCS="\n".join(item["docs"]), QUERY=item["query"])}],
                             max_tokens=200)
        short = self.doc_answer(item["query"], item["docs"])
        entail, _ = self.support(item["docs"], item["query"], short)
        return {"id": item["id"], "base_reply": base, "short_answer": short, "reader_answer": best["answer"],
                "reader_margin": best["span_score"] - best["null_score"], "support": entail}


def decide(s: dict, t: dict) -> str:
    """The reply, from signals `s` and thresholds `t` (margin, support)."""
    evidence = s["reader_margin"] >= t["margin"] or s["support"] >= t["support"]
    speaker_rejected = "insufficient information" in s["base_reply"]
    if not evidence:
        return REJECT
    if speaker_rejected and s["reader_margin"] >= t["margin"] and s["support"] >= t["support"]:
        return s["short_answer"]
    return s["base_reply"]


GRID = {"margin": [-6.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 99.0], "support": [0.2, 0.3, 0.4, 0.5, 0.6, 99.0]}  # 99 = off


def evaluate(signals: dict, items: dict, t: dict) -> dict:
    out = {}
    for setting, rows in signals.items():
        judged = [rgb.judge(decide(s, t), items[setting][s["id"]]["answer"]) for s in rows]
        out[setting] = rgb.score(setting, judged)
    return out


def objective(scores: dict) -> float:
    """The pre-registered primary metric: mean net score over the testbeds (design/mk1-v2-spec.md, 6)."""
    return statistics.mean(sc["net"] for sc in scores.values())


def tune(signals: dict, items: dict) -> dict:
    best = None
    for m in GRID["margin"]:
        for su in GRID["support"]:
            t = {"margin": m, "support": su}
            value = objective(evaluate(signals, items, t))
            if best is None or value > best[0]:
                best = (value, t)
    return best[1]


def run_mk1(part: str, settings: list[str]) -> dict:
    from cognitive_lab.mk1v2.llm_service import LLMService

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache = rgb.RESULTS / f"mk1v2-rgb-signals-v2_{part}.json"
    items = {s: {it["id"]: it for it in rgb.exam(s, part)} for s in settings}
    if cache.exists():
        signals = json.loads(cache.read_text(encoding="utf-8"))
    else:
        signals = {}
        with LLMService("e2b") as llm:
            sat = Satellites(device, llm)
            for setting in settings:
                signals[setting] = [sat.signals(it) for it in items[setting].values()]
                print("signals", setting, len(signals[setting]), flush=True)
        cache.write_text(json.dumps(signals, ensure_ascii=False), encoding="utf-8")
    thresholds_file = rgb.RESULTS / "mk1v2-rgb-thresholds.json"
    if part == "dev":
        thresholds = tune(signals, items)
        thresholds_file.write_text(json.dumps(thresholds), encoding="utf-8")
    else:
        thresholds = json.loads(thresholds_file.read_text(encoding="utf-8"))  # frozen on dev
    result = {"system": "MK1 v2", "part": part, "thresholds": thresholds, "settings": {}}
    for setting in settings:
        replies = [decide(s, thresholds) for s in signals[setting]]
        judged = [{"id": s["id"], "reply": r, **rgb.judge(r, items[setting][s["id"]]["answer"])} for s, r in zip(signals[setting], replies)]
        result["settings"][setting] = {**rgb.score(setting, judged), "replies": judged}
    return result
