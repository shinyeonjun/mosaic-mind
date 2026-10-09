"""MK1 v2 on RGB: satellites with separate jobs, combined by explicit decisions (design/agi-blueprint-2026-10-09.md, 1).

Per question (5 documents):
  read (two channels)   Gemma 4 E2B writes a short answer from the documents; the KLUE-trained mDeBERTa reader
                        finds the best span per document with a no-answer score
  verify                mDeBERTa NLI: does a document entail "the answer to <question> is <answer>"?
  recall                Gemma 4 E2B without the documents: what it already knew
  decide                evidence = reader margin (best span score - no-answer score) and NLI support;
                        no evidence -> reject; documents vs recall disagree, recall absent from every document and
                        the documents' answer -> report factual errors and give the recalled answer
  speak                 the decision in the exam's wording
The thresholds are set on the development 10% only (`tune`), then frozen for the sealed 90%.
"""

import json
import statistics

import torch

from cognitive_lab.mk1v2 import rgb

REJECT = "I can not answer the question because of the insufficient information in documents."
ERRORS = "There are factual errors in the provided documents."
NLI_DIR = rgb.ROOT / "model" / "mDeBERTa-v3-base-mnli-xnli"


class Satellites:
    def __init__(self, device: torch.device, llm):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        from cognitive_lab.world5.reader_qa import Reader

        self.device, self.llm = device, llm
        self.reader = Reader(device)
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
        llm_answer = self.doc_answer(item["query"], item["docs"])
        recalled = self.recall(item["query"])
        entail, _ = self.support(item["docs"], item["query"], llm_answer)
        recall_entail, recall_contra = self.support(item["docs"], item["query"], recalled)
        in_docs = any(recalled.lower() in d.lower() for d in item["docs"]) if recalled else False
        return {"id": item["id"], "llm_answer": llm_answer, "reader_answer": best["answer"],
                "reader_margin": best["span_score"] - best["null_score"], "support": entail, "recall": recalled,
                "recall_support": recall_entail, "recall_contradiction": recall_contra, "recall_in_docs": in_docs,
                "recall_agrees": bool(recalled) and (recalled.lower() in llm_answer.lower() or llm_answer.lower() in recalled.lower())}


def decide(s: dict, t: dict) -> str:
    """The reply, from signals `s` and thresholds `t` (margin, support, contradiction)."""
    evidence = s["reader_margin"] >= t["margin"] or s["support"] >= t["support"]
    if not evidence:
        return REJECT
    if (not s["recall_agrees"]) and (not s["recall_in_docs"]) and s["recall_contradiction"] >= t["contradiction"]:
        return f"{ERRORS} The answer is {s['recall']}."
    return s["llm_answer"] if s["support"] >= t["support"] or not s["reader_answer"] else (
        s["llm_answer"] if s["reader_margin"] < t["margin"] else s["reader_answer"] + " / " + s["llm_answer"])


GRID = {"margin": [-4.0, -3.0, -2.0, -1.0, 0.0, 1.0], "support": [0.2, 0.3, 0.4, 0.5, 0.6],
        "contradiction": [0.5, 0.7, 0.9, 1.01]}  # 1.01 = never report errors


def evaluate(signals: dict, items: dict, t: dict) -> dict:
    out = {}
    for setting, rows in signals.items():
        judged = [rgb.judge(decide(s, t), items[setting][s["id"]]["answer"]) for s in rows]
        out[setting] = rgb.score(setting, judged)
    return out


def objective(scores: dict) -> float:
    """Mean of the primary metric per testbed (accuracy; rejection rate; error detection for counterfactual)."""
    values = []
    for setting, sc in scores.items():
        values.append(sc.get("rejection_rate", sc.get("error_detection_rate") if setting == "counterfactual" else sc.get("accuracy")))
    return statistics.mean(values)


def tune(signals: dict, items: dict) -> dict:
    best = None
    for m in GRID["margin"]:
        for su in GRID["support"]:
            for c in GRID["contradiction"]:
                t = {"margin": m, "support": su, "contradiction": c}
                value = objective(evaluate(signals, items, t))
                if best is None or value > best[0]:
                    best = (value, t)
    return best[1]


def run_mk1(part: str, settings: list[str]) -> dict:
    from cognitive_lab.mk1v2.llm_service import LLMService

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache = rgb.RESULTS / f"mk1v2-rgb-signals_{part}.json"
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
