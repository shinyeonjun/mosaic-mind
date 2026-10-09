"""MK1 v2 on multi-hop questions: split, read hop by hop, chain, and answer only with evidence at every hop.

  speak   Gemma 4 E2B answers from the documents with the exam's prompt (what "E2B alone" does), and also splits
          the question into single-hop sub-questions, where "#1" stands for the previous answer
  read    the English reading satellite answers each sub-question from the 5 documents (best span over documents,
          evidence margin = span score - no-answer score); the answer replaces "#k" in later sub-questions
  decide  the speaker answered -> keep it; the speaker rejected and every hop has evidence (margin >= threshold)
          -> answer with the last hop's answer; otherwise reject
Thresholds are set on the development 10% only. Nothing is learned from MuSiQue.
"""

import json
import statistics

import torch

from cognitive_lab.mk1v2 import musique
from cognitive_lab.mk1v2.reader_en import CHECKPOINT as READER_EN
from cognitive_lab.mk1v2.rgb import INSTRUCTION, SYSTEM

REJECT = "I can not answer the question because of the insufficient information in documents."
SPLIT_PROMPT = ("Split a multi-hop question into simple single-hop questions, in order, one per line, nothing else. "
                "Write #1 for the answer of the first line, #2 for the second.\n"
                "Example question: Who is the mother of the director of Jaws?\nAnswer:\nWho is the director of Jaws?\nWho is the mother of #1?\n"
                "Example question: In what country is the city where Picasso was born?\nAnswer:\nIn what city was Picasso born?\nIn what country is #1?")
# Plain lines, not a JSON schema: with llama.cpp's schema-constrained path, Gemma 4 went back to its thinking mode
# (enable_thinking=False was ignored) and returned an empty answer, so a first run never split any question.


def hop_signals(llm, reader, item: dict) -> dict:
    base = llm.chat([{"role": "system", "content": SYSTEM},
                     {"role": "user", "content": INSTRUCTION.format(DOCS="\n".join(item["docs"]), QUERY=item["query"])}], max_tokens=200)
    raw = llm.chat([{"role": "system", "content": SPLIT_PROMPT}, {"role": "user", "content": item["query"]}], max_tokens=200)
    subs = [line.strip().lstrip("-0123456789.) ").removeprefix("#1 ").removeprefix("#2 ").removeprefix("#3 ").strip()
            for line in raw.splitlines() if line.strip().endswith("?")] or [item["query"]]
    hops, answers = [], []
    for sub in subs[:4]:
        for k, a in enumerate(answers, 1):
            sub = sub.replace(f"#{k}", a)
        reads = reader.read_many([(sub, d) for d in item["docs"]])
        best = max(reads, key=lambda r: r["span_score"] - r["null_score"])
        answers.append(best["answer"])
        hops.append({"question": sub, "answer": best["answer"], "margin": best["span_score"] - best["null_score"]})
    return {"id": item["id"], "base_reply": base, "hops": hops}


def decide(s: dict, threshold: float) -> str:
    if "insufficient information" not in s["base_reply"]:
        return s["base_reply"]
    if s["hops"] and min(h["margin"] for h in s["hops"]) >= threshold:
        return s["hops"][-1]["answer"]
    return REJECT


THRESHOLDS = [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 99.0]  # 99 = never override the speaker


def run_mk1_musique(part: str) -> dict:
    from cognitive_lab.mk1v2.llm_service import LLMService
    from cognitive_lab.world5.reader_qa import Reader

    items = musique.exam(part)
    cache = musique.RESULTS / f"mk1v2-musique-signals_{part}.json"
    if cache.exists():
        signals = json.loads(cache.read_text(encoding="utf-8"))
    else:
        reader = Reader(torch.device("cuda" if torch.cuda.is_available() else "cpu"), checkpoint=READER_EN)
        signals = []
        with LLMService("e2b") as llm:
            for n, it in enumerate(items, 1):
                signals.append(hop_signals(llm, reader, it))
                if n % 50 == 0:
                    print(f"  {n}/{len(items)}", flush=True)
        cache.write_text(json.dumps(signals, ensure_ascii=False), encoding="utf-8")
    threshold_file = musique.RESULTS / "mk1v2-musique-threshold.json"
    if part == "dev":
        scored = {t: statistics.mean(musique.judge(decide(s, t), it)["net"] for s, it in zip(signals, items)) for t in THRESHOLDS}
        threshold = max(THRESHOLDS, key=lambda t: (scored[t], t))  # ties -> the more cautious (higher) threshold
        threshold_file.write_text(json.dumps({"threshold": threshold, "dev_scores": scored}), encoding="utf-8")
    else:
        threshold = json.loads(threshold_file.read_text(encoding="utf-8"))["threshold"]  # frozen on dev
    replies = [decide(s, threshold) for s in signals]
    judged = [{"id": s["id"], "reply": r, **musique.judge(r, it)} for s, r, it in zip(signals, replies, items)]
    return {"system": "MK1 v2 (multi-hop)", "part": part, "threshold": threshold, "summary": musique.summary(items, judged), "replies": judged}
