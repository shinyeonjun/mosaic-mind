"""MK1 v2 with a learned thinking satellite, on MuSiQue (design/mk1-v2-spec.md, 9).

Per question (5 documents):
  speak    Gemma 4 E2B answers from the documents with the exam's prompt (what "E2B alone" does)
  select   the hop-by-hop selector (HotpotQA-trained) chains documents: best first paragraph, then the best next one
           given it, then a third while it scores high -> chain scores s1, s2 (, s3)
  read     the multi-hop reader (HotpotQA-trained) reads the chained paragraphs joined -> answer, evidence margin
  decide   evidence = weakest chain score >= t_chain and reader margin >= t_margin
           speaker answered and evidence            -> keep the speaker's reply
           speaker answered, no evidence            -> keep it too, unless `doubt` is on (then reject)
           speaker rejected and evidence            -> the reader's answer
           speaker rejected, no evidence            -> reject
Thresholds (and `doubt`) are set on the development 10% only and frozen for the sealed 90%.
Nothing is trained on MuSiQue.
"""

import itertools
import json
import statistics

import torch

from cognitive_lab.mk1v2 import musique
from cognitive_lab.mk1v2.reader_hop import CHECKPOINT as READER_HOP
from cognitive_lab.mk1v2.rgb import INSTRUCTION, SYSTEM
from cognitive_lab.mk1v2.selector import Selector

REJECT = "I can not answer the question because of the insufficient information in documents."
GRID = {"chain": [0.05, 0.1, 0.2, 0.3, 0.5, 0.7], "margin": [-2.0, 0.0, 1.0, 2.0, 3.0, 99.0], "doubt": [False, True]}


def signals(llm, selector: Selector, reader, item: dict) -> dict:
    base = llm.chat([{"role": "system", "content": SYSTEM},
                     {"role": "user", "content": INSTRUCTION.format(DOCS="\n".join(item["docs"]), QUERY=item["query"])}], max_tokens=200)
    chain = selector.chain(item["query"], item["docs"])
    context = "\n".join(item["docs"][i] for i, _ in chain)
    read = reader.read_many([(item["query"], context)])[0]
    return {"id": item["id"], "base_reply": base, "chain": [[i, s] for i, s in chain], "answer": read["answer"],
            "margin": read["span_score"] - read["null_score"]}


def decide(s: dict, t: dict) -> str:
    evidence = min(score for _, score in s["chain"][:2]) >= t["chain"] and s["margin"] >= t["margin"]
    speaker_rejected = "insufficient information" in s["base_reply"]
    if not speaker_rejected:
        return REJECT if (t["doubt"] and not evidence) else s["base_reply"]
    return s["answer"] if evidence else REJECT


def run(part: str) -> dict:
    from cognitive_lab.mk1v2.llm_service import LLMService
    from cognitive_lab.world5.reader_qa import Reader

    items = musique.exam(part)
    cache = musique.RESULTS / f"mk1v2-think-signals_{part}.json"
    if cache.exists():
        sig = json.loads(cache.read_text(encoding="utf-8"))
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        selector, reader = Selector(device), Reader(device, checkpoint=READER_HOP)
        sig = []
        with LLMService("e2b") as llm:
            for n, it in enumerate(items, 1):
                sig.append(signals(llm, selector, reader, it))
                if n % 50 == 0:
                    print(f"  {n}/{len(items)}", flush=True)
        cache.write_text(json.dumps(sig, ensure_ascii=False), encoding="utf-8")
    threshold_file = musique.RESULTS / "mk1v2-think-thresholds.json"
    if part == "dev":
        best = None
        for c, m, d in itertools.product(GRID["chain"], GRID["margin"], GRID["doubt"]):
            t = {"chain": c, "margin": m, "doubt": d}
            value = statistics.mean(musique.judge(decide(s, t), it)["net"] for s, it in zip(sig, items))
            if best is None or value > best[0] + 1e-9:  # ties keep the earlier, more permissive setting
                best = (value, t)
        thresholds = best[1]
        threshold_file.write_text(json.dumps({"thresholds": thresholds, "dev_net": best[0]}), encoding="utf-8")
    else:
        thresholds = json.loads(threshold_file.read_text(encoding="utf-8"))["thresholds"]  # frozen on dev
    judged = [{"id": s["id"], "reply": decide(s, thresholds), **musique.judge(decide(s, thresholds), it)} for s, it in zip(sig, items)]
    return {"system": "MK1 v2 + thinking satellite", "part": part, "thresholds": thresholds,
            "summary": musique.summary(items, judged), "replies": judged}
