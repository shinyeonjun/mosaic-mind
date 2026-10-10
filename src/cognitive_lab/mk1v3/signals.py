"""v3 step 1: the signal recorder for document questions (design/processing-flow-v3-2026-10-10.md, 4).

For a question with documents, every satellite that applies is run and its output is written down with the features a
calibrator needs. The same function serves two uses: building calibration data (here, on the satellites' own
held-out data, never on an exam) and, later, the v3 loop itself.

Calibration data (not exams; the satellites were not trained on these splits):
  squad    SQuAD 2.0 dev: the question's paragraph + 4 paragraphs of other articles (unanswerable SQuAD questions stay
           unanswerable, as SQuAD means them)
  hotpot   HotpotQA validation (bridge questions, no yes/no answers): both supporting paragraphs + 3 distractors; its
           twin drops one supporting paragraph for another distractor (unanswerable, as MuSiQue's twins)
Signals per question:
  speaker  Gemma 4 E2B with the RGB prompt (temperature 0) and its token log-probabilities: mean, minimum, first token,
           mean top-1 - top-2 gap, length; rejected or not
  samples  3 more replies at temperature 0.8 (60 tokens): share with the same rejected/answered choice, mean word overlap
           with the speaker's reply
  reader   the English reading satellite on each document: best span margin, its answer, whether the speaker's reply
           contains it
  chain    the hop-by-hop selector's scores for the first two hops
Labels: answerable; right (an answer alias inside a reply that is not a rejection).

python -m cognitive_lab.mk1v3.signals --per-source 300
"""

import argparse
import json
import random
import statistics
import time
import urllib.request

ROOT = __import__("pathlib").Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts" / "results" / "mk1v3-calibration-signals.jsonl"
PASSAGES = 5


def squad_items(n: int) -> list[dict]:
    data = json.loads((ROOT / "data" / "squad2" / "dev-v2.0.json").read_text(encoding="utf-8"))["data"]
    paragraphs = [(a["title"], p["context"]) for a in data for p in a["paragraphs"]]
    qs = [(a["title"], p["context"], q) for a in data for p in a["paragraphs"] for q in p["qas"]]
    rng = random.Random("v3-calib-squad")
    rng.shuffle(qs)
    items = []
    for title, context, q in qs[:n]:
        others = [p for p in rng.sample(paragraphs, 12) if p[0] != title][:PASSAGES - 1]
        docs = [f"{title}: {context}"] + [f"{t}: {c}" for t, c in others]
        rng.shuffle(docs)
        answers = sorted({a["text"] for a in q["answers"]})
        items.append({"id": f"squad|{q['id']}", "source": "squad", "query": q["question"], "docs": docs,
                      "answers": answers, "answerable": bool(answers)})
    return items


def hotpot_items(n: int) -> list[dict]:
    from cognitive_lab.mk1v2.selector import questions, text

    qs = [q for q in questions("validation") if q["type"] == "bridge" and q["answer"].lower() not in ("yes", "no")]
    rng = random.Random("v3-calib-hotpot")
    rng.shuffle(qs)
    items = []
    for q in qs[:n // 2]:
        others = [t for t in q["paragraphs"] if t not in q["support"]]
        rng.shuffle(others)
        for answerable in (True, False):
            keep = q["support"] if answerable else [rng.choice(q["support"])]
            titles = keep + others[:PASSAGES - len(keep)]
            rng.shuffle(titles)
            items.append({"id": f"hotpot|{q['id']}|{'a' if answerable else 'u'}", "source": "hotpot", "query": q["question"],
                          "docs": [text(t, q["paragraphs"][t]) for t in titles], "answers": [q["answer"]], "answerable": answerable})
    return items


def words(t: str) -> set:
    return set("".join(c.lower() if c.isalnum() else " " for c in t).split())


def overlap(a: str, b: str) -> float:
    a, b = words(a), words(b)
    return len(a & b) / max(1, len(a | b))


class Recorder:
    def __init__(self, llm, device):
        from cognitive_lab.mk1v2.reader_en import CHECKPOINT as READER_EN
        from cognitive_lab.mk1v2.selector import Selector
        from cognitive_lab.world5.reader_qa import Reader

        self.llm = llm
        self.reader = Reader(device, checkpoint=READER_EN)
        self.selector = Selector(device)

    def _chat(self, messages, max_tokens, temperature, logprobs=False, seed=0):
        body = {"messages": messages, "max_tokens": max_tokens, "temperature": temperature, "seed": seed,
                "chat_template_kwargs": {"enable_thinking": False}}
        if logprobs:
            body.update({"logprobs": True, "top_logprobs": 2})
        req = urllib.request.Request(self.llm.url + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        return json.load(urllib.request.urlopen(req, timeout=300))["choices"][0]

    def record(self, item: dict) -> dict:
        from cognitive_lab.mk1v2.rgb import INSTRUCTION, SYSTEM

        messages = [{"role": "system", "content": SYSTEM},
                    {"role": "user", "content": INSTRUCTION.format(DOCS="\n".join(item["docs"]), QUERY=item["query"])}]
        base = self._chat(messages, 200, 0.0, logprobs=True)
        reply = base["message"]["content"]
        toks = (base.get("logprobs") or {}).get("content") or []
        lps = [t["logprob"] for t in toks if t.get("token")]
        gaps = [t["top_logprobs"][0]["logprob"] - t["top_logprobs"][1]["logprob"] for t in toks
                if t.get("token") and len(t.get("top_logprobs", [])) > 1]
        rejected = "insufficient information" in reply
        samples = [self._chat(messages, 60, 0.8, seed=s + 1)["message"]["content"] for s in range(3)]
        reads = self.reader.read_many([(item["query"], d) for d in item["docs"]])
        best = max(reads, key=lambda r: r["span_score"] - r["null_score"])
        chain = self.selector.chain(item["query"], item["docs"], hops=2)
        right = (not rejected) and any(a.lower() in reply.lower() for a in item["answers"] if a)
        return {
            "id": item["id"], "source": item["source"], "answerable": item["answerable"], "reply": reply,
            "right": right, "rejected": rejected,
            "lp_mean": statistics.mean(lps) if lps else 0.0, "lp_min": min(lps) if lps else 0.0,
            "lp_first": lps[0] if lps else 0.0, "gap_mean": statistics.mean(gaps) if gaps else 0.0, "length": len(lps),
            "sample_same_choice": statistics.mean(float(("insufficient information" in s) == rejected) for s in samples),
            "sample_overlap": statistics.mean(overlap(reply, s) for s in samples),
            "reader_margin": best["span_score"] - best["null_score"], "reader_answer": best["answer"],
            "reply_has_reader": float(bool(best["answer"]) and best["answer"].lower() in reply.lower()),
            "chain1": chain[0][1], "chain2": chain[1][1] if len(chain) > 1 else 0.0,
        }


def main() -> None:
    import torch

    from cognitive_lab.mk1v2.llm_service import LLMService

    parser = argparse.ArgumentParser(description="Record satellite signals for calibration (not on exams)")
    parser.add_argument("--per-source", type=int, default=300)
    args = parser.parse_args()
    items = squad_items(args.per_source) + hotpot_items(args.per_source)
    done = {json.loads(l)["id"] for l in OUT.read_text(encoding="utf-8").splitlines()} if OUT.exists() else set()
    todo = [it for it in items if it["id"] not in done]
    print(f"{len(items)} questions, {len(done)} already recorded, {len(todo)} to go", flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    started = time.perf_counter()
    with LLMService("e2b") as llm, OUT.open("a", encoding="utf-8") as log:
        rec = Recorder(llm, device)
        for n, it in enumerate(todo, 1):
            log.write(json.dumps(rec.record(it), ensure_ascii=False) + "\n")
            log.flush()
            if n % 25 == 0:
                rate = n / (time.perf_counter() - started)
                print(f"  {n}/{len(todo)} ({(len(todo) - n) / rate / 60:.0f} min left)", flush=True)
    print("Saved:", OUT)


if __name__ == "__main__":
    main()
