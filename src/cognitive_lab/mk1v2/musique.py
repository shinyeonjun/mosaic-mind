"""External exam: MuSiQue-Full (Trivedi et al., TACL 2022), multi-hop questions and their unanswerable twins.

Data: StonyBrookNLP/musique, musique_full_v1.0_dev.jsonl (CC BY 4.0). Every question id comes twice: an answerable
version (2-4 supporting paragraphs among 20) and an unanswerable twin whose needed paragraphs were removed.
Exam built here (fixed before any system was run on it):
  - 500 twin pairs drawn with seed "musique-sample" (1,000 questions); 10% of the pairs (seed "musique-split") are
    the development set for engineering, the other 90% are sealed and read only with --final, once;
  - 5 documents per question, shuffled with the question id as seed: answerable = all supporting paragraphs + random
    other paragraphs of that question; unanswerable = 5 random paragraphs of the twin (related, but not enough);
  - the RGB system prompt and wording ("insufficient information" = a rejection);
  - right = the answer or an alias inside the reply (case-insensitive), as RGB;
  - net score (design/mk1-v2-spec.md, 6): answerable +1 right / 0 rejected / -1 wrong; unanswerable +1 rejected / -1.
Nothing is trained or tuned on MuSiQue except thresholds on the development 10%.

python -m cognitive_lab.mk1v2.musique --system e2b --set dev
"""

import argparse
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path

from cognitive_lab.mk1v2.rgb import INSTRUCTION, SYSTEM

ROOT = Path(__file__).resolve().parents[3]
FILE = ROOT / "data" / "musique" / "data" / "musique_full_v1.0_dev.jsonl"
RESULTS = ROOT / "artifacts" / "results"
PAIRS, PASSAGES = 500, 5


def exam(part: str) -> list[dict]:
    by_id = defaultdict(list)
    for line in FILE.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        by_id[row["id"]].append(row)
    ids = sorted(by_id)
    random.Random("musique-sample").shuffle(ids)
    ids = ids[:PAIRS]
    dev = set(sorted(ids, key=lambda i: random.Random(f"musique-split|{i}").random())[:PAIRS // 10])
    chosen = [i for i in ids if (i in dev) == (part == "dev")]
    if part not in ("dev", "final"):
        raise ValueError(part)
    items = []
    for qid in chosen:
        for row in sorted(by_id[qid], key=lambda r: not r["answerable"]):
            rng = random.Random(f"musique-docs|{qid}|{row['answerable']}")
            paragraphs = row["paragraphs"]
            support = [p for p in paragraphs if p["is_supporting"]] if row["answerable"] else []
            others = [p for p in paragraphs if not p["is_supporting"]]
            rng.shuffle(others)
            docs = support + others[:PASSAGES - len(support)]
            rng.shuffle(docs)
            items.append({"id": f"{qid}|{'answerable' if row['answerable'] else 'unanswerable'}", "query": row["question"],
                          "answers": [row["answer"]] + row.get("answer_aliases", []), "answerable": row["answerable"],
                          "docs": [f"{p['title']}: {p['paragraph_text']}" for p in docs]})
    return items


def judge(reply: str, item: dict) -> dict:
    rejected = "insufficient information" in reply
    right = (not rejected) and any(a.lower() in reply.lower() for a in item["answers"] if a)
    if item["answerable"]:
        net = 1 if right else (0 if rejected else -1)
    else:
        net = 1 if rejected else -1
    return {"rejected": rejected, "right": right, "net": net}


def summary(items: list[dict], judged: list[dict]) -> dict:
    ans = [j for it, j in zip(items, judged) if it["answerable"]]
    una = [j for it, j in zip(items, judged) if not it["answerable"]]
    return {"questions": len(items), "net": round(statistics.mean(j["net"] for j in judged), 4),
            "answerable_accuracy": round(statistics.mean(j["right"] for j in ans), 4),
            "answerable_rejected": round(statistics.mean(j["rejected"] for j in ans), 4),
            "unanswerable_rejection_rate": round(statistics.mean(j["rejected"] for j in una), 4)}


def run_llm(name: str, part: str) -> dict:
    from cognitive_lab.mk1v2.llm_service import LLMService

    items = exam(part)
    judged = []
    with LLMService(name) as llm:
        for n, it in enumerate(items, 1):
            reply = llm.chat([{"role": "system", "content": SYSTEM},
                              {"role": "user", "content": INSTRUCTION.format(DOCS="\n".join(it["docs"]), QUERY=it["query"])}],
                             max_tokens=200)
            judged.append({"id": it["id"], "reply": reply, **judge(reply, it)})
            if n % 100 == 0:
                print(f"  {n}/{len(items)}", flush=True)
        cost = llm.cost()
    return {"system": f"gemma-4-{name} alone", "part": part, "summary": summary(items, judged), "replies": judged, "cost": cost}


def main() -> None:
    parser = argparse.ArgumentParser(description="MuSiQue-Full as an external exam")
    parser.add_argument("--system", required=True, help="e2b | e4b | mk1 | mk1-think")
    parser.add_argument("--set", choices=("dev", "final"), default="dev")
    parser.add_argument("--final", action="store_true", help="required to read the sealed 90%%")
    args = parser.parse_args()
    if args.set == "final" and not args.final:
        raise SystemExit("the final set is sealed: pass --final, once, when the system is frozen")
    if args.system in ("e2b", "e4b"):
        result = run_llm(args.system, args.set)
    elif args.system == "mk1-think":  # the learned thinking satellite (mk1v2/think.py)
        from cognitive_lab.mk1v2.think import run

        result = run(args.set)
    else:  # the first, prompt-split multi-hop pipeline (kept for the record)
        from cognitive_lab.mk1v2.multihop import run_mk1_musique

        result = run_mk1_musique(args.set)
    path = RESULTS / f"exam-musique_{args.system}_{args.set}.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(result["summary"], indent=1))
    print("Saved:", path)


if __name__ == "__main__":
    main()
