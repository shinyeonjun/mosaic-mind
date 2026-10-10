"""v3 step 4: the sealed mixed exam (design/processing-flow-v3-2026-10-10.md, 11). Never-used data only:
  musique     MuSiQue-Full dev twin pairs 500-700 of the "musique-sample" order (the first 500 were the earlier exam):
              200 pairs = 400 questions, documents built exactly as mk1v2/musique.py, net score
  taskstream  the five tasks other than TREC (all 500 TREC rows are used), rows 500-700 of each task's
              "taskstream|<task>" order, both conditions (named, symbolic), online accuracy with feedback
RGB and ARC-AGI-1 evaluation are used up; the grid kind is not part of this exam.
Systems: v3 (one frozen MK1v3, the two streams interleaved with seed "v3-sealed", never told which is which),
Gemma 4 E4B alone and Gemma 4 E2B alone (MuSiQue: the exam prompt; task stream: the 16 most recent feedback examples).
Records are appended after every input, so a stopped run resumes.

python -m cognitive_lab.mk1v3.sealed --system v3 --final
python -m cognitive_lab.mk1v3.sealed --compare
"""

import argparse
import json
import random
import statistics
from collections import defaultdict

from cognitive_lab.mk1v3.signals import ROOT

RESULTS = ROOT / "artifacts" / "results"
PAIRS = (500, 700)
ROWS = (500, 700)


def musique_items() -> list[dict]:
    from cognitive_lab.mk1v2.musique import FILE, PASSAGES

    by_id = defaultdict(list)
    for line in FILE.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        by_id[row["id"]].append(row)
    ids = sorted(by_id)
    random.Random("musique-sample").shuffle(ids)  # the same order as the earlier exam
    items = []
    for qid in ids[PAIRS[0]:PAIRS[1]]:
        for row in sorted(by_id[qid], key=lambda r: not r["answerable"]):
            rng = random.Random(f"musique-docs|{qid}|{row['answerable']}")
            support = [p for p in row["paragraphs"] if p["is_supporting"]] if row["answerable"] else []
            others = [p for p in row["paragraphs"] if not p["is_supporting"]]
            rng.shuffle(others)
            docs = support + others[:PASSAGES - len(support)]
            rng.shuffle(docs)
            items.append({"id": f"musique|{qid}|{'answerable' if row['answerable'] else 'unanswerable'}",
                          "query": row["question"], "answers": [row["answer"]] + row.get("answer_aliases", []),
                          "answerable": row["answerable"], "docs": [f"{p['title']}: {p['paragraph_text']}" for p in docs]})
    return items


def taskstream_items() -> list[dict]:
    from cognitive_lab.mk1v2.taskstream import SYMBOLS, TASKS, _rows

    order = sorted(TASKS)
    random.Random("taskstream-order").shuffle(order)
    out = []
    for condition in ("named", "symbolic"):
        for task in order:
            rows = _rows(task)
            if len(rows) < ROWS[1]:
                continue  # TREC: no unused rows left
            random.Random(f"taskstream|{task}").shuffle(rows)
            names = TASKS[task][3] if condition == "named" else list(SYMBOLS[:len(TASKS[task][3])])
            for n, x in enumerate(rows[ROWS[0]:ROWS[1]]):
                out.append({"id": f"taskstream|{condition}|{task}|{n}", "text": x["text"], "label": x["label"],
                            "labels": names, "task": task, "condition": condition})
    return out


def stream() -> list[dict]:
    """The two kinds interleaved (each keeps its own order), fixed seed."""
    queues = [musique_items(), taskstream_items()]
    rng = random.Random("v3-sealed")
    out = []
    while any(queues):
        live = [q for q in queues if q]
        q = rng.choices(live, weights=[len(x) for x in live])[0]
        out.append(q.pop(0))
    return out


def score_item(item: dict, answer) -> int:
    from cognitive_lab.mk1v2 import musique

    if item["id"].startswith("musique"):
        return musique.judge(answer, item)["net"]
    return int(answer == item["label"])


def run(system: str) -> None:
    import torch

    from cognitive_lab.mk1v2.llm_service import LLMService

    path = RESULTS / f"mk1v3-sealed_{system}.jsonl"
    done = {json.loads(l)["id"] for l in path.read_text(encoding="utf-8").splitlines()} if path.exists() else set()
    items = stream()
    print(f"{len(items)} inputs, {len(done)} done", flush=True)
    with LLMService("e2b" if system == "v3" else system) as llm, path.open("a", encoding="utf-8") as log:
        if system == "v3":
            from cognitive_lab.mk1v3.loop import MK1v3

            mk1 = MK1v3(llm, torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            replay = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()] if done else []
            # resuming: the label stream relearns from the feedback it already got (calls repeat; deterministic, cache off)
            for rec in replay:
                if rec["id"].startswith("taskstream"):
                    item = next(i for i in items if i["id"] == rec["id"])
                    board = mk1.solve(item)
                    mk1.feedback(board, item["label"])
        from cognitive_lab.mk1v2.rgb import INSTRUCTION, SYSTEM
        from cognitive_lab.mk1v2.taskstream import K_EXAMPLES, parse

        seen = defaultdict(list)  # Gemma alone: the feedback examples of each task
        for n, item in enumerate(items, 1):
            is_doc = item["id"].startswith("musique")
            if item["id"] in done:
                if not is_doc:
                    seen[(item["task"], item["condition"])].append((item["text"], item["label"]))
                continue
            if system == "v3":
                board = mk1.solve(item)
                answer = board.read("reply") if is_doc else board.read("answer")
                extra = {"calls": board.calls(), "seconds": round(board.cost(), 2)}
                if not is_doc:
                    mk1.feedback(board, item["label"])
            elif is_doc:
                answer = llm.chat([{"role": "system", "content": SYSTEM},
                                   {"role": "user", "content": INSTRUCTION.format(DOCS="\n".join(item["docs"]), QUERY=item["query"])}],
                                  max_tokens=200)
                extra = {}
            else:
                key = (item["task"], item["condition"])
                shots = "\n".join(f"Text: {t}\nLabel: {item['labels'][y]}" for t, y in seen[key][-K_EXAMPLES:])
                prompt = (f"Classify the text into exactly one of these labels: {', '.join(item['labels'])}.\n"
                          f"{'Labelled examples:' + chr(10) + shots + chr(10) if shots else ''}"
                          f"Text: {item['text']}\nAnswer with the label only.")
                reply = llm.chat([{"role": "user", "content": prompt}], max_tokens=8).strip()
                g = parse(reply, item["labels"], item["condition"])
                answer = -1 if g is None else g
                extra = {"reply": reply}
                seen[key].append((item["text"], item["label"]))
            rec = {"id": item["id"], "answer": answer if is_doc else int(answer), "score": score_item(item, answer), **extra}
            log.write(json.dumps(rec, ensure_ascii=False) + "\n")
            log.flush()
            if n % 200 == 0:
                print(f"  {n}/{len(items)}", flush=True)


def compare() -> dict:
    def load(system):
        return {r["id"]: r for r in (json.loads(l) for l in (RESULTS / f"mk1v3-sealed_{system}.jsonl").read_text(encoding="utf-8").splitlines())}

    systems = {s: load(s) for s in ("v3", "e4b", "e2b")}
    groups = {"musique": lambda i: i.startswith("musique"),
              "taskstream named": lambda i: i.startswith("taskstream|named"),
              "taskstream symbolic": lambda i: i.startswith("taskstream|symbolic")}
    out = {"score": {}, "versus": {}}
    for g, test in groups.items():
        ids = sorted(i for i in systems["v3"] if test(i))
        assert all(set(i for i in s if test(i)) == set(ids) for s in systems.values()), "systems saw different inputs"
        out["score"][g] = {s: round(statistics.mean(v[i]["score"] for i in ids), 4) for s, v in systems.items()}
        for opp in ("e4b", "e2b"):
            # MuSiQue: resample questions; task stream: resample positions within each task (as taskstream_compare)
            clusters = defaultdict(list)
            for i in ids:
                clusters[i.rsplit("|", 1)[0] if g != "musique" else "all"].append(i)
            rng = random.Random(0)
            draws = []
            for _ in range(2000):
                per = []
                for members in clusters.values():
                    pick = [rng.choice(members) for _ in members]
                    per.append(statistics.mean(systems["v3"][i]["score"] - systems[opp][i]["score"] for i in pick))
                draws.append(statistics.mean(per))
            draws.sort()
            diff = statistics.mean(statistics.mean(systems["v3"][i]["score"] - systems[opp][i]["score"] for i in m) for m in clusters.values())
            out["versus"][f"{g}: v3 - {opp}"] = {"difference": round(diff, 4), "ci95": [round(draws[50], 4), round(draws[1949], 4)]}
    v3 = systems["v3"]
    doc = [r for i, r in v3.items() if i.startswith("musique")]
    out["v3_speaker_calls_per_musique_question"] = round(statistics.mean(sum(c == "speaker" for c in r["calls"]) for r in doc), 3)
    out["v3_speaker_skipped_share"] = round(statistics.mean("speaker-skipped" in r["calls"] for r in doc), 3)
    print(json.dumps(out, indent=1))
    (RESULTS / "mk1v3-sealed_verdict.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="v3 sealed mixed exam")
    parser.add_argument("--system", choices=("v3", "e4b", "e2b"))
    parser.add_argument("--final", action="store_true", help="required: the inputs are sealed, run once per system")
    parser.add_argument("--compare", action="store_true")
    args = parser.parse_args()
    if args.compare:
        compare()
        return
    if not args.final:
        raise SystemExit("sealed: pass --final, once per system, after the pre-registration commit")
    run(args.system)


if __name__ == "__main__":
    main()

