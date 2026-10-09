"""External exam: ARC-AGI-1 (Chollet 2019), abstract grid puzzles: find the rule from 2-5 demonstrations, apply it.

Data: github.com/fchollet/ARC-AGI (Apache-2.0), data/training (400 tasks) and data/evaluation (400 tasks).
  dev     the 400 training tasks (engineering, primitives, thresholds)
  final   the 400 evaluation tasks, sealed: read only with --final, once
Scoring (ARC's own): two attempts per test input; a test input is solved when an attempt equals the output grid
exactly; task score = solved test inputs / test inputs; exam score = mean over tasks.

Systems:
  e2b / e4b   Gemma 4 alone (thinking mode off, visible reasoning allowed, then "ANSWER:" and the grid, up to 3,000
              tokens): the demonstrations and the test input as rows of digits; attempt 1 at temperature 0,
              attempt 2 sampled at temperature 0.7 (seeded); a reply that is not a grid counts as wrong
  mk1         the search satellite (mk1v2/arc_search.py) thinks first; its programs give the attempts; when it has
              fewer than two answers, Gemma E2B's saved attempts (gemma-e2b run of the same part) fill the rest

python -m cognitive_lab.mk1v2.arc --system e2b --set dev
"""

import argparse
import json
import re
import statistics
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data" / "arc" / "ARC-AGI-master" / "data"
RESULTS = ROOT / "artifacts" / "results"
CONTEXT = 16384
REPLY = 3000  # room to reason before the answer (Gemma writes its reasoning even when asked for the grid only)


def tasks(part: str, limit: int | None = None) -> list[tuple[str, dict]]:
    folder = {"dev": "training", "final": "evaluation"}[part]
    files = sorted((DATA / folder).glob("*.json"))[:limit]
    return [(f.stem, json.loads(f.read_text(encoding="utf-8"))) for f in files]


def show(g) -> str:
    return "\n".join(" ".join(str(v) for v in row) for row in g)


def prompt(task: dict, test_input) -> str:
    parts = ["Each example shows an input grid and its output grid (digits are colours). Find the rule that turns every "
             "input into its output, then apply it to the test input.\n"]
    for n, p in enumerate(task["train"], 1):
        parts.append(f"Example {n} input:\n{show(p['input'])}\nExample {n} output:\n{show(p['output'])}\n")
    parts.append(f"Test input:\n{show(test_input)}\nYou may reason first. End your reply with a line ANSWER: followed by the "
                 "test output grid, one row per line, digits separated by spaces, and nothing after it.")
    return "\n".join(parts)


def parse(reply: str):
    """The grid after the last ANSWER:, else the last block of consecutive digit rows; ragged rows = no answer."""
    text = reply.rsplit("ANSWER:", 1)[1] if "ANSWER:" in reply else reply
    blocks, current = [], []
    for line in text.splitlines():
        line = line.strip().strip("`")
        if line and re.fullmatch(r"[\d\s,\[\]]+", line) and re.search(r"\d", line):
            current.append([int(v) for v in re.findall(r"\d", line)])
        elif current:
            blocks.append(current)
            current = []
    if current:
        blocks.append(current)
    rows = blocks[-1] if blocks else None
    return rows if rows and len({len(r) for r in rows}) == 1 else None


def gemma_attempts(llm, task: dict, test_input) -> tuple[list, list]:
    """Two parsed attempts and the last 300 characters of each reply (for audit)."""
    text = prompt(task, test_input)
    if len(text) / 2.2 > CONTEXT - REPLY - 100:  # about 2 characters per token for spaced digits; too long = no answer
        return [None, None], ["(prompt too long)", "(prompt too long)"]
    out, tails = [], []
    for temperature in (0.0, 0.7):
        try:
            reply = llm.chat([{"role": "user", "content": text}], max_tokens=REPLY, temperature=temperature)
        except Exception as e:  # a server error counts as no answer
            reply = f"(error: {e})"
        out.append(parse(reply))
        tails.append(reply[-300:])
    return out, tails


def score(attempts: list[list], truth: list) -> float:
    return statistics.mean(float(any(a == t for a in att if a is not None)) for att, t in zip(attempts, truth))


def run(system: str, part: str, limit: int | None, tag: str) -> dict:
    """Records are appended to a .partial.jsonl file after every task, so a stopped run resumes where it was.
    mk1 calls no language model: when the search has fewer than two answers, it takes the remaining attempts from the
    saved gemma-e2b run of the same part (the same call, made once)."""
    items = tasks(part, limit)
    partial = RESULTS / f"exam-arc_{system}_{part}{tag}.partial.jsonl"
    records = [json.loads(line) for line in partial.read_text(encoding="utf-8").splitlines()] if partial.exists() else []
    done = {r["id"] for r in records}
    started = time.perf_counter()
    if system == "mk1":
        from cognitive_lab.mk1v2.arc_search import solve

        speaker = {r["id"]: r for r in json.loads((RESULTS / f"exam-arc_e2b_{part}{tag}.json").read_text(encoding="utf-8"))["records"]}
        llm = None
    else:
        from cognitive_lab.mk1v2.llm_service import LLMService

        llm = LLMService(system, context=CONTEXT).__enter__()
    try:
        with partial.open("a", encoding="utf-8") as log:
            for n, (tid, task) in enumerate(items, 1):
                if tid in done:
                    continue
                truth = [p["output"] for p in task["test"]]
                if system == "mk1":
                    found = solve(task)
                    attempts = []
                    for i in range(len(task["test"])):
                        mine = [outs[i] for _, outs in found]
                        attempts.append(mine + speaker[tid]["attempts"][i][:max(0, 2 - len(mine))])
                    record = {"id": tid, "programs": [" > ".join(p) for p, _ in found], "thought": bool(found)}
                else:
                    pairs = [gemma_attempts(llm, task, t["input"]) for t in task["test"]]
                    attempts = [a for a, _ in pairs]
                    record = {"id": tid, "tails": [t for _, t in pairs]}
                record.update({"score": score(attempts, truth), "attempts": attempts})
                records.append(record)
                log.write(json.dumps(record) + "\n")
                log.flush()
                if n % 20 == 0:
                    print(f"  {n}/{len(items)} score {statistics.mean(r['score'] for r in records):.3f} "
                          f"({(time.perf_counter() - started) / 60:.0f} min)", flush=True)
        cost = llm.cost() if llm else None
    finally:
        if llm:
            llm.__exit__()
    order = {tid: i for i, (tid, _) in enumerate(items)}
    records.sort(key=lambda r: order[r["id"]])
    summary = {"tasks": len(records), "score": round(statistics.mean(r["score"] for r in records), 4)}
    if system == "mk1":
        thought = [r for r in records if r["thought"]]
        summary.update({"search_answered": len(thought),
                        "search_answered_score": round(statistics.mean(r["score"] for r in thought), 4) if thought else None})
    return {"system": system, "part": part, "summary": summary, "records": records, "cost": cost}


def main() -> None:
    parser = argparse.ArgumentParser(description="ARC-AGI-1 as an external exam")
    parser.add_argument("--system", required=True, help="e2b | e4b | mk1")
    parser.add_argument("--set", choices=("dev", "final"), default="dev")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--final", action="store_true", help="required to read the sealed evaluation tasks")
    args = parser.parse_args()
    if args.set == "final" and not args.final:
        raise SystemExit("the final set is sealed: pass --final, once, when the system is frozen")
    tag = f"_{args.limit}" if args.limit else ""
    result = run(args.system, args.set, args.limit, tag)
    path = RESULTS / f"exam-arc_{args.system}_{args.set}{tag}.json"
    path.write_text(json.dumps(result), encoding="utf-8")
    print(json.dumps(result["summary"]))
    print("Saved:", path)


if __name__ == "__main__":
    main()
