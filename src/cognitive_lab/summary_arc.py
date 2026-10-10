"""Collect the ARC-AGI-1 exam results for notebook 22 (design/mk1-v2-spec.md, 12). No model runs here.

Reads the saved result files (exam-arc_*) and the task JSON (for the true test outputs, to say which attempt scored).
MK1's attempts per test input are: the search satellite's answers first (one per program found), then E2B's saved
attempts filling the rest up to two (mk1v2/arc.py). Everything returns plain Python structures.
"""

import json
from collections import Counter

from cognitive_lab.mk1v2.arc import RESULTS, tasks

SYSTEMS = [("e2b", "E2B 혼자"), ("e4b", "E4B 혼자"), ("mk1", "MK1")]
PARTS = {"dev": "_dev_100", "final": "_final"}  # dev = first 100 training tasks, final = all 400 evaluation tasks
# The standard ARC palette, colours 0-9.
PALETTE = ["#000000", "#0074D9", "#FF4136", "#2ECC40", "#FFDC00", "#AAAAAA", "#F012BE", "#FF851B", "#7FDBFF", "#870C25"]


def records(system: str, part: str) -> dict:
    """{task id: record} in task order."""
    data = json.loads((RESULTS / f"exam-arc_{system}{PARTS[part]}.json").read_text(encoding="utf-8"))
    return {r["id"]: r for r in data["records"]}


def summary(system: str, part: str) -> dict:
    data = json.loads((RESULTS / f"exam-arc_{system}{PARTS[part]}.json").read_text(encoding="utf-8"))
    return {"summary": data["summary"], "cost": data["cost"]}


def verdict(part: str) -> dict:
    return json.loads((RESULTS / f"exam-arc_verdict{PARTS[part]}.json").read_text(encoding="utf-8"))


def task_data(part: str) -> dict:
    """{task id: task JSON} for the tasks the exam used."""
    limit = 100 if part == "dev" else None
    return dict(tasks(part, limit))


def sources(part: str) -> dict:
    """{task id: {"search": points, "fill": points}} for MK1: per test input, the point goes to the search satellite
    when one of its own answers matches, else to the E2B fill when a filled attempt matches; task points are the mean
    over its test inputs (as in ARC scoring)."""
    mk1, data = records("mk1", part), task_data(part)
    out = {}
    for tid, r in mk1.items():
        truth = [p["output"] for p in data[tid]["test"]]
        k = len(r["programs"])  # the search's answers come first
        s = f = 0.0
        for att, t in zip(r["attempts"], truth):
            if any(a == t for a in att[:k]):
                s += 1
            elif any(a == t for a in att[k:] if a is not None):
                f += 1
        out[tid] = {"search": s / len(truth), "fill": f / len(truth)}
    return out


def solved(system: str, part: str) -> set:
    """Task ids with a score above zero."""
    return {tid for tid, r in records(system, part).items() if r["score"] > 0}


def search_judgement(part: str) -> dict:
    """How the search satellite judged: silent tasks, answered-and-right, answered-and-wrong (by task score)."""
    mk1, src = records("mk1", part), sources(part)
    answered = [tid for tid, r in mk1.items() if r["thought"]]
    right = [tid for tid in answered if src[tid]["search"] > 0]
    return {"tasks": len(mk1), "silent": len(mk1) - len(answered), "answered": len(answered),
            "right": len(right), "wrong": len(answered) - len(right),
            "wrong_ids": sorted(set(answered) - set(right))}


def programs(part: str, only_right: bool = True) -> Counter:
    """Count of the first program string per answered task (the one that gave attempt 1), optionally only for tasks
    the search got right."""
    mk1, src = records("mk1", part), sources(part)
    c = Counter()
    for tid, r in mk1.items():
        if r["programs"] and (not only_right or src[tid]["search"] > 0):
            c[r["programs"][0]] += 1
    return c


def primitive_counts(part: str) -> Counter:
    """How often each primitive appears in the first program of tasks the search got right."""
    c = Counter()
    for prog, n in programs(part).items():
        for step in prog.split(" > "):
            c[step] += n
    return c


def format_health(system: str, part: str) -> dict:
    """Share of Gemma attempts that parsed into a grid (None = the reply was not a grid)."""
    att = [a for r in records(system, part).values() for test in r["attempts"] for a in test]
    return {"attempts": len(att), "grids": sum(a is not None for a in att)}


def examples(part: str, n: int = 3, max_cells: int = 150) -> list[dict]:
    """Small tasks the search solved and E4B did not, one per distinct program, smallest grids first:
    {id, task, answer (MK1 attempt 1 per test input), program}."""
    mk1, src, e4b, data = records("mk1", part), sources(part), records("e4b", part), task_data(part)
    found = []
    for tid, r in mk1.items():
        if src[tid]["search"] < 1 or e4b[tid]["score"] > 0:
            continue
        t = data[tid]
        size = max(len(g) * len(g[0]) for p in t["train"] + t["test"] for g in (p["input"], p["output"]))
        if size <= max_cells:
            found.append((size, {"id": tid, "task": t, "answer": [a[0] for a in r["attempts"]], "program": r["programs"][0]}))
    out, seen = [], set()
    for _, e in sorted(found, key=lambda x: x[0]):
        if e["program"] not in seen and len(out) < n:
            seen.add(e["program"])
            out.append(e)
    return out
