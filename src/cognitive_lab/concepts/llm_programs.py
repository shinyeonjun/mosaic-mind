"""Probe: does a frozen Gemma 4 E4B, writing Python programs that are checked on every demonstration, solve ARC tasks
the search satellite does not answer? (design/new-concepts-2026-10-12.md section 3)

  per task  16 samples (temperature 0.8, seeds 0-15), thinking off; each reply's ```python``` block must define
            transform(grid); it runs in a separate isolated process (5 s, numpy/math/collections/itertools/copy only)
  check     keep a program only if it reproduces every demonstration output exactly
  answer    vote over the kept programs' test outputs; two attempts per test input

Data: 60 ARC-AGI-1 training tasks (spent development data) that the search does not answer, random.Random(0).

python -m cognitive_lab.concepts.llm_programs run
python -m cognitive_lab.concepts.llm_programs report
"""

import argparse
import json
import random
import re
import subprocess
import sys
import tempfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from cognitive_lab.concepts.arc_learner import SEARCH_CACHE, arc_tasks, score
from cognitive_lab.concepts.learner import RESULT_DIR
from cognitive_lab.mk1v2.llm_service import LLMService

SAMPLES, TEMPERATURE, TASKS = 16, 0.8, 60
MAX_TOKENS = 1536
OUT = RESULT_DIR / "llm-programs-probe.jsonl"
ALLOWED = {"numpy", "math", "collections", "itertools", "copy"}
BANNED = re.compile(r"\b(open|eval|exec|compile|input|globals|locals|getattr|setattr|__import__)\s*\(|__")

PROMPT = """You are solving an abstract reasoning puzzle. Each example maps an input grid to an output grid by one hidden rule.
Grids are lists of rows; each number 0-9 is a colour (0 is usually the background).

{examples}

Write a Python function `transform(grid)` that takes the input grid (a list of lists of ints) and returns the output
grid (a list of lists of ints) for every example above. First describe the rule in one or two sentences, then give
the code in a single ```python block. You may use numpy. Do not print anything."""

RUNNER = r"""
import json, sys
data = json.load(sys.stdin)
env = {}
exec(data["code"], env)
out = []
for g in data["inputs"]:
    try:
        r = env["transform"]([row[:] for row in g])
        r = [[int(v) for v in row] for row in (r.tolist() if hasattr(r, "tolist") else r)]
        out.append(r)
    except Exception:
        out.append(None)
print(json.dumps(out))
"""


def grid_text(g) -> str:
    return "\n".join(" ".join(str(v) for v in row) for row in g)


def prompt(t: dict) -> str:
    parts = [f"Example {i + 1}\nInput:\n{grid_text(p['input'])}\nOutput:\n{grid_text(p['output'])}"
             for i, p in enumerate(t["train"])]
    return PROMPT.format(examples="\n\n".join(parts))


def extract(reply: str) -> str | None:
    blocks = re.findall(r"```[ \t]*[A-Za-z0-9]*[ \t]*\r?\n(.*?)(?:```|\Z)", reply, re.S)  # an unclosed block runs to the end
    code = next((b for b in reversed(blocks) if "def transform" in b), None)
    if code is None or BANNED.search(code):
        return None
    for m in re.finditer(r"^\s*(?:from\s+(\w+)|import\s+(\w+))", code, re.M):
        if (m.group(1) or m.group(2)) not in ALLOWED:
            return None
    return code


def execute(code: str, inputs: list) -> list | None:
    with tempfile.TemporaryDirectory() as cwd:
        try:
            r = subprocess.run([sys.executable, "-I", "-c", RUNNER], input=json.dumps({"code": code, "inputs": inputs}),
                               capture_output=True, text=True, timeout=5, cwd=cwd)
            return json.loads(r.stdout) if r.returncode == 0 else None
        except (subprocess.TimeoutExpired, json.JSONDecodeError):
            return None


def tasks() -> list[tuple[str, dict]]:
    search = json.loads(SEARCH_CACHE.read_text())
    pool = [(k, t) for k, t in arc_tasks("training")
            if not any(search[k]["attempts"][i] for i in range(len(t["test"])))]
    return random.Random(0).sample(pool, TASKS)


def solve(llm: LLMService, t: dict) -> dict:
    msg = [{"role": "user", "content": prompt(t)}]

    def one(seed: int) -> dict:
        try:
            reply = llm.chat(msg, max_tokens=MAX_TOKENS, temperature=TEMPERATURE, seed=seed)
        except Exception as e:  # prompt too long for a slot, server error
            return {"seed": seed, "error": type(e).__name__}
        code = extract(reply)
        if code is None:
            return {"seed": seed, "code": None}
        inputs = [p["input"] for p in t["train"]] + [p["input"] for p in t["test"]]
        outs = execute(code, inputs)
        n = len(t["train"])
        ok = outs is not None and all(o == p["output"] for o, p in zip(outs[:n], t["train"]))
        demos_right = 0 if outs is None else sum(o == p["output"] for o, p in zip(outs[:n], t["train"]))
        return {"seed": seed, "code": code, "demos_right": demos_right, "passed": ok,
                "test_out": outs[n:] if ok else None}

    with ThreadPoolExecutor(llm.parallel) as pool:
        samples = list(pool.map(one, range(SAMPLES)))
    attempts = []
    kept = [s for s in samples if s.get("passed")]
    for i in range(len(t["test"])):
        votes = Counter(json.dumps(s["test_out"][i]) for s in kept if s["test_out"][i] is not None)
        attempts.append([json.loads(g) for g, _ in votes.most_common(2)])
    return {"samples": samples, "kept": len(kept), "attempts": attempts if kept else []}


def run():
    done = {r["task"] for r in map(json.loads, OUT.read_text().splitlines())} if OUT.exists() else set()
    with LLMService("e4b", context=32768, parallel=4) as llm:
        for k, t in tasks():
            if k in done:
                continue
            r = solve(llm, t)
            row = {"task": k, "kept": r["kept"], "score": score(t, r["attempts"]), "answered": bool(r["attempts"]),
                   "samples": r["samples"], "cost": llm.cost()}
            with OUT.open("a") as fh:
                fh.write(json.dumps(row) + "\n")
            errs = sum("error" in s for s in r["samples"])
            nocode = sum(s.get("code", 1) is None for s in r["samples"])
            best = max((s.get("demos_right", 0) for s in r["samples"]), default=0)
            print(f"{k} kept {r['kept']:2d} score {row['score']:.2f} best demos {best}/{len(t['train'])} "
                  f"no-code {nocode} errors {errs} {llm.cost()['seconds'] / 60:.0f}m", flush=True)


def report():
    rows = {}
    for r in map(json.loads, OUT.read_text().splitlines()):
        rows.setdefault(r["task"], r)  # a task written twice (two runs at once on 2026-10-12): the first row counts
    rows = list(rows.values())
    solved = sum(r["score"] > 0 for r in rows)
    answered = [r for r in rows if r["answered"]]
    prec = sum(r["score"] > 0 for r in answered) / len(answered) if answered else float("nan")
    print(f"tasks {len(rows)}  answered {len(answered)}  solved {solved}  precision {prec:.3f}")
    print(f"rule: go if solved >= 5 and precision >= 0.8 -> {'GO' if solved >= 5 and prec >= 0.8 else 'STOP'}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["run", "report"])
    a = p.parse_args()
    run() if a.cmd == "run" else report()


if __name__ == "__main__":
    main()
