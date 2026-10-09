"""Collect the new-task stream exam results for notebook 21 (design/mk1-v2-spec.md, 10 and 11). No model runs here.

Gemma runs are rescored from their saved replies with taskstream.parse (via taskstream_compare.rights), so every
system is scored by the same parser. Everything returns plain Python / numpy structures.
"""

import json

import numpy as np

from cognitive_lab.mk1v2.taskstream import RESULTS
from cognitive_lab.mk1v2.taskstream_compare import rights, score

SYSTEMS = [  # (result name, plain label)
    ("e2b", "E2B 혼자"),
    ("e4b", "E4B 혼자"),
    ("mk1", "MK1 v1"),
    ("mk1-v2", "MK1 v2"),
]
CONDITIONS = [("named", "이름 라벨"), ("symbolic", "기호 라벨 (A, B, C…)")]
TASK_NAMES = {"subj": "SUBJ\n(객관/주관)", "ag_news": "AG News\n(뉴스 주제)", "trec": "TREC\n(질문 종류)",
              "sst2": "SST-2\n(영화평 감정)", "tweet_sentiment": "TweetEval\n(트윗 감정)", "emotion": "Emotion\n(감정 6개)"}

# Section 11 dev table: the growing part alone, symbolic labels, 100 texts per task (None = not measured).
EYES = [  # (eye, raw class means, normalised, normalised + centred)
    ("e5-small (v1)", 0.602, 0.657, 0.643),
    ("E2B 내부 표현", 0.545, None, 0.618),
    ("e5-small + E2B 이어 붙이기", 0.575, None, 0.652),
    ("bge-large (v2)", None, 0.703, 0.713),
]


def load(part: str) -> dict:
    """{system: {condition: {task: [0/1 per text in stream order]}}} for the systems that ran on this part."""
    return {s: rights(s, part) for s, _ in SYSTEMS if (RESULTS / f"exam-taskstream_{s}_{part}.json").exists()}


def task_order(part: str) -> list[str]:
    """Tasks in the order they arrived (as saved in the result files)."""
    data = json.loads((RESULTS / f"exam-taskstream_e2b_{part}.json").read_text(encoding="utf-8"))["conditions"]
    return [t["task"] for t in data["named"]["tasks"]]


def accuracy(r: dict) -> dict:
    """{system: {condition: online accuracy, mean over the six tasks}}"""
    return {s: {c: score(v) for c, v in by.items()} for s, by in r.items()}


def verdict(part: str) -> dict:
    """The saved verdict; older files (final) key MK1 v1's comparisons by the opponent alone ("e4b"), so those are
    renamed to the newer "mk1 - e4b" form."""
    v = json.loads((RESULTS / f"exam-taskstream_verdict_{part}.json").read_text(encoding="utf-8"))
    v["versus"] = {(k if " - " in k else f"mk1 - {k}"): d for k, d in v["versus"].items()}
    return v


def per_task(r: dict, condition: str) -> dict:
    """{system: {task: accuracy}}"""
    return {s: {t: float(np.mean(x)) for t, x in by[condition].items()} for s, by in r.items()}


def curve(r: dict, condition: str, window: int = 20) -> dict:
    """{system: (positions 1..n, accuracy)}: per position the mean over the six tasks of right/wrong, then a trailing
    rolling mean over the last `window` positions (fewer at the start)."""
    out = {}
    for s, by in r.items():
        per_position = np.mean([np.array(x, dtype=float) for x in by[condition].values()], axis=0)
        c = np.cumsum(np.insert(per_position, 0, 0.0))
        n = np.arange(1, len(per_position) + 1)
        lo = np.maximum(0, n - window)
        out[s] = (n, (c[n] - c[lo]) / (n - lo))
    return out


def blocks(r: dict, condition: str, size: int = 50) -> dict:
    """{system: [accuracy in positions 1-50, 51-100, ...]} averaged over the six tasks."""
    out = {}
    for s, by in r.items():
        per_position = np.mean([np.array(x, dtype=float) for x in by[condition].values()], axis=0)
        out[s] = [float(per_position[i:i + size].mean()) for i in range(0, len(per_position), size)]
    return out
