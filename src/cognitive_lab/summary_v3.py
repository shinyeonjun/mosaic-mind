"""Collect the MK1 v3 results for notebook 23 (design/processing-flow-v3-2026-10-10.md, 1-12). No model runs here.

Reads the saved result files (mk1v3-*) in artifacts/results. Numbers that only live in the design document (the v2
hand-wired dev nets, the headroom table of section 8, the scoreboard rows of README.md) are copied here as constants
with their section. growth.stream() rebuilds the self-growth stream order from data/taskstream (CPU only).
Everything returns plain Python structures.
"""

import json
from collections import Counter

from cognitive_lab.mk1v3.signals import ROOT

RESULTS = ROOT / "artifacts" / "results"
ARCHIVE = RESULTS / "archive" / "v3-cache-on"

# design/flow-analysis-2026-10-10.md 1 and processing-flow-v3 3: the four hand-wired v2 flows and the one v3 loop.
FLOWS_V2 = [
    ("RGB", "말하기 + 읽기 + 짧은 답 -> NLI 지지도", "손 문턱 2개 (dev 격자 탐색)"),
    ("MuSiQue", "말하기 + 선택기 사슬 -> 다단계 독해기", "손 문턱 3개 (dev 격자 탐색)"),
    ("과제 흐름", "말하기 + 과제마다 키우는 부품", "세는 중재자 (피드백으로 학습)"),
    ("ARC", "탐색기 먼저 -> 못 찾으면 말하기", "고정 순서"),
]
LOOP_V3 = ["형태 보기", "할 일 목록", "다음 고르기 (싼 것 먼저)", "실행 -> 칠판",
           "검증", "보정 P(맞음)", "멈춤: 기대 점수", "말하기", "배우기 (피드백)"]

# processing-flow-v3 7 (v2 hand-wired dev nets on the same dev items; v2 thresholds tuned on this dev set).
V2_DEV_NET = {"rgb": 0.823, "musique": 0.560}

# processing-flow-v3 8: headroom from the recorded signals (perfect decision, with / without rescue).
HEADROOM = [  # (dataset, v3, perfect decision without rescue, + reader candidate, better deciding, rescue) as printed
    ("SQuAD", 0.713, 0.973, 0.987, 0.260, 0.013),
    ("HotpotQA", 0.557, 0.837, 0.863, 0.280, 0.027),
    ("RGB 개발용", 0.777, 0.954, 0.962, 0.177, 0.008),
    ("MuSiQue 개발용", 0.570, 0.680, 0.800, 0.110, 0.120),
]

SIGNAL_NAMES = {
    "lp_mean": "로그확률 평균", "lp_min": "로그확률 최소", "lp_first": "첫 토큰 로그확률", "gap_mean": "1-2위 차이",
    "length": "답 길이", "sample_same_choice": "샘플 같은 답", "sample_overlap": "샘플 겹침",
    "reader_margin": "읽기 위성 여백", "reply_has_reader": "답 안에 읽기 답", "chain1": "사슬 점수 1",
    "chain2": "사슬 점수 2", "rejected": "말하기 위성 거절",
}

GROWTH_SYSTEMS = [  # (key, plain name) in the order of the doc's table
    ("oracle", "과제를 알려 줌 (상한)"), ("e4b", "E4B 혼자"), ("mk1", "MK1 (자라기 + 말하기)"),
    ("growth", "자라는 부품 혼자"), ("mk1-fixed", "MK1 자라기 끔"), ("one", "부품 하나"), ("e2b", "E2B 혼자"),
]

# README.md rows for MK1 v2 / v3 (external exams, sealed once each), result vs E4B.
SCOREBOARD = [
    ("판단 (언제 거절할지)", "RGB", "MK1 v2", "0.441 vs 0.440", "비김"),
    ("생각 (여러 단계 추론)", "MuSiQue", "MK1 v2 + 생각 위성", "0.506 vs 0.587", "짐"),
    ("공부 (피드백으로 새 과제)", "새 과제 흐름 기호", "MK1 v2 + 키우는 부품 (bge 눈)", "0.752 vs 0.630", "이김"),
    ("공부 (피드백으로 새 과제)", "새 과제 흐름 이름", "MK1 v2 + 키우는 부품 (bge 눈)", "0.764 vs 0.788", "비김"),
    ("생각 (격자 규칙 찾기)", "ARC-AGI-1 eval", "MK1 v2 + 격자 생각 위성", "0.074 vs 0.023", "이김"),
    ("엮기 (종류 모르고 하나로)", "섞인 봉인: 기호", "MK1 v3", "0.725 vs 0.662", "이김"),
    ("엮기 (종류 모르고 하나로)", "섞인 봉인: 이름", "MK1 v3", "0.750 vs 0.796", "짐"),
    ("엮기 (종류 모르고 하나로)", "섞인 봉인: MuSiQue", "MK1 v3", "0.588 vs 0.583", "비김"),
    ("성장 (과제 이름 없이 자라기)", "자라기 봉인 흐름", "MK1 v3 + 스스로 자라기", "0.556 vs 0.599", "비김"),
]


def _json(name: str, folder=RESULTS):
    return json.loads((folder / name).read_text(encoding="utf-8"))


def _jsonl(name: str) -> list[dict]:
    return [json.loads(line) for line in (RESULTS / name).read_text(encoding="utf-8").splitlines() if line.strip()]


def calibrator() -> dict:
    return _json("mk1v3-calibrator-report.json")


def exam_dev() -> dict:
    """Transfer to exam dev (nothing learned on exam data), with the v2 hand-wired nets from the doc."""
    d = _json("mk1v3-exam-dev-check.json")
    return {k: {**d[k], "net_v2_hand_wired": V2_DEV_NET[k]} for k in ("rgb", "musique")}


def mixed_dev() -> dict:
    """The mixed four-exam dev stream (cache off). The file's per_exam_reference for RGB/MuSiQue is the cache-on value
    (written one minute before mixed.py switched to the cache-off reference), so they are replaced by the cache-off
    per-exam v3 nets from mk1v3-exam-dev-check.json, which is what the doc's table (10) compares against."""
    d = _json("mk1v3-mixed-dev.json")
    ref = exam_dev()
    out = {}
    for k, v in d.items():
        v = dict(v)
        v["file_reference"] = v["per_exam_reference"]
        if k in ref:
            v["per_exam_reference"] = ref[k]["net_v3_calibrated"]
        out[k] = v
    return out


def mixed_dev_cache_on() -> dict:
    """The first mixed run with llama.cpp's prompt cache on (archive): the determinism story of section 10."""
    return _json("mk1v3-mixed-dev.json", ARCHIVE)


def sealed() -> dict:
    """Sealed mixed exam: the verdict plus per-item speaker calls for v3 on MuSiQue (re-counted from the records)."""
    v = _json("mk1v3-sealed_verdict.json")
    recs = _jsonl("mk1v3-sealed_v3.jsonl")
    mus = [r for r in recs if r["id"].startswith("musique|")]
    calls = [sum(c == "speaker" for c in r["calls"]) for r in mus]
    v["musique_speaker_calls"] = Counter(calls)
    v["musique_speaker_calls_mean"] = sum(calls) / len(calls)
    v["musique_skipped_share"] = sum("speaker-skipped" in r["calls"] for r in mus) / len(mus)
    v["items"] = {"musique": len(mus), "named": sum("|named|" in r["id"] for r in recs),
                  "symbolic": sum("|symbolic|" in r["id"] for r in recs)}
    return v


def growth_verdict(part: str) -> dict:
    return _json(f"mk1v3-growth_verdict_{part}.json")


def growth_stream(part: str) -> dict:
    """The self-growth stream with every system's per-input right/wrong (and the part used, for growers), in stream
    order. Returns {"tasks": [...], "blocks": [...], "right": {system: [...]}, "part": {system: [...]}}."""
    from cognitive_lab.mk1v3.growth import stream

    items = stream(part)
    order = {it["id"]: i for i, it in enumerate(items)}
    right, used = {}, {}
    for key, _ in GROWTH_SYSTEMS:
        d = _json(f"mk1v3-growth_{key}_{part}.json")
        r = [None] * len(items)
        p = [None] * len(items)
        for rec in d["records"]:
            i = order[rec["id"]]
            r[i] = rec["right"]
            p[i] = rec.get("part")
        assert None not in r, f"{key}: records do not cover the stream"
        right[key] = r
        if any(x is not None for x in p):
            used[key] = p
    return {"tasks": [it["task"] for it in items], "blocks": [it["block"] for it in items], "right": right, "part": used}


def rolling(xs: list, w: int = 20) -> list:
    """Trailing mean over the last w inputs (shorter at the start)."""
    out, s = [], 0.0
    for i, x in enumerate(xs):
        s += x
        if i >= w:
            s -= xs[i - w]
        out.append(s / min(i + 1, w))
    return out


def part_task_counts(tasks: list, parts: list) -> tuple[list, list, list]:
    """(task names, part ids, counts[task][part])."""
    tnames = sorted(set(tasks), key=tasks.index)
    pids = sorted(set(parts))
    c = Counter(zip(tasks, parts))
    return tnames, pids, [[c[(t, p)] for p in pids] for t in tnames]


def signed(x: float, places: int = 3) -> str:
    """+0.063 style, rounding halves away from zero as the doc does (-0.0075 -> -0.008), not Python's float rounding."""
    from decimal import ROUND_HALF_UP, Decimal

    q = Decimal(str(x)).quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_UP)
    return f"{q:+}"
