"""v3 steps 3-4 on development data: ONE MK1v3 instance answers a stream that mixes the development questions of the
four exams (RGB five testbeds, MuSiQue, ARC first 100 training tasks, the task stream in both conditions). It is never
told which exam a question comes from. Each exam is then scored by its own rule and compared with its per-exam number.
Task-stream items keep their order inside each task (the stream learns from feedback); everything else is interleaved
with a fixed seed. ARC: the speaker's attempts are the saved gemma-e2b dev attempts (identical calls, made once).

python -m cognitive_lab.mk1v3.mixed
"""

import json
import random
import statistics
import time

import torch

from cognitive_lab.mk1v3.signals import ROOT

RESULTS = ROOT / "artifacts" / "results"


def streams() -> dict[str, list[tuple[dict, object]]]:
    """exam -> [(input, truth)] in that exam's own order."""
    from cognitive_lab.mk1v2 import arc, taskstream
    from cognitive_lab.mk1v3.exam_check import items as doc_items

    out = {"rgb": [], "musique": [], "arc": [], "taskstream": []}
    for it in doc_items():
        out[it["source"]].append((it, it))
    for tid, task in arc.tasks("dev", 100):
        out["arc"].append(({**task, "id": tid}, [p["output"] for p in task["test"]]))
    for condition in ("named", "symbolic"):
        for task in taskstream.stream("dev", condition):
            for x in task["items"]:
                out["taskstream"].append(({"text": x["text"], "labels": task["labels"], "task": task["task"],
                                           "condition": condition}, x["label"]))
    return out


def merge_streams(per_exam: dict, seed: str = "v3-mixed") -> list[tuple[str, dict, object]]:
    rng = random.Random(seed)
    queues = {k: list(v) for k, v in per_exam.items()}
    out = []
    while any(queues.values()):
        k = rng.choices([k for k in queues if queues[k]], weights=[len(queues[k]) for k in queues if queues[k]])[0]
        out.append((k, *queues[k].pop(0)))
    return out


def doc_net(board, item) -> int:
    reply = board.read("reply")
    rejected = "insufficient information" in reply
    if not item["answerable"]:
        return 1 if rejected else -1
    if rejected:
        return 0
    if item["source"] == "rgb":
        from cognitive_lab.mk1v2 import rgb

        return 1 if rgb.judge(reply, item["truth"])["right"] else -1
    return 1 if any(a.lower() in reply.lower() for a in item["answers"] if a) else -1


def main() -> None:
    from cognitive_lab.mk1v2.arc import score as arc_score
    from cognitive_lab.mk1v2.llm_service import LLMService
    from cognitive_lab.mk1v3.loop import MK1v3

    stream = merge_streams(streams())
    print(f"{len(stream)} inputs, mixed", flush=True)
    saved = json.loads((RESULTS / "exam-arc_e2b_dev_100.json").read_text(encoding="utf-8"))["records"]
    scores = {"rgb": [], "musique": [], "arc": [], "taskstream|named": [], "taskstream|symbolic": []}
    calls = {k: [] for k in scores}
    seconds = {k: 0.0 for k in scores}
    started = time.perf_counter()
    with LLMService("e2b") as llm:
        mk1 = MK1v3(llm, torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        mk1.grid_speaker_cache = {r["id"]: r["attempts"] for r in saved}
        for n, (exam, item, truth) in enumerate(stream, 1):
            board = mk1.solve(item)
            key = f"taskstream|{item['condition']}" if exam == "taskstream" else exam
            if exam in ("rgb", "musique"):
                scores[key].append(doc_net(board, item))
            elif exam == "arc":
                scores[key].append(arc_score(board.read("attempts"), truth))
            else:
                scores[key].append(int(board.read("answer") == truth))
                mk1.feedback(board, truth)
            calls[key].append(sum(w in ("speaker",) for w in board.calls()))
            seconds[key] += board.cost()
            if n % 200 == 0:
                print(f"  {n}/{len(stream)} ({(time.perf_counter() - started) / 60:.0f} min)", flush=True)
    reference = {"rgb": 0.80, "musique": 0.58,  # per-exam v3 dev after the cache-off re-recording (exam_check)
                  "arc": 0.25, "taskstream|named": 0.715, "taskstream|symbolic": 0.6883}
    out = {k: {"n": len(v), "score": round(statistics.mean(v), 4), "per_exam_reference": reference[k],
               "speaker_calls_per_input": round(statistics.mean(calls[k]), 3), "seconds": round(seconds[k], 1)}
           for k, v in scores.items()}
    print(json.dumps(out, indent=1))
    (RESULTS / "mk1v3-mixed-dev.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()

