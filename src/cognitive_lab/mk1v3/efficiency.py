"""Step 0 of the antiques path (design/antiques-rethink-2026-10-12.md, 7): the efficiency numbers that were never
measured. The same fixed development items, three systems, one machine, nothing else on the GPU:

  v3    MK1 v3 (E2B speaker + reader + selector + bge-large + grid search), the single loop, live speaker on grids
  e2b   Gemma 4 E2B alone      e4b   Gemma 4 E4B alone   (the RGB prompt on documents; two attempts on grids;
        the 16 most recent feedback examples on labels)
Items (seeded subsets of the v3 mixed development stream): RGB 40, MuSiQue 40, ARC training 10, task stream 40 named
+ 40 symbolic (the first 40 items of the first task in stream order, with feedback, so learning happens as usual).
Measured: wall seconds per item by kind, speaker calls per item, GPU memory used (nvidia-smi, sampled every 0.5 s):
after the model server is up, and the peak over the run. Every system gets the same llama-server context (16384,
what grids need), so KV-cache memory is the same condition for all.

python -m cognitive_lab.mk1v3.efficiency --system v3
python -m cognitive_lab.mk1v3.efficiency --compare
"""

import argparse
import json
import random
import statistics
import subprocess
import threading
import time

import torch

from cognitive_lab.mk1v3.mixed import RESULTS, doc_net, streams

N = {"rgb": 40, "musique": 40, "arc": 10, "taskstream": 40}
CONTEXT = 16384


class Vram:
    """Peak GPU memory in MiB from nvidia-smi while running."""

    def __init__(self):
        self.peak, self.samples, self.stop = 0, [], False
        self.thread = threading.Thread(target=self._loop, daemon=True)

    @staticmethod
    def read() -> int:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True)
        return int(out.stdout.strip().splitlines()[0])

    def _loop(self):
        while not self.stop:
            v = self.read()
            self.samples.append(v)
            self.peak = max(self.peak, v)
            time.sleep(0.5)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop = True
        self.thread.join(timeout=2)


def items() -> list[tuple[str, dict, object]]:
    per = streams()
    rng = random.Random("efficiency")
    out = []
    for exam in ("rgb", "musique", "arc"):
        out += [(exam, it, truth) for it, truth in rng.sample(per[exam], N[exam])]
    for condition in ("named", "symbolic"):
        first = [(it, truth) for it, truth in per["taskstream"] if it["condition"] == condition]
        task0 = first[0][0]["task"]
        out += [("taskstream", it, truth) for it, truth in first if it["task"] == task0][:N["taskstream"]]
    return out


class Reply:
    def __init__(self, reply: str):
        self.reply = reply

    def read(self, _slot: str):
        return self.reply


def alone(llm, exam: str, item: dict, recent: dict) -> tuple[object, int]:
    """The single model's answer and how many speaker calls it took."""
    from cognitive_lab.mk1v2 import rgb
    from cognitive_lab.mk1v2.arc import gemma_attempts
    from cognitive_lab.mk1v2.taskstream import K_EXAMPLES, parse

    if exam in ("rgb", "musique"):
        msgs = [{"role": "system", "content": rgb.SYSTEM},
                {"role": "user", "content": rgb.INSTRUCTION.format(DOCS="\n".join(item["docs"]), QUERY=item["query"])}]
        return llm.chat(msgs, max_tokens=200), 1
    if exam == "arc":
        attempts = [gemma_attempts(llm, item, t["input"])[0] for t in item["test"]]
        return attempts, 2 * len(item["test"])
    labels = item["labels"]
    key = "/".join(labels)
    shots = "\n".join(f"Text: {x}\nLabel: {labels[y]}" for x, y in recent.get(key, [])[-K_EXAMPLES:])
    prompt = (f"Classify the text into exactly one of these labels: {', '.join(labels)}.\n"
              f"{'Labelled examples:' + chr(10) + shots + chr(10) if shots else ''}"
              f"Text: {item['text']}\nAnswer with the label only.")
    g = parse(llm.chat([{"role": "user", "content": prompt}], max_tokens=8).strip(), labels,
              "symbolic" if all(len(l) == 1 for l in labels) else "named")
    return (-1 if g is None else g), 1


def run(system: str) -> dict:
    from cognitive_lab.mk1v2.arc import score as arc_score
    from cognitive_lab.mk1v2.llm_service import LLMService

    rows = []
    before = Vram.read()
    with Vram() as vram, LLMService("e2b" if system == "v3" else system, context=CONTEXT) as llm:
        after_server = Vram.read()
        mk1 = None
        if system == "v3":
            from cognitive_lab.mk1v3.loop import MK1v3

            mk1 = MK1v3(llm, torch.device("cuda"))
        recent: dict[str, list] = {}
        started = time.perf_counter()
        for n, (exam, item, truth) in enumerate(items(), 1):
            key = f"taskstream|{item['condition']}" if exam == "taskstream" else exam
            t = time.perf_counter()
            if mk1 is not None:
                board = mk1.solve(item)
                calls = sum(w == "speaker" for w in board.calls())
                if exam in ("rgb", "musique"):
                    score = doc_net(board, item)
                elif exam == "arc":
                    score = arc_score(board.read("attempts"), truth)
                else:
                    score = int(board.read("answer") == truth)
                    mk1.feedback(board, truth)
            else:
                out, calls = alone(llm, exam, item, recent)
                if exam in ("rgb", "musique"):
                    score = doc_net(Reply(out), item)
                elif exam == "arc":
                    score = arc_score(out, truth)
                else:
                    score = int(out == truth)
                    recent.setdefault("/".join(item["labels"]), []).append((item["text"], truth))
            rows.append({"key": key, "seconds": round(time.perf_counter() - t, 3), "speaker_calls": calls, "score": score})
            if n % 20 == 0:
                print(f"  {n} items, {(time.perf_counter() - started) / 60:.1f} min, peak {vram.peak} MiB", flush=True)
    out = {"system": system, "context": CONTEXT, "vram_mib": {"before": before, "after_server": after_server, "peak": vram.peak},
           "by_kind": {}, "rows": rows}
    for key in sorted({r["key"] for r in rows}):
        rs = [r for r in rows if r["key"] == key]
        out["by_kind"][key] = {"n": len(rs), "seconds_per_item": round(statistics.mean(r["seconds"] for r in rs), 2),
                               "speaker_calls_per_item": round(statistics.mean(r["speaker_calls"] for r in rs), 2),
                               "score": round(statistics.mean(r["score"] for r in rs), 3)}
    (RESULTS / f"efficiency_{system}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=1))
    return out


def compare() -> None:
    res = {s: json.loads((RESULTS / f"efficiency_{s}.json").read_text(encoding="utf-8"))
           for s in ("v3", "e2b", "e4b") if (RESULTS / f"efficiency_{s}.json").exists()}
    print("system   after-server MiB   peak MiB")
    for s, r in res.items():
        print(f"{s:6s}   {r['vram_mib']['after_server']:>14}   {r['vram_mib']['peak']:>8}")
    kinds = sorted({k for r in res.values() for k in r["by_kind"]})
    print("\nkind                  " + "".join(f"{s:>22s}" for s in res))
    for k in kinds:
        print(f"{k:20s}  " + "".join(f"{r['by_kind'][k]['seconds_per_item']:>8.2f}s {r['by_kind'][k]['speaker_calls_per_item']:>4.2f}c {r['by_kind'][k]['score']:>5.2f}"
                                    for r in res.values()))
    (RESULTS / "efficiency_compare.json").write_text(json.dumps({s: {"vram_mib": r["vram_mib"], "by_kind": r["by_kind"]} for s, r in res.items()}, indent=1), encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--system", choices=("v3", "e2b", "e4b"))
    p.add_argument("--compare", action="store_true")
    a = p.parse_args()
    compare() if a.compare else run(a.system)


if __name__ == "__main__":
    main()
