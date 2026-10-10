"""MK1 v3, step 3: one processing loop for every input (design/processing-flow-v3-2026-10-10.md, 3).

    mk1 = MK1v3(llm, device)
    board = mk1.solve(item)          # any input; the kind is read from its structure, never told
    mk1.feedback(item, truth)        # optional: the world reveals the answer; the loop learns from it

One board per input: typed slots plus a write log (who wrote what, at what cost in seconds).
  1 form       the kind from the input's structure: grid task / documents + question / labelled stream item
  2 cheap      cheap satellites first (reader, selector; grid search; the label part)
  3 value      call the expensive speaker only when it can change the decision (documents: skip when the cheap
               P(answerable) < 0.2, chosen on calibration data with no loss there)
  4 calibrate  calibrators learned off-exam turn signals into probabilities
  5 decide     the expected score under the task's scoring rule; documents: answer iff 2 P(right) - 1 > 1 - P(answerable)
  6 speak      the speaker's words or the rejection
  7 learn      feedback updates the parts that learn online (label stream: the grown part and the counting weights)
Kinds and their satellites:
  documents    reading satellite, hop selector, speaker (E2B) with log-probabilities and 3 samples, two calibrators
  grid         the grid search satellite (exact verifier: re-running the program on the demonstrations); the speaker's
               attempts fill when it has fewer than two answers (ARC scores no abstention)
  label        the part grown for this task (bge-large prototypes, normalised, centred), the speaker with the 16 most
               recent examples, counting weights per channel; there is always an answer
"""

import json
import time

import numpy as np
import torch

from cognitive_lab.mk1v3.calibrate import FEATURES, WEIGHTS, Logistic

REJECT = "I can not answer the question because of the insufficient information in documents."
CHEAP_FEATURES = ["reader_margin", "chain1", "chain2"]
SKIP_SPEAKER_BELOW = 0.2


class Board:
    def __init__(self, item: dict):
        self.slots, self.log = {"input": item}, []

    def write(self, slot: str, value, who: str, seconds: float = 0.0) -> None:
        self.slots[slot] = value
        self.log.append({"slot": slot, "who": who, "seconds": round(seconds, 3)})

    def read(self, slot: str):
        return self.slots[slot]

    def cost(self) -> float:
        return sum(e["seconds"] for e in self.log)

    def calls(self) -> list[str]:
        return [e["who"] for e in self.log]


def kind(item: dict) -> str:
    if "train" in item and "test" in item:
        return "grid"
    if "docs" in item and "query" in item:
        return "documents"
    if "text" in item and "labels" in item:
        return "label"
    raise ValueError("unknown input form")


class LabelPart:
    """The part grown for one labelled task (as mk1v2/taskstream.py run_mk1, version v2)."""

    def __init__(self, n: int, dim: int):
        self.sums, self.counts, self.total = torch.zeros(n, dim), torch.zeros(n), torch.zeros(dim)
        self.seen = 0
        self.hits = {"proto": [0.0, 0.0], "speaker": [0.0, 0.0]}
        self.examples: list[tuple[str, int]] = []

    def guess(self, v: torch.Tensor) -> int | None:
        if self.counts.sum() == 0:
            return None
        m = self.total / self.seen
        means = torch.nn.functional.normalize(self.sums / self.counts.clamp(min=1)[:, None] - m, dim=-1)
        sims = torch.where(self.counts > 0, means @ torch.nn.functional.normalize(v - m, dim=0), torch.full((len(self.counts),), -9.0))
        return int(sims.argmax())

    def learn(self, v: torch.Tensor, label: int, text: str) -> None:
        self.sums[label] += v
        self.counts[label] += 1
        self.total += v
        self.seen += 1
        self.examples.append((text, label))


class MK1v3:
    def __init__(self, llm, device: torch.device):
        self.llm, self.device = llm, device
        saved = json.loads(WEIGHTS.read_text(encoding="utf-8"))
        self.models = {h: Logistic.from_dict(saved["models"][h]) for h in FEATURES}
        self.cheap = None
        self._recorder = self._embed = None
        self.parts: dict[str, LabelPart] = {}
        self.grid_speaker_cache: dict[str, list] = {}  # task id -> the speaker's saved attempts (identical calls)

    # ---------- satellites, loaded when first needed ----------
    def recorder(self):
        if self._recorder is None:
            from cognitive_lab.mk1v3.signals import Recorder

            self._recorder = Recorder(self.llm, self.device)
        return self._recorder

    def embed(self):
        if self._embed is None:
            from cognitive_lab.mk1v2.taskstream import Embedder

            self._embed = Embedder(self.device, "bge-large")
        return self._embed

    def cheap_model(self):
        """P(answerable) from cheap signals only, learned on the calibration data (not exams)."""
        if self.cheap is None:
            from cognitive_lab.mk1v3.calibrate import SIGNALS, merge

            rows = merge([json.loads(l) for l in SIGNALS.read_text(encoding="utf-8").splitlines()])
            X = np.array([[r[f] for f in CHEAP_FEATURES] for r in rows])
            self.cheap = Logistic().fit(X, np.array([float(r["answerable"]) for r in rows]))
        return self.cheap

    # ---------- the loop ----------
    def solve(self, item: dict) -> Board:
        board = Board(item)
        k = kind(item)
        board.write("kind", k, "form")
        getattr(self, f"_{k}")(board, item)
        return board

    def _documents(self, board: Board, item: dict) -> None:
        rec = self.recorder()
        t = time.perf_counter()
        reads = rec.reader.read_many([(item["query"], d) for d in item["docs"]])
        best = max(reads, key=lambda r: r["span_score"] - r["null_score"])
        board.write("reader", {"answer": best["answer"], "margin": best["span_score"] - best["null_score"]}, "reader", time.perf_counter() - t)
        t = time.perf_counter()
        chain = rec.selector.chain(item["query"], item["docs"], hops=2)
        board.write("chain", [c for _, c in chain], "selector", time.perf_counter() - t)
        cheap = [board.read("reader")["margin"], chain[0][1], chain[1][1] if len(chain) > 1 else 0.0]
        p_cheap = float(self.cheap_model().predict(np.array([cheap]))[0])
        board.write("p_answerable_cheap", p_cheap, "calibrator")
        if p_cheap < SKIP_SPEAKER_BELOW:  # the speaker cannot make answering worth it: do not call it
            board.write("decision", "reject", "decide")
            board.write("reply", REJECT, "speaker-skipped")
            return
        t = time.perf_counter()
        signals = rec.record(item)  # the speaker with log-probabilities and samples (re-reads documents; cheap)
        board.write("signals", signals, "speaker", time.perf_counter() - t)
        signals["rejected"] = float(signals["rejected"])
        from cognitive_lab.mk1v3.calibrate import matrix

        p_ans = float(self.models["answerable"].predict(matrix([signals], "answerable")[0])[0])
        board.write("p_answerable", p_ans, "calibrator")
        if signals["rejected"]:
            board.write("decision", "reject", "decide")
            board.write("reply", signals["reply"], "speaker")
            return
        p_right = float(self.models["right"].predict(matrix([signals], "right")[0])[0])
        board.write("p_right", p_right, "calibrator")
        answer = 2 * p_right - 1 > 1 - p_ans
        board.write("decision", "answer" if answer else "reject", "decide")
        board.write("reply", signals["reply"] if answer else REJECT, "speaker")

    def _grid(self, board: Board, task: dict) -> None:
        from cognitive_lab.mk1v2.arc import gemma_attempts
        from cognitive_lab.mk1v2.arc_search import solve

        t = time.perf_counter()
        found = solve(task)
        board.write("programs", [" > ".join(p) for p, _ in found], "grid-search", time.perf_counter() - t)
        attempts = []
        for i, test in enumerate(task["test"]):
            mine = [outs[i] for _, outs in found]
            if len(mine) < 2:  # no abstention cost in ARC: let the speaker fill the remaining tries
                t = time.perf_counter()
                cached = self.grid_speaker_cache.get(task.get("id", ""))
                extra = cached[i] if cached else gemma_attempts(self.llm, task, test["input"])[0]
                board.write(f"speaker_attempts_{i}", extra, "speaker", time.perf_counter() - t)
                mine = mine + extra[:2 - len(mine)]
            attempts.append(mine)
        board.write("attempts", attempts, "decide")

    def _label(self, board: Board, item: dict) -> None:
        from cognitive_lab.mk1v2.taskstream import K_EXAMPLES, parse

        labels, task = item["labels"], item["task"]
        t = time.perf_counter()
        v = self.embed()([item["text"]])[0]
        part = self.parts.setdefault(f"{task}|{'/'.join(labels)}", LabelPart(len(labels), v.shape[0]))
        board.write("vector", v, "eye", time.perf_counter() - t)
        said = {}
        if (g := part.guess(v)) is not None:
            said["proto"] = g
        shots = "\n".join(f"Text: {x}\nLabel: {labels[y]}" for x, y in part.examples[-K_EXAMPLES:])
        prompt = (f"Classify the text into exactly one of these labels: {', '.join(labels)}.\n"
                  f"{'Labelled examples:' + chr(10) + shots + chr(10) if shots else ''}"
                  f"Text: {item['text']}\nAnswer with the label only.")
        t = time.perf_counter()
        reply = self.llm.chat([{"role": "user", "content": prompt}], max_tokens=8).strip()
        condition = "symbolic" if all(len(l) == 1 for l in labels) else "named"
        if (g := parse(reply, labels, condition)) is not None:
            said["speaker"] = g
        board.write("said", said, "speaker", time.perf_counter() - t)
        n = len(labels)
        log_p = torch.zeros(n)
        for channel, k in said.items():  # counting weights, Beta(1, 1) start
            r, s = part.hits[channel]
            acc = (r + 1.0) / (s + 2.0)
            like = torch.full((n,), max(1e-6, (1 - acc) / (n - 1)))
            like[k] = max(1e-6, acc)
            log_p += like.log()
        board.write("answer", int(log_p.argmax()) if said else 0, "decide")

    def feedback(self, board: Board, truth) -> None:
        """The world reveals the answer. Only the label stream learns online in this version."""
        if board.read("kind") != "label":
            return
        item = board.read("input")
        part = self.parts[f"{item['task']}|{'/'.join(item['labels'])}"]
        for channel, k in board.read("said").items():
            part.hits[channel][0] += float(k == truth)
            part.hits[channel][1] += 1
        part.learn(board.read("vector"), truth, item["text"])
