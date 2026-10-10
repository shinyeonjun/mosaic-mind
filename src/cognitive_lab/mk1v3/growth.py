"""Self-growth (design/mk1-v3-growth, design/processing-flow-v3-2026-10-10.md 12): grow parts without being told tasks.

Stream: five text-classification tasks (SUBJ, AG News, SST-2, TweetEval sentiment, Emotion; TREC has no rows left)
in blocks of random length (20-60), switching without notice and coming back; every task uses the SAME labels A-F, so
nothing but the text and the feedback says which task is on. Each task's rows are used once, in order.
  dev     rows 0-160 of each task (opened before)          sealed  rows 700-860 (never used; SST-2 has 872 rows)

The growing learner (contextual inference, as the COIN model of Heald et al. 2021 and Bayesian change-point detection):
  parts         one prototype part per inferred context (bge-large, normalised, centred)
  which part    after each feedback, the last W items are scored under every hypothesis:
                  an existing part: its label log-likelihood + kappa * (cosine to its centre - c0) per item
                  the current part: its live record + the same look term, + a stickiness bonus
                  a NEW context: a fresh part learning prequentially over the window (it earns every prediction)
                plus a prior from usage (Chinese restaurant: existing by count, new by alpha)
                the best hypothesis becomes the current part; a new one is grown from the window
  speaker       Gemma 4 E2B with the 16 most recent feedback examples (whatever task they came from)
  answer        counting weights per (part, channel), Beta(1, 1) start, as the label stream in loop.py
  take-back     when a switch is detected, the window's items leave the part that learned them during the lag
mk1-fixed is the same MK1 with growth switched off (one part + the speaker): the ablation for "does growing help".
Frozen settings, chosen on dev by accuracy: W = 5, kappa = 20, c0 = 0.65, alpha = 0.5, stay = 2, beta = 12.

python -m cognitive_lab.mk1v3.growth --system growth --set dev
"""

import argparse
import json
import math
import random
import statistics

import torch

from cognitive_lab.mk1v3.signals import ROOT

RESULTS = ROOT / "artifacts" / "results"
TASK_NAMES = ["ag_news", "emotion", "sst2", "subj", "tweet_sentiment"]
SYMBOLS = list("ABCDEF")
L = 6
ROW_RANGE = {"dev": (0, 160), "sealed": (700, 860)}
K_EXAMPLES = 16
SETTINGS = {"W": 5, "kappa": 20.0, "c0": 0.65, "alpha": 0.5, "stay": 2.0, "beta": 12.0}


def stream(part: str) -> list[dict]:
    from cognitive_lab.mk1v2.taskstream import _rows

    lo, hi = ROW_RANGE[part]
    rows = {}
    for t in TASK_NAMES:
        r = _rows(t)
        random.Random(f"taskstream|{t}").shuffle(r)
        rows[t] = r[lo:hi]
    rng = random.Random(f"growth-blocks|{part}")
    pos = {t: 0 for t in TASK_NAMES}
    out, prev, block = [], None, 0
    while any(pos[t] < len(rows[t]) for t in TASK_NAMES):
        live = [t for t in TASK_NAMES if pos[t] < len(rows[t]) and t != prev] or [t for t in TASK_NAMES if pos[t] < len(rows[t])]
        t = rng.choice(live)
        n = min(rng.randint(20, 60), len(rows[t]) - pos[t])
        for _ in range(n):
            x = rows[t][pos[t]]
            out.append({"id": f"{part}|{t}|{pos[t]}", "text": x["text"], "label": x["label"], "task": t, "block": block})
            pos[t] += 1
        prev, block = t, block + 1
    return out


class Part:
    def __init__(self, dim: int):
        self.s, self.c, self.tot, self.n = torch.zeros(L, dim), torch.zeros(L), torch.zeros(dim), 0
        self.hits = {"part": [0.0, 0.0], "speaker": [0.0, 0.0]}

    def probs(self, v: torch.Tensor, beta: float, eps: float = 0.02) -> torch.Tensor:
        if self.n == 0:
            return torch.full((L,), 1.0 / L)
        m = self.tot / self.n
        means = torch.nn.functional.normalize(self.s / self.c.clamp(min=1)[:, None] - m, dim=-1)
        sims = means @ torch.nn.functional.normalize(v - m, dim=0)
        seen = self.c > 0
        p = torch.zeros(L)
        p[seen] = torch.softmax(beta * sims[seen], 0)
        if seen.all():
            return (1 - eps) * p + eps / L
        return torch.where(seen, (1 - eps) * p, torch.full((L,), eps / float((~seen).sum())))

    def centre(self) -> torch.Tensor:
        return self.tot / max(self.n, 1)

    def learn(self, v: torch.Tensor, y: int, sign: float = 1.0) -> None:
        self.s[y] += sign * v
        self.c[y] += sign
        self.tot += sign * v
        self.n += int(sign)


class Grower:
    """Infers the context, grows parts, and keeps per-part channel counts."""

    def __init__(self, dim: int, **settings):
        self.cfg = {**SETTINGS, **settings}
        self.dim = dim
        self.parts, self.usage, self.cur = [Part(dim)], [0.0], 0
        self.window: list[tuple[torch.Tensor, int, bool]] = []  # (vector, label, current part was right)
        self.grown = 0

    def look(self, part: Part, items) -> float:
        if part.n == 0:
            return 0.0
        c = part.centre()
        return self.cfg["kappa"] * sum(float(torch.nn.functional.cosine_similarity(v, c, dim=0)) - self.cfg["c0"] for v, _, _ in items)

    def guess(self, v: torch.Tensor) -> int:
        return int(self.parts[self.cur].probs(v, self.cfg["beta"]).argmax())

    def feedback(self, v: torch.Tensor, y: int, part_right: bool) -> None:
        cfg = self.cfg
        part = self.parts[self.cur]
        part.learn(v, y)
        self.usage[self.cur] += 1
        self.window = (self.window + [(v, y, part_right)])[-cfg["W"]:]
        if len(self.window) < cfg["W"] or not cfg.get("grow", True):
            return
        tot = sum(self.usage)
        best_j = self.cur
        best = (math.log(self.usage[self.cur] / (tot + cfg["alpha"])) + cfg["stay"]
                + sum(math.log(0.8 if r else 0.2 / (L - 1)) for _, _, r in self.window) + self.look(part, self.window))
        for j, p in enumerate(self.parts):
            if j == self.cur or p.n == 0:
                continue
            s = (math.log(self.usage[j] / (tot + cfg["alpha"]))
                 + sum(math.log(float(p.probs(x, cfg["beta"])[yy])) for x, yy, _ in self.window) + self.look(p, self.window))
            if s > best:
                best_j, best = j, s
        fresh, ll = Part(self.dim), 0.0
        for x, yy, _ in self.window:
            ll += math.log(float(fresh.probs(x, cfg["beta"])[yy]))
            fresh.learn(x, yy)
        if math.log(cfg["alpha"] / (tot + cfg["alpha"])) + ll > best:
            self._take_back()
            self.parts.append(fresh)
            self.usage.append(float(fresh.n))
            self.cur, self.window, self.grown = len(self.parts) - 1, [], self.grown + 1
        elif best_j != self.cur:
            self._take_back(to=best_j)
            self.cur, self.window = best_j, []

    def _take_back(self, to: int | None = None) -> None:
        """The window's items belonged to the new context: remove them from the part that learned them during the
        detection lag (and give them to the chosen existing part; a new part has learned them already)."""
        if not self.cfg.get("reassign", True):
            return
        old = self.parts[self.cur]
        for x, yy, _ in self.window:
            old.learn(x, yy, sign=-1.0)
            self.usage[self.cur] -= 1
            if to is not None:
                self.parts[to].learn(x, yy)
                self.usage[to] += 1


def combine(part: Part, said: dict) -> int:
    log_p = torch.zeros(L)
    for channel, k in said.items():
        r, s = part.hits[channel]
        acc = (r + 1.0) / (s + 2.0)
        like = torch.full((L,), max(1e-6, (1 - acc) / (L - 1)))
        like[k] = max(1e-6, acc)
        log_p += like.log()
    return int(log_p.argmax())


def speaker_prompt(examples, text: str) -> str:
    shots = "\n".join(f"Text: {t}\nLabel: {SYMBOLS[y]}" for t, y in examples[-K_EXAMPLES:])
    return (f"Classify the text into exactly one of these labels: {', '.join(SYMBOLS)}.\n"
            f"{'Labelled examples:' + chr(10) + shots + chr(10) if shots else ''}"
            f"Text: {text}\nAnswer with the label only.")


def run(system: str, part: str) -> dict:
    items = stream(part)
    records = []
    needs_llm = system in ("mk1", "mk1-fixed", "e2b", "e4b")
    needs_eye = system in ("mk1", "mk1-fixed", "growth", "one", "oracle")
    llm = None
    if needs_llm:
        from cognitive_lab.mk1v2.llm_service import LLMService

        llm = LLMService("e2b" if system.startswith("mk1") else system).__enter__()
    try:
        vectors = None
        if needs_eye:
            from cognitive_lab.mk1v2.taskstream import Embedder

            eye = Embedder(torch.device("cuda" if torch.cuda.is_available() else "cpu"), "bge-large")
            vectors = eye([it["text"] for it in items])
        grower = (Grower(vectors.shape[1], grow=system != "mk1-fixed") if system in ("mk1", "mk1-fixed", "growth") else None)
        fixed: dict[str, Part] = {}
        examples: list[tuple[str, int]] = []
        from cognitive_lab.mk1v2.taskstream import parse

        for n, it in enumerate(items):
            y = it["label"]
            said = {}
            if system in ("one", "oracle"):
                p = fixed.setdefault("one" if system == "one" else it["task"], Part(vectors.shape[1]))
                answer = int(p.probs(vectors[n], SETTINGS["beta"]).argmax())
                p.learn(vectors[n], y)
            elif system in ("growth", "mk1", "mk1-fixed"):
                said["part"] = grower.guess(vectors[n])
                answer = said["part"]
            if system in ("mk1", "mk1-fixed", "e2b", "e4b"):
                reply = llm.chat([{"role": "user", "content": speaker_prompt(examples, it["text"])}], max_tokens=8).strip()
                g = parse(reply, SYMBOLS, "symbolic")
                if system.startswith("mk1"):
                    if g is not None:
                        said["speaker"] = g
                    answer = combine(grower.parts[grower.cur], said)
                else:
                    answer = -1 if g is None else g
            if grower is not None:
                part_obj = grower.parts[grower.cur]
                for channel, k in said.items():
                    part_obj.hits[channel][0] += float(k == y)
                    part_obj.hits[channel][1] += 1
                cur_before = grower.cur
                grower.feedback(vectors[n], y, said["part"] == y)
                records.append({"id": it["id"], "right": int(answer == y), "part": cur_before})
            else:
                records.append({"id": it["id"], "right": int(answer == y)})
            examples.append((it["text"], y))
            if n % 200 == 0:
                print(f"  {n}/{len(items)}", flush=True)
    finally:
        if llm is not None:
            llm.__exit__()
    out = {"system": system, "part": part, "inputs": len(items), "accuracy": round(statistics.mean(r["right"] for r in records), 4),
           "records": records}
    # recovery: accuracy in the first 10 items of blocks whose task was seen before
    seen_tasks, first10 = set(), []
    for it, r in zip(items, records):
        if it["block"] != getattr(run, "_b", None):
            run._b, returning = it["block"], it["task"] in seen_tasks
            k = 0
        if returning and k < 10:
            first10.append(r["right"])
        k += 1
        seen_tasks.add(it["task"])
    out["returning_first10_accuracy"] = round(statistics.mean(first10), 4) if first10 else None
    if grower is not None:
        by_part = {}
        for it, r in zip(items, records):
            by_part.setdefault(r["part"], []).append(it["task"])
        out["parts"] = len(grower.parts)
        out["purity"] = round(sum(max(map(ts.count, set(ts))) for ts in by_part.values()) / len(items), 4)
    return out


def compare(part: str) -> dict:
    """Paired bootstrap over blocks (2,000 draws, seed 0): resample whole blocks, the unit the stream is built from."""
    systems = {}
    for sname in ("mk1", "mk1-fixed", "e2b", "e4b", "growth", "one", "oracle"):
        path = RESULTS / f"mk1v3-growth_{sname}_{part}.json"
        if path.exists():
            systems[sname] = json.loads(path.read_text(encoding="utf-8"))
    items = stream(part)
    blocks = {}
    for i, it in enumerate(items):
        blocks.setdefault(it["block"], []).append(i)
    out = {"part": part, "accuracy": {k: v["accuracy"] for k, v in systems.items()},
           "returning_first10": {k: v["returning_first10_accuracy"] for k, v in systems.items()},
           "parts": {k: v.get("parts") for k, v in systems.items() if "parts" in v}, "versus": {}}
    rng = random.Random(0)
    keys = list(blocks)
    for opp in ("mk1-fixed", "e2b", "e4b"):
        if "mk1" not in systems or opp not in systems:
            continue
        a = [r["right"] for r in systems["mk1"]["records"]]
        b = [r["right"] for r in systems[opp]["records"]]
        draws = []
        for _ in range(2000):
            idx = [i for k in (rng.choice(keys) for _ in keys) for i in blocks[k]]
            draws.append(statistics.mean(a[i] - b[i] for i in idx))
        draws.sort()
        out["versus"][f"mk1 - {opp}"] = {"difference": round(statistics.mean(a) - statistics.mean(b), 4),
                                          "ci95": [round(draws[50], 4), round(draws[1949], 4)]}
    print(json.dumps(out, indent=1))
    (RESULTS / f"mk1v3-growth_verdict_{part}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Self-growth stream (no task ids)")
    parser.add_argument("--system", choices=("growth", "mk1", "mk1-fixed", "one", "oracle", "e2b", "e4b"))
    parser.add_argument("--set", choices=("dev", "sealed"), default="dev")
    parser.add_argument("--final", action="store_true")
    parser.add_argument("--compare", action="store_true")
    args = parser.parse_args()
    if args.compare:
        compare(args.set)
        return
    if args.set == "sealed" and not args.final:
        raise SystemExit("sealed: pass --final, once per system, after the pre-registration commit")
    out = run(args.system, args.set)
    path = RESULTS / f"mk1v3-growth_{args.system}_{args.set}.json"
    path.write_text(json.dumps(out), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "records"}))


if __name__ == "__main__":
    main()
