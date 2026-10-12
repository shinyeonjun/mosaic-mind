"""Self-directed study (design/self-study-2026-10-12.md): with the same budget of questions ("what is the label of
this text?"), does choosing what to study beat studying at random?

World: five text-classification tasks (AG News, Emotion, MR, SUBJ, TweetEval sentiment) mixed in one unlabelled pool,
no task names, and the SAME label symbols A-F across tasks. Per task: 200 pool texts, 60 exam texts (never seen while
studying). The learner is the same for every policy; only the choice of what to ask differs.

  learner      k nearest labelled neighbours on bge-large vectors (normalised, centred on the pool mean),
               cosine-weighted votes; uncertainty = 1 - top vote share
  random       ask about a random pool text
  uncertain    ask about the pool text the learner is least sure of (standard active learning)
  self         the system splits the pool into its own topics (k-means, K regions, no task names), keeps per topic
               whether its guess was right before each answer came back (an honest, prequential record), and asks
               where   value = topic share x recent error x progress
               recent error  = (wrong + 1) / (n + 2) over the last 8 questions in the topic
               progress      = 1 until the topic has 8 questions; then max(floor, 1 + 2 * (older error - newer error))
                               over the last 8 split 4/4 (topics where asking stopped helping lose priority)
               then the least-sure text inside that topic

Rows (fixed shuffled order of mk1v3/growth.py; earlier exams used rows 0-500 and 700-1020, MR 0-160):
  dev     rows 1020-1280 (MR 160-420)          sealed  rows 1280-1540 (MR 420-680)

python -m cognitive_lab.mk1v3.study --set dev                # all policies, all seeds, curves
python -m cognitive_lab.mk1v3.study --set dev --e4b          # Gemma 4 E4B reference at budgets 50 and 150
"""

import argparse
import json
import random
import statistics

import numpy as np

from cognitive_lab.mk1v3.growth import RESULTS, ROOT, task_rows

TASKS = ["ag_news", "emotion", "mr", "subj", "tweet_sentiment"]
SYMBOLS = list("ABCDEF")
ROWS = {"dev": (1020, 1280), "sealed": (1280, 1540)}
MR_ROWS = {"dev": (160, 420), "sealed": (420, 680)}
POOL, EXAM = 200, 60
BUDGETS = list(range(10, 151, 10))
SEEDS = 20
K_NEIGHBOURS = 5
SETTINGS = {"regions": 10, "progress": True, "floor": 0.2, "pick": "random",
            "prior_tasks": 2, "prior_n": 100}  # dev chooses regions in {5, 10} and progress on/off
CACHE = ROOT / "artifacts" / "results" / "study-vectors"


def world(part: str) -> tuple[list[dict], list[dict]]:
    pool, exam = [], []
    for t in TASKS:
        lo, hi = (MR_ROWS if t == "mr" else ROWS)[part]
        rows = task_rows(t)[lo:hi]
        pool += [{"text": r["text"], "label": r["label"], "task": t} for r in rows[:POOL]]
        exam += [{"text": r["text"], "label": r["label"], "task": t} for r in rows[POOL:POOL + EXAM]]
    return pool, exam


def vectors(part: str, pool: list[dict], exam: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    path = CACHE / f"{part}.npz"
    if not path.exists():
        import torch

        from cognitive_lab.mk1v2.taskstream import Embedder

        eye = Embedder(torch.device("cuda" if torch.cuda.is_available() else "cpu"), "bge-large")
        v = eye([x["text"] for x in pool + exam]).float().cpu().numpy()
        CACHE.mkdir(parents=True, exist_ok=True)
        np.savez(path, v=v)
    v = np.load(path)["v"]
    v = v / np.linalg.norm(v, axis=1, keepdims=True)
    v = v - v[:len(pool)].mean(0)  # centred on the pool (the system sees the pool, never the exam)
    v = v / np.linalg.norm(v, axis=1, keepdims=True)
    return v[:len(pool)], v[len(pool):]


class Learner:
    def __init__(self, pool_v: np.ndarray):
        self.pool_v = pool_v
        self.asked: list[int] = []
        self.labels: list[int] = []

    def votes(self, q: np.ndarray) -> np.ndarray:
        """Vote shares (rows of q) over the six symbols from the k nearest asked texts."""
        out = np.full((len(q), len(SYMBOLS)), 1.0 / len(SYMBOLS))
        if not self.asked:
            return out
        sims = q @ self.pool_v[self.asked].T
        k = min(K_NEIGHBOURS, len(self.asked))
        idx = np.argpartition(-sims, k - 1, axis=1)[:, :k]
        lab = np.array(self.labels)
        out = np.zeros((len(q), len(SYMBOLS)))
        for j in range(k):
            w = np.maximum(sims[np.arange(len(q)), idx[:, j]], 1e-3)
            np.add.at(out, (np.arange(len(q)), lab[idx[:, j]]), w)
        return out / out.sum(1, keepdims=True)

    def predict(self, q: np.ndarray) -> np.ndarray:
        return self.votes(q).argmax(1)

    def uncertainty(self, q: np.ndarray) -> np.ndarray:
        return 1.0 - self.votes(q).max(1)

    def learn(self, i: int, y: int) -> None:
        self.asked.append(i)
        self.labels.append(y)


def kmeans(v: np.ndarray, k: int, rng: np.random.Generator, iters: int = 30) -> np.ndarray:
    centres = v[rng.choice(len(v), k, replace=False)]
    for _ in range(iters):
        assign = (v @ centres.T).argmax(1)
        for j in range(k):
            members = v[assign == j]
            if len(members):
                c = members.mean(0)
                centres[j] = c / np.linalg.norm(c)
    return (v @ centres.T).argmax(1)


def study(policy: str, pool: list[dict], pool_v: np.ndarray, exam_v: np.ndarray, exam_y: np.ndarray, seed: int,
          settings: dict | None = None) -> dict:
    cfg = {**SETTINGS, **(settings or {})}
    rng = np.random.default_rng(seed)
    learner = Learner(pool_v)
    left = set(range(len(pool)))
    region = kmeans(pool_v, cfg["regions"], rng) if policy == "self" else None
    history: dict[int, list[int]] = {}
    curve, asked_tasks = {}, []
    # earlier experience: the same two tasks per seed (whatever the policy) were studied before, 100 texts each,
    # learned in a random order with the same honest record
    prior_rng = np.random.default_rng(1000 + seed)
    known = prior_rng.choice(TASKS, cfg["prior_tasks"], replace=False) if cfg["prior_tasks"] else []
    prior = [i for t in known for i in prior_rng.choice([j for j, x in enumerate(pool) if x["task"] == t],
                                                         cfg["prior_n"], replace=False)]
    for i in prior_rng.permutation(prior) if prior else []:
        i = int(i)
        if region is not None:
            right = bool(learner.asked) and int(learner.predict(pool_v[i:i + 1])[0]) == pool[i]["label"]
            history.setdefault(int(region[i]), []).append(int(right))
        learner.learn(i, pool[i]["label"])
        left.discard(i)
    for step in range(1, BUDGETS[-1] + 1):
        cand = np.array(sorted(left))
        if policy == "random" or not learner.asked:
            i = int(rng.choice(cand))
        elif policy == "uncertain":
            u = learner.uncertainty(pool_v[cand]) + rng.random(len(cand)) * 1e-9
            i = int(cand[u.argmax()])
        else:
            best, pick = -1.0, None
            for r in np.unique(region[cand]):
                h = history.get(int(r), [])[-8:]
                err = (len(h) - sum(h) + 1) / (len(h) + 2)
                prog = 1.0
                if cfg["progress"] and len(h) == 8:
                    older, newer = 1 - statistics.mean(h[:4]), 1 - statistics.mean(h[4:])
                    prog = max(cfg["floor"], 1 + 2 * (older - newer))
                value = (region[cand] == r).mean() * err * prog + rng.random() * 1e-9
                if value > best:
                    best, pick = value, int(r)
            inside = cand[region[cand] == pick]
            if cfg["pick"] == "uncertain":
                u = learner.uncertainty(pool_v[inside]) + rng.random(len(inside)) * 1e-9
                i = int(inside[u.argmax()])
            else:
                i = int(rng.choice(inside))
        y = pool[i]["label"]
        if region is not None:
            right = bool(learner.asked) and int(learner.predict(pool_v[i:i + 1])[0]) == y
            history.setdefault(int(region[i]), []).append(int(right))
        learner.learn(i, y)
        left.discard(i)
        asked_tasks.append(pool[i]["task"])
        if step in BUDGETS:
            curve[step] = (learner.predict(exam_v) == exam_y).astype(int).tolist()
    return {"curve": curve, "asked_tasks": asked_tasks, "asked": learner.asked}


def run(part: str, settings: dict | None = None, policies=("random", "uncertain", "self")) -> dict:
    pool, exam = world(part)
    pool_v, exam_v = vectors(part, pool, exam)
    exam_y = np.array([x["label"] for x in exam])
    out = {}
    for policy in policies:
        runs = [study(policy, pool, pool_v, exam_v, exam_y, seed, settings) for seed in range(SEEDS)]
        out[policy] = runs
        acc = [statistics.mean(statistics.mean(r["curve"][b]) for r in runs) for b in BUDGETS]
        share = {t: round(statistics.mean(r["asked_tasks"].count(t) for r in runs) / BUDGETS[-1], 3) for t in TASKS}
        print(f"{policy:9s} AUC {statistics.mean(acc):.4f}  @50 {acc[4]:.3f}  @150 {acc[-1]:.3f}  asked {share}")
    return out


def per_item(runs: list[dict]) -> np.ndarray:
    """Exam items x budgets, mean over seeds: the unit for the paired bootstrap is the exam item."""
    return np.array([[statistics.mean(r["curve"][b][i] for r in runs) for b in BUDGETS]
                     for i in range(len(runs[0]["curve"][BUDGETS[0]]))])


def versus(a: list[dict], b: list[dict], reps: int = 10000) -> dict:
    d = (per_item(a) - per_item(b)).mean(1)  # AUC difference per exam item
    rng = np.random.default_rng(0)
    boots = d[rng.integers(0, len(d), (reps, len(d)))].mean(1)
    return {"diff": round(float(d.mean()), 4), "ci95": [round(float(np.percentile(boots, 2.5)), 4),
                                                       round(float(np.percentile(boots, 97.5)), 4)]}


def e4b(part: str, budgets=(50, 150)) -> dict:
    """Gemma 4 E4B with the random policy's labelled texts (seed 0); the 16 most similar go in the prompt."""
    from cognitive_lab.mk1v2.llm_service import LLMService
    from cognitive_lab.mk1v2.taskstream import parse

    pool, exam = world(part)
    pool_v, exam_v = vectors(part, pool, exam)
    exam_y = np.array([x["label"] for x in exam])
    asked = study("random", pool, pool_v, exam_v, exam_y, 0)["asked"]
    out = {}
    with LLMService("e4b") as llm:
        for b in budgets:
            have = asked[:b]
            right = []
            for q, x in zip(exam_v, exam):
                near = [have[j] for j in np.argsort(-(pool_v[have] @ q))[:16][::-1]]
                shots = "\n".join(f"Text: {pool[j]['text']}\nLabel: {SYMBOLS[pool[j]['label']]}" for j in near)
                prompt = (f"Classify the text into exactly one of these labels: {', '.join(SYMBOLS)}.\n"
                          f"Labelled examples:\n{shots}\nText: {x['text']}\nAnswer with the label only.")
                g = parse(llm.chat([{"role": "user", "content": prompt}], max_tokens=8).strip(), SYMBOLS, "symbolic")
                right.append(int(g == x["label"]))
            out[b] = right
            print(f"e4b @{b} {statistics.mean(right):.3f}", flush=True)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--set", choices=("dev", "sealed"), default="dev")
    p.add_argument("--e4b", action="store_true")
    p.add_argument("--final", action="store_true")
    a = p.parse_args()
    if a.set == "sealed" and not a.final:
        raise SystemExit("sealed: pass --final once, after the pre-registration commit")
    if a.e4b:
        r = e4b(a.set)
        (RESULTS / f"study-e4b_{a.set}.json").write_text(json.dumps(r))
        return
    if a.set == "dev":
        for regions in (5, 10):
            for progress in (True, False):
                print(f"-- regions {regions} progress {progress}")
                run("dev", {"regions": regions, "progress": progress}, policies=("self",))
    out = run(a.set)
    verdict = {"self - random": versus(out["self"], out["random"]),
               "self - uncertain": versus(out["self"], out["uncertain"]),
               "uncertain - random": versus(out["uncertain"], out["random"])}
    print(json.dumps(verdict))
    (RESULTS / f"study_{a.set}.json").write_text(json.dumps({"settings": SETTINGS, "verdict": verdict, "runs": out}))


if __name__ == "__main__":
    main()
