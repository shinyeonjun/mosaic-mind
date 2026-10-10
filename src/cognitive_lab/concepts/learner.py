"""Small concept learner trained from scratch (design/build-from-scratch-2026-10-10.md section 10, step 2).

A task is 8 grids padded to 14 x 14: 3 demonstration pairs, the test input and the test output slot. The transformer
reads all 1568 cells at once and predicts the 196 cells of the output slot (colour 0-9, or OUTSIDE for cells beyond
the output's size). No search, no hand-made primitives: whatever it knows about concepts it learned from tasks.

python -m cognitive_lab.concepts.learner speed
python -m cognitive_lab.concepts.learner pretrain --steps 20000
python -m cognitive_lab.concepts.learner adapt --families dev --n 8 32
"""

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from cognitive_lab.concepts.generator import task
from cognitive_lab.concepts.split import SPLIT

SIDE = 14
CELLS = SIDE * SIDE
OUTSIDE, UNKNOWN = 10, 11
GRIDS = 8
CKPT_DIR = Path("artifacts/checkpoints")
RESULT_DIR = Path("artifacts/concepts")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ----------------------------------------------------------------------------- encoding
def pad(g) -> np.ndarray:
    out = np.full((SIDE, SIDE), OUTSIDE, dtype=np.int64)
    g = np.asarray(g)
    out[:g.shape[0], :g.shape[1]] = g
    return out


def encode(t: dict, order=None) -> tuple[np.ndarray, np.ndarray]:
    """(8, 14, 14) input cells and (14, 14) target. `order` picks which of the 4 pairs is the test (leave-one-out)."""
    pairs = t["train"] + t["test"]
    if order is not None:
        pairs = [pairs[i] for i in order]
    grids = []
    for p in pairs[:3]:
        grids += [pad(p["input"]), pad(p["output"])]
    grids += [pad(pairs[3]["input"]), np.full((SIDE, SIDE), UNKNOWN, dtype=np.int64)]
    return np.stack(grids), pad(pairs[3]["output"])


def permute_colours(x: np.ndarray, y: np.ndarray, rng: random.Random):
    """Same random relabelling of colours 1-9 on every grid of a task (background, OUTSIDE, UNKNOWN kept)."""
    perm = list(range(1, 10))
    rng.shuffle(perm)
    table = np.arange(12)
    table[1:10] = perm
    return table[x], table[y]


def batch_of(tasks: list[dict], rng: random.Random | None = None, augment: bool = False):
    xs, ys = [], []
    for t in tasks:
        order = None
        if augment:
            order = list(range(4))
            rng.shuffle(order)
        x, y = encode(t, order)
        if augment:
            x, y = permute_colours(x, y, rng)
        xs.append(x)
        ys.append(y)
    return torch.from_numpy(np.stack(xs)), torch.from_numpy(np.stack(ys))


# ----------------------------------------------------------------------------- model
class Learner(nn.Module):
    def __init__(self, d: int = 192, layers: int = 6, heads: int = 6):
        super().__init__()
        self.colour = nn.Embedding(12, d)
        self.row = nn.Embedding(SIDE, d)
        self.col = nn.Embedding(SIDE, d)
        self.slot = nn.Embedding(GRIDS, d)
        block = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.0, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.body = nn.TransformerEncoder(block, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, 11)
        r = torch.arange(SIDE)
        self.register_buffer("rows", r.repeat_interleave(SIDE).repeat(GRIDS), persistent=False)
        self.register_buffer("cols", r.repeat(SIDE).repeat(GRIDS), persistent=False)
        self.register_buffer("slots", torch.arange(GRIDS).repeat_interleave(CELLS), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 8, 14, 14) -> logits (B, 14, 14, 11) for the output slot."""
        b = x.shape[0]
        h = self.colour(x.reshape(b, -1)) + self.row(self.rows) + self.col(self.cols) + self.slot(self.slots)
        h = self.norm(self.body(h))[:, -CELLS:]
        return self.head(h).reshape(b, SIDE, SIDE, 11)


def loss_of(model, x, y):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(x.to(DEVICE))
    return F.cross_entropy(logits.float().reshape(-1, 11), y.to(DEVICE).reshape(-1))


@torch.no_grad()
def exact(model, tasks: list[dict], bs: int = 32) -> list[int]:
    """1 if every one of the 196 output-slot cells is right (so the size is right too), per task."""
    model.eval()
    out = []
    for i in range(0, len(tasks), bs):
        x, y = batch_of(tasks[i:i + bs])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = model(x.to(DEVICE)).argmax(-1).cpu()
        out += (pred == y).reshape(len(x), -1).all(1).int().tolist()
    model.train()
    return out


# ----------------------------------------------------------------------------- pretraining
class Stream(torch.utils.data.IterableDataset):
    """Endless fresh tasks from the training families; each worker draws its own seeds."""

    def __init__(self, families, bs):
        self.families, self.bs = families, bs

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        wid = info.id if info else 0
        rng = random.Random(f"stream-{wid}")
        i = 0
        while True:
            tasks = []
            while len(tasks) < self.bs:
                t = task(rng.choice(self.families), f"train-{wid}-{i}")
                i += 1
                if t:
                    tasks.append(t)
            yield batch_of(tasks, rng, augment=True)


def pretrain(steps: int, bs: int = 32, lr: float = 3e-4, out: str = "concept-learner", log_every: int = 1000,
             families=None, d=192, layers=6, heads=6):
    families = families or SPLIT["train"]
    torch.manual_seed(0)
    model = Learner(d, layers, heads).to(DEVICE)
    print(f"params {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M  families {len(families)}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    warm = min(1000, steps // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps))))
    loader = torch.utils.data.DataLoader(Stream(families, bs), batch_size=None, num_workers=4,
                                         persistent_workers=True, prefetch_factor=4)
    val = {f: [t for t in (task(f, f"val-{i}") for i in range(50)) if t] for f in families}
    log = []
    t0, run = time.time(), 0.0
    for step, (x, y) in enumerate(loader, 1):
        loss = loss_of(model, x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        run = 0.98 * run + 0.02 * loss.item() if step > 1 else loss.item()
        if step % log_every == 0 or step == steps:
            acc = {f: float(np.mean(exact(model, v))) for f, v in val.items()}
            mean = float(np.mean(list(acc.values())))
            log.append({"step": step, "loss": run, "val_exact": mean, "per_family": acc,
                        "minutes": (time.time() - t0) / 60})
            print(f"step {step:6d} loss {run:.4f} val exact {mean:.3f} {(time.time() - t0) / 60:.1f}m", flush=True)
            CKPT_DIR.mkdir(parents=True, exist_ok=True)
            torch.save({"model": model.state_dict(), "config": {"d": d, "layers": layers, "heads": heads},
                        "step": step, "families": families}, CKPT_DIR / f"{out}.pt")
            RESULT_DIR.mkdir(parents=True, exist_ok=True)
            (RESULT_DIR / f"{out}-log.json").write_text(json.dumps(log, indent=1))
        if step >= steps:
            break
    return model


def load(name: str = "concept-learner") -> Learner:
    ck = torch.load(CKPT_DIR / f"{name}.pt", map_location=DEVICE, weights_only=True)
    model = Learner(**ck["config"]).to(DEVICE)
    model.load_state_dict(ck["model"])
    return model


# ----------------------------------------------------------------------------- adaptation
RECIPE = {"steps": 1500, "lr": 1e-4, "bs": 16}  # frozen on dev families only: the one where from-scratch did best (section 10)


def adapt(model: Learner, family: str, n: int, run: int, steps: int, lr: float, bs: int) -> Learner:
    """Fine-tune on n tasks of one family (leave-one-out pair order and colour relabelling as augmentation)."""
    tasks = [t for t in (task(family, f"adapt-{run}-{i}") for i in range(n * 2)) if t][:n]
    rng = random.Random(f"adapt-{family}-{run}")
    torch.manual_seed(run)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    model.train()
    for _ in range(steps):
        x, y = batch_of([rng.choice(tasks) for _ in range(bs)], rng, augment=True)
        loss = loss_of(model, x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    return model


def evaluate(families, ns, ckpt="concept-learner", runs=3, n_eval=200, recipe=None, tag="dev"):
    """Per family, per N, per condition: exact-match per eval task, averaged over adaptation runs."""
    recipe = recipe or RECIPE
    ck = torch.load(CKPT_DIR / f"{ckpt}.pt", map_location=DEVICE, weights_only=True)
    results = {}
    for f in families:
        evals = [t for t in (task(f, f"eval-{i}") for i in range(n_eval * 2)) if t][:n_eval]
        res = {"zero_shot": exact(load(ckpt), evals)}
        for n in ns:
            for cond in ("pretrained", "scratch"):
                per_run = []
                for run in range(runs):
                    if cond == "pretrained":
                        m = load(ckpt)
                    else:
                        torch.manual_seed(1000 + run)
                        m = Learner(**ck["config"]).to(DEVICE)
                    adapt(m, f, n, run, **recipe)
                    per_run.append(exact(m, evals))
                res[f"{cond}_n{n}"] = np.mean(per_run, axis=0).tolist()
        results[f] = res
        print(f"{f:24s} zero {np.mean(res['zero_shot']):.3f}  " +
              "  ".join(f"N={n} pre {np.mean(res[f'pretrained_n{n}']):.3f} scr {np.mean(res[f'scratch_n{n}']):.3f}"
                        for n in ns), flush=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    (RESULT_DIR / f"step2-{tag}.json").write_text(json.dumps({"recipe": recipe, "results": results}))
    return results


def bootstrap(results: dict, a: str, b: str, reps: int = 10000, seed: int = 0):
    """Difference of means a - b, resampling eval tasks within each family (families weighted equally)."""
    rng = np.random.default_rng(seed)
    fams = list(results)
    diffs = [np.array(results[f][a]) - np.array(results[f][b]) for f in fams]
    point = float(np.mean([d.mean() for d in diffs]))
    boots = np.mean([d[rng.integers(0, len(d), (reps, len(d)))].mean(1) for d in diffs], axis=0)
    return point, float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def speed(bs: int = 32, steps: int = 30):
    model = Learner().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    rng = random.Random(0)
    t0 = time.time()
    tasks = [task(f, f"speed-{i}") for i, f in enumerate(rng.choice(SPLIT["train"]) for _ in range(bs * 4))]
    print(f"generate {bs * 4} tasks {time.time() - t0:.2f}s", flush=True)
    x, y = batch_of([t for t in tasks if t][:bs], rng, augment=True)
    for i in range(steps + 3):
        if i == 3:
            torch.cuda.synchronize()
            t0 = time.time()
        loss = loss_of(model, x, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    torch.cuda.synchronize()
    print(f"params {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M  {(time.time() - t0) / steps * 1000:.0f} ms/step "
          f"bs {bs}  peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GB", flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["speed", "pretrain", "adapt"])
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--families", default="dev", choices=["dev", "test"])
    p.add_argument("--n", type=int, nargs="+", default=[8, 32])
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--final", action="store_true")
    p.add_argument("--adapt-steps", type=int, default=None, help="recipe search on dev only")
    p.add_argument("--lr", type=float, default=None, help="recipe search on dev only")
    a = p.parse_args()
    if a.cmd == "speed":
        speed()
    elif a.cmd == "pretrain":
        pretrain(a.steps)
    else:
        if a.families == "test" and not a.final:
            raise SystemExit("test families are the one verdict: pass --final once the recipe is frozen")
        recipe, tag = dict(RECIPE), a.families
        if a.adapt_steps or a.lr:
            if a.families == "test":
                raise SystemExit("the recipe is chosen on dev only")
            recipe.update({k: v for k, v in (("steps", a.adapt_steps), ("lr", a.lr)) if v})
            tag = f"dev-s{recipe['steps']}-lr{recipe['lr']:g}"
        res = evaluate(SPLIT[a.families], a.n, runs=a.runs, recipe=recipe, tag=tag)
        for n in a.n:
            d, lo, hi = bootstrap(res, f"pretrained_n{n}", f"scratch_n{n}")
            print(f"N={n}: pretrained - scratch {d:+.3f} [{lo:+.3f}, {hi:+.3f}]")


if __name__ == "__main__":
    main()
