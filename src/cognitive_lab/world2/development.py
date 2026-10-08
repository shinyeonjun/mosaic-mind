"""Development track D1: does a judge raised in the clean world learn the messy world faster?

- grown:   start from the N2 checkpoint trained on 3000 standard v2 sessions (same seed).
- newborn: same N2 architecture from random initialization.
Both then train on k misspecified (v2-M) sessions for the same number of optimizer steps,
with the checkpoint chosen on the v2-M validation part. We report the v2-M test score
(acquisition and transfer) and the standard v2 test score afterwards (retention).

python -m cognitive_lab.world2.development --seed 42
"""

import argparse
import json
import time
from pathlib import Path

import torch
from torch import nn

from cognitive_lab.world2 import misspecified
from cognitive_lab.world2.generator import generate_part
from cognitive_lab.world2.neural_judge import N2Model, calibrated_score, checkpoint_path, session_tensors

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = PROJECT_ROOT / "artifacts" / "results"
SESSION_BUDGETS = (25, 50, 100, 300, 1000, 3000)


def fit(model: N2Model, train, validation, steps: int, batch_size: int, lr: float, seed: int,
        eval_every: int = 25) -> tuple[dict, float, int]:
    """Train for a fixed number of steps; keep the state with the best validation score.
    Step 0 (the starting point) is a candidate, so training can never make the result worse
    than where the learner started, as judged on validation."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    generator = torch.Generator().manual_seed(seed)
    count = train[0].shape[0]

    def validate() -> float:
        model.eval()
        with torch.no_grad():
            return calibrated_score(model(*validation), validation[2])

    best = (validate(), 0, {k: v.detach().clone() for k, v in model.state_dict().items()})
    for step in range(1, steps + 1):
        model.train()
        batch = torch.randint(count, (min(batch_size, count),), generator=generator)
        kinds, keys, truths = (tensor[batch] for tensor in train)
        loss = nn.functional.cross_entropy(model(kinds, keys, truths).reshape(-1, 3), truths.reshape(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % eval_every == 0:
            score = validate()
            if score > best[0]:
                best = (score, step, {k: v.detach().clone() for k, v in model.state_dict().items()})
    return best[2], best[0], best[1]


def test_score(model: N2Model, tensors) -> float:
    model.eval()
    with torch.no_grad():
        return calibrated_score(model(*tensors), tensors[2])


def main() -> None:
    parser = argparse.ArgumentParser(description="Development track D1: clean world -> messy world")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    args = parser.parse_args()

    started = time.perf_counter()
    grown_path = checkpoint_path(args.seed)  # N2 trained on the standard world
    if not grown_path.exists():
        raise SystemExit(f"missing {grown_path}; run `python -m cognitive_lab.world2.neural_judge --seed {args.seed}`")
    saved = torch.load(grown_path, map_location="cpu")
    pool = session_tensors(misspecified.generate_part("train", args.seed, sessions=max(SESSION_BUDGETS)))
    validation = session_tensors(misspecified.generate_part("validation", args.seed))
    messy_test = session_tensors(misspecified.generate_part("test", args.seed))
    clean_test = session_tensors(generate_part("test", args.seed))

    grown_start = N2Model(saved["hidden"], saved["memory"])
    grown_start.load_state_dict(saved["state"])
    rows = [{"learner": "grown", "sessions": 0, "messy_test": test_score(grown_start, messy_test),
             "clean_test": test_score(grown_start, clean_test)}]
    print(rows[-1], flush=True)
    for k in SESSION_BUDGETS:
        train = tuple(tensor[:k] for tensor in pool)
        for learner in ("grown", "newborn"):
            torch.manual_seed(args.seed + k)
            model = N2Model(saved["hidden"], saved["memory"])
            if learner == "grown":
                model.load_state_dict(saved["state"])
            state, val_score, step = fit(model, train, validation, args.steps, args.batch_size,
                                         args.learning_rate, args.seed + k)
            model.load_state_dict(state)
            rows.append({"learner": learner, "sessions": k, "validation": val_score, "selected_step": step,
                         "messy_test": test_score(model, messy_test), "clean_test": test_score(model, clean_test)})
            print(rows[-1], flush=True)

    result = {"world": "v2 -> v2-M", "experiment": "development-d1", "seed": args.seed,
              "steps_per_run": args.steps, "batch_size": args.batch_size, "learning_rate": args.learning_rate,
              "grown_from": str(grown_path.relative_to(PROJECT_ROOT)), "rows": rows,
              "seconds": round(time.perf_counter() - started, 1)}
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"world-v2_development-d1_seed-{args.seed}.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
