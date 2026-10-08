"""Run the active-probing environment and non-neural baselines."""

import argparse
import json
from pathlib import Path

from cognitive_lab.active.baselines import evaluate_baselines

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate simple policies in the Experiment 006 active-probing world"
    )
    parser.add_argument("--episodes", type=int, default=512)
    parser.add_argument("--probe-budget", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.episodes <= 0 or args.probe_budget < 0:
        parser.error("--episodes must be positive and --probe-budget non-negative")

    result = evaluate_baselines(args.episodes, args.probe_budget, args.seed)
    results_dir = PROJECT_ROOT / "artifacts" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    output_path = results_dir / f"experiment-006_active-probing_seed-{args.seed}.json"
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
