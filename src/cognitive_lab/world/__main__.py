"""CLI: generate data, audit splits for shortcuts, and score agents.

python -m cognitive_lab.world generate --seed 42
python -m cognitive_lab.world audit --seed 42
python -m cognitive_lab.world score --agent s1 --split iid --seed 42
python -m cognitive_lab.world score --agent l1-choice --split iid --seed 42 --device cuda
"""

import argparse
import json
import sys
from pathlib import Path

from cognitive_lab.world.audit import audit_split
from cognitive_lab.world.generator import CASE_WEIGHTS, PARTS, SPLITS, generate_part
from cognitive_lab.world.scoreboard import AGENTS, run_agent

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "artifacts" / "world-v1" / "data"
RESULTS_DIR = PROJECT_ROOT / "artifacts" / "results"
SIZES = {"train": 4000, "validation": 500, "test": 1000}


def part_path(split: str, seed: int, part: str) -> Path:
    return DATA_DIR / split / f"seed-{seed}" / f"{part}.jsonl"


def load_part(split: str, seed: int, part: str) -> list[dict]:
    path = part_path(split, seed, part)
    if not path.exists():
        raise SystemExit(f"missing {path}; run `python -m cognitive_lab.world generate --seed {seed}` first")
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file]


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved: {path}")


def generate(args: argparse.Namespace) -> None:
    report = {"world": "v1", "seed": args.seed, "sizes": SIZES, "parts": {}}
    for split in args.splits:
        for part in PARTS:
            episodes, stats = generate_part(split, part, SIZES[part], args.seed)
            report["parts"][f"{split}/{part}"] = stats
            path = part_path(split, args.seed, part)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", encoding="utf-8") as file:
                for episode in episodes:
                    file.write(json.dumps(episode, ensure_ascii=False) + "\n")
            print(f"{split}/{part}: {len(episodes)} episodes, {stats['resampled_candidates']} resampled")
    print("Every episode's label matched the exact solver.")
    write_json(DATA_DIR / f"generation_seed-{args.seed}.json", report)


def audit(args: argparse.Namespace) -> None:
    report = {"world": "v1", "seed": args.seed, "splits": {}}
    for split in args.splits:
        train = load_part(split, args.seed, "train")
        report["splits"][split] = {
            part: audit_split(train, load_part(split, args.seed, part), args.seed, CASE_WEIGHTS)
            for part in ("validation", "test")
        }
    write_json(RESULTS_DIR / f"world-v1_audit_seed-{args.seed}.json", report)
    failed = [
        f"{split}/{part}: {', '.join(result['rejected_by'])}"
        for split, parts in report["splits"].items()
        for part, result in parts.items()
        if not result["accepted"]
    ]
    if failed:
        print("REJECTED splits:\n  " + "\n  ".join(failed))
        sys.exit(1)
    print("All audited splits passed the shortcut checks.")


def score(args: argparse.Namespace) -> None:
    args.fewshot_examples = []
    if args.shots:
        from cognitive_lab.world.llm_agent import select_examples

        args.fewshot_examples = select_examples(load_part(args.split, args.seed, "train"), args.shots)
    agent, gold_input = AGENTS[args.agent](args)
    episodes = load_part(args.split, args.seed, args.part)
    if args.limit:
        episodes = episodes[: args.limit]
    result = {
        "world": "v1",
        "agent": args.agent,
        "split": args.split,
        "part": args.part,
        "seed": args.seed,
        "limit": args.limit,
        "shots": args.shots,
        **run_agent(agent, episodes, gold_input),
    }
    print(json.dumps({k: result[k] for k in ("mean_score", "accuracy", "by_case")}, ensure_ascii=False, indent=2))
    suffix = (f"_shots-{args.shots}" if args.shots else "") + (f"_limit-{args.limit}" if args.limit else "")
    write_json(
        RESULTS_DIR / f"world-v1_{args.agent}_{args.split}-{args.part}_seed-{args.seed}{suffix}.json",
        result,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Korean rule world v1")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("generate", "audit", "score"):
        command = commands.add_parser(name)
        command.add_argument("--seed", type=int, default=42)
        if name in ("generate", "audit"):
            command.add_argument("--splits", nargs="+", choices=SPLITS, default=list(SPLITS))
    score_parser = commands.choices["score"]
    score_parser.add_argument("--agent", choices=sorted(AGENTS), required=True)
    score_parser.add_argument("--split", choices=SPLITS, default="iid")
    score_parser.add_argument("--part", choices=PARTS, default="test")
    score_parser.add_argument("--limit", type=int, default=0, help="score only the first N episodes")
    score_parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    score_parser.add_argument("--shots", type=int, default=0, help="few-shot training examples for l1 agents")
    score_parser.add_argument("--adapter", type=Path, default=None,
                              help="LoRA checkpoint for l1-lora agents (default: this split and seed)")
    score_parser.add_argument("--model-dir", type=Path, default=PROJECT_ROOT / "model" / "LFM2.5-1.2B-Instruct")
    args = parser.parse_args()
    {"generate": generate, "audit": audit, "score": score}[args.command](args)


if __name__ == "__main__":
    main()
