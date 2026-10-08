"""CLI for rule world v2.

python -m cognitive_lab.world2 generate --seed 42
python -m cognitive_lab.world2 audit --seed 42
python -m cognitive_lab.world2 score --agent b2-learn --seed 42
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from cognitive_lab.world2.agents import AGENTS as CODE_AGENTS
from cognitive_lab.world2.bayes_learner import BayesLearnerJudge, InformedBayesJudge
from cognitive_lab.world2.partial import ADFJudge, CountPartialJudge, JointBayesJudge
from cognitive_lab.world2.generator import ACCURACY_LEVELS, NAME_POOLS, PARTS, SPEAKERS_PER_SESSION, generate_part
from cognitive_lab.world2.scoreboard import run_sessions

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "artifacts" / "world-v2" / "data"
RESULTS_DIR = PROJECT_ROOT / "artifacts" / "results"


def _n2(memory: bool, feedback: str = "full", world: str = "standard", context: bool = False):
    def build(args):
        from cognitive_lab.world2.neural_judge import N2Judge, checkpoint_path

        path = checkpoint_path(args.seed, memory, feedback, args.n2_speakers, args.n2_sparse, world, context)
        if not path.exists():
            raise SystemExit(f"missing {path}; run `python -m cognitive_lab.world2.neural_judge --seed {args.seed}` first")
        return N2Judge(path)

    return build


AGENTS = {
    **CODE_AGENTS,
    "b3-bayes": lambda args: BayesLearnerJudge(),
    "b3-bayes-h01": lambda args: BayesLearnerJudge(hazard=0.01),
    "b3-bayes-h03": lambda args: BayesLearnerJudge(hazard=0.03),
    "b3-informed-h02": lambda args: InformedBayesJudge(hazard=0.02),
    "b3-informed-h05": lambda args: InformedBayesJudge(hazard=0.05),
    "b3-informed-h10": lambda args: InformedBayesJudge(hazard=0.10),
    "b4-joint": lambda args: JointBayesJudge(explore_until=args.explore_until,
                                             explore_threshold=args.explore_threshold),
    "b5-adf": lambda args: ADFJudge(explore_until=args.explore_until, explore_threshold=args.explore_threshold),
    "b2-partial": lambda args: CountPartialJudge(explore_until=args.explore_until,
                                                 explore_threshold=args.explore_threshold),
    "n2": _n2(memory=True),
    "n2-no-memory": _n2(memory=False),
    "n2-partial": _n2(memory=True, feedback="partial"),
    "n2-mis": _n2(memory=True, world="misspecified"),
    "n2-mis-ctx": _n2(memory=True, world="misspecified", context=True),
    "i1": lambda args: __import__("cognitive_lab.world2.integrated", fromlist=["x"]).I1Judge(
        __import__("cognitive_lab.world2.integrated", fromlist=["x"]).checkpoint_path(
            args.seed, "misspecified" if args.misspecified else "standard",
            __import__("cognitive_lab.world2.integrated", fromlist=["x"]).DEFAULT_READER)),
    "o2-mis": lambda args: __import__("cognitive_lab.world2.misspecified", fromlist=["x"]).OracleMJudge(),
}


def part_path(seed: int, part: str, speakers: int = SPEAKERS_PER_SESSION, sparse: bool = False,
              misspecified: bool = False) -> Path:
    scale = (("" if speakers == SPEAKERS_PER_SESSION else f"_s{speakers}") + ("_sparse" if sparse else "")
             + ("_mis" if misspecified else ""))
    return DATA_DIR / f"seed-{seed}{scale}" / f"{part}.jsonl"


def load_part(seed: int, part: str, speakers: int = SPEAKERS_PER_SESSION, sparse: bool = False,
              misspecified: bool = False) -> list[dict]:
    path = part_path(seed, part, speakers, sparse, misspecified)
    if not path.exists():
        raise SystemExit(f"missing {path}; run `python -m cognitive_lab.world2 generate --seed {seed} "
                         f"--speakers {speakers}{' --sparse' if sparse else ''}` first")
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file]


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved: {path}")


def generate(args: argparse.Namespace) -> None:
    for part in PARTS:
        if args.misspecified:
            from cognitive_lab.world2 import misspecified

            sessions = misspecified.generate_part(part, args.seed)
        else:
            sessions = generate_part(part, args.seed, speakers=args.speakers, sparse=args.sparse)
        path = part_path(args.seed, part, args.speakers, args.sparse, args.misspecified)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as file:
            for session in sessions:
                file.write(json.dumps(session, ensure_ascii=False) + "\n")
        episodes = sum(len(s["episodes"]) for s in sessions)
        print(f"{part}: {len(sessions)} sessions, {episodes} episodes -> {path}")


def audit(args: argparse.Namespace) -> None:
    """Checks that do not depend on an agent: label balance, name pools, trust/name independence."""
    report, problems = {"world": "v2", "seed": args.seed, "parts": {}}, []
    pools = {part: set(names) for part, names in NAME_POOLS.items()}
    for a in PARTS:
        for b in PARTS:
            if a < b and pools[a] & pools[b]:
                problems.append(f"name pools overlap: {a}/{b}")
    for part in PARTS:
        sessions = load_part(args.seed, part, args.speakers, args.sparse, args.misspecified)
        episodes = [e for s in sessions for e in s["episodes"]]
        labels = Counter(e["split_keys"]["answer_index"] for e in episodes)
        shares = {k: round(labels[k] / len(episodes), 4) for k in range(3)}
        level_by_name: dict[str, Counter] = {}
        for session in sessions:
            hidden = session["hidden"]
            labels = (hidden["accuracies"] if "accuracies" in hidden
                      else [p["type"] for p in hidden["profiles"]])
            for name, label in zip(session["speakers"], labels):
                level_by_name.setdefault(name, Counter())[label] += 1
        # The most frequent level (or type) per name should not stand out from chance.
        name_bias = max(max(c.values()) / sum(c.values()) for c in level_by_name.values())
        level_counts = Counter(a for s in sessions for a in s["hidden"].get("accuracies", []))
        drift_share = sum(s["hidden"]["drift"] is not None for s in sessions) / len(sessions)
        cases = Counter(e["split_keys"]["case"] for e in episodes)
        report["parts"][part] = {
            "sessions": len(sessions),
            "episodes": len(episodes),
            "answer_shares": shares,
            "accuracy_level_counts": {str(k): level_counts[k] for k in ACCURACY_LEVELS},
            "max_single_level_share_per_name": round(name_bias, 4),
            "drift_session_share": round(drift_share, 4),
            "speaker_count_cases": dict(sorted(cases.items())),
        }
        if any(abs(share - 1 / 3) > 0.02 for share in shares.values()):
            problems.append(f"{part}: unbalanced answers {shares}")
        from cognitive_lab.world2.generator import name_pool

    if any(name not in name_pool(part, args.speakers) for s in sessions for name in s["speakers"]):
            problems.append(f"{part}: speaker outside its name pool")
    report["problems"] = problems
    scale = (("" if args.speakers == SPEAKERS_PER_SESSION else f"_s{args.speakers}") + ("_sparse" if args.sparse else "")
             + ("_mis" if args.misspecified else ""))
    write_json(RESULTS_DIR / f"world-v2_audit_seed-{args.seed}{scale}.json", report)
    if problems:
        print("PROBLEMS:\n  " + "\n  ".join(problems))
        sys.exit(1)
    print("Audit passed.")


def score(args: argparse.Namespace) -> None:
    agent = AGENTS[args.agent](args)
    if not args.gold_input:
        from cognitive_lab.world2.agents import NLIReader

        agent.reader = NLIReader(args.reader_checkpoint, args.device)
    sessions = load_part(args.seed, args.part, args.speakers, args.sparse, args.misspecified)
    if args.limit:
        sessions = sessions[: args.limit]
    result = {"world": "v2", "agent": args.agent, "part": args.part, "seed": args.seed,
              "speakers": args.speakers, "sparse": args.sparse, "n2_trained_speakers": args.n2_speakers,
              "n2_trained_sparse": args.n2_sparse,
              "limit": args.limit, "explore_until": args.explore_until,
              "explore_threshold": args.explore_threshold,
              **run_sessions(agent, sessions, gold_input=args.gold_input, feedback=args.feedback)}
    o = result["overall"]
    print(f"{args.agent}: score {o['mean_score']} (oracle {o['oracle_mean_score']}, regret {o['regret']}), "
          f"abstain {o['abstain_rate']}")
    print("  by bucket:", {k: v["mean_score"] for k, v in result["by_episode_bucket"].items()})
    print("  drift:", {k: v["mean_score"] for k, v in result["drift_sessions"].items()})
    suffix = (("" if args.speakers == SPEAKERS_PER_SESSION else f"_s{args.speakers}")
              + ("_sparse" if args.sparse else "") + ("_mis" if args.misspecified else "")
              + ("" if args.n2_speakers == SPEAKERS_PER_SESSION else f"_n2s{args.n2_speakers}")
              + ("_n2sparse" if args.n2_sparse else "")
              + ("" if args.feedback == "full" else "_partial")
              + (f"_explore-{args.explore_until}-{args.explore_threshold}" if args.explore_until else "")
              + ("" if args.gold_input else "_text") + (f"_limit-{args.limit}" if args.limit else ""))
    write_json(RESULTS_DIR / f"world-v2_{args.agent}_{args.part}_seed-{args.seed}{suffix}.json", result)


def main() -> None:
    parser = argparse.ArgumentParser(description="Rule world v2: learning whom to trust")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("generate", "audit", "score"):
        command = commands.add_parser(name)
        command.add_argument("--seed", type=int, default=42)
        command.add_argument("--speakers", type=int, default=SPEAKERS_PER_SESSION,
                             help="speakers per session (4 is the standard world; others are scaling runs)")
        command.add_argument("--sparse", action="store_true",
                             help="scaling runs: keep 1.75 claims per episode instead of a fixed per-speaker rate")
        command.add_argument("--misspecified", action="store_true",
                             help="rule world v2-M: speakers break the standard statement model")
    score_parser = commands.choices["score"]
    score_parser.add_argument("--agent", choices=sorted(AGENTS), required=True)
    score_parser.add_argument("--part", choices=PARTS, default="test")
    score_parser.add_argument("--limit", type=int, default=0, help="score only the first N sessions")
    score_parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    score_parser.add_argument("--feedback", choices=("full", "partial"), default="full")
    score_parser.add_argument("--n2-speakers", type=int, default=SPEAKERS_PER_SESSION,
                              help="load the N2 checkpoint trained with this many speakers")
    score_parser.add_argument("--n2-sparse", action="store_true", help="load the N2 checkpoint trained on sparse data")
    score_parser.add_argument("--explore-until", type=int, default=0,
                              help="episodes at the start of a session that use --explore-threshold")
    score_parser.add_argument("--explore-threshold", type=float, default=0.5)
    score_parser.add_argument("--text", dest="gold_input", action="store_false",
                              help="agents read utterance text through their language module")
    score_parser.add_argument(
        "--reader-checkpoint", type=Path,
        default=PROJECT_ROOT / "artifacts" / "checkpoints" / "world-v1_p2-nli_iid_seed-42.pt",
        help="language module for --text (default: the rule world v1 NLI reader, reused unchanged)")
    args = parser.parse_args()
    {"generate": generate, "audit": audit, "score": score}[args.command](args)


if __name__ == "__main__":
    main()
