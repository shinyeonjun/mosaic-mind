"""Sanity checks for the anchored-interface experiments (run after training, before trusting results).

1. reproduce: reload a run's saved reader and judge, re-score the test part, and require the
   exact score recorded in that run's result file. Catches save/load mistakes.
2. cache: for a frozen reader, messages taken through the message cache must equal messages
   computed directly. Catches cache-key or eval-mode mistakes.

python -m cognitive_lab.world.interface_checks --stem anchored-learned-sourcefold-staged-frozen --seed 42 \
    --reader world-v1_reader-anchored-learned_composition_seed-42.pt --judge-kind source-fold
"""

import argparse
import json

import torch

from cognitive_lab.world.interface import ENCODER_DIRS, RESULTS_DIR, load_part
from cognitive_lab.world.interface_anchored import (
    CHECKPOINT_DIR, EquivariantJudge, PairEncoder, SourceFoldJudge, batch_inputs, evaluate, hypothesis,
    load_reader, queried_door,
)


def build(reader_file: str, judge_file: str, judge_kind: str, device: torch.device):
    saved_reader = torch.load(CHECKPOINT_DIR / reader_file, map_location="cpu")
    encoder = PairEncoder(encoder_dir=ENCODER_DIRS[saved_reader["encoder_name"]]).to(device)
    load_reader(encoder, saved_reader["reader"])
    saved_judge = torch.load(CHECKPOINT_DIR / judge_file, map_location="cpu")
    if saved_judge["judge_kind"] != judge_kind:
        raise SystemExit(f"judge checkpoint is {saved_judge['judge_kind']!r}, expected {judge_kind!r}")
    judge = (SourceFoldJudge if judge_kind == "source-fold" else EquivariantJudge)(encoder.message_size).to(device)
    judge.load_state_dict(saved_judge["judge"])
    return encoder, judge


def check_reproduce(stem: str, split: str, seed: int, encoder, judge, device, frozen: bool) -> bool:
    recorded = json.loads((RESULTS_DIR / f"world-v1_interface-{stem}_{split}_seed-{seed}.json").read_text(encoding="utf-8"))
    if frozen:
        for parameter in encoder.parameters():
            parameter.requires_grad_(False)
    test = load_part(split, seed, "test")
    score = evaluate(judge, encoder, test, device)["mean_score"]
    ok = score == recorded["test"]["mean_score"]
    print(f"[reproduce] recorded {recorded['test']['mean_score']} reloaded {score} -> {'OK' if ok else 'MISMATCH'}")
    return ok


def check_cache(encoder, split: str, seed: int, device, tolerance: float = 1e-4) -> bool:
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    encoder.eval()
    episodes = load_part(split, seed, "test")[:32]
    encoder.__dict__.pop("message_cache", None)
    cached, _, _, _ = batch_inputs(episodes, encoder, device)
    worst = 0.0
    with torch.no_grad():
        for b, episode in enumerate(episodes):
            door = queried_door(episode)
            for t, utterance in enumerate(episode["utterances"]):
                direct = encoder([utterance["text"]] * 3, [hypothesis(door, k) for k in episode["entities"]["keys"]],
                                 precise=True)
                worst = max(worst, (cached[b, t] - direct).abs().max().item())
    ok = worst <= tolerance
    print(f"[cache] max |cached - direct| = {worst:.6f} (tolerance {tolerance}, fp32) -> {'OK' if ok else 'MISMATCH'}")
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(description="Sanity checks for anchored-interface runs")
    parser.add_argument("--stem", required=True, help="result stem after 'world-v1_interface-'")
    parser.add_argument("--split", default="composition")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reader", required=True, help="reader checkpoint file name in artifacts/checkpoints")
    parser.add_argument("--judge", default=None, help="judge checkpoint file name (default: from the stem)")
    parser.add_argument("--judge-kind", choices=("flat", "source-fold"), required=True)
    parser.add_argument("--frozen", action="store_true", help="the run kept the reader frozen (uses the cache path)")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    judge_file = args.judge or f"world-v1_judge-{args.stem}_{args.split}_seed-{args.seed}.pt"
    encoder, judge = build(args.reader, judge_file, args.judge_kind, device)
    results = [check_reproduce(args.stem, args.split, args.seed, encoder, judge, device, args.frozen)]
    if args.frozen:
        results.append(check_cache(encoder, args.split, args.seed, device))
    print("ALL CHECKS PASSED" if all(results) else "CHECK FAILED")
    raise SystemExit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
