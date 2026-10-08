"""Whom does the curiosity head ask? Asked speakers' hidden accuracy vs the silent candidates.

python -m cognitive_lab.world2.asking_analysis
"""

import json
from collections import Counter

import torch

from cognitive_lab.world2 import hedged
from cognitive_lab.world2.asking_system import CuriosityHead, load_stage1, run_policy, tensors
from cognitive_lab.world2.generator import ACCURACY_LEVELS, accuracies_at
from cognitive_lab.world2.integrated import CHECKPOINT_DIR, RESULTS_DIR


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    report = {}
    for seed in (42, 43, 44):
        system, rows = load_stage1(seed, device)
        with torch.no_grad():
            table = system.messages()
        sessions = hedged.generate_part("test", seed)
        base, answers = tensors(sessions, rows, device)
        head = CuriosityHead().to(device)
        head.load_state_dict(torch.load(CHECKPOINT_DIR / f"world-v2a_curiosity_cost-0.2_seed-{seed}.pt")["state"])
        head.eval()
        trace: list = []
        run_policy(system, table, base, answers, "curiosity", head, trace=trace)
        targets = torch.stack(trace, 1)  # [B,E]
        silent = answers[1].cpu()
        asked, candidates = Counter(), Counter()
        for b, session in enumerate(sessions):
            for t, episode in enumerate(session["episodes"]):
                if not silent[b, t].any():
                    continue
                levels = accuracies_at(session, episode["index"])
                for s in range(len(levels)):
                    if silent[b, t, s]:
                        candidates[levels[s]] += 1
                if targets[b, t] >= 0:
                    asked[levels[int(targets[b, t])]] += 1
        report[seed] = {str(level): {"asked": asked[level], "candidates": candidates[level],
                                     "ask_rate": round(asked[level] / max(candidates[level], 1), 3)}
                        for level in ACCURACY_LEVELS}
        print(seed, {k: v["ask_rate"] for k, v in report[seed].items()}, flush=True)
    (RESULTS_DIR / "world-v2a_curiosity_whom.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
