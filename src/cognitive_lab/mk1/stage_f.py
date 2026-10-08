"""MK1 stage F: conflicting channels, weighed by a learned arbiter (world v11, design/mk1-integration.md, F).

python -m cognitive_lab.mk1.stage_f      # after python -m cognitive_lab.mk1.arbiter
"""

import json
import time

import torch

from cognitive_lab.mk1.arbiter import BOARD_ACCURACIES
from cognitive_lab.mk1.integration import bootstrap
from cognitive_lab.mk1.system import MK1, episode_score
from cognitive_lab.world2.integrated import RESULTS_DIR
from cognitive_lab.world7.routing import COST
from cognitive_lab.world10 import world

POLICIES = {"think": "안내판만", "article-only": "기사만", "ask-available": "합치기", "arbitrate": "중재자"}


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    started = time.perf_counter()
    sessions = world.generate("test", 42, board_accuracies=BOARD_ACCURACIES)
    episodes = [e for s in sessions for e in s["episodes"]]
    accuracy = torch.tensor([s["board_accuracy"] for s in sessions for _ in s["episodes"]])
    tags = [e["tag"] for e in episodes]
    position = torch.tensor([t for s in sessions for t in range(len(s["episodes"]))])
    model = MK1(device)  # live fp32 article reading
    nets = {}
    for policy in POLICIES:
        out, _ = model(sessions, policy)
        raw = torch.tensor([episode_score(e, a) for e, a in zip(episodes, out["answer"])])
        nets[policy] = raw - COST * out["asks"].cpu()
        print(policy, round(nets[policy].mean().item(), 4), flush=True)
    # Guide: knows, per session, which channel is better for C (board if r >= 0.75, else the article alone).
    nets["guide"] = torch.where(torch.tensor([t == "C" for t in tags]) & (accuracy < 0.75), nets["article-only"],
                                torch.where(torch.tensor([t == "C" for t in tags]), nets["think"], nets["article-only"]))
    result = {"stage": "MK1-F", "problems": len(episodes), "total": {p: round(v.mean().item(), 4) for p, v in nets.items()},
              "by_accuracy": {}, "by_type": {}}
    for r in BOARD_ACCURACIES:
        m = accuracy == r
        result["by_accuracy"][str(r)] = {p: round(v[m].mean().item(), 4) for p, v in nets.items()}
        c = m & torch.tensor([t == "C" for t in tags])
        result["by_accuracy"][str(r)]["C_only"] = {p: round(v[c].mean().item(), 4) for p, v in nets.items()}
    for kind in world.TYPES:
        m = torch.tensor([t == kind for t in tags])
        result["by_type"][kind] = {p: round(v[m].mean().item(), 4) for p, v in nets.items()}
    hard_c = (accuracy == 0.5) & torch.tensor([t == "C" for t in tags])
    early, late = hard_c & (position < 20), hard_c & (position >= 20)
    arb = nets["arbitrate"]
    generator = torch.Generator().manual_seed(0)
    diffs = torch.stack([arb[late][torch.randint(int(late.sum()), (int(late.sum()),), generator=generator)].mean()
                         - arb[early][torch.randint(int(early.sum()), (int(early.sum()),), generator=generator)].mean()
                         for _ in range(2000)])
    result["learning_within_session_r0.5_C"] = {"first_half": round(arb[early].mean().item(), 4),
                                                "second_half": round(arb[late].mean().item(), 4),
                                                "difference_ci95": [round(diffs.quantile(0.025).item(), 4),
                                                                    round(diffs.quantile(0.975).item(), 4)]}
    aac = torch.tensor([t in ("A", "AC") for t in tags])
    by_r = result["by_accuracy"]
    result["arbiter_minus_merge_ci95"] = bootstrap(nets["arbitrate"] - nets["ask-available"])
    result["criteria"] = {
        "1_beats_merge": result["arbiter_minus_merge_ci95"][0] > 0,
        "2_no_worse_than_best_single_channel": all(by_r[str(r)]["arbitrate"] >= max(by_r[str(r)]["think"], by_r[str(r)]["article-only"]) - 0.01
                                                   for r in BOARD_ACCURACIES),
        "3_learns_within_session": result["learning_within_session_r0.5_C"]["difference_ci95"][0] > 0,
        "4_no_conflict_unchanged": abs(nets["arbitrate"][aac].mean().item() - nets["article-only"][aac].mean().item()) <= 0.01,
    }
    result["graduated"] = all(result["criteria"].values())
    result["seconds"] = round(time.perf_counter() - started, 1)
    path = RESULTS_DIR / "mk1-f_conflict.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
