"""MK1 stage I: belief that holds outside experience (design/mk1-integration.md, I).

The learned arbiter (stage F, GRU memory) against the counting arbiter (counts + a Beta prior fitted on the same
training sessions), on fresh worlds (seed + 300) mixing board accuracies seen in training (1.0 / 0.75 / 0.5) and
never seen (0.9 / 0.6 / 0.25 / 0.1). Three part seeds, session-level intervals.

python -m cognitive_lab.mk1.stage_i     # after python -m cognitive_lab.mk1.arbiter --seed N --kind counting (N = 42, 43, 44)
"""

import json

import torch

from cognitive_lab.mk1.audit import cluster_bootstrap, session_ids
from cognitive_lab.mk1.stage_d import net_scores
from cognitive_lab.mk1.system import MK1
from cognitive_lab.world2.integrated import RESULTS_DIR
from cognitive_lab.world10 import world

SEEN, UNSEEN = (1.0, 0.75, 0.5), (0.9, 0.6, 0.25, 0.1)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result = {"stage": "MK1-I", "seeds": {}}
    criteria = {"1_outside_experience": True, "2_inside_experience": True, "3_never_below_both_single_channels": True}
    for seed in (42, 43, 44):
        sessions = world.generate("test", seed + 300, board_accuracies=SEEN + UNSEEN)
        accuracy = torch.tensor([s["board_accuracy"] for s in sessions for _ in s["episodes"]])
        sid = session_ids(sessions)
        nets = {}
        learned, counting = MK1(device, seed=seed), MK1(device, seed=seed, arbiter="counting")
        nets["learned arbiter"] = net_scores(sessions, learned(sessions, "arbitrate")[0])
        nets["counting arbiter"] = net_scores(sessions, counting(sessions, "arbitrate")[0])
        nets["board only"] = net_scores(sessions, learned(sessions, "think")[0])
        nets["article only"] = net_scores(sessions, learned(sessions, "article-only")[0])
        prior = counting.arbiter.log_prior.exp().tolist()
        row = {"prior_alpha_beta": [round(x, 2) for x in prior], "by_accuracy": {}}
        for r in SEEN + UNSEEN:
            m = accuracy == r
            row["by_accuracy"][str(r)] = {k: round(v[m].mean().item(), 4) for k, v in nets.items()}
            row["by_accuracy"][str(r)]["sessions"] = int(sum(s["board_accuracy"] == r for s in sessions))
            low = min(row["by_accuracy"][str(r)]["board only"], row["by_accuracy"][str(r)]["article only"])
            criteria["3_never_below_both_single_channels"] &= row["by_accuracy"][str(r)]["counting arbiter"] >= low
        diff = nets["counting arbiter"] - nets["learned arbiter"]
        outside = torch.tensor([a.item() in (0.25, 0.1) for a in accuracy])
        inside = torch.tensor([a.item() in SEEN for a in accuracy])
        row["outside_0.25_0.1"] = {"counting_minus_learned": round(diff[outside].mean().item(), 4),
                                   "ci_sessions": cluster_bootstrap(diff[outside], sid[outside])}
        row["inside_seen_levels"] = {"counting_minus_learned": round(diff[inside].mean().item(), 4),
                                     "ci_sessions": cluster_bootstrap(diff[inside], sid[inside])}
        criteria["1_outside_experience"] &= row["outside_0.25_0.1"]["ci_sessions"][0] > 0
        criteria["2_inside_experience"] &= row["inside_seen_levels"]["counting_minus_learned"] >= -0.02
        result["seeds"][seed] = row
        print(seed, json.dumps(row, ensure_ascii=False), flush=True)
    result["criteria"] = criteria
    result["graduated"] = all(criteria.values())
    path = RESULTS_DIR / "mk1-i_outside_experience.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("criteria", "graduated")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
