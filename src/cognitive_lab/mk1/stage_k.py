"""MK1 stage K: noticing an unexpected change in a channel (design/mk1-integration.md, K).

Counting arbiter vs change-point arbiter (both fitted on the same mixed experience: no change / mild changes /
betrayals to 0.25), judged on fresh worlds (test seed + 300) by the final net score through the thinker.

python -m cognitive_lab.mk1.stage_k     # after python -m cognitive_lab.mk1.arbiter --seed N --kind counting|change --switch
"""

import json

import torch

from cognitive_lab.mk1.arbiter import BOARD_ACCURACIES
from cognitive_lab.mk1.audit import cluster_bootstrap, session_ids
from cognitive_lab.mk1.stage_d import net_scores
from cognitive_lab.mk1.system import MK1
from cognitive_lab.world2.integrated import RESULTS_DIR
from cognitive_lab.world10 import world

WORLDS = {
    "betrayal -> 0.25": dict(board_accuracies=(1.0, 0.75), switch=True, switch_to=(0.25,)),
    "betrayal -> 0.1": dict(board_accuracies=(1.0, 0.75), switch=True, switch_to=(0.1,)),
    "mild change": dict(board_accuracies=BOARD_ACCURACIES, switch=True),
    "no change": dict(board_accuracies=BOARD_ACCURACIES),
}


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result = {"stage": "MK1-K", "seeds": {}}
    criteria = {"1_betrayal_change_beats_counting": True, "2_no_cost_without_betrayal": True}
    for seed in (42, 43, 44):
        models = {k: MK1(device, seed=seed, arbiter=k) for k in ("counting-switch", "change-switch")}
        row, betrayal_diff, betrayal_sid, offset = {}, [], [], 0
        for name, kw in WORLDS.items():
            sessions = world.generate("test", seed + 300, **kw)
            nets = {k: net_scores(sessions, m(sessions, "arbitrate")[0]) for k, m in models.items()}
            diff = nets["change-switch"] - nets["counting-switch"]
            sid = session_ids(sessions)
            t = torch.tensor([i for s in sessions for i in range(len(s["episodes"]))])
            switch = torch.tensor([s["switch_at"] if s["switch_at"] is not None else 99 for s in sessions for _ in s["episodes"]])
            c = torch.tensor([e["tag"] == "C" for s in sessions for e in s["episodes"]])
            later = c & (t >= switch + 10)
            row[name] = {k: round(v.mean().item(), 4) for k, v in nets.items()}
            row[name]["change_minus_counting"] = round(diff.mean().item(), 4)
            row[name]["ci_sessions"] = cluster_bootstrap(diff, sid)
            if later.any():
                row[name]["C_10_plus_after_change"] = {k: round(v[later].mean().item(), 4) for k, v in nets.items()}
            if name.startswith("betrayal"):
                betrayal_diff.append(diff)
                betrayal_sid.append(sid + offset)
                offset += len(sessions)
            else:
                criteria["2_no_cost_without_betrayal"] &= row[name]["change_minus_counting"] >= -0.01
        both = torch.cat(betrayal_diff)
        row["betrayal (both)"] = {"change_minus_counting": round(both.mean().item(), 4),
                                  "ci_sessions": cluster_bootstrap(both, torch.cat(betrayal_sid))}
        criteria["1_betrayal_change_beats_counting"] &= row["betrayal (both)"]["ci_sessions"][0] > 0
        row["fitted"] = {k: {"alpha_beta": [round(x, 2) for x in m.arbiter.log_prior.exp().tolist()],
                             "hazard": round(torch.sigmoid(m.arbiter.logit_hazard).item(), 4) if hasattr(m.arbiter, "logit_hazard") else None}
                         for k, m in models.items()}
        result["seeds"][seed] = row
        print(seed, json.dumps(row, ensure_ascii=False), flush=True)
    result["criteria"] = criteria
    result["graduated"] = all(criteria.values())
    path = RESULTS_DIR / "mk1-k_change.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("criteria", "graduated")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
