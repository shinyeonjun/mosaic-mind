"""MK1 stage G: should MK1 ask to verify what the board already says? (world v11, design/mk1-integration.md, G)

Measured before building anything: the stage-7 router, unchanged, seeing the arbiter's confidence (which
knows how often the board has been right this session) instead of the thinker's raw belief, against guides
that know each session's board accuracy.

python -m cognitive_lab.mk1.stage_g
"""

import json

import torch

from cognitive_lab.mk1.arbiter import BOARD_ACCURACIES
from cognitive_lab.mk1.integration import bootstrap
from cognitive_lab.mk1.stage_d import net_scores
from cognitive_lab.mk1.system import MK1
from cognitive_lab.world2.integrated import RESULTS_DIR
from cognitive_lab.world10 import world


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sessions = world.generate("test", 42, board_accuracies=BOARD_ACCURACIES)
    episodes = [e for s in sessions for e in s["episodes"]]
    accuracy = torch.tensor([s["board_accuracy"] for s in sessions for _ in s["episodes"]])
    c = torch.tensor([e["tag"] == "C" for e in episodes])
    model = MK1(device)
    nets, asks = {}, {}
    for policy in ("verify-never", "verify-always", "verify-router", "verify-hindsight"):
        out, _ = model(sessions, policy)
        nets[policy], asks[policy] = net_scores(sessions, out), out["asks"].cpu()
    guides = {}
    for rule in ((), (0.5,), (0.5, 0.75), (0.5, 0.75, 1.0)):
        verify = c & torch.tensor([a.item() in rule for a in accuracy])
        guides[f"verify C when r in {list(rule)}"] = torch.where(verify | ~c, nets["verify-always"], nets["verify-never"])
    best = max(guides, key=lambda k: guides[k].mean().item())
    result = {"stage": "MK1-G",
              "policies": {p: {"net": round(v.mean().item(), 4), "asks_in_C": round(asks[p][c].mean().item(), 3),
                               "by_accuracy": {str(r): round(v[accuracy == r].mean().item(), 4) for r in BOARD_ACCURACIES}}
                           for p, v in nets.items()},
              "guides": {k: round(v.mean().item(), 4) for k, v in guides.items()}, "best_guide": best,
              "best_guide_minus_router_ci95": bootstrap(guides[best] - nets["verify-router"])}
    path = RESULTS_DIR / "mk1-g_verify.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
