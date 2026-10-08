"""MK1 stage E: a real article opens a door, a board of chains carries it on (world10, design/mk1-integration.md, E).

Nothing is trained: the stage-6 real-text trust part, the stage-4 adapter, the stage-3 thinker and the
stage-7 router are used as they are.

python -m cognitive_lab.mk1.stage_e
"""

import json
import time

import torch

from cognitive_lab.mk1.integration import bootstrap
from cognitive_lab.mk1.system import MK1, episode_score
from cognitive_lab.world2.integrated import RESULTS_DIR
from cognitive_lab.world7.routing import COST
from cognitive_lab.world4.composite import BASE_DOORS
from cognitive_lab.world10 import world
from cognitive_lab.world10.world import TYPES

POLICIES = {"think": "안 묻기", "ask-available": "기사 문 항상 묻기", "route": "사령탑"}


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    started = time.perf_counter()
    sessions = world.generate("test", 42)
    episodes = [e for s in sessions for e in s["episodes"]]
    tags = [e["tag"] for e in episodes]
    masks = {k: torch.tensor([t == k for t in tags]) for k in TYPES}
    model = MK1(device)
    nets, asks, runs = {}, {}, {}
    for policy in POLICIES:
        t0 = time.perf_counter()
        out, boards = model(sessions, policy)
        runs[policy] = (out, boards)
        raw = torch.tensor([episode_score(e, a) for e, a in zip(episodes, out["answer"])])
        asks[policy] = out["asks"].cpu()
        nets[policy] = raw - COST * asks[policy]
        print(policy, f"{time.perf_counter() - t0:.1f}s, new article reads {model.articles.new}", flush=True)
    needs = masks["A"] | masks["AC"]  # the ideal guide: ask about the article door exactly when the board lacks it
    nets["oracle"] = torch.where(needs, nets["ask-available"], nets["think"])
    asks["oracle"] = torch.where(needs, asks["ask-available"], asks["think"])
    result = {"stage": "MK1-E", "problems": len(episodes), "by_type": {}, "trace": boards[0].trace()}
    for kind, m in masks.items():
        result["by_type"][kind] = {"problems": int(m.sum()),
                                   **{p: {"net": round(nets[p][m].mean().item(), 4), "asks": round(asks[p][m].mean().item(), 3)}
                                      for p in nets}}
    result["total"] = {p: round(v.mean().item(), 4) for p, v in nets.items()}
    # Criterion 2: once asked, does the chain pass the article's conclusion on without loss?
    asked = nets["ask-available"] + COST * asks["ask-available"]
    a, ac = asked[masks["A"]], asked[masks["AC"]]
    result["asked_A_minus_AC"] = round(a.mean().item() - ac.mean().item(), 4)
    by = result["by_type"]
    result["criteria"] = {
        "1_unseen_combination_AC": abs(by["AC"]["route"]["net"] - by["AC"]["oracle"]["net"]) <= 0.02,
        "2_chain_passes_without_loss": abs(result["asked_A_minus_AC"]) <= 0.02,
        "3_no_needless_asks_C": by["C"]["route"]["asks"] <= 0.1,
        "4_router_beats_fixed": result["total"]["route"] > max(result["total"]["think"], result["total"]["ask-available"]),
    }
    # Diagnostic added after criterion 2 failed (A and AC are different questions, so their scores also
    # differ by how well the article part reads them): fidelity, paired within each episode. Of the
    # episodes where the article part concluded a key, how often is the thinker's final answer that key?
    out, boards = runs["ask-available"]
    conclusions = boards[0].read("doors.conclusions")
    result["fidelity"] = {}
    for kind in ("A", "AC"):
        told = same = silent = silent_kept = right = 0
        for i, e in enumerate(episodes):
            if e["tag"] != kind:
                continue
            c = conclusions[i, BASE_DOORS.index(e["article"]["door"])]
            key = e["keys"][int(c.argmax())] if c.max() > 0 else "모름"
            if key == "모름":
                silent += 1
                silent_kept += out["answer"][i] == "모름"
            else:
                told += 1
                same += out["answer"][i] == key
                right += key == e["answer"]
        result["fidelity"][kind] = {"article_concluded": round(told / (told + silent), 4), "conclusion_right": round(right / told, 4),
                                    "answer_equals_conclusion": round(same / told, 4), "unknown_passed_on": round(silent_kept / silent, 4)}
    result["route_minus_oracle_ci95"] = bootstrap(nets["route"] - nets["oracle"])
    result["graduated"] = all(result["criteria"].values())
    result["seconds"] = round(time.perf_counter() - started, 1)
    path = RESULTS_DIR / "mk1-e_article_doors.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "trace"}, ensure_ascii=False, indent=1))
    print(result["trace"])


if __name__ == "__main__":
    main()
