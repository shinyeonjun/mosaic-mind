"""MK1 stage D: real Korean articles on the board, read live, mixed with the rule worlds (design/mk1-integration.md, D).

python -m cognitive_lab.mk1.stage_d
"""

import json
import random
import time

import torch

from cognitive_lab.mk1.articles import article_sessions
from cognitive_lab.mk1.integration import bootstrap
from cognitive_lab.mk1.system import MK1, chain_sessions, composite_sessions, episode_score
from cognitive_lab.world2.asking_system import COST as ASK_COST
from cognitive_lab.world2.integrated import RESULTS_DIR
from cognitive_lab.world3 import chains
from cognitive_lab.world6 import world
from cognitive_lab.world7.routing import COST, generate


def net_scores(sessions: list[dict], out: dict) -> torch.Tensor:
    episodes = [e for s in sessions for e in s["episodes"]]
    raw = torch.tensor([episode_score(e, a) for e, a in zip(episodes, out["answer"])])
    return raw - COST * out["asks"].cpu() - ASK_COST * out["speaker_asks"].cpu()


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    started = time.perf_counter()
    articles = article_sessions(world.generate("test", 42), "test")
    result = {"stage": "MK1-D"}

    cached = MK1(device, articles_from_cache=True)
    cached_out = cached(articles)[0]
    by_cache = net_scores(articles, cached_out)
    del cached
    model = MK1(device)  # live fp32 article reading
    before = model.articles.new
    t0 = time.perf_counter()
    live_out, _ = model(articles)
    seconds = time.perf_counter() - t0
    by_live = net_scores(articles, live_out)
    read = model.articles.new - before
    result["D2_live_reading"] = {"cached_bf16": round(by_cache.mean().item(), 4), "live_fp32": round(by_live.mean().item(), 4),
                                 "live_minus_cached_ci95": bootstrap(by_live - by_cache),
                                 "answers_changed": sum(a != b for a, b in zip(live_out["answer"], cached_out["answer"])),
                                 "passages_read": read, "seconds": round(seconds, 1),
                                 "passages_per_second": round(read / seconds, 1) if read else None}
    print("D2", json.dumps(result["D2_live_reading"], ensure_ascii=False), flush=True)

    rule = composite_sessions(generate("test", 42)) + chain_sessions(chains.generate("test", 45))
    alone = {"rule": net_scores(rule, model(rule, "route", curious=True)[0]), "article": by_live}
    mixed = rule + articles
    order = list(range(len(mixed)))
    random.Random(0).shuffle(order)
    shuffled = [mixed[i] for i in order]
    t0 = time.perf_counter()
    out, _ = model(shuffled, "route", curious=True)
    seconds = time.perf_counter() - t0
    net = net_scores(shuffled, out)
    tags = [e["tag"] for s in shuffled for e in s["episodes"]]
    expected = torch.cat([alone["rule"], alone["article"]])
    starts = torch.tensor([0] + [len(s["episodes"]) for s in mixed]).cumsum(0)
    expected = torch.cat([expected[starts[i]:starts[i + 1]] for i in order])
    result["D3_mixed"] = {"problems": len(tags), "seconds": round(seconds, 1), "problems_per_second": round(len(tags) / seconds, 1),
                          "by_kind": {}}
    for kind in sorted(set(tags)):
        m = torch.tensor([t == kind for t in tags])
        result["D3_mixed"]["by_kind"][kind] = {"problems": int(m.sum()), "mixed": round(net[m].mean().item(), 4),
                                               "alone": round(expected[m].mean().item(), 4)}
    print("D3", json.dumps(result["D3_mixed"], ensure_ascii=False), flush=True)
    result["criteria"] = {
        "D2_live_not_worse": result["D2_live_reading"]["live_minus_cached_ci95"][1] >= 0,
        "D3_mixing_breaks_nothing": all(abs(v["mixed"] - v["alone"]) <= 0.002 for v in result["D3_mixed"]["by_kind"].values()),
    }
    result["graduated"] = all(result["criteria"].values())
    result["seconds"] = round(time.perf_counter() - started, 1)
    out_path = RESULTS_DIR / "mk1-d_articles.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("criteria", "graduated")}, ensure_ascii=False))
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
