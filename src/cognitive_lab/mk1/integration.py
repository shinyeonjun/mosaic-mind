"""MK1 stage C: one MK1, one mixed world, no problem-type labels (design/mk1-integration.md, C).

Mixed test set: the stage-7 test sessions (types T / C / TC) and the stage-3 test episodes (board only,
nobody talking), handed to MK1 in one list. Runs: curiosity off, curiosity on, and (diagnostic) the
"ask only parts that heard something" gate off. Also measures speed and peak GPU memory.

python -m cognitive_lab.mk1.integration
"""

import json
import random
import time

import torch

from cognitive_lab.mk1.system import MK1, chain_sessions, composite_sessions
from cognitive_lab.world2.asking_system import COST as ASK_COST
from cognitive_lab.world2.integrated import RESULTS_DIR
from cognitive_lab.world3 import chains
from cognitive_lab.world3.thinker import scores
from cognitive_lab.world7.routing import COST, generate

STAGE_SCORES = {"T": 0.3926, "C": 1.0, "TC": 0.4109, "board-only": 0.975}


def bootstrap(diff: torch.Tensor, seed: int = 0, rounds: int = 2000) -> list[float]:
    generator = torch.Generator().manual_seed(seed)
    diff = diff.cpu()
    means = torch.stack([diff[torch.randint(len(diff), (len(diff),), generator=generator)].mean() for _ in range(rounds)])
    return [round(means.quantile(0.025).item(), 4), round(means.quantile(0.975).item(), 4)]


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    started = time.perf_counter()
    raw7 = generate("test", 42)
    sessions = composite_sessions(raw7) + chain_sessions(chains.generate("test", 45))
    random.Random(0).shuffle(sessions)  # one mixed list; MK1 is not told which is which
    kinds = [e["tag"] for s in sessions for e in s["episodes"]]
    kinds_t = {k: torch.tensor([x == k for x in kinds], device=device) for k in STAGE_SCORES}
    print(f"mixed set: {len(sessions)} sessions, {len(kinds)} problems "
          f"({ {k: int(v.sum()) for k, v in kinds_t.items()} }) ({time.perf_counter() - started:.0f}s)", flush=True)

    model = MK1(device)
    model(sessions[:5], "route", curious=True)  # load the reader / warm up
    runs, result = {}, {"stage": "MK1-C", "problems": len(kinds), "by_kind": {}, "runs": {}}
    for name, options in (("curiosity-off", {"curious": False}), ("curiosity-on", {"curious": True}),
                          ("gate-off (diagnostic)", {"curious": True, "gate": False})):
        if device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        reads_before = len(model.reader.memo)
        t0 = time.perf_counter()
        out, _ = model(sessions, "route", **options)
        if device.type == "cuda":
            torch.cuda.synchronize()
        seconds = time.perf_counter() - t0
        net = scores(out["logits"], out["targets"]) - COST * out["asks"] - ASK_COST * out["speaker_asks"]
        runs[name] = net
        result["runs"][name] = {
            "seconds": round(seconds, 2), "problems_per_second": round(len(kinds) / seconds, 1),
            "new_reads": len(model.reader.memo) - reads_before,
            "peak_gpu_mb": round(torch.cuda.max_memory_allocated() / 2**20) if device.type == "cuda" else None,
            "by_kind": {k: {"net": round(net[m].mean().item(), 4),
                            "router_asks": round(out["asks"][m].mean().item(), 3),
                            "speaker_asks": round(out["speaker_asks"][m].mean().item(), 3)} for k, m in kinds_t.items()}}
        print(name, json.dumps(result["runs"][name], ensure_ascii=False), flush=True)

    off, on = runs["curiosity-off"], runs["curiosity-on"]
    for k, m in kinds_t.items():
        result["by_kind"][k] = {"stage_score": STAGE_SCORES[k], "off": round(off[m].mean().item(), 4),
                                "on": round(on[m].mean().item(), 4),
                                "on_minus_off": round((on[m] - off[m]).mean().item(), 4),
                                "on_minus_off_ci95": bootstrap(on[m] - off[m])}
    criteria = {
        "1_mixing_breaks_nothing": all(abs(result["by_kind"][k]["off"] - STAGE_SCORES[k]) <= 0.002 for k in STAGE_SCORES),
        "2a_asking_pays_on_T": result["by_kind"]["T"]["on_minus_off_ci95"][0] > 0,
        "2b_no_drop_elsewhere": all(result["by_kind"][k]["on_minus_off"] >= -0.01 for k in ("C", "TC", "board-only")),
    }
    result["criteria"] = criteria
    result["graduated"] = all(criteria.values())
    # Cold reading cost: a fresh service reading a sample of board pairs live.
    from cognitive_lab.mk1.reader import ReaderService
    from cognitive_lab.mk1.system import THINKER_HYPOTHESES

    cold = ReaderService(device)
    cold._load()
    sample = list(dict.fromkeys(x for s in sessions[:40] for e in s["episodes"] for x in e["board"]))[:30]
    t0 = time.perf_counter()
    cold.table(sample, THINKER_HYPOTHESES)
    if device.type == "cuda":
        torch.cuda.synchronize()
    result["cold_reader_pairs_per_second"] = round(len(sample) * len(THINKER_HYPOTHESES) / (time.perf_counter() - t0), 1)
    result["seconds"] = round(time.perf_counter() - started, 1)
    out_path = RESULTS_DIR / "mk1-c_integration.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("by_kind", "criteria", "graduated", "cold_reader_pairs_per_second")},
                     ensure_ascii=False, indent=1))
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
