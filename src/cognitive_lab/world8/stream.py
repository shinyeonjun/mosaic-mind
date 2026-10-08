"""Rule world v8: an unannounced change in the problem stream, and when to grow (design/world-v8-growth.md).

40 blocks of 200 episodes, the answer revealed after each episode:
  blocks 1-10 same phrasing | 11-20 + new phrasing (50%) | 21-40 + next-key links (50%)
Policies (each runs on the same stream from the frozen stage-3 thinker):
  never     never grows
  failure   (ours) grows when the share of failures exceeds 0.05 in two blocks in a row; a failure is
            a wrong answer, or 모름 when the revealed answer shows there was one. (A first version
            counted only confidently wrong answers; on seed 45 the old thinker answers next-key
            questions with 모름, so it never fired. Kept in results/archive.)
  novelty   grows when the share of episodes with a link sentence unlike anything seen (reader
            features far from the reference sentences) exceeds 0.05 in two blocks in a row; after a
            growth, everything seen so far joins the reference
  oracle    knows the change comes at block 21
Growing: the two triggering blocks plus the next three (1,000 episodes) are the study material; a
fresh next-key branch is trained on them (900 train / 100 validation) and installed. Until then the
current part keeps answering.

python -m cognitive_lab.world8.stream --seed 45
"""

import argparse
import json
import random
import time

import torch

from cognitive_lab.world2.integrated import RESULTS_DIR
from cognitive_lab.world3.thinker import episode_tensors, fixed_point_stop
from cognitive_lab.world8.chains import episode
from cognitive_lab.world8.grow import TEST_STEPS, load_branched, train_branch

BLOCKS, BLOCK_SIZE = 40, 200
THRESHOLD, RUN = 0.05, 2
STUDY_BLOCKS = 5


def family_mix(block: int) -> tuple[str, ...]:
    return ("same",) if block <= 10 else (("same", "same-new") if block <= 20 else ("same", "next"))


def stream(seed: int) -> list[list[dict]]:
    rng = random.Random(f"world-v8|stream|{seed}")
    blocks = []
    for b in range(1, BLOCKS + 1):
        mix = family_mix(b)
        blocks.append([episode(rng, mix[n % len(mix)], rng.choice((1, 2, 3)), rng.random() >= 0.2, f"s{seed}-b{b}-{n}")
                       for n in range(BLOCK_SIZE)])
    return blocks


@torch.no_grad()
def answer(model, data) -> dict:
    index, query, target, _ = data
    logits, _ = model.think(index, query, TEST_STEPS)
    stop = fixed_point_stop(model.last_changes)
    q = logits[stop, torch.arange(len(query), device=query.device)].softmax(-1)
    best, choice = q.max(-1)
    answered = best > 0.5
    right = torch.where(target >= 0, choice == target, torch.zeros_like(answered))
    score = torch.where(answered, torch.where(right, 1.0, -1.0), torch.where(target >= 0, 0.0, 1.0))
    missed = (answered & ~right) | (~answered & (target >= 0))  # wrong, or 모름 although there was an answer
    return {"score": score.mean().item(), "surprise": (answered & ~right).float().mean().item(),
            "failure": missed.float().mean().item()}


class Novelty:
    """Share of episodes with a link sentence far from every reference sentence (cosine on the mean
    reader feature over hypotheses). The threshold is the 99th percentile of distances of blocks 6-10
    sentences to blocks 1-5 sentences: both phase 1, so it measures ordinary variety."""

    def __init__(self, table: dict, device: torch.device):
        mean = table["features"].float().mean(1)
        self.vectors = torch.nn.functional.normalize(mean, dim=-1).to(device)
        self.lookup = {s["text"]: i for i, s in enumerate(table["sentences"])}
        self.reference: set[int] = set()
        self.threshold = None

    def links(self, episodes) -> list[list[int]]:
        return [[self.lookup[s["text"]] for s in e["gold"] if s["kind"] != "fact"] for e in episodes]

    def distance(self, ids: list[int]) -> torch.Tensor:
        ref = torch.tensor(sorted(self.reference), device=self.vectors.device)
        return 1 - (self.vectors[ids] @ self.vectors[ref].T).amax(1)

    def calibrate(self, early: list[dict], later: list[dict]) -> None:
        self.reference = {i for ids in self.links(early) for i in ids}
        later_ids = sorted({i for ids in self.links(later) for i in ids})
        self.threshold = torch.quantile(self.distance(later_ids), 0.99).item()

    def rate(self, episodes) -> float:
        novel = [bool((self.distance(ids) > self.threshold).any()) for ids in self.links(episodes)]
        return sum(novel) / len(novel)

    def absorb(self, episodes) -> None:
        self.reference |= {i for ids in self.links(episodes) for i in ids}


def run(policy: str, blocks, seed: int, device: torch.device) -> dict:
    model, table = load_branched(seed, device)
    model.branch_enabled = False
    tensors = lambda eps: tuple(t.to(device) for t in episode_tensors(eps, table))
    novelty = Novelty(table, device) if policy == "novelty" else None
    log, growths, signal_run, studying, buffer = [], [], 0, None, []
    for b, episodes in enumerate(blocks, start=1):
        result = answer(model, tensors(episodes))
        if novelty is not None and b == 10:
            novelty.calibrate([e for blk in blocks[:5] for e in blk], [e for blk in blocks[5:10] for e in blk])
        signal = {"failure": result["failure"], "novelty": novelty.rate(episodes) if novelty and novelty.threshold else 0.0,
                  "never": 0.0, "oracle": 0.0}[policy]
        entry = {"block": b, "score": round(result["score"], 4), "surprise": round(result["surprise"], 4),
                 "failure": round(result["failure"], 4),
                 "signal": round(signal, 4), "branch": model.branch_enabled}
        if studying is not None:  # collecting study material
            buffer += episodes
            studying -= 1
        else:
            if policy == "oracle":
                decide = b == 22  # the change starts at block 21; two blocks seen, like the others
            elif policy in ("failure", "novelty"):
                watching = policy == "failure" or novelty.threshold is not None
                signal_run = signal_run + 1 if watching and signal > THRESHOLD else 0
                decide = signal_run >= RUN
            else:
                decide = False
            if decide:
                buffer = blocks[b - 2] + blocks[b - 1]
                studying = STUDY_BLOCKS - RUN
                entry["decided_to_grow"] = True
                signal_run = 0
        if studying == 0:  # study material complete: grow a fresh branch and install it
            model, table = load_branched(seed, device)
            rng = random.Random(f"study|{seed}|{b}")
            rng.shuffle(buffer)
            started = time.perf_counter()
            epoch = train_branch(model, tensors(buffer[100:]), tensors(buffer[:100]), seed)
            model.branch_enabled = True
            growths.append({"decided_at": b - (STUDY_BLOCKS - RUN), "installed_before": b + 1, "selected_epoch": epoch,
                            "seconds": round(time.perf_counter() - started, 1),
                            "families_studied": sorted({e["family"] for e in buffer})})
            if novelty is not None:
                novelty.absorb([e for blk in blocks[:b] for e in blk])
            studying, buffer = None, []
        log.append(entry)
    return {"blocks": log, "growths": growths}


def main() -> None:
    parser = argparse.ArgumentParser(description="Stream with an unannounced change: when to grow (v8)")
    parser.add_argument("--seed", type=int, default=45, help="stage-3 thinker seed")
    parser.add_argument("--stream-offset", type=int, default=1000,
                        help="the stream is generated from seed + offset (offset 0 was used while designing)")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    started = time.perf_counter()
    blocks = stream(args.seed + args.stream_offset)
    results = {policy: run(policy, blocks, args.seed, device) for policy in ("never", "failure", "novelty", "oracle")}

    def mean(policy, lo, hi):
        values = [e["score"] for e in results[policy]["blocks"] if lo <= e["block"] <= hi]
        return round(sum(values) / len(values), 4)

    summary = {p: {"phase1": mean(p, 1, 10), "phase2": mean(p, 11, 20), "phase3_early": mean(p, 21, 30),
                   "phase3_late": mean(p, 31, 40), "growths": [(g["decided_at"], g["installed_before"], g["families_studied"])
                                                               for g in results[p]["growths"]]} for p in results}
    out = {"world": "v8", "seed": args.seed, "stream_seed": args.seed + args.stream_offset, "summary": summary, "results": results,
           "seconds": round(time.perf_counter() - started, 1)}
    (RESULTS_DIR / f"world-v8_stream_seed-{args.seed}.json").write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                                                       encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
