"""Graduation checks: re-measure each stage's graduation score from the saved parts, without training.

Run after any change to code a part depends on; every score must match its part card in
`mk1/registry.py` (tolerance 0.002). A missing part is reported, not skipped silently.
The board checks ("board:" names) re-measure stages 1-4, 6 and 7 with one MK1 whose parts talk only
through the blackboard and read live (mk1/system.py): wiring must not change any score.

python -m cognitive_lab.mk1.checks            # all
python -m cognitive_lab.mk1.checks stage3     # names containing "stage3"
python -m cognitive_lab.mk1.checks board      # the board checks only
"""

import sys
import time

import torch

from cognitive_lab.mk1.registry import PARTS

TOLERANCE = 0.002


def stage1_trust(device):
    from cognitive_lab.world2 import hedged
    from cognitive_lab.world2.hedged_system import ConnectedSystem, checkpoint_path, feature_table, session_index
    from cognitive_lab.world2.integrated import calibrated_score

    saved = torch.load(checkpoint_path(42, 8), map_location="cpu")
    rows, features, head = feature_table(saved["reader"], device)
    model = ConnectedSystem(features, head, saved["message_size"]).to(device)
    model.load_state_dict(saved["state"])
    model.eval()
    test = tuple(t.to(device) for t in session_index(hedged.generate_part("test", 42), rows))
    with torch.no_grad():
        return calibrated_score(model(*test), test[2])


def stage2_curiosity(device):
    from cognitive_lab.world2 import hedged
    from cognitive_lab.world2.asking_system import CuriosityHead, load_stage1, run_policy, tensors
    from cognitive_lab.world2.integrated import CHECKPOINT_DIR

    system, rows = load_stage1(42, device)
    with torch.no_grad():
        table = system.messages()
    base, answers = tensors(hedged.generate_part("test", 42), rows, device)
    head = CuriosityHead().to(device)
    head.load_state_dict(torch.load(CHECKPOINT_DIR / "world-v2a_curiosity_cost-0.2_seed-42.pt")["state"])
    head.eval()
    return run_policy(system, table, base, answers, "curiosity", head, 0.2, seed=42)["net"]


def stage3_thinker(device):
    from cognitive_lab.world2.integrated import CHECKPOINT_DIR
    from cognitive_lab.world3 import chains
    from cognitive_lab.world3.features import feature_table
    from cognitive_lab.world3.thinker import Thinker, episode_tensors, evaluate

    table = feature_table(device)
    model = Thinker(table["features"].to(device), message="state").to(device)
    model.load_state_dict(torch.load(CHECKPOINT_DIR / "world-v3_thinker-grown_seed-45.pt")["state"])
    model.eval()
    data = tuple(t.to(device) for t in episode_tensors(chains.generate("test", 45), table))
    return evaluate(model, data, 16)["adaptive"]


def stage4_composite(device):
    from cognitive_lab.world4 import composite
    from cognitive_lab.world4.system import CHECKPOINT_DIR, CompositeSystem, evaluate, load_parts, session_tensors

    trust, thinker, rows, chain_table = load_parts(42, "grown", device)
    model = CompositeSystem(trust, thinker).to(device)
    model.load_state_dict(torch.load(CHECKPOINT_DIR / "world-v4_grown_seed-42.pt")["state"])
    model.eval()
    sessions = composite.generate_part("test", 42 + 200)
    data = {k: v.to(device) for k, v in session_tensors(sessions, rows, chain_table).items()}
    return evaluate(model, data)["score"]


def stage5_reader(device):
    """200 test questions are re-read live; at least 95% must give the cached answer. The cache was read
    in bf16, 32 questions at a time, and bf16 answers shift with batch composition (read one at a time,
    10 of 300 differed). Live reading is now fp32 (MK1 stage B), which does not depend on batching, so the
    remaining differences are the cache's bf16 rounding. The score is the share of answerable test
    questions the cached reading gets right (the graduation record)."""
    from cognitive_lab.world5 import klue
    from cognitive_lab.world5.reader_qa import Reader, load_predictions

    cached = load_predictions()
    questions = klue.load("test")
    reader = Reader(device)
    same = sum(reader.read(q["question"], q["context"])["answer"] == cached[q["guid"]]["answer"] for q in questions[:200])
    if same < 0.95 * 200:
        raise AssertionError(f"only {same}/200 live answers match the cached predictions")
    answerable = [q for q in questions if not q["impossible"]]
    return round(sum(klue.is_right(cached[q["guid"]]["answer"], q["answers"]) for q in answerable) / len(answerable), 4)


def stage6_trust_real(device):
    from cognitive_lab.world6 import world
    from cognitive_lab.world6.trust import CHECKPOINT_DIR, TrustPart, choices, episode_scores, tensors

    data = {k: v.to(device) for k, v in tensors(world.generate("test", 42)).items()}
    model = TrustPart().to(device)
    model.load_state_dict(torch.load(CHECKPOINT_DIR / "world-v6_trust_seed-42.pt")["state"])
    model.eval()
    with torch.no_grad():
        return round(episode_scores(choices(model(data)), data).mean().item(), 4)


def stage7_router(device):
    from cognitive_lab.world4.system import CHECKPOINT_DIR, CompositeSystem, load_parts, session_tensors
    from cognitive_lab.world7.routing import COST, TYPES, Router, generate, run_all_subsets, subset_index

    trust, thinker, rows, chain_table = load_parts(42, "grown", device)
    model = CompositeSystem(trust, thinker).to(device)
    model.load_state_dict(torch.load(CHECKPOINT_DIR / "world-v4_grown_seed-42.pt")["state"])
    model.eval()
    sessions = generate("test", 42)
    data = {k: v.to(device) for k, v in session_tensors(sessions, rows, chain_table).items()}
    data["types"] = torch.tensor([[TYPES.index(e["composite"]["type"]) for e in s["episodes"]] for s in sessions], device=device)
    run = run_all_subsets(model, data)
    router = Router().to(device)
    router.load_state_dict(torch.load(CHECKPOINT_DIR / "world-v7_router_seed-42.pt")["router"])
    router.eval()
    with torch.no_grad():
        asked = router(run["features"]) > COST
    scores = run["subset_scores"].gather(1, subset_index(asked)[:, None]).squeeze(1)
    return round(scores[run["types"] == TYPES.index("TC")].mean().item(), 4)


def reader_parity(device):
    from cognitive_lab.mk1.reader import parity

    report = parity(device, samples=1000)
    return max(report["stage 1-2 (v2-H reader table)"], report["stage 3-4 (v3 thinker table)"])


_MK1 = {}


def mk1(device):
    from cognitive_lab.mk1.system import MK1

    if device not in _MK1:
        _MK1[device] = MK1(device)
    return _MK1[device]


def board_stage1(device):
    from cognitive_lab.mk1.system import composite_sessions
    from cognitive_lab.world2.integrated import calibrated_score
    from cognitive_lab.world4 import composite

    sessions = composite_sessions(composite.generate_part("test", 42))  # the v2-H test sessions, chains added
    out, _ = mk1(device)(sessions, "trust")
    shape = (len(sessions), -1)
    return calibrated_score(out["logits"].view(*shape, 3), out["targets"].view(shape))


def board_stage2(device):
    from cognitive_lab.mk1.system import speaker_sessions
    from cognitive_lab.world2 import hedged
    from cognitive_lab.world2.asking_system import COST
    from cognitive_lab.world3.thinker import scores

    out, _ = mk1(device)(speaker_sessions(hedged.generate_part("test", 42)), "trust", curious=True)
    return round((scores(out["logits"], out["targets"]) - COST * out["speaker_asks"]).mean().item(), 4)


def board_stage3(device):
    from cognitive_lab.mk1.system import chain_sessions
    from cognitive_lab.world3 import chains
    from cognitive_lab.world3.thinker import scores

    out, _ = mk1(device)(chain_sessions(chains.generate("test", 45)), "think")
    return round(scores(out["logits"], out["targets"]).mean().item(), 4)


def board_stage4(device):
    from cognitive_lab.mk1.system import composite_sessions
    from cognitive_lab.world3.thinker import scores
    from cognitive_lab.world4 import composite

    out, _ = mk1(device)(composite_sessions(composite.generate_part("test", 42 + 200)), "ask-all")
    return round(scores(out["logits"], out["targets"]).mean().item(), 4)


def board_stage6(device):
    """Real articles: sources hand over passages and MK1 reads them. The reading memory is seeded from the
    v6 cache (bf16), so the trust part sees what it graduated on; live fp32 reading is measured in stage D."""
    from cognitive_lab.mk1.articles import article_sessions
    from cognitive_lab.mk1.system import MK1, episode_score
    from cognitive_lab.world6 import world

    if ("cache", device) not in _MK1:
        _MK1[("cache", device)] = MK1(device, articles_from_cache=True)
    sessions = article_sessions(world.generate("test", 42), "test")
    out, _ = _MK1[("cache", device)](sessions)
    episodes = [e for s in sessions for e in s["episodes"]]
    return round(sum(episode_score(e, a) for e, a in zip(episodes, out["answer"])) / len(episodes), 4)


def board_stage7(device):
    from cognitive_lab.mk1.system import composite_sessions
    from cognitive_lab.world3.thinker import scores
    from cognitive_lab.world7.routing import COST, generate

    raw = generate("test", 42)
    out, board = mk1(device)(composite_sessions(raw), "route")
    net = scores(out["logits"], out["targets"]) - COST * out["asks"]
    tc = torch.tensor([e["composite"]["type"] == "TC" for s in raw for e in s["episodes"]], device=net.device)
    return round(net[tc].mean().item(), 4)


BOARD_CHECKS = {"stage1-trust": board_stage1, "stage2-curiosity": board_stage2, "stage3-thinker": board_stage3, "stage4-composite": board_stage4,
                "stage6-trust-real": board_stage6, "stage7-router": board_stage7}


CHECKS = {"reader-parity": reader_parity, "stage1-trust": stage1_trust, "stage2-curiosity": stage2_curiosity,
          "stage3-thinker": stage3_thinker, "stage4-composite": stage4_composite, "stage5-reader": stage5_reader,
          "stage6-trust-real": stage6_trust_real, "stage7-router": stage7_router}


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wanted = sys.argv[1:]
    failures = 0
    for part in PARTS:
        name, expected = part.graduation.get("check"), part.graduation.get("score")
        if wanted and not any(w in name for w in wanted):
            continue
        if expected is None or name not in CHECKS:
            print(f"MISSING  {name:20s} {part.role}: no saved part to re-measure")
            continue
        started = time.perf_counter()
        try:
            got = float(CHECKS[name](device))
            ok = abs(got - expected) <= TOLERANCE
        except Exception as error:  # report and keep going
            got, ok = float("nan"), False
            print(f"ERROR    {name}: {error}")
        failures += not ok
        print(f"{'PASS' if ok else 'FAIL':8s} {name:20s} expected {expected:.4f} got {got:.4f} "
              f"({time.perf_counter() - started:.0f}s)  {part.role}", flush=True)
    for part in PARTS:
        name, expected = part.graduation.get("check"), part.graduation.get("score")
        if name not in BOARD_CHECKS or wanted and not any(w in "board:" + name for w in wanted):
            continue
        started = time.perf_counter()
        try:
            got = float(BOARD_CHECKS[name](device))
            ok = abs(got - expected) <= TOLERANCE
        except Exception as error:  # report and keep going
            got, ok = float("nan"), False
            print(f"ERROR    board:{name}: {error!r}")
        failures += not ok
        print(f"{'PASS' if ok else 'FAIL':8s} {'board:' + name:20s} expected {expected:.4f} got {got:.4f} "
              f"({time.perf_counter() - started:.0f}s)  MK1 on the board", flush=True)
    print("all passed" if failures == 0 else f"{failures} failed")
    return failures


if __name__ == "__main__":
    raise SystemExit(main())
