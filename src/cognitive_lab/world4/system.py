"""Stage 4 of the ladder: grow by connecting grown parts (rule world v4).

Parts
- trust part: the stage-1 system (frozen reader body, connection head, memory, judge). Asked
  about each base door, it gives key probabilities from what the speakers said and what it has
  learned about them this session.
- thinking part: the stage-3 thinker (link head, thinking loop, answer head, fixed-point stop).
- adapter (new, small): turns the trust part's conclusion for a base door (one-hot key if it is
  sure, p > 0.5, else uniform) into fact messages the thinker can take in, [p_k, max_k p] -> M
  per (door, key), shared over doors/keys. (Passing graded probabilities failed: the thinker only
  ever saw certain facts.)
Routing is by source: speakers' utterances go to the trust part, the board's links to the thinker.

Arms (same data, same architecture):
  grown   : both parts loaded and frozen; only the adapter learns
  retune  : both parts loaded; everything learns
  scratch : both parts freshly initialized (reader body still frozen); everything learns

python -m cognitive_lab.world4.system --seed 42 --arm grown
"""

import argparse
import json
import random
import statistics
import time

import torch
from torch import nn

from cognitive_lab.world2 import hedged
from cognitive_lab.world2.hedged_system import ConnectedSystem
from cognitive_lab.world2.hedged_system import checkpoint_path as trust_checkpoint
from cognitive_lab.world2.hedged_system import feature_table as trust_features
from cognitive_lab.world2.integrated import CHECKPOINT_DIR, MAX_UTTERANCES, PROJECT_ROOT, RESULTS_DIR, calibrated_score
from cognitive_lab.world3 import chains
from cognitive_lab.world3.features import feature_table as chain_features
from cognitive_lab.world3.thinker import MAX_UTTERANCES as BOARD_SLOTS
from cognitive_lab.world3.thinker import Thinker, fixed_point_stop, scores
from cognitive_lab.world4 import composite

TRAIN_STEPS, TEST_STEPS = 8, 16


def session_tensors(sessions: list[dict], trust_rows: list, chain_table: dict) -> dict:
    """Trust inputs per base door, anchor and truth per episode, board and query per episode."""
    lookup = {row: i for i, row in enumerate(trust_rows)}
    board_lookup = {s["text"]: i for i, s in enumerate(chain_table["sentences"])}
    batch, episodes = len(sessions), len(sessions[0]["episodes"])
    index = torch.full((batch, episodes, 3, MAX_UTTERANCES, 3), len(trust_rows), dtype=torch.long)
    speakers = torch.full((batch, episodes, MAX_UTTERANCES), -1, dtype=torch.long)
    anchor = torch.zeros(batch, episodes, dtype=torch.long)
    truths = torch.zeros(batch, episodes, dtype=torch.long)
    board = torch.full((batch, episodes, BOARD_SLOTS), len(chain_table["sentences"]), dtype=torch.long)
    query = torch.zeros(batch, episodes, dtype=torch.long)
    length = torch.zeros(batch, episodes, dtype=torch.long)
    for b, session in enumerate(sessions):
        names = {name: s for s, name in enumerate(session["speakers"])}
        for t, episode in enumerate(session["episodes"]):
            keys = episode["entities"]["keys"]
            for u, utterance in enumerate(episode["utterances"][:MAX_UTTERANCES]):
                speakers[b, t, u] = names[utterance["source"]]
                for d, door in enumerate(composite.BASE_DOORS):
                    index[b, t, d, u] = torch.tensor([lookup[(utterance["text"], door, key)] for key in keys])
            info = episode["composite"]
            anchor[b, t] = composite.BASE_DOORS.index(info["anchor"])
            truths[b, t] = keys.index(episode["answer"])
            board[b, t, :len(episode["board"])] = torch.tensor([board_lookup[x] for x in episode["board"]])
            query[b, t] = chains.DOORS.index(info["query"])
            length[b, t] = info["length"]
    return {"index": index, "speakers": speakers, "anchor": anchor, "truths": truths,
            "board": board, "query": query, "length": length}


class CompositeSystem(nn.Module):
    def __init__(self, trust: ConnectedSystem, thinker: Thinker, size: int = 8, width: int = 16):
        super().__init__()
        self.trust, self.thinker = trust, thinker
        self.adapter = nn.Sequential(nn.Linear(2, width), nn.Tanh(), nn.Linear(width, size))

    def trust_logits(self, data: dict) -> torch.Tensor:
        """Key logits for each base door in each episode [B,E,3,3], memory updated from feedback."""
        judge = self.trust.judge
        table = self.trust.messages()
        index, speakers, anchor, truths = data["index"], data["speakers"], data["anchor"], data["truths"]
        batch, episodes = truths.shape
        memory = judge.initial_memory(batch)
        rows = torch.arange(batch, device=truths.device)
        out = []
        for t in range(episodes):
            reads = [judge.read(table[index[:, t, d]], speakers[:, t]) for d in range(3)]
            out.append(torch.stack([judge.logits(h, spoke, memory) for h, spoke in reads], 1))
            h = torch.stack([h for h, _ in reads], 1)[rows, anchor[:, t]]
            memory = judge.update(memory, h, reads[0][1], truths[:, t])
        return torch.stack(out, 1)

    def answer_logits(self, door_logits: torch.Tensor, board: torch.Tensor, query: torch.Tensor,
                      steps: int) -> torch.Tensor:
        """door_logits [N,3,3] -> per-step answer logits [T,N,3] from the thinker."""
        # Pass conclusions, not graded beliefs: the thinker grew up on certain facts only and
        # cannot carry "70% sun key" (trained on such inputs it hides in 모름). A base door the
        # trust part is sure about (p > 0.5, its own answer rule) becomes a certain fact; otherwise
        # no fact. Straight-through: arms that train the trust part still get its gradient.
        q = door_logits.softmax(-1)
        sure = q.amax(-1, keepdim=True) > 0.5
        crisp = torch.where(sure, nn.functional.one_hot(q.argmax(-1), 3).float(), torch.full_like(q, 1 / 3))
        p = crisp + q - q.detach()
        features = torch.stack([p, p.amax(-1, keepdim=True).expand_as(p)], -1)  # [N,3,3,2]
        messages = torch.tanh(self.adapter(features))
        count = board.shape[0]
        extra = messages.new_zeros(count, len(chains.DOORS), 3, messages.shape[-1])
        extra[:, :3] = messages
        mask = torch.zeros(count, extra.shape[1], dtype=torch.bool, device=board.device)
        mask[:, :3] = True
        logits, _ = self.thinker.think(board, query, steps, extra_facts=extra, extra_mask=mask)
        return logits


def flat(x: torch.Tensor) -> torch.Tensor:
    return x.reshape(-1, *x.shape[2:])


@torch.no_grad()
def evaluate(model: CompositeSystem, data: dict, door_logits: torch.Tensor | None = None) -> dict:
    door_logits = model.trust_logits(data) if door_logits is None else door_logits
    logits = model.answer_logits(flat(door_logits), flat(data["board"]), flat(data["query"]), TEST_STEPS)
    stop = fixed_point_stop(model.thinker.last_changes)
    rows = torch.arange(logits.shape[1], device=logits.device)
    target = flat(data["truths"])
    result = scores(logits[stop, rows], target)
    anchor_logits = door_logits[torch.arange(door_logits.shape[0])[:, None], torch.arange(door_logits.shape[1]),
                                data["anchor"]]
    length = flat(data["length"])
    return {"score": round(result.mean().item(), 4),
            "trust_part_alone": calibrated_score(anchor_logits, data["truths"]),
            "by_length": {int(l): round(result[length == l].mean().item(), 4) for l in length.unique()},
            "mean_steps": round((stop.float() + 1).mean().item(), 2)}


def load_parts(seed: int, arm: str, device: torch.device) -> tuple[ConnectedSystem, Thinker, list, dict]:
    saved = torch.load(trust_checkpoint(seed, 8), map_location="cpu")
    rows, features, head = trust_features(saved["reader"], device)
    trust = ConnectedSystem(features, head, saved["message_size"])
    chain_table = chain_features(device)
    thinker = Thinker(chain_table["features"].to(device), message="state")
    if arm != "scratch":
        trust.load_state_dict(saved["state"])
        thinker.load_state_dict(torch.load(CHECKPOINT_DIR / f"world-v3_thinker-grown_seed-{seed + 3}.pt")["state"])
    return trust.to(device), thinker.to(device), rows, chain_table


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 4: connect grown parts for a new problem (v4)")
    parser.add_argument("--seed", type=int, default=42, help="trust part seed; the thinker uses seed + 3")
    parser.add_argument("--arm", choices=("grown", "retune", "scratch"), default="grown")
    parser.add_argument("--train-sessions", type=int, default=300)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--link-cost", type=float, default=0.1)
    parser.add_argument("--no-test", action="store_true", help="development: validation only")
    parser.add_argument("--test-offset", type=int, default=200,
                        help="test sessions come from v2-H test seed (seed + offset): fresh sessions")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    device = torch.device("cuda" if args.device != "cpu" and torch.cuda.is_available() else "cpu")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    started = time.perf_counter()

    trust, thinker, rows, chain_table = load_parts(args.seed, args.arm, device)
    model = CompositeSystem(trust, thinker).to(device)
    if args.arm == "grown":
        for p in list(model.trust.parameters()) + list(model.thinker.parameters()):
            p.requires_grad_(False)
    parts = {}
    for part, count in (("train", args.train_sessions), ("validation", None), ("test", None)):
        part_seed = args.seed + (args.test_offset if part == "test" else 0)
        sessions = composite.generate_part(part, part_seed, sessions=count)
        parts[part] = {k: v.to(device) for k, v in session_tensors(sessions, rows, chain_table).items()}
        parts[part]["sessions"] = sessions
    frozen_logits = {}
    if args.arm == "grown":  # the trust part is frozen: its answers can be computed once
        with torch.no_grad():
            frozen_logits = {part: model.trust_logits(data) for part, data in parts.items()}
    before = {part: evaluate(model, parts[part], frozen_logits.get(part)) for part in ("validation",)}
    print(f"tensors ready ({time.perf_counter() - started:.0f}s); start validation {before['validation']}", flush=True)

    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.learning_rate)
    generator = torch.Generator().manual_seed(args.seed)
    train = parts["train"]
    history, best = [], (-2.0, 0, None)
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = torch.randperm(args.train_sessions, generator=generator).to(device)
        losses = []
        for start in range(0, len(order), args.batch_size):
            batch = order[start:start + args.batch_size]
            data = {k: v[batch] for k, v in train.items() if k != "sessions"}
            door_logits = frozen_logits["train"][batch] if frozen_logits else model.trust_logits(data)
            logits = model.answer_logits(flat(door_logits), flat(data["board"]), flat(data["query"]), TRAIN_STEPS)
            if args.arm == "grown":
                # A connection's job is faithful transmission: the thinker should report what the
                # trust part concluded about the anchor (its key if sure, else 모름), right or wrong.
                # Being right is the trust part's job. (Trained on the true answers instead, the
                # adapter hides in 모름 whenever the trust part might be wrong.) Needs no answers.
                rows = torch.arange(door_logits.shape[0], device=device)[:, None]
                cols = torch.arange(door_logits.shape[1], device=device)
                anchor_q = flat(door_logits[rows, cols, data["anchor"]]).softmax(-1)
                sure = anchor_q.amax(-1, keepdim=True) > 0.5
                goal = torch.where(sure, nn.functional.one_hot(anchor_q.argmax(-1), 3).float(),
                                   torch.full_like(anchor_q, 1 / 3))
                loss = -(goal * logits.log_softmax(-1)).sum(-1).mean()
            else:
                target = flat(data["truths"])
                loss = nn.functional.cross_entropy(logits.reshape(-1, 3), target.repeat(TRAIN_STEPS))
            if args.arm != "grown":  # as in stage 3: keep only the links the answers need
                loss = loss + args.link_cost * model.thinker.last_strength.mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())
        model.eval()
        validation = evaluate(model, parts["validation"], frozen_logits.get("validation"))
        history.append({"epoch": epoch, "train_loss": round(statistics.mean(losses), 4), **validation})
        print(history[-1], flush=True)
        if validation["score"] > best[0]:
            best = (validation["score"], epoch, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()})

    model.load_state_dict(best[2])
    model.eval()
    if args.no_test:
        print(f"selected epoch {best[1]}; validation {best[0]}")
        return
    test = evaluate(model, parts["test"], frozen_logits.get("test"))
    ideal = hedged.ceilings(parts["test"]["sessions"])
    # Old abilities after this training: the thinker on its own stage-3 test (the trust part's
    # own v2-H test score is test["trust_part_alone"]: v4 test sessions are the v2-H test sessions).
    from cognitive_lab.world3.thinker import episode_tensors
    from cognitive_lab.world3.thinker import evaluate as think_evaluate

    v3_test = tuple(t.to(device) for t in episode_tensors(chains.generate("test", args.seed + 3), chain_table))
    old_thinker = think_evaluate(model.thinker, v3_test, TEST_STEPS)["adaptive"]
    result = {"world": "v4", "experiment": f"composite-{args.arm}", "seed": args.seed, "thinker_seed": args.seed + 3,
              "test_session_seed": args.seed + args.test_offset,
              "train_sessions": args.train_sessions, "selected_epoch": best[1],
              "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
              "history": history, "test": test, "ideal_trust_learner_with_cues": ideal["learner+cue"],
              "old_abilities": {"trust_v2h_test": test["trust_part_alone"], "thinker_v3_test": old_thinker},
              "training_seconds": round(time.perf_counter() - started, 1)}
    path = CHECKPOINT_DIR / f"world-v4_{args.arm}_seed-{args.seed}.pt"
    torch.save({"state": best[2], "arm": args.arm, "epoch": best[1]}, path)
    result["checkpoint"] = str(path.relative_to(PROJECT_ROOT))
    out = RESULTS_DIR / f"world-v4_{args.arm}_seed-{args.seed}.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"selected epoch {best[1]}; test {test}; ideal (trust learner with cues) {ideal['learner+cue']}")
    print(f"Saved: {path}\nSaved: {out}")


if __name__ == "__main__":
    main()
