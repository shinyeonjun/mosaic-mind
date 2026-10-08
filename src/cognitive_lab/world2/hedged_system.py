"""The integrated system in rule world v2-H, with a learned connection head on the frozen reader.

Reader body: the language module grown in rule world v1, frozen. Its pooled feature for every
possible (sentence, queried door, key) pair is computed once in fp32.
Connection head: a new linear map from the 768-d feature to an M-d message (tanh), trained
end to end from final answers only. Rows 0-3 start as the v1 reader's own 4-d head, so the
system starts exactly where the integrated system in plain v2 started; rows 4.. start near 0.
Memory + judge: the integrated system's (I1Model), unchanged.

Nobody tells the system that tone matters. If the head learns to pass the cue on, the
system can beat the designed fact format, which has no place for it.

python -m cognitive_lab.world2.hedged_system --seed 42
python -m cognitive_lab.world2.hedged_system --probe       # does the frozen reader carry the cue?
"""

import argparse
import json
import random
import statistics
import time
from pathlib import Path

import torch
from torch import nn

from cognitive_lab.world.templates import ENTITY_SETS, UNKNOWN, door_name, key_name
from cognitive_lab.world2 import hedged
from cognitive_lab.world2.agents import BaseJudge
from cognitive_lab.world2.integrated import (CACHE_DIR, CHECKPOINT_DIR, DEFAULT_READER, MAX_UTTERANCES,
                                             PROJECT_ROOT, RESULTS_DIR, I1Judge, I1Model, all_sentences,
                                             calibrated_score, episode_arrays, message_table, queried_door)

MESSAGE_SIZE = 8


def all_hedged_sentences() -> list[str]:
    texts = set()
    for text in all_sentences():
        if _is_chatter(text):
            texts.add(text)
            continue
        for prefixes in hedged.CUE_PREFIXES.values():
            texts.update(prefix + text for prefix in prefixes)
    return sorted(texts)


def _is_chatter(text: str) -> bool:
    from cognitive_lab.world.templates import TEMPLATES

    return text in {template for _, template in TEMPLATES["chatter"]}


def entries() -> list[tuple[str, str, str]]:
    doors = [door_name(d) for d in ENTITY_SETS["base"]["doors"]]
    keys = [key_name(k) for k in ENTITY_SETS["base"]["keys"]]
    return [(text, door, key) for text in all_hedged_sentences() for door in doors for key in keys]


def feature_table(reader_file: str, device: torch.device) -> tuple[list, torch.Tensor, dict]:
    """entries, pooled fp32 features [N, 768], and the v1 reader's head weights."""
    cache_path = CACHE_DIR / f"hedged-features_{Path(reader_file).stem}.pt"
    if cache_path.exists():
        saved = torch.load(cache_path)
        return saved["entries"], saved["features"], saved["head"]
    from cognitive_lab.world.interface import ENCODER_DIRS
    from cognitive_lab.world.interface_anchored import PairEncoder, hypothesis, load_reader

    saved = torch.load(CHECKPOINT_DIR / reader_file, map_location="cpu")
    encoder = PairEncoder(encoder_dir=ENCODER_DIRS[saved["encoder_name"]]).to(device)
    load_reader(encoder, saved["reader"])
    encoder.eval()
    pooled = {}
    encoder.head.register_forward_hook(lambda module, inputs, output: pooled.setdefault("x", inputs[0]))
    rows, features = entries(), []
    with torch.no_grad():
        for start in range(0, len(rows), 256):
            chunk = rows[start:start + 256]
            pooled.clear()
            encoder([t for t, _, _ in chunk], [hypothesis(d, k) for _, d, k in chunk], precise=True)
            features.append(pooled["x"].float().cpu())
    head = {k: v.detach().cpu().clone() for k, v in encoder.head.state_dict().items()}
    result = {"entries": rows, "features": torch.cat(features), "head": head}
    torch.save(result, cache_path)
    return rows, result["features"], head


def check_features(rows, features, head, reader_file: str, device: torch.device) -> float:
    """The v1 head on cached features must reproduce the integrated system's cached messages."""
    old = message_table(reader_file, device)
    index = {row: i for i, row in enumerate(rows)}
    shared = [row for row in old if row in index]
    weight, bias = head["weight"], head["bias"]
    rebuilt = torch.tanh(features[[index[r] for r in shared]] @ weight.T + bias)
    error = (rebuilt - torch.stack([old[r] for r in shared])).abs().max().item()
    if len(shared) != len(old) or error > 1e-4:
        raise RuntimeError(f"feature cache mismatch: {len(shared)}/{len(old)} entries, max error {error:.2e}")
    return error


class ConnectedSystem(nn.Module):
    def __init__(self, features: torch.Tensor, head: dict, size: int = MESSAGE_SIZE):
        super().__init__()
        self.register_buffer("features", features, persistent=False)
        self.head = nn.Linear(features.shape[1], size)
        with torch.no_grad():
            self.head.weight.normal_(0, 0.01)
            self.head.bias.zero_()
            self.head.weight[:4] = head["weight"]
            self.head.bias[:4] = head["bias"]
        self.judge = I1Model(message_size=size)

    def messages(self) -> torch.Tensor:
        """[N + 1, M]; the last row is the zero message for padding."""
        table = torch.tanh(self.head(self.features))
        return torch.cat([table, table.new_zeros(1, table.shape[1])])

    def forward(self, index, speakers, truths):
        return self.judge(self.messages()[index], speakers, truths)


def session_index(sessions: list[dict], rows: list) -> tuple[torch.Tensor, ...]:
    """entry index [B,E,U,3] (pad = N), speakers [B,E,U] (-1 = pad), truths [B,E]."""
    lookup = {row: i for i, row in enumerate(rows)}
    pad = len(rows)
    batch, episodes = len(sessions), len(sessions[0]["episodes"])
    index = torch.full((batch, episodes, MAX_UTTERANCES, 3), pad, dtype=torch.long)
    speakers = torch.full((batch, episodes, MAX_UTTERANCES), -1, dtype=torch.long)
    truths = torch.zeros(batch, episodes, dtype=torch.long)
    for b, session in enumerate(sessions):
        names = {name: s for s, name in enumerate(session["speakers"])}
        for t, episode in enumerate(session["episodes"]):
            door, keys = queried_door(episode), episode["entities"]["keys"]
            for u, utterance in enumerate(episode["utterances"][:MAX_UTTERANCES]):
                index[b, t, u] = torch.tensor([lookup[(utterance["text"], door, key)] for key in keys])
                speakers[b, t, u] = names[utterance["source"]]
            truths[b, t] = keys.index(episode["answer"])
    return index, speakers, truths


class ConnectedJudge(I1Judge):
    """Scoreboard agent: the integrated system's judge with a message table from the learned head."""

    def __init__(self, checkpoint: Path, device: str = "cpu"):
        saved = torch.load(checkpoint, map_location="cpu")
        rows, features, head = feature_table(saved["reader"], torch.device(device))
        system = ConnectedSystem(features, head, saved["message_size"])
        system.load_state_dict(saved["state"])
        system.eval()
        with torch.no_grad():
            table = system.messages()
        BaseJudge.__init__(self, None)
        self.model, self.table, self.size = system.judge, dict(zip(rows, table[:-1])), saved["message_size"]
        self.checkpoint = str(checkpoint)

    def act(self, episode: dict) -> str:
        messages, speaker = episode_arrays(episode, self.speakers, self.table, self.size)
        with torch.no_grad():
            self._h, self._spoke = self.model.read(messages[None], speaker[None])
            q = self.model.logits(self._h, self._spoke, self.memory).softmax(-1)[0]
        best = int(q.argmax())
        return episode["entities"]["keys"][best] if q[best] > 0.5 else UNKNOWN


def checkpoint_path(seed: int, size: int) -> Path:
    return CHECKPOINT_DIR / f"world-v2h_connected-m{size}_seed-{seed}.pt"


def probe(reader_file: str, device: torch.device) -> None:
    """Linear probe for the cue (sure / plain / hedge), held out by base sentence."""
    rows, features, head = feature_table(reader_file, device)
    cue_of = {prefix: cue for cue, prefixes in hedged.CUE_PREFIXES.items() for prefix in prefixes if prefix}
    labels, groups = [], []
    for text, _, _ in rows:
        prefix = next((p for p in cue_of if text.startswith(p)), "")
        labels.append(hedged.CUES.index(cue_of.get(prefix, "plain")))
        groups.append(text[len(prefix):])
    labels = torch.tensor(labels)
    keep = torch.tensor([not _is_chatter(g) for g in groups])
    bases = sorted(set(groups))
    rng = random.Random(0)
    held = set(rng.sample(bases, len(bases) // 4))
    test = torch.tensor([g in held for g in groups]) & keep
    train = ~torch.tensor([g in held for g in groups]) & keep
    old = torch.tanh(features @ head["weight"].T + head["bias"])
    for name, x in (("v1 4-d message", old), ("768-d pooled feature", features)):
        x = (x - x[train].mean(0)) / (x[train].std(0) + 1e-6)
        model = nn.Linear(x.shape[1], 3)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        for _ in range(500):
            loss = nn.functional.cross_entropy(model(x[train]), labels[train])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        accuracy = (model(x[test]).argmax(-1) == labels[test]).float().mean().item()
        print(f"cue probe from {name}: held-out accuracy {accuracy:.3f} (chance about 0.43)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Integrated system with a learned connection head in v2-H")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reader", default=DEFAULT_READER)
    parser.add_argument("--message-size", type=int, default=MESSAGE_SIZE)
    parser.add_argument("--train-sessions", type=int, default=3000)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--head-learning-rate", type=float, default=1e-3)
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    device = torch.device("cuda" if args.device != "cpu" and torch.cuda.is_available() else "cpu")
    if args.probe:
        probe(args.reader, device)
        return
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    started = time.perf_counter()

    rows, features, head = feature_table(args.reader, device)
    error = check_features(rows, features, head, args.reader, device)
    train_sessions = hedged.generate_part("train", args.seed, sessions=args.train_sessions)
    validation_sessions = hedged.generate_part("validation", args.seed)
    train = tuple(t.to(device) for t in session_index(train_sessions, rows))
    validation = tuple(t.to(device) for t in session_index(validation_sessions, rows))
    print(f"entries {len(rows)} (feature check max error {error:.1e}), train {train[0].shape[0]} sessions "
          f"({time.perf_counter() - started:.0f}s)", flush=True)

    model = ConnectedSystem(features, head, args.message_size).to(device)
    optimizer = torch.optim.Adam([{"params": model.head.parameters(), "lr": args.head_learning_rate},
                                  {"params": model.judge.parameters(), "lr": args.learning_rate}])
    history, best = [], (-2.0, 0, None)
    generator = torch.Generator().manual_seed(args.seed)
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = torch.randperm(train[0].shape[0], generator=generator)
        losses = []
        for start in range(0, len(order), args.batch_size):
            batch = order[start:start + args.batch_size].to(device)
            index, speakers, truths = (tensor[batch] for tensor in train)
            loss = nn.functional.cross_entropy(model(index, speakers, truths).reshape(-1, 3), truths.reshape(-1))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())
        model.eval()
        with torch.no_grad():
            score = calibrated_score(model(*validation), validation[2])
        history.append({"epoch": epoch, "train_loss": round(statistics.mean(losses), 4), "validation_score": score})
        print(history[-1], flush=True)
        if score > best[0]:
            best = (score, epoch, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()})

    path = checkpoint_path(args.seed, args.message_size)
    torch.save({"state": best[2], "reader": args.reader, "message_size": args.message_size,
                "epoch": best[1]}, path)

    # Test: batched score, then the episode-by-episode agent must reproduce it exactly.
    from cognitive_lab.world2.scoreboard import run_sessions

    test_sessions = hedged.generate_part("test", args.seed)
    model.load_state_dict(best[2])
    model.eval()
    test = tuple(t.to(device) for t in session_index(test_sessions, rows))
    with torch.no_grad():
        batched = calibrated_score(model(*test), test[2])
    agent_score = run_sessions(ConnectedJudge(path), test_sessions, gold_input=False)["overall"]["mean_score"]
    if abs(agent_score - batched) > 1e-3:
        raise RuntimeError(f"agent re-score {agent_score} != batched {batched}")
    result = {"world": "v2-H", "experiment": "connected-head", "seed": args.seed, "reader": args.reader,
              "message_size": args.message_size, "train_sessions": args.train_sessions,
              "parameters": {"head": sum(p.numel() for p in model.head.parameters()),
                             "judge": sum(p.numel() for p in model.judge.parameters())},
              "selected_epoch": best[1], "history": history, "test_score": agent_score,
              "test_score_batched": batched, "ceilings": hedged.ceilings(test_sessions),
              "training_seconds": round(time.perf_counter() - started, 1),
              "checkpoint": str(path.relative_to(PROJECT_ROOT))}
    out = RESULTS_DIR / f"world-v2h_connected-m{args.message_size}_seed-{args.seed}.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"selected epoch {best[1]}; test {agent_score} (batched {batched}); ceilings {result['ceilings']}")
    print(f"Saved: {path}\nSaved: {out}")


if __name__ == "__main__":
    main()
