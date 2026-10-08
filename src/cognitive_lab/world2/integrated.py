"""Integrated system I1 (design/integrated-system-v1.md): read + remember + judge, no designed facts.

Reader: the language module grown in rule world v1 (anchored pair messages), moved to v2
without retraining. Because it is frozen and v2 sentences come from the v1 templates, every
possible (sentence, queried door, key) message is computed once in fp32 and looked up.

Judge + memory (trained on v2 sessions from final answers only):
  h[s,k]  one shared GRU reads speaker s's messages about key k in utterance order
  mem[s]  one shared GRU cell per speaker, updated after feedback from
          [spoke, h[s, true key], mean of h[s, other keys]]
  c[s,k]  = f(h[s,k], mean_k h[s,.], mem[s], spoke);  logit_k = sum_s c[s,k]
Speaker- and key-permutation equivariant; no trust order is given. Decision: answer the
top key when its probability exceeds 0.5, else 모름 (expected-score optimal, as in v2).

python -m cognitive_lab.world2.integrated --seed 42 [--world misspecified]
"""

import argparse
import json
import random
import statistics
import time
from pathlib import Path

import torch
from torch import nn

from cognitive_lab.world.templates import ENTITY_SETS, TEMPLATES, UNKNOWN, door_name, key_name
from cognitive_lab.world2.agents import BaseJudge
from cognitive_lab.world2.generator import SESSION_LENGTH, generate_part

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CHECKPOINT_DIR = PROJECT_ROOT / "artifacts" / "checkpoints"
RESULTS_DIR = PROJECT_ROOT / "artifacts" / "results"
CACHE_DIR = PROJECT_ROOT / "artifacts" / "cache"
DEFAULT_READER = "world-v1_reader-anchored-learned-sourcefold-staged-tuned_composition_seed-42.pt"
MAX_UTTERANCES = 10


def all_sentences() -> list[str]:
    """Every utterance text the v1/v2 templates can produce with the base entities."""
    doors, keys = ENTITY_SETS["base"]["doors"], ENTITY_SETS["base"]["keys"]
    texts = set()
    for kind in ("assert", "negate", "correct"):
        for _, template in TEMPLATES[kind]:
            for d in doors:
                for k in keys:
                    texts.add(template.format(D=door_name(d), K=key_name(k)))
    texts.update(template for _, template in TEMPLATES["chatter"])
    return sorted(texts)


def message_table(reader_file: str, device: torch.device) -> dict[tuple[str, str, str], torch.Tensor]:
    """(sentence, queried door name, key name) -> 4-d message from the frozen v1 reader (fp32)."""
    cache_path = CACHE_DIR / f"integrated-messages_{Path(reader_file).stem}.pt"
    if cache_path.exists():
        return torch.load(cache_path)
    from cognitive_lab.world.interface import ENCODER_DIRS
    from cognitive_lab.world.interface_anchored import PairEncoder, hypothesis, load_reader

    saved = torch.load(CHECKPOINT_DIR / reader_file, map_location="cpu")
    encoder = PairEncoder(encoder_dir=ENCODER_DIRS[saved["encoder_name"]]).to(device)
    load_reader(encoder, saved["reader"])
    encoder.eval()
    doors = [door_name(d) for d in ENTITY_SETS["base"]["doors"]]
    keys = [key_name(k) for k in ENTITY_SETS["base"]["keys"]]
    entries = [(text, door, key) for text in all_sentences() for door in doors for key in keys]
    table = {}
    with torch.no_grad():
        for start in range(0, len(entries), 256):
            chunk = entries[start:start + 256]
            values = encoder([t for t, _, _ in chunk], [hypothesis(d, k) for _, d, k in chunk], precise=True)
            table.update(zip(chunk, values.cpu()))
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(table, cache_path)
    return table


def queried_door(episode: dict) -> str:
    hits = [d for d in episode["entities"]["doors"] if d in episode["question"]]
    if len(hits) != 1:
        raise ValueError(f"cannot find the queried door in {episode['question']!r}")
    return hits[0]


def episode_arrays(episode: dict, speakers: list[str], table: dict, size: int = 4):
    """messages [U,3,M], speaker index [U] (-1 = padding), using text only (no gold structure)."""
    door = queried_door(episode)
    keys = episode["entities"]["keys"]
    messages = torch.zeros(MAX_UTTERANCES, 3, size)
    speaker = torch.full((MAX_UTTERANCES,), -1, dtype=torch.long)
    index = {name: s for s, name in enumerate(speakers)}
    for u, utterance in enumerate(episode["utterances"][:MAX_UTTERANCES]):
        messages[u] = torch.stack([table[(utterance["text"], door, key)] for key in keys])
        speaker[u] = index[utterance["source"]]
    return messages, speaker


def session_tensors(sessions: list[dict], table: dict):
    """messages [B,E,U,3,M], speakers [B,E,U], truths [B,E]."""
    batch, episodes = len(sessions), len(sessions[0]["episodes"])
    messages = torch.zeros(batch, episodes, MAX_UTTERANCES, 3, 4)
    speakers = torch.full((batch, episodes, MAX_UTTERANCES), -1, dtype=torch.long)
    truths = torch.zeros(batch, episodes, dtype=torch.long)
    for b, session in enumerate(sessions):
        for t, episode in enumerate(session["episodes"]):
            messages[b, t], speakers[b, t] = episode_arrays(episode, session["speakers"], table)
            truths[b, t] = episode["entities"]["keys"].index(episode["answer"])
    return messages, speakers, truths


class I1Model(nn.Module):
    def __init__(self, message_size: int = 4, width: int = 32, memory: int = 32, speakers: int = 4):
        super().__init__()
        self.speakers = speakers
        self.embed = nn.Linear(message_size, width)
        self.reader = nn.GRUCell(width, width)
        self.reader_initial = nn.Parameter(torch.zeros(width))
        self.memory_initial = nn.Parameter(torch.zeros(memory))
        self.memory_cell = nn.GRUCell(1 + 2 * width, memory)
        self.contribution = nn.Sequential(nn.Linear(2 * width + memory + 1, width), nn.Tanh(), nn.Linear(width, 1))

    def initial_memory(self, batch: int) -> torch.Tensor:
        return self.memory_initial.expand(batch, self.speakers, -1).clone()

    def read(self, messages: torch.Tensor, speakers: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """messages [B,U,3,M], speakers [B,U] -> h [B,S,3,W], spoke [B,S]."""
        batch, steps, keys, _ = messages.shape
        x = self.embed(messages)
        states, spoke = [], []
        for s in range(self.speakers):
            state = self.reader_initial.expand(batch * keys, -1)
            active = speakers == s
            for u in range(steps):
                updated = self.reader(x[:, u].reshape(batch * keys, -1), state)
                keep = active[:, u].repeat_interleave(keys).unsqueeze(-1)
                state = torch.where(keep, updated, state)
            states.append(state.view(batch, keys, -1))
            spoke.append(active.any(1).float())
        return torch.stack(states, 1), torch.stack(spoke, 1)

    def logits(self, h: torch.Tensor, spoke: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        keys = h.shape[2]
        z = h.mean(2, keepdim=True).expand_as(h)
        mem = memory.unsqueeze(2).expand(-1, -1, keys, -1)
        flag = spoke[:, :, None, None].expand(-1, -1, keys, 1)
        contribution = self.contribution(torch.cat([h, z, mem, flag], -1)).squeeze(-1)  # [B,S,3]
        return (contribution * spoke.unsqueeze(-1)).sum(1)

    def update(self, memory: torch.Tensor, h: torch.Tensor, spoke: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
        batch, speakers, keys, width = h.shape
        onehot = nn.functional.one_hot(truth, keys).float()[:, None, :, None]  # [B,1,3,1]
        true_h = (h * onehot).sum(2)
        other_h = (h * (1 - onehot)).sum(2) / (keys - 1)
        features = torch.cat([spoke.unsqueeze(-1), true_h, other_h], -1)
        updated = self.memory_cell(features.reshape(batch * speakers, -1), memory.reshape(batch * speakers, -1))
        return updated.view(batch, speakers, -1)

    def forward(self, messages, speakers, truths) -> torch.Tensor:
        batch, episodes = truths.shape
        memory = self.initial_memory(batch)
        outputs = []
        for t in range(episodes):
            h, spoke = self.read(messages[:, t], speakers[:, t])
            outputs.append(self.logits(h, spoke, memory))
            memory = self.update(memory, h, spoke, truths[:, t])
        return torch.stack(outputs, 1)


def calibrated_score(logits: torch.Tensor, truths: torch.Tensor) -> float:
    q = logits.softmax(-1)
    best, choice = q.max(-1)
    scores = torch.where(best > 0.5, torch.where(choice == truths, 1.0, -1.0), torch.zeros_like(best))
    return round(scores.mean().item(), 4)


def checkpoint_path(seed: int, world: str, reader_file: str) -> Path:
    tag = ("" if world == "standard" else "-mis") + f"_reader-{Path(reader_file).stem.split('_seed-')[-1]}"
    return CHECKPOINT_DIR / f"world-v2_i1{tag}_seed-{seed}.pt"


class I1Judge(BaseJudge):
    """Scoreboard agent: reads only utterance text, speaker names, and the question."""

    def __init__(self, checkpoint: Path, device: str = "cpu"):
        super().__init__(None)
        saved = torch.load(checkpoint, map_location="cpu")
        self.model = I1Model()
        self.model.load_state_dict(saved["state"])
        self.model.eval()
        self.table = message_table(saved["reader"], torch.device(device))
        self.checkpoint = str(checkpoint)

    def start_session(self, meta: dict) -> None:
        super().start_session(meta)
        self.memory = self.model.initial_memory(1).detach()

    def act(self, episode: dict) -> str:
        messages, speaker = episode_arrays(episode, self.speakers, self.table)
        with torch.no_grad():
            self._h, self._spoke = self.model.read(messages[None], speaker[None])
            q = self.model.logits(self._h, self._spoke, self.memory).softmax(-1)[0]
        best = int(q.argmax())
        return episode["entities"]["keys"][best] if q[best] > 0.5 else UNKNOWN

    def feedback(self, episode: dict, answer: str) -> None:
        truth = torch.tensor([episode["entities"]["keys"].index(answer)])
        with torch.no_grad():
            self.memory = self.model.update(self.memory, self._h, self._spoke, truth)

    def cost_report(self) -> dict:
        return {"reader": "v1 anchored reader (frozen, cached messages)", "checkpoint": self.checkpoint,
                "parameters": sum(p.numel() for p in self.model.parameters())}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the integrated read+remember+judge system I1 on v2")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--world", choices=("standard", "misspecified"), default="standard")
    parser.add_argument("--reader", default=DEFAULT_READER, help="v1 reader checkpoint file in artifacts/checkpoints")
    parser.add_argument("--train-sessions", type=int, default=3000)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    device = torch.device("cuda" if args.device != "cpu" and torch.cuda.is_available() else "cpu")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    started = time.perf_counter()

    table = message_table(args.reader, device)
    if args.world == "misspecified":
        from cognitive_lab.world2 import misspecified

        train_sessions = misspecified.generate_part("train", args.seed, sessions=args.train_sessions)
        validation_sessions = misspecified.generate_part("validation", args.seed)
    else:
        train_sessions = generate_part("train", args.seed, sessions=args.train_sessions)
        validation_sessions = generate_part("validation", args.seed)
    train = tuple(t.to(device) for t in session_tensors(train_sessions, table))
    validation = tuple(t.to(device) for t in session_tensors(validation_sessions, table))
    print(f"messages {len(table)}, train {train[0].shape[0]} sessions, validation {validation[0].shape[0]} "
          f"({time.perf_counter() - started:.0f}s)", flush=True)

    model = I1Model().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    history, best = [], (-2.0, 0, None)
    generator = torch.Generator().manual_seed(args.seed)
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = torch.randperm(train[0].shape[0], generator=generator)
        losses = []
        for start in range(0, len(order), args.batch_size):
            batch = order[start:start + args.batch_size].to(device)
            messages, speakers, truths = (tensor[batch] for tensor in train)
            loss = nn.functional.cross_entropy(model(messages, speakers, truths).reshape(-1, 3), truths.reshape(-1))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())
        model.eval()
        with torch.no_grad():
            score = calibrated_score(model(*validation), validation[2])
        history.append({"epoch": epoch, "train_loss": round(statistics.mean(losses), 4), "validation_score": score})
        if epoch == 1 or epoch % 5 == 0:
            print(history[-1], flush=True)
        if score > best[0]:
            best = (score, epoch, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()})

    path = checkpoint_path(args.seed, args.world, args.reader)
    torch.save({"state": best[2], "reader": args.reader, "world": args.world, "epoch": best[1]}, path)
    result = {"world": "v2" + ("-M" if args.world == "misspecified" else ""), "experiment": "integrated-i1",
              "seed": args.seed, "reader": args.reader, "train_sessions": args.train_sessions,
              "parameters": sum(p.numel() for p in model.parameters()), "selected_epoch": best[1],
              "history": history, "training_seconds": round(time.perf_counter() - started, 1),
              "checkpoint": str(path.relative_to(PROJECT_ROOT))}
    out = RESULTS_DIR / f"world-v2_i1-train{'-mis' if args.world == 'misspecified' else ''}_seed-{args.seed}.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"selected epoch {best[1]} (validation {best[0]})\nSaved: {path}\nSaved: {out}")


if __name__ == "__main__":
    main()
