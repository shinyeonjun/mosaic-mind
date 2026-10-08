"""N2: a meta-learned neural judge for rule world v2.

The network learns *how to learn trust* from many training sessions. Its structure
encodes the world's symmetries, not its numbers:

- Speaker equivariance: one GRU cell with shared weights keeps a memory vector h_s per
  speaker. After feedback it reads x_s = [spoke, positive, negative, right, wrong]
  (all zero when s was silent, so the cell can also learn forgetting).
- Key equivariance: an evidence head maps h_s to four numbers
  (pos_match, pos_other, neg_match, neg_other). A positive claim on key c adds
  pos_match to logit c and pos_other to the other two keys; a negative claim likewise.
  The Bayes learner B3 is inside this family: with h_s encoding the level posterior,
  pos_match = log sum_l P(l) l and pos_other = log sum_l P(l) (1 - l) / 2.
- Output: softmax over the 3 keys, trained with cross-entropy (a proper scoring rule,
  so probabilities are pushed toward calibration). The answer/abstain rule is the
  expected-score-optimal one shared with O2/B3: answer when max probability > 0.5.

`--no-memory` freezes h_s at its learned initial value: the same network without the
ability to learn within a session (an ablation that isolates the value of memory).

`--feedback partial` trains for the right/wrong-only world. The memory input becomes
x_s = [spoke, positive, negative, known_right, known_wrong, unknown, soft_right], where
soft_right = sum_k q'_k * 1[claim_s right | T = k] and q' is the model's own prediction
conditioned on the feedback (one-hot if "right", answered key zeroed if "wrong",
unchanged if it abstained). This is the expected sufficient statistic used by EM; it
lets each speaker's memory learn from agreement with the others even without feedback.
During training the answers that determine feedback come from the model's own
(detached) decisions; the loss still uses the true key, which is available offline.

`--context` adds one generic per-claim input: how many *other* speakers made exactly the
same claim in this episode (divided by 3). It enters both the evidence head and the
memory update. It is not specific to any error type, but it is the signal needed to
notice correlated speakers. `--world misspecified` trains on rule world v2-M.

python -m cognitive_lab.world2.neural_judge --seed 42
"""

import argparse
import json
import random
import time
from pathlib import Path

import torch
from torch import nn

from cognitive_lab.world.templates import UNKNOWN
from cognitive_lab.world2.agents import BaseJudge, final_claims
from cognitive_lab.world2.generator import SPEAKERS_PER_SESSION, generate_part

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CHECKPOINT_DIR = PROJECT_ROOT / "artifacts" / "checkpoints"
RESULTS_DIR = PROJECT_ROOT / "artifacts" / "results"
NONE, POSITIVE, NEGATIVE = 0, 1, 2
FEEDBACK_SIZE = {"full": 5, "partial": 7}


def checkpoint_path(seed: int, memory: bool = True, feedback: str = "full",
                    speakers: int = SPEAKERS_PER_SESSION, sparse: bool = False,
                    world: str = "standard", context: bool = False) -> Path:
    tag = (("" if memory else "-no-memory") + ("" if feedback == "full" else "-partial")
           + ("" if speakers == SPEAKERS_PER_SESSION else f"-s{speakers}") + ("-sparse" if sparse else "")
           + ("" if world == "standard" else "-mis") + ("-ctx" if context else ""))
    return CHECKPOINT_DIR / f"world-v2_n2{tag}_seed-{seed}.pt"


def claim_right(kinds: torch.Tensor, keys: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
    """Whether each speaker's claim is right if the truth were `truth` ([B] or [B,1])."""
    truth = truth.reshape(-1, 1)
    return ((kinds == POSITIVE) & (keys == truth)) | ((kinds == NEGATIVE) & (keys != truth))


def full_features(kinds: torch.Tensor, keys: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
    spoke = kinds != NONE
    right = claim_right(kinds, keys, truth)
    return torch.stack([spoke.float(), (kinds == POSITIVE).float(), (kinds == NEGATIVE).float(),
                        (spoke & right).float(), (spoke & ~right).float()], dim=-1)


def partial_features(kinds: torch.Tensor, keys: torch.Tensor, q: torch.Tensor,
                     answered: torch.Tensor, correct: torch.Tensor) -> torch.Tensor:
    """q [B,3] model prediction (detached); answered [B] key or -1; correct [B] bool."""
    did_answer = answered >= 0
    onehot = nn.functional.one_hot(answered.clamp_min(0), 3).float()
    conditioned = torch.where((did_answer & correct).unsqueeze(-1), onehot, q)
    conditioned = torch.where((did_answer & ~correct).unsqueeze(-1), conditioned * (1 - onehot), conditioned)
    conditioned = conditioned / conditioned.sum(-1, keepdim=True).clamp_min(1e-9)
    spoke = kinds != NONE
    soft = sum(conditioned[:, k : k + 1] * claim_right(kinds, keys, torch.full_like(answered, k)).float()
               for k in range(3))
    known_right = spoke & (soft > 0.999)
    known_wrong = spoke & (soft < 0.001)
    unknown = spoke & ~known_right & ~known_wrong
    return torch.stack([spoke.float(), (kinds == POSITIVE).float(), (kinds == NEGATIVE).float(),
                        known_right.float(), known_wrong.float(), unknown.float(), soft * spoke.float()], dim=-1)


def duplicates(kinds: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
    """[B,S] -> [B,S,1]: number of other speakers with the identical claim, divided by 3."""
    same = (kinds.unsqueeze(-1) == kinds.unsqueeze(-2)) & (keys.unsqueeze(-1) == keys.unsqueeze(-2))
    same &= (kinds != NONE).unsqueeze(-1) & (kinds != NONE).unsqueeze(-2)
    count = same.sum(-1) - (kinds != NONE).long()
    return (count.float() / 3).unsqueeze(-1)


def decide_tensor(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Expected-score-optimal answers (key index, or -1 to abstain) and the probabilities."""
    q = logits.softmax(-1)
    best, choice = q.max(-1)
    return torch.where(best > 0.5, choice, torch.full_like(choice, -1)), q


class N2Model(nn.Module):
    def __init__(self, hidden: int = 32, memory: bool = True, feedback: str = "full", context: bool = False):
        super().__init__()
        self.memory = memory
        self.feedback = feedback
        self.context = context
        extra = 1 if context else 0
        self.initial = nn.Parameter(torch.zeros(hidden))
        self.cell = nn.GRUCell(FEEDBACK_SIZE[feedback] + extra, hidden)
        self.evidence = nn.Sequential(nn.Linear(hidden + extra, hidden), nn.Tanh(), nn.Linear(hidden, 4))

    def initial_state(self, batch: int, speakers: int) -> torch.Tensor:
        return self.initial.expand(batch, speakers, -1).clone()

    def logits(self, state: torch.Tensor, kinds: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
        """state [B,S,H], kinds/keys [B,S] -> key logits [B,3]."""
        head_input = torch.cat([state, duplicates(kinds, keys)], dim=-1) if self.context else state
        pos_match, pos_other, neg_match, neg_other = self.evidence(head_input).unbind(-1)
        onehot = nn.functional.one_hot(keys, 3).float()
        positive = (kinds == POSITIVE).float().unsqueeze(-1)
        negative = (kinds == NEGATIVE).float().unsqueeze(-1)
        contribution = (
            positive * (pos_match.unsqueeze(-1) * onehot + pos_other.unsqueeze(-1) * (1 - onehot))
            + negative * (neg_match.unsqueeze(-1) * onehot + neg_other.unsqueeze(-1) * (1 - onehot))
        )
        return contribution.sum(dim=1)

    def update(self, state: torch.Tensor, kinds: torch.Tensor, keys: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
        return self.update_features(state, self.with_context(full_features(kinds, keys, truth), kinds, keys))

    def with_context(self, features: torch.Tensor, kinds: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
        return torch.cat([features, duplicates(kinds, keys)], dim=-1) if self.context else features

    def update_features(self, state: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        if not self.memory:
            return state
        batch, speakers, hidden = state.shape
        updated = self.cell(features.reshape(batch * speakers, -1), state.reshape(batch * speakers, hidden))
        return updated.reshape(batch, speakers, hidden)

    def forward(self, kinds: torch.Tensor, keys: torch.Tensor, truths: torch.Tensor) -> torch.Tensor:
        """kinds/keys [B,E,S], truths [B,E] -> logits [B,E,3]; feedback follows each answer."""
        batch, episodes, speakers = kinds.shape
        state = self.initial_state(batch, speakers)
        outputs = []
        for t in range(episodes):
            logits = self.logits(state, kinds[:, t], keys[:, t])
            outputs.append(logits)
            if self.feedback == "full":
                state = self.update(state, kinds[:, t], keys[:, t], truths[:, t])
            else:
                answered, q = decide_tensor(logits.detach())
                features = partial_features(kinds[:, t], keys[:, t], q, answered, answered == truths[:, t])
                state = self.update_features(state, self.with_context(features, kinds[:, t], keys[:, t]))
        return torch.stack(outputs, dim=1)


def session_tensors(sessions: list[dict]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Gold final claims per (session, episode, speaker) and the true keys."""
    episodes = len(sessions[0]["episodes"])
    speakers = len(sessions[0]["speakers"])
    if any(len(s["speakers"]) != speakers for s in sessions):
        raise ValueError("sessions in one tensor batch must have the same number of speakers")
    kinds = torch.zeros(len(sessions), episodes, speakers, dtype=torch.long)
    keys = torch.zeros_like(kinds)
    truths = torch.zeros(len(sessions), episodes, dtype=torch.long)
    for b, session in enumerate(sessions):
        index = {name: s for s, name in enumerate(session["speakers"])}
        for t, episode in enumerate(session["episodes"]):
            key_labels = [k.removesuffix(" 열쇠") for k in episode["entities"]["keys"]]
            for speaker, key, polarity in final_claims(episode["gold_events"], episode["gold_question"]["door"], key_labels):
                kinds[b, t, index[speaker]] = POSITIVE if polarity else NEGATIVE
                keys[b, t, index[speaker]] = key
            truths[b, t] = episode["split_keys"]["answer_index"]
    return kinds, keys, truths


def calibrated_score(logits: torch.Tensor, truths: torch.Tensor) -> float:
    probabilities = logits.softmax(dim=-1)
    best, choice = probabilities.max(dim=-1)
    answered = best > 0.5
    scores = torch.where(answered, torch.where(choice == truths, 1.0, -1.0), torch.zeros_like(best))
    return round(scores.mean().item(), 4)


class N2Judge(BaseJudge):
    def __init__(self, checkpoint: Path, reader=None):
        super().__init__(reader)
        saved = torch.load(checkpoint, map_location="cpu")
        self.model = N2Model(saved["hidden"], saved["memory"], saved.get("feedback", "full"), saved.get("context", False))
        self.model.load_state_dict(saved["state"])
        self.model.eval()
        self.checkpoint = str(checkpoint)

    def start_session(self, meta: dict) -> None:
        super().start_session(meta)
        self.index = {name: s for s, name in enumerate(self.speakers)}
        self.state = self.model.initial_state(1, len(self.speakers)).detach()

    def _tensors(self, claims) -> tuple[torch.Tensor, torch.Tensor]:
        kinds = torch.zeros(1, len(self.speakers), dtype=torch.long)
        keys = torch.zeros_like(kinds)
        for speaker, key, polarity in claims:
            if speaker in self.index:
                kinds[0, self.index[speaker]] = POSITIVE if polarity else NEGATIVE
                keys[0, self.index[speaker]] = key
        return kinds, keys

    def act(self, episode: dict) -> str:
        claims, names = self.claims(episode)
        self._pending = self._tensors(claims)
        with torch.no_grad():
            self._answered, self._q = decide_tensor(self.model.logits(self.state, *self._pending))
        choice = int(self._answered[0])
        return names[choice] if choice >= 0 else UNKNOWN

    def feedback(self, episode: dict, answer: str) -> None:
        truth = torch.tensor([episode["entities"]["keys"].index(answer)])
        with torch.no_grad():
            self.state = self.model.update(self.state, *self._pending, truth)

    def feedback_partial(self, episode: dict, answered: str | None, correct: bool | None) -> None:
        if self.model.feedback != "partial":
            raise RuntimeError("this N2 checkpoint was trained for full feedback")
        key = -1 if answered is None else episode["entities"]["keys"].index(answered)
        features = partial_features(*self._pending, self._q, torch.tensor([key]), torch.tensor([bool(correct)]))
        features = self.model.with_context(features, *self._pending)
        with torch.no_grad():
            self.state = self.model.update_features(self.state, features)

    def cost_report(self) -> dict:
        return {**super().cost_report(), "checkpoint": self.checkpoint,
                "parameters": sum(p.numel() for p in self.model.parameters())}


def main() -> None:
    parser = argparse.ArgumentParser(description="Meta-train the N2 neural judge")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-sessions", type=int, default=3000,
                        help="training sessions generated from the train name pool (the first 300 equal the saved train part)")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--no-memory", dest="memory", action="store_false")
    parser.add_argument("--feedback", choices=("full", "partial"), default="full")
    parser.add_argument("--speakers", type=int, default=SPEAKERS_PER_SESSION,
                        help="speakers per training session (the network itself works for any count)")
    parser.add_argument("--sparse", action="store_true", help="1.75 claims per episode regardless of speakers")
    parser.add_argument("--world", choices=("standard", "misspecified"), default="standard")
    parser.add_argument("--context", action="store_true", help="add the duplicate-claim count input")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    started = time.perf_counter()
    if args.world == "misspecified":
        from cognitive_lab.world2 import misspecified

        train = session_tensors(misspecified.generate_part("train", args.seed, sessions=args.train_sessions))
        validation = session_tensors(misspecified.generate_part("validation", args.seed))
    else:
        train = session_tensors(generate_part("train", args.seed, sessions=args.train_sessions,
                                              speakers=args.speakers, sparse=args.sparse))
        validation = session_tensors(generate_part("validation", args.seed, speakers=args.speakers,
                                                   sparse=args.sparse))
    print(f"train sessions {train[0].shape[0]}, validation sessions {validation[0].shape[0]} "
          f"(generated in {time.perf_counter() - started:.1f}s)", flush=True)

    model = N2Model(args.hidden, args.memory, args.feedback, args.context)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    history, best = [], (-2.0, 0, None)
    order = torch.randperm(train[0].shape[0], generator=torch.Generator().manual_seed(args.seed))
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = order[torch.randperm(len(order), generator=torch.Generator().manual_seed(args.seed + epoch))]
        losses = []
        for start in range(0, len(order), args.batch_size):
            batch = order[start : start + args.batch_size]
            kinds, keys, truths = (tensor[batch] for tensor in train)
            logits = model(kinds, keys, truths)
            loss = nn.functional.cross_entropy(logits.reshape(-1, 3), truths.reshape(-1))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())
        model.eval()
        with torch.no_grad():
            val_logits = model(*validation)
            val_loss = nn.functional.cross_entropy(val_logits.reshape(-1, 3), validation[2].reshape(-1)).item()
            val_score = calibrated_score(val_logits, validation[2])
        record = {"epoch": epoch, "train_loss": round(sum(losses) / len(losses), 4),
                  "validation_loss": round(val_loss, 4), "validation_score": val_score}
        history.append(record)
        if epoch % 5 == 0 or epoch == 1:
            print(record, flush=True)
        if val_score > best[0]:
            best = (val_score, epoch, {k: v.detach().clone() for k, v in model.state_dict().items()})

    path = checkpoint_path(args.seed, args.memory, args.feedback, args.speakers, args.sparse, args.world, args.context)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state": best[2], "hidden": args.hidden, "memory": args.memory, "feedback": args.feedback,
                "context": args.context, "world": args.world,
                "epoch": best[1]}, path)
    result = {
        "world": "v2", "experiment": "n2-meta-training", "seed": args.seed, "memory": args.memory,
        "feedback": args.feedback, "speakers": args.speakers, "sparse": args.sparse,
        "world_variant": args.world, "context": args.context,
        "train_sessions": args.train_sessions, "hidden": args.hidden,
        "parameters": sum(p.numel() for p in model.parameters()),
        "optimizer": {"name": "Adam", "learning_rate": args.learning_rate, "batch_size": args.batch_size},
        "selected_epoch": best[1], "selection_metric": "validation calibrated score",
        "history": history, "training_seconds": round(time.perf_counter() - started, 1),
        "checkpoint": str(path.relative_to(PROJECT_ROOT)),
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    tag = path.stem.removeprefix("world-v2_").removesuffix(f"_seed-{args.seed}")
    out = RESULTS_DIR / f"world-v2_{tag}-train_seed-{args.seed}.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"selected epoch {best[1]} (validation score {best[0]})\nSaved: {path}\nSaved: {out}")


if __name__ == "__main__":
    main()
