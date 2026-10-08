"""Anchored messages: fix *what* a message is about, learn *what it says* (research log, section 4).

For each utterance and each key k, the language module cross-encodes the pair
(utterance, "<queried door>은 <k>로 열린다.") and emits a small vector m_tk. The queried
door is found by matching the given door names in the question (no gold structure).

Judge (key-equivariant): for every key k the same GRU reads x_tk = [m_tk, source one-hot]
in utterance order; then a DeepSets step lets keys exchange information
(z = mean_k h_k, score_k = f([h_k, z]), score_unknown = g(z)), which is what elimination
needs. Output: 3 key scores + 모름.

Systems:
- anchored-designed: m_tk is the gold pair label one-hot (entailed / contradicted /
  unrelated, as for the v1 NLI reader). Tests whether this judge structure alone gives
  compositional generalization.
- anchored-learned: m_tk is learned from final answers only (mDeBERTa, 4-d tanh message).
  Tests whether the *content* of an anchored message emerges without supervision.

`--judge source-fold` adds source symmetry. Instead of a learned source embedding, the
same GRU reads each source's utterances separately (so "a later statement by the same
speaker replaces the earlier one" is one shared skill for every source). The per-source
states are then folded in trust order (passerby -> sign -> manager) by one shared GRU
cell; trust enters only through that order, never through source-specific weights.

Staged growth (development track): learning the messages jointly with the source-fold
judge from scratch stalled at the label prior (a no-communication equilibrium), although
each part learned on its own. `--init-encoder` loads a reader trained in an earlier run
(every anchored-learned run saves its best reader to artifacts/checkpoints), and
`--freeze-encoder` keeps it fixed while a new judge learns on top: read first, judge later.
`--init-judge` loads a judge saved by an earlier run (every run saves its best judge), so a
third stage can fine-tune reader and judge together from the staged starting point.
Whenever a run starts from checkpoints, the starting point itself is scored on validation
first (epoch 0) and stays a candidate, so further training can never be selected unless it
is better on validation.

python -m cognitive_lab.world.interface_anchored --system anchored-learned --split composition --seed 42
"""

import argparse
import json
import random
import statistics
import time

from pathlib import Path

import torch
from torch import nn

from cognitive_lab.world.interface import (
    ENCODER_DIRS, NLI_DIR, PROJECT_ROOT, RESULTS_DIR, calibrated, label_of, lesson, load_part, tokenizer_dir,
)

CHECKPOINT_DIR = PROJECT_ROOT / "artifacts" / "checkpoints"
from cognitive_lab.world.templates import SOURCES

SYSTEMS = ("anchored-designed", "anchored-learned")
ENTAIL, CONTRADICT, UNRELATED = 0, 1, 2


def queried_door(episode: dict) -> str:
    hits = [d for d in episode["entities"]["doors"] if d in episode["question"]]
    return hits[0] if len(hits) == 1 else episode["entities"]["doors"][0]


def hypothesis(door: str, key: str) -> str:
    return f"{door}은 {key}로 열린다."


def pair_label(episode: dict, event: dict, key: str) -> int:
    """Gold relation of one utterance to the pair (queried door, key)."""
    door = queried_door(episode).removesuffix(" 문")
    if event["type"] == "chatter" or event["door"] != door:
        return UNRELATED
    same = event["key"] == key.removesuffix(" 열쇠")
    if event["polarity"]:
        return ENTAIL if same else CONTRADICT
    return CONTRADICT if same else UNRELATED


class PairEncoder(nn.Module):
    def __init__(self, size: int = 4, encoder_dir=NLI_DIR):
        super().__init__()
        from transformers import AutoModel, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir(encoder_dir))
        self.backbone = AutoModel.from_pretrained(encoder_dir, dtype=torch.float32)
        self.backbone.embeddings.word_embeddings.weight.requires_grad_(False)
        self.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self.head = nn.Linear(self.backbone.config.hidden_size, size)
        self.message_size = size

    def forward(self, premises: list[str], hypotheses: list[str], precise: bool = False) -> torch.Tensor:
        """bf16 autocast for training speed; `precise=True` runs in fp32. Under bf16 the same pair
        can differ by up to ~0.2 in a message component depending on batch composition, so cached
        messages of a frozen reader are always computed precisely."""
        device = self.head.weight.device
        batch = self.tokenizer(premises, hypotheses, return_tensors="pt", padding=True, truncation=True,
                               max_length=64).to(device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=device.type == "cuda" and not precise):
            states = self.backbone(**batch).last_hidden_state
        mask = batch["attention_mask"].unsqueeze(-1).float()
        pooled = (states.float() * mask).sum(1) / mask.sum(1)
        return torch.tanh(self.head(pooled))


class EquivariantJudge(nn.Module):
    def __init__(self, message_size: int, width: int = 32):
        super().__init__()
        self.embed = nn.Linear(message_size + len(SOURCES), width)
        self.cell = nn.GRUCell(width, width)
        self.initial = nn.Parameter(torch.zeros(width))
        self.key_score = nn.Sequential(nn.Linear(2 * width, width), nn.Tanh(), nn.Linear(width, 1))
        self.unknown_score = nn.Sequential(nn.Linear(width, width), nn.Tanh(), nn.Linear(width, 1))

    def forward(self, messages: torch.Tensor, sources: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """messages [B,T,3,M], sources [B,T,3(sources)], mask [B,T] -> logits [B,4]."""
        batch, steps, keys, _ = messages.shape
        state = self.initial.expand(batch * keys, -1)
        source = sources.unsqueeze(2).expand(-1, -1, keys, -1)
        x = self.embed(torch.cat([messages, source], -1))  # [B,T,3,W]
        for t in range(steps):
            updated = self.cell(x[:, t].reshape(batch * keys, -1), state)
            keep = mask[:, t].repeat_interleave(keys).unsqueeze(-1)
            state = torch.where(keep, updated, state)
        h = state.view(batch, keys, -1)
        z = h.mean(1, keepdim=True)
        scores = self.key_score(torch.cat([h, z.expand_as(h)], -1)).squeeze(-1)  # [B,3]
        return torch.cat([scores, self.unknown_score(z.squeeze(1))], -1)


def load_reader(encoder: "PairEncoder", reader: dict) -> None:
    """Load a saved reader; every tensor except the frozen word embeddings must match exactly."""
    expected = {k for k in encoder.state_dict() if "word_embeddings" not in k}
    missing, unexpected = expected - set(reader), set(reader) - expected
    if missing or unexpected:
        raise RuntimeError(f"reader checkpoint mismatch: missing {sorted(missing)[:5]}, "
                           f"unexpected {sorted(unexpected)[:5]}")
    encoder.load_state_dict(reader, strict=False)  # word embeddings come from the base checkpoint


class SourceFoldJudge(nn.Module):
    """Key- and source-equivariant judge: shared per-source reader, fold in trust order."""

    def __init__(self, message_size: int, width: int = 32):
        super().__init__()
        self.embed = nn.Linear(message_size, width)
        self.reader = nn.GRUCell(width, width)
        self.reader_initial = nn.Parameter(torch.zeros(width))
        self.spoke = nn.Linear(1, width)
        self.fold = nn.GRUCell(width, width)
        self.fold_initial = nn.Parameter(torch.zeros(width))
        self.key_score = nn.Sequential(nn.Linear(2 * width, width), nn.Tanh(), nn.Linear(width, 1))
        self.unknown_score = nn.Sequential(nn.Linear(width, width), nn.Tanh(), nn.Linear(width, 1))

    def forward(self, messages: torch.Tensor, sources: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch, steps, keys, _ = messages.shape
        x = self.embed(messages)  # [B,T,3,W]
        folded = self.fold_initial.expand(batch * keys, -1)
        for source in range(sources.shape[-1]):  # SOURCES order is the trust order, lowest first
            state = self.reader_initial.expand(batch * keys, -1)
            active = mask & (sources[..., source] > 0)  # [B,T]
            for t in range(steps):
                updated = self.reader(x[:, t].reshape(batch * keys, -1), state)
                keep = active[:, t].repeat_interleave(keys).unsqueeze(-1)
                state = torch.where(keep, updated, state)
            spoke = active.any(1).float().repeat_interleave(keys).unsqueeze(-1)
            folded = self.fold(state + self.spoke(spoke), folded)
        h = folded.view(batch, keys, -1)
        z = h.mean(1, keepdim=True)
        scores = self.key_score(torch.cat([h, z.expand_as(h)], -1)).squeeze(-1)
        return torch.cat([scores, self.unknown_score(z.squeeze(1))], -1)


def batch_inputs(episodes: list[dict], encoder: PairEncoder | None, device: torch.device):
    steps = max(len(e["utterances"]) for e in episodes)
    mask = torch.zeros(len(episodes), steps, dtype=torch.bool, device=device)
    sources = torch.zeros(len(episodes), steps, len(SOURCES), device=device)
    for b, e in enumerate(episodes):
        mask[b, : len(e["utterances"])] = True
        for t, u in enumerate(e["utterances"]):
            sources[b, t, SOURCES.index(u["source"])] = 1.0
    if encoder is None:
        messages = torch.zeros(len(episodes), steps, 3, 3, device=device)
        for b, e in enumerate(episodes):
            for t, event in enumerate(e["gold_events"]):
                for k, key in enumerate(e["entities"]["keys"]):
                    messages[b, t, k, pair_label(e, event, key)] = 1.0
    else:
        pairs, where = [], []
        for b, e in enumerate(episodes):
            door = queried_door(e)
            for t, u in enumerate(e["utterances"]):
                for k, key in enumerate(e["entities"]["keys"]):
                    pairs.append((u["text"], hypothesis(door, key)))
                    where.append((b, t, k))
        distinct = sorted(set(pairs))
        position = {p: i for i, p in enumerate(distinct)}
        trainable = any(p.requires_grad for p in encoder.parameters())
        if trainable:
            encoded = encoder([p for p, _ in distinct], [h for _, h in distinct])
        else:
            # A frozen reader always gives the same message for a pair, and v1 has few distinct
            # pairs, so messages are computed once and reused (stage 2 then trains only the judge).
            cache = encoder.__dict__.setdefault("message_cache", {})
            missing = [p for p in distinct if p not in cache]
            if missing:
                with torch.no_grad():
                    for start in range(0, len(missing), 256):
                        chunk = missing[start:start + 256]
                        values = encoder([p for p, _ in chunk], [h for _, h in chunk], precise=True)
                        cache.update(zip(chunk, values))
            encoded = torch.stack([cache[p] for p in distinct])
        index = torch.tensor([position[p] for p in pairs], device=device)
        messages = torch.zeros(len(episodes), steps, 3, encoder.message_size, device=device)
        b, t, k = (torch.tensor(column, device=device) for column in zip(*where))
        messages = messages.index_put((b, t, k), encoded[index])
    labels = torch.tensor([label_of(e) for e in episodes], device=device)
    return messages, sources, mask, labels


def evaluate(judge, encoder, episodes, device, batch_size: int = 32) -> dict:
    judge.eval()
    if encoder is not None:
        encoder.eval()
    predictions, labels, cases = [], [], []
    with torch.no_grad():
        for start in range(0, len(episodes), batch_size):
            chunk = episodes[start : start + batch_size]
            messages, sources, mask, y = batch_inputs(chunk, encoder, device)
            predictions += judge(messages, sources, mask).argmax(-1).tolist()
            labels += y.tolist()
            cases += [e["split_keys"]["case"] for e in chunk]
    by_case = {c: calibrated([p for p, cc in zip(predictions, cases) if cc == c],
                             [y for y, cc in zip(labels, cases) if cc == c]) for c in sorted(set(cases))}
    return {"mean_score": calibrated(predictions, labels),
            "accuracy": round(sum(p == y for p, y in zip(predictions, labels)) / len(labels), 4), "by_case": by_case}


def probe(encoder: PairEncoder, train: list[dict], test: list[dict]) -> dict:
    """Linear probe: does the learned pair message separate entailed / contradicted / unrelated?"""
    def table(episodes):
        seen = {}
        for e in episodes:
            door = queried_door(e)
            for u, ev in zip(e["utterances"], e["gold_events"]):
                for key in e["entities"]["keys"]:
                    seen.setdefault((u["text"], hypothesis(door, key)), pair_label(e, ev, key))
        return list(seen), torch.tensor(list(seen.values()))

    encoder.eval()
    with torch.no_grad():
        pairs_tr, y_tr = table(train)
        pairs_te, y_te = table(test)
        x_tr = torch.cat([encoder([p for p, _ in pairs_tr[i:i + 64]], [h for _, h in pairs_tr[i:i + 64]])
                          for i in range(0, len(pairs_tr), 64)]).cpu()
        x_te = torch.cat([encoder([p for p, _ in pairs_te[i:i + 64]], [h for _, h in pairs_te[i:i + 64]])
                          for i in range(0, len(pairs_te), 64)]).cpu()
    classifier = nn.Linear(x_tr.shape[1], 3)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=0.05)
    for _ in range(300):
        optimizer.zero_grad()
        nn.functional.cross_entropy(classifier(x_tr), y_tr).backward()
        optimizer.step()
    with torch.no_grad():
        accuracy = (classifier(x_te).argmax(-1) == y_te).float().mean().item()
    counts = torch.bincount(y_te, minlength=3).tolist()
    return {"pair_relation_probe_accuracy": round(accuracy, 4), "majority_baseline": round(max(counts) / sum(counts), 4),
            "test_pairs": len(pairs_te), "label_counts_entail_contradict_unrelated": counts}


def main() -> None:
    parser = argparse.ArgumentParser(description="Anchored module messages on rule world v1")
    parser.add_argument("--system", choices=SYSTEMS, required=True)
    parser.add_argument("--split", default="composition", choices=("iid", "phrasing", "composition", "entity"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--epochs", type=int, default=0, help="default: 30 designed, 8 learned")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--judge-lr", type=float, default=1e-3)
    parser.add_argument("--encoder-lr", type=float, default=2e-5)
    parser.add_argument("--curriculum", action="store_true", help="stage 1: 3 epochs, stages 2-3: 1 each, then full")
    parser.add_argument("--judge", choices=("flat", "source-fold"), default="flat")
    parser.add_argument("--init-encoder", type=Path, default=None,
                        help="reader checkpoint saved by an earlier anchored-learned run")
    parser.add_argument("--freeze-encoder", action="store_true", help="keep the reader fixed; train only the judge")
    parser.add_argument("--init-judge", type=Path, default=None, help="judge checkpoint saved by an earlier run")
    parser.add_argument("--encoder", choices=sorted(ENCODER_DIRS), default="nli",
                        help="nli: MNLI+XNLI fine-tuned (default); base: same model before NLI; e5: e5-small")
    args = parser.parse_args()

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else
                          ("cpu" if args.device == "auto" else args.device))
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    train, validation, test = (load_part(args.split, args.seed, part) for part in ("train", "validation", "test"))
    encoder = PairEncoder(encoder_dir=ENCODER_DIRS[args.encoder]).to(device) if args.system == "anchored-learned" else None
    if encoder is not None and args.init_encoder is not None:
        saved = torch.load(args.init_encoder, map_location="cpu")
        if saved["encoder_name"] != args.encoder:
            raise SystemExit(f"checkpoint reader is {saved['encoder_name']!r}, not {args.encoder!r}")
        load_reader(encoder, saved["reader"])
    if encoder is not None and args.freeze_encoder:
        for parameter in encoder.parameters():
            parameter.requires_grad_(False)
    judge_class = SourceFoldJudge if args.judge == "source-fold" else EquivariantJudge
    judge = judge_class(3 if encoder is None else encoder.message_size).to(device)
    if args.init_judge is not None:
        saved_judge = torch.load(args.init_judge, map_location="cpu")
        if saved_judge["judge_kind"] != args.judge:
            raise SystemExit(f"checkpoint judge is {saved_judge['judge_kind']!r}, not {args.judge!r}")
        judge.load_state_dict(saved_judge["judge"])  # strict: every tensor must match
    groups = [{"params": judge.parameters(), "lr": args.judge_lr}]
    if encoder is not None and not args.freeze_encoder:
        groups += [{"params": encoder.head.parameters(), "lr": args.judge_lr},
                   {"params": [p for p in encoder.backbone.parameters() if p.requires_grad], "lr": args.encoder_lr}]
    optimizer = torch.optim.AdamW(groups, weight_decay=0.01)
    epochs = args.epochs or (30 if encoder is None else 8)
    stage_epochs = {1: 3, 2: 1, 3: 1}

    def stage_of(epoch: int) -> int:
        if not args.curriculum:
            return 4
        boundary = 0
        for stage in (1, 2, 3):
            boundary += stage_epochs[stage]
            if epoch <= boundary:
                return stage
        return 4

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started, history, best = time.perf_counter(), [], (-2.0, 0, None)

    def snapshot() -> dict:
        state = {"judge": {k: v.detach().cpu().clone() for k, v in judge.state_dict().items()}}
        if encoder is not None:
            state["encoder"] = {k: v.detach().cpu().clone() for k, v in encoder.state_dict().items()}
        return state

    if args.init_encoder is not None or args.init_judge is not None:
        start_metrics = evaluate(judge, encoder, validation, device)
        history.append({"epoch": 0, "stage": "start", "train_loss": None, "validation": start_metrics["mean_score"]})
        print(history[-1], flush=True)
        best = (start_metrics["mean_score"], 0, snapshot())
    for epoch in range(1, epochs + 1):
        stage = stage_of(epoch)
        rng = random.Random(args.seed * 1000 + epoch)
        lessons = [v for e in train if (v := lesson(e, stage, rng)) is not None]
        rng.shuffle(lessons)
        judge.train()
        if encoder is not None:
            encoder.train(not args.freeze_encoder)
        losses, epoch_started = [], time.perf_counter()
        for step, start in enumerate(range(0, len(lessons), args.batch_size), 1):
            messages, sources, mask, labels = batch_inputs(lessons[start:start + args.batch_size], encoder, device)
            loss = nn.functional.cross_entropy(judge(messages, sources, mask), labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            if encoder is not None and step % 50 == 0:
                print(f"  epoch {epoch} step {step} loss {statistics.mean(losses[-50:]):.4f} "
                      f"({time.perf_counter() - epoch_started:.0f}s)", flush=True)
        metrics = evaluate(judge, encoder, validation, device)
        history.append({"epoch": epoch, "stage": stage, "train_loss": round(statistics.mean(losses), 4),
                        "validation": metrics["mean_score"]})
        print(history[-1], flush=True)
        if metrics["mean_score"] > best[0]:
            best = (metrics["mean_score"], epoch, snapshot())

    judge.load_state_dict(best[2]["judge"])
    if encoder is not None:
        encoder.load_state_dict(best[2]["encoder"])
    test_metrics = evaluate(judge, encoder, test, device)
    result = {"world": "v1", "experiment": "interface-anchored", "system": args.system, "split": args.split,
              "seed": args.seed, "epochs": epochs, "curriculum": args.curriculum, "judge": args.judge,
              "encoder": args.encoder, "init_encoder": str(args.init_encoder) if args.init_encoder else None,
              "init_judge": str(args.init_judge) if args.init_judge else None,
              "freeze_encoder": args.freeze_encoder, "judge_lr": args.judge_lr, "encoder_lr": args.encoder_lr,
              "selected_epoch": best[1],
              "history": history, "test": test_metrics,
              "probe": probe(encoder, train, test) if encoder is not None else None,
              "training_seconds": round(time.perf_counter() - started, 1),
              "peak_vram_gib": round(torch.cuda.max_memory_allocated(device) / 2**30, 3) if device.type == "cuda" else None}
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    tag = (("-curriculum" if args.curriculum else "") + ("-sourcefold" if args.judge == "source-fold" else "")
           + ("" if args.encoder == "nli" else f"-enc-{args.encoder}")
           + ("-staged" if args.init_encoder else "") + ("-frozen" if args.freeze_encoder else "")
           + ("-tuned" if args.init_judge and not args.freeze_encoder else ""))
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    judge_path = CHECKPOINT_DIR / f"world-v1_judge-{args.system}{tag}_{args.split}_seed-{args.seed}.pt"
    torch.save({"judge": best[2]["judge"], "judge_kind": args.judge,
                "message_size": 3 if encoder is None else encoder.message_size,
                "encoder_name": args.encoder if encoder is not None else None}, judge_path)
    print(f"Saved judge: {judge_path}")
    if encoder is not None and not args.freeze_encoder:
        # Save the reader (without the frozen word embeddings) so later stages can grow on it.
        reader = {k: v for k, v in best[2]["encoder"].items() if "word_embeddings" not in k}
        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        reader_path = CHECKPOINT_DIR / f"world-v1_reader-{args.system}{tag}_{args.split}_seed-{args.seed}.pt"
        torch.save({"reader": reader, "encoder_name": args.encoder}, reader_path)
        print(f"Saved reader: {reader_path}")
    out = RESULTS_DIR / f"world-v1_interface-{args.system}{tag}_{args.split}_seed-{args.seed}.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"test {test_metrics}\nprobe {result['probe']}\nSaved: {out}")


if __name__ == "__main__":
    main()
