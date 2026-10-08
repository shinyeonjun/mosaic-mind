"""Live reading service: the frozen v1 reader, called on demand instead of precomputed tables.

Every grown part reads through the same frozen reader (rule-world v1, staged-tuned, seed 42): the
pooled 768-d feature of (sentence, hypothesis), computed in fp32 (bf16 shifted cached values by up
to 0.19 depending on batch composition; see design/research-log-2026-10-06.md, pitfalls). Results
are memoised, so a sentence is read once per hypothesis; with `store`, the memo is kept on disk
between runs (a reading memory: the same frozen reader in fp32, so a remembered reading equals a live one).

python -m cognitive_lab.mk1.reader      # parity with the precomputed tables
"""

import random

import torch

from cognitive_lab.world2.integrated import CACHE_DIR, CHECKPOINT_DIR, DEFAULT_READER

READING_MEMORY = CACHE_DIR / "mk1-reading-memory.pt"


class ReaderService:
    """`precision`: "fp32" (default, what every part was trained on) or "fp16" (about 1.8x faster on the
    laptop GPU, where fp32 reading is compute-bound at about 800 pairs/s; see design/mk1-integration.md).
    Pairs are read sorted by length, so a batch carries little padding (fp32 results do not depend on it)."""

    def __init__(self, device: torch.device, reader_file: str = DEFAULT_READER, batch: int = 256, store=None,
                 precision: str = "fp32"):
        if precision not in ("fp32", "fp16"):
            raise ValueError(precision)
        self.device, self.reader_file, self.batch, self.precision = device, reader_file, batch, precision
        self.store = None if store is None else store.with_name(f"{store.stem}-{precision}{store.suffix}") if precision != "fp32" else store
        self.memo: dict[tuple[str, str], torch.Tensor] = {}
        if self.store is not None and self.store.exists():
            saved = torch.load(self.store)
            if saved["reader"] == reader_file:
                self.memo = dict(zip(saved["pairs"], saved["features"]))
        self._encoder = None
        self.calls = 0

    def _load(self):
        if self._encoder is None:
            from cognitive_lab.world.interface import ENCODER_DIRS
            from cognitive_lab.world.interface_anchored import PairEncoder, load_reader

            saved = torch.load(CHECKPOINT_DIR / self.reader_file, map_location="cpu")
            encoder = PairEncoder(encoder_dir=ENCODER_DIRS[saved["encoder_name"]]).to(self.device)
            load_reader(encoder, saved["reader"])
            encoder.eval()
            self._encoder = encoder
            self.head = {k: v.detach().cpu().clone() for k, v in encoder.head.state_dict().items()}
        return self._encoder

    @torch.no_grad()
    def features(self, pairs: list[tuple[str, str]]) -> torch.Tensor:
        """Pooled features [len(pairs), 768] (fp32 tensors) on the CPU."""
        missing = sorted((p for p in dict.fromkeys(pairs) if p not in self.memo), key=lambda p: len(p[0]) + len(p[1]))
        if missing:
            encoder = self._load()
            for start in range(0, len(missing), self.batch):
                chunk = missing[start:start + self.batch]
                self.calls += len(chunk)
                for pair, row in zip(chunk, self._pool(encoder, chunk)):
                    self.memo[pair] = row
        return torch.stack([self.memo[p] for p in pairs])

    def _pool(self, encoder, chunk: list[tuple[str, str]]) -> torch.Tensor:
        """The reader's pooled input to its head (as PairEncoder.forward computes it)."""
        batch = encoder.tokenizer([s for s, _ in chunk], [h for _, h in chunk], return_tensors="pt", padding=True,
                                  truncation=True, max_length=64).to(self.device)
        with torch.autocast(device_type=self.device.type, dtype=torch.float16,
                            enabled=self.device.type == "cuda" and self.precision == "fp16"):
            states = encoder.backbone(**batch).last_hidden_state
        mask = batch["attention_mask"].unsqueeze(-1).float()
        return ((states.float() * mask).sum(1) / mask.sum(1)).cpu()

    def save(self) -> None:
        """Write the reading memory to `store` if anything new was read."""
        if self.store is not None and self.calls:
            torch.save({"reader": self.reader_file, "pairs": list(self.memo),
                        "features": torch.stack(list(self.memo.values()))}, self.store)
            self.calls = 0

    def table(self, sentences: list[str], hypotheses: list[str]) -> torch.Tensor:
        """[S, H, 768] for every sentence against every hypothesis."""
        rows = self.features([(s, h) for s in sentences for h in hypotheses])
        return rows.view(len(sentences), len(hypotheses), -1)


def parity(device: torch.device, samples: int = 3000) -> dict:
    """Max absolute difference between live reading and each precomputed table, on random entries."""
    from cognitive_lab.world2.hedged_system import feature_table as hedged_table
    from cognitive_lab.world3 import features as v3
    from cognitive_lab.world8.language import ORDERED, relation_hypothesis, v9_tables
    from cognitive_lab.world.interface_anchored import hypothesis as fact_hypothesis

    service = ReaderService(device)
    rng = random.Random(0)
    report = {}

    rows, feats, _ = hedged_table(DEFAULT_READER, device)  # (sentence, door, key) -> [768]
    pick = rng.sample(range(len(rows)), samples)
    live = service.features([(rows[i][0], fact_hypothesis(rows[i][1], rows[i][2])) for i in pick])
    report["stage 1-2 (v2-H reader table)"] = (live - feats[pick]).abs().max().item()

    table = v3.feature_table(device)
    hyps = [v3.hypothesis_text(h) for h in table["hypotheses"]]
    pick = [(rng.randrange(len(table["sentences"])), rng.randrange(len(hyps))) for _ in range(samples)]
    live = service.features([(table["sentences"][s]["text"], hyps[h]) for s, h in pick])
    cached = torch.stack([table["features"][s, h] for s, h in pick])
    report["stage 3-4 (v3 thinker table)"] = (live - cached).abs().max().item()

    t9, relation = v9_tables(device)
    pick = [(rng.randrange(len(t9["sentences"])), rng.randrange(len(ORDERED))) for _ in range(samples)]
    live = service.features([(t9["sentences"][s]["text"], relation_hypothesis(*ORDERED[h])) for s, h in pick])
    cached = torch.stack([relation[s, h].float() for s, h in pick])
    report["stage 9 (relation table, stored fp16)"] = (live - cached).abs().max().item()
    report["reader calls"] = service.calls
    return report


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for name, value in parity(device).items():
        print(f"{name}: {value:.2e}" if isinstance(value, float) else f"{name}: {value}")
