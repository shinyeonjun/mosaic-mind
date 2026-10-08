"""Live reading service: the frozen v1 reader, called on demand instead of precomputed tables.

Every grown part reads through the same frozen reader (rule-world v1, staged-tuned, seed 42): the
pooled 768-d feature of (sentence, hypothesis), computed in fp32 (bf16 shifted cached values by up
to 0.19 depending on batch composition; see design/research-log-2026-10-06.md, pitfalls). Results
are memoised, so a sentence is read once per hypothesis.

python -m cognitive_lab.mk1.reader      # parity with the precomputed tables
"""

import random

import torch

from cognitive_lab.world2.integrated import CHECKPOINT_DIR, DEFAULT_READER


class ReaderService:
    def __init__(self, device: torch.device, reader_file: str = DEFAULT_READER, batch: int = 256):
        self.device, self.reader_file, self.batch = device, reader_file, batch
        self.memo: dict[tuple[str, str], torch.Tensor] = {}
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
            self._pooled = {}
            encoder.head.register_forward_hook(lambda _m, inputs, _o: self._pooled.__setitem__("x", inputs[0]))
            self._encoder = encoder
            self.head = {k: v.detach().cpu().clone() for k, v in encoder.head.state_dict().items()}
        return self._encoder

    @torch.no_grad()
    def features(self, pairs: list[tuple[str, str]]) -> torch.Tensor:
        """Pooled fp32 features [len(pairs), 768] on the CPU."""
        missing = [p for p in dict.fromkeys(pairs) if p not in self.memo]
        if missing:
            encoder = self._load()
            for start in range(0, len(missing), self.batch):
                chunk = missing[start:start + self.batch]
                encoder([s for s, _ in chunk], [h for _, h in chunk], precise=True)
                self.calls += len(chunk)
                for pair, row in zip(chunk, self._pooled["x"].float().cpu()):
                    self.memo[pair] = row
        return torch.stack([self.memo[p] for p in pairs])

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
