"""Frozen v1 reader features for rule world v3: every (sentence, hypothesis) pair, fp32, cached.

Hypotheses: "door opens with key" (doors x 3, the v1 reader's own anchor form) and
"door A and door B open with the same key" (all unordered pairs, new to the reader).

python -m cognitive_lab.world3.features     # build the cache + linear probes
"""

import itertools
import random

import torch
from torch import nn

from cognitive_lab.world2.integrated import CACHE_DIR, CHECKPOINT_DIR, DEFAULT_READER
from cognitive_lab.world3.chains import DOORS, FACT_TEMPLATES, KEYS, SAME_TEMPLATES

PAIRS = tuple(itertools.combinations(DOORS, 2))


def fact_hypothesis(door: str, key: str) -> str:
    return f"{door}은 {key}로 열린다."


def same_hypothesis(a: str, b: str) -> str:
    return f"{a}과 {b}은 같은 열쇠로 열린다."


def all_sentences() -> list[dict]:
    rows = [{"text": t.format(D=d, K=k), "kind": "fact", "doors": (d,), "key": k}
            for t in FACT_TEMPLATES for d in DOORS for k in KEYS]
    rows += [{"text": t.format(A=a, B=b), "kind": "same", "doors": tuple(sorted((a, b), key=DOORS.index))}
             for t in SAME_TEMPLATES for a in DOORS for b in DOORS if a != b]
    return rows


def hypotheses() -> list[tuple]:
    return [("fact", d, k) for d in DOORS for k in KEYS] + [("same", a, b) for a, b in PAIRS]


def hypothesis_text(h: tuple) -> str:
    return fact_hypothesis(h[1], h[2]) if h[0] == "fact" else same_hypothesis(h[1], h[2])


def feature_table(device: torch.device, reader_file: str = DEFAULT_READER) -> dict:
    """{"sentences", "hypotheses", "features" [S, H, 768], "head"} (v1 head for reference)."""
    path = CACHE_DIR / "world3-features.pt"
    if path.exists():
        return torch.load(path)
    from cognitive_lab.world.interface import ENCODER_DIRS
    from cognitive_lab.world.interface_anchored import PairEncoder, load_reader

    saved = torch.load(CHECKPOINT_DIR / reader_file, map_location="cpu")
    encoder = PairEncoder(encoder_dir=ENCODER_DIRS[saved["encoder_name"]]).to(device)
    load_reader(encoder, saved["reader"])
    encoder.eval()
    pooled = {}
    encoder.head.register_forward_hook(lambda _m, inputs, _o: pooled.__setitem__("x", inputs[0]))
    sentences, hyps = all_sentences(), hypotheses()
    pairs = [(s["text"], hypothesis_text(h)) for s in sentences for h in hyps]
    chunks = []
    with torch.no_grad():
        for start in range(0, len(pairs), 512):
            chunk = pairs[start:start + 512]
            encoder([p for p, _ in chunk], [h for _, h in chunk], precise=True)
            chunks.append(pooled["x"].float().cpu())
    table = {"sentences": sentences, "hypotheses": hyps, "reader": reader_file,
             "features": torch.cat(chunks).view(len(sentences), len(hyps), -1),
             "head": {k: v.detach().cpu().clone() for k, v in encoder.head.state_dict().items()}}
    torch.save(table, path)
    return table


def supports(sentence: dict, h: tuple) -> bool:
    if h[0] == "fact":
        return sentence["kind"] == "fact" and sentence["doors"][0] == h[1] and sentence["key"] == h[2]
    return sentence["kind"] == "same" and set(sentence["doors"]) == {h[1], h[2]}


def probe(table: dict) -> None:
    """Does the feature say "this sentence states this hypothesis"? Held out by door pair."""
    sentences, hyps, features = table["sentences"], table["hypotheses"], table["features"]
    held = set(random.Random(0).sample(PAIRS, len(PAIRS) // 4))
    for kind in ("fact", "same"):
        cols = [j for j, h in enumerate(hyps) if h[0] == kind]
        x = features[:, cols].reshape(-1, features.shape[-1])
        y = torch.tensor([float(supports(s, hyps[j])) for s in sentences for j in cols])
        held_mask = torch.tensor([kind == "same" and (hyps[j][1], hyps[j][2]) in held or
                                  kind == "fact" and DOORS.index(hyps[j][1]) < 3 for _ in sentences for j in cols])
        for name, inputs in (("v1 4-d message", torch.tanh(x @ table["head"]["weight"].T + table["head"]["bias"])),
                             ("768-d feature", x)):
            mean, std = inputs[~held_mask].mean(0), inputs[~held_mask].std(0) + 1e-6
            z = (inputs - mean) / std
            model = nn.Linear(z.shape[1], 1)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
            positive = y[~held_mask].mean()
            weight = torch.where(y[~held_mask] > 0, 0.5 / positive, 0.5 / (1 - positive))
            for _ in range(300):
                loss = (nn.functional.binary_cross_entropy_with_logits(model(z[~held_mask]).squeeze(-1),
                                                                       y[~held_mask], reduction="none") * weight).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            pred = model(z[held_mask]).squeeze(-1) > 0
            truth = y[held_mask] > 0
            recall = (pred & truth).sum() / truth.sum()
            specificity = (~pred & ~truth).sum() / (~truth).sum()
            print(f"{kind:4s} probe from {name}: held-out recall {recall:.3f}, specificity {specificity:.3f}")


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    table = feature_table(device)
    print("features", tuple(table["features"].shape))
    probe(table)


if __name__ == "__main__":
    main()
