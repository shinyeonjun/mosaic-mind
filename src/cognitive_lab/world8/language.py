"""Sentences of rule world v8 and frozen-reader features for them.

Three link phrasings:
  same      the v3 "same key" templates (the thinker grew up on these)
  same-new  "same key" in phrasings the thinker never saw (should need no new part)
  next      "B's key is the one after A's" in the cycle 해 -> 달 -> 별 -> 해 (a relation the thinker
            cannot pass on: it only carries a key unchanged along a link)
Features: the frozen v1 reader's pooled 768-d feature for (sentence, hypothesis), as in v3, for the
v3 hypotheses (facts, unordered "same" pairs) and, for the new part, ordered "next" pairs.

python -m cognitive_lab.world8.language     # build the caches
"""

import itertools

import torch

from cognitive_lab.world2.integrated import CACHE_DIR, CHECKPOINT_DIR, DEFAULT_READER
from cognitive_lab.world3 import features as v3
from cognitive_lab.world3.chains import DOORS, KEYS

SAME_NEW_TEMPLATES = (
    "{A}과 {B}은 열쇠가 매한가지야.",
    "{A}을 연 열쇠면 {B}도 열린다.",
    "{A}, {B} 둘 다 같은 열쇠를 써요.",
)
NEXT_TEMPLATES = (  # B's key comes right after A's
    "{B} 열쇠는 {A} 열쇠의 다음 차례야.",
    "{A} 열쇠 바로 다음 순서가 {B} 열쇠예요.",
    "{B}은 {A} 열쇠 다음 차례의 열쇠로 열린다.",
)
ORDERED = tuple(itertools.permutations(DOORS, 2))


def next_key(key: str) -> str:
    return KEYS[(KEYS.index(key) + 1) % len(KEYS)]


def new_sentences() -> list[dict]:
    rows = [{"text": t.format(A=a, B=b), "kind": "same-new", "doors": tuple(sorted((a, b), key=DOORS.index))}
            for t in SAME_NEW_TEMPLATES for a in DOORS for b in DOORS if a != b]
    rows += [{"text": t.format(A=a, B=b), "kind": "next", "doors": (a, b)}
             for t in NEXT_TEMPLATES for a in DOORS for b in DOORS if a != b]
    return rows


def next_hypothesis(a: str, b: str) -> str:
    return f"{b} 열쇠는 {a} 열쇠의 다음이다."


@torch.no_grad()
def pair_features(premises: list[str], hypotheses: list[str], device: torch.device) -> torch.Tensor:
    """Pooled fp32 features [len(premises), len(hypotheses), 768] from the frozen v1 reader."""
    from cognitive_lab.world.interface import ENCODER_DIRS
    from cognitive_lab.world.interface_anchored import PairEncoder, load_reader

    saved = torch.load(CHECKPOINT_DIR / DEFAULT_READER, map_location="cpu")
    encoder = PairEncoder(encoder_dir=ENCODER_DIRS[saved["encoder_name"]]).to(device)
    load_reader(encoder, saved["reader"])
    encoder.eval()
    pooled = {}
    encoder.head.register_forward_hook(lambda _m, inputs, _o: pooled.__setitem__("x", inputs[0]))
    pairs = [(p, h) for p in premises for h in hypotheses]
    chunks = []
    for start in range(0, len(pairs), 512):
        chunk = pairs[start:start + 512]
        encoder([p for p, _ in chunk], [h for _, h in chunk], precise=True)
        chunks.append(pooled["x"].half().cpu())
    return torch.cat(chunks).view(len(premises), len(hypotheses), -1)


def thinker_table(device: torch.device) -> dict:
    """The v3 table extended with the new sentences (same hypotheses), for the existing thinker."""
    path = CACHE_DIR / "world8-thinker-features.pt"
    if path.exists():
        return torch.load(path)
    base = v3.feature_table(device)
    extra = new_sentences()
    hyps = [v3.hypothesis_text(h) for h in base["hypotheses"]]
    added = pair_features([s["text"] for s in extra], hyps, device).float()
    table = {"sentences": base["sentences"] + extra, "hypotheses": base["hypotheses"], "reader": base["reader"],
             "features": torch.cat([base["features"], added]), "head": base["head"]}
    torch.save(table, path)
    return table


def next_table(device: torch.device, sentences: list[dict]) -> torch.Tensor:
    """Features of every sentence against the ordered "next" hypotheses [S, 132, 768] (fp16), for the new part."""
    path = CACHE_DIR / "world8-next-features.pt"
    if path.exists():
        return torch.load(path)
    features = pair_features([s["text"] for s in sentences], [next_hypothesis(a, b) for a, b in ORDERED], device)
    torch.save(features, path)
    return features


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    table = thinker_table(device)
    print("thinker table", tuple(table["features"].shape))


# ---- rule world v9: a second new relation and relation-neutral hypotheses ---------------

SWAP_TEMPLATES = (  # B's key is A's with 해 and 별 swapped (달 stays)
    "{B} 열쇠는 {A} 열쇠에서 해와 별을 맞바꾼 거야.",
    "{A} 열쇠의 해와 별을 서로 바꾸면 {B} 열쇠가 돼요.",
    "{B}은 {A} 열쇠를 해·별 맞바꿈한 열쇠로 열린다.",
)


def swap_key(key: str) -> str:
    return {"해 열쇠": "별 열쇠", "별 열쇠": "해 열쇠"}.get(key, key)


def swap_sentences() -> list[dict]:
    return [{"text": t.format(A=a, B=b), "kind": "swap", "doors": (a, b)}
            for t in SWAP_TEMPLATES for a in DOORS for b in DOORS if a != b]


def relation_hypothesis(a: str, b: str) -> str:
    """Relation-neutral: says only that b's key is fixed by a's, not how."""
    return f"{b} 열쇠는 {a} 열쇠에 따라 정해진다."


def v9_tables(device: torch.device) -> tuple[dict, torch.Tensor]:
    """The thinker table extended with swap sentences, and every sentence against the relation-neutral
    ordered hypotheses [S, 132, 768] (fp16) for the grown branches."""
    path = CACHE_DIR / "world9-tables.pt"
    if path.exists():
        saved = torch.load(path)
        return saved["table"], saved["relation"]
    base = thinker_table(device)
    extra = swap_sentences()
    hyps = [v3.hypothesis_text(h) for h in base["hypotheses"]]
    added = pair_features([s["text"] for s in extra], hyps, device).float()
    table = {**base, "sentences": base["sentences"] + extra, "features": torch.cat([base["features"], added])}
    relation = pair_features([s["text"] for s in table["sentences"]], [relation_hypothesis(a, b) for a, b in ORDERED], device)
    torch.save({"table": table, "relation": relation}, path)
    return table, relation
