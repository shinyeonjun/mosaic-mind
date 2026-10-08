"""Diagnostics for the thinker: is the learned link graph sharp, and when does the answer arrive?

python -m cognitive_lab.world3.diagnose --checkpoint world-v3_thinker_seed-0.pt
"""

import argparse

import torch

from cognitive_lab.world2.integrated import CHECKPOINT_DIR
from cognitive_lab.world3 import chains
from cognitive_lab.world3.features import PAIRS, feature_table
from cognitive_lab.world3.thinker import Thinker, episode_tensors


def link_strengths(model: Thinker, episodes: list[dict], index: torch.Tensor) -> dict:
    """Learned link strength for stated pairs, pairs sharing a door with a stated pair, and others."""
    _, links = model.messages()
    pad = index >= model.features.shape[0]
    strength = torch.sigmoid(model.link_score(model.psi(links[index]))).squeeze(-1)
    strength = strength.masked_fill(pad[:, :, None], 0.0).amax(1).cpu()
    strength = (strength > 0.5).float()  # the links actually used (0/1)
    groups = {"stated": [], "near (shares a door)": [], "other": []}
    for b, episode in enumerate(episodes):
        stated = {frozenset(s["doors"]) for s in episode["gold"] if s["kind"] == "same"}
        mentioned = {d for pair in stated for d in pair}
        for j, (a, c) in enumerate(PAIRS):
            pair = frozenset((a, c))
            group = "stated" if pair in stated else "near (shares a door)" if pair & mentioned else "other"
            groups[group].append(strength[b, j].item())
    return {k: (round(sum(v) / len(v), 3), round(max(v), 3)) for k, v in groups.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    table = feature_table(device)
    saved = torch.load(CHECKPOINT_DIR / args.checkpoint)
    model = Thinker(table["features"].to(device), message=saved["args"].get("message", "belief")).to(device)
    model.load_state_dict(saved["state"])
    model.eval()
    episodes = chains.generate("development", 0)[:600]
    index, query, target, length = (t.to(device) for t in episode_tensors(episodes, table))
    with torch.no_grad():
        print("link strength (mean, max):", link_strengths(model, episodes, index))
        logits, _ = model.think(index, query, 12)
        p = logits.softmax(-1)
        rows = torch.arange(len(query), device=device)
        answerable = target >= 0
        for value in (4, 5, 6):
            sel = (length == value) & answerable
            print(f"L={value}: p(true key) by step", [round(x, 2) for x in p[:, rows, target.clamp(min=0)][:, sel].mean(1).tolist()])


if __name__ == "__main__":
    main()
