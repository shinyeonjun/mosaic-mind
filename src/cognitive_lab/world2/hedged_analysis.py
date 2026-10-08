"""Checks that the v2-H gain comes from the tone cue (design/world-v2h-tone-cues.md).

1. Cue removed: every utterance's message is replaced by the message of the same sentence
   without its tone prefix. If the gain is the cue, the score falls to the designed-format level.
2. Probe: can the cue be read off the learned 8-d message (held out by base sentence), over all
   pairs and over the pair each statement is about?

python -m cognitive_lab.world2.hedged_analysis
"""

import json
import random

import torch
from torch import nn

from cognitive_lab.world2 import hedged
from cognitive_lab.world2.hedged_system import (ConnectedSystem, _is_chatter, checkpoint_path, feature_table,
                                                session_index)
from cognitive_lab.world2.integrated import RESULTS_DIR, calibrated_score

PREFIX_CUE = {p: c for c, ps in hedged.CUE_PREFIXES.items() for p in ps if p}


def strip(text: str) -> tuple[str, str]:
    prefix = next((p for p in PREFIX_CUE if text.startswith(p)), "")
    return text[len(prefix):], PREFIX_CUE.get(prefix, "plain")


def probe(x: torch.Tensor, labels: torch.Tensor, train: torch.Tensor, test: torch.Tensor) -> float:
    x = (x - x[train].mean(0)) / (x[train].std(0) + 1e-6)
    model = nn.Linear(x.shape[1], 3)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    for _ in range(500):
        loss = nn.functional.cross_entropy(model(x[train]), labels[train])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return round((model(x[test]).argmax(-1) == labels[test]).float().mean().item(), 4)


def main() -> None:
    saved0 = torch.load(checkpoint_path(42, 8), map_location="cpu")
    rows, features, head = feature_table(saved0["reader"], torch.device("cpu"))
    lookup = {row: i for i, row in enumerate(rows)}
    plain_of = torch.tensor([lookup[(strip(t)[0], d, k)] for t, d, k in rows] + [len(rows)])
    bases = [strip(t)[0] for t, _, _ in rows]
    labels = torch.tensor([hedged.CUES.index(strip(t)[1]) for t, _, _ in rows])
    keep = torch.tensor([not _is_chatter(b) for b in bases])
    # The pair a statement is about (its own door and key): where the cue matters.
    claim = torch.tensor([d in b and k in b and not _is_chatter(b) for b, (_, d, k) in zip(bases, rows)])
    held = set(random.Random(0).sample(sorted(set(bases)), len(set(bases)) // 4))
    in_held = torch.tensor([b in held for b in bases])
    v1 = torch.tanh(features @ head["weight"].T + head["bias"])
    report = {"v1_message_probe_claim_pair": probe(v1, labels, claim & ~in_held, claim & in_held),
              "feature_probe_claim_pair": probe(features, labels, claim & ~in_held, claim & in_held)}
    print(report, flush=True)
    for seed in (42, 43, 44):
        saved = torch.load(checkpoint_path(seed, 8), map_location="cpu")
        system = ConnectedSystem(features, head, 8)
        system.load_state_dict(saved["state"])
        system.eval()
        test = session_index(hedged.generate_part("test", seed), rows)
        with torch.no_grad():
            messages = system.messages()
            full = calibrated_score(system.judge(messages[test[0]], test[1], test[2]), test[2])
            removed = calibrated_score(system.judge(messages[plain_of][test[0]], test[1], test[2]), test[2])
        report[seed] = {"test": full, "cue_removed": removed,
                        "probe_8d": probe(messages[:-1], labels, keep & ~in_held, keep & in_held),
                        "probe_dims_0_3": probe(messages[:-1, :4], labels, keep & ~in_held, keep & in_held),
                        "probe_dims_4_7": probe(messages[:-1, 4:], labels, keep & ~in_held, keep & in_held),
                        "probe_claim_pair": probe(messages[:-1], labels, claim & ~in_held, claim & in_held)}
        print(seed, report[seed], flush=True)
    (RESULTS_DIR / "world-v2h_connected-m8_analysis.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
