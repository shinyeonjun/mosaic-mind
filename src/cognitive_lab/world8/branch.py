"""A branch grown on the frozen stage-3 thinker: a "next key" channel (rule world v8).

While thinking, a door hears, besides its "same"-linked neighbors' states z[e, k], the states of
"next"-linked neighbors shifted by one key: z[e, k-1] from the door before it and z[e, k+1] from the
door after it (해 -> 달 -> 별 -> 해). Everything of the old thinker stays frozen; with no next links the
branch adds nothing.

Which links are "next" (version 2, after version 1 failed; see design/world-v8-growth.md and
artifacts/results/archive/world8_branch_v1.py):
  every link sentence states one link, so per sentence u
      relation  r(u)      = P(u states a "next" link)
      pair      a(u, ab)  = softmax over the 132 ordered pairs: which pair and which way
      next(a -> b) = max_u r(u) a(u, ab)
  Version 1 judged the 132 ordered pairs independently; spurious pairs kept switching on, and the
  "not both" rule then erased the old same links. One link per sentence rules that out.
Messages m(u, ab) = tanh(head(feature(u, "b's key is the one after a's"))). The head starts as the
reader's own 4-d head (rows 0-3; it already marks the mentioned pair in row 0, +1 vs -0.97, and
next-ness in row 3, +1 vs -0.89, but not the direction); rows 4-7 start near 0 and must learn the
direction, which the 768-d feature holds (probe 1.0). The pair scorer starts reading row 0 and the
relation scorer row 3; both stay trainable.
A pair cannot be both "same" and "next": a next link removes the old head's same link (the old head
mistakes 39% of next sentences for same ones).
Training uses the soft links; testing uses 0/1 links (> 0.5).
"""

import torch
from torch import nn

from cognitive_lab.world3 import chains
from cognitive_lab.world3.thinker import DOOR_COUNT, Thinker
from cognitive_lab.world8.language import ORDERED


class BranchedThinker(Thinker):
    def __init__(self, features: torch.Tensor, next_features: torch.Tensor, reader_head: dict,
                 size: int = 8, width: int = 32):
        super().__init__(features, size=size, width=width, message="state")
        self.register_buffer("next_features", next_features, persistent=False)  # [S, 132, 768] fp16
        self.next_head = nn.Linear(next_features.shape[-1], size)
        self.pair_score = nn.Linear(size, 1)
        self.relation_score = nn.Linear(size, 1)
        with torch.no_grad():
            self.next_head.weight.normal_(0, 0.01)
            self.next_head.bias.zero_()
            self.next_head.weight[:4] = reader_head["weight"]
            self.next_head.bias[:4] = reader_head["bias"]
            for layer, row in ((self.pair_score, 0), (self.relation_score, 3)):
                layer.weight.zero_()
                layer.bias.zero_()
                layer.weight[0, row] = 4.0
        rows, cols = zip(*[(chains.DOORS.index(a), chains.DOORS.index(b)) for a, b in ORDERED])
        self.register_buffer("next_rows", torch.tensor(rows), persistent=False)
        self.register_buffer("next_cols", torch.tensor(cols), persistent=False)
        self.branch_enabled = True

    def branch_parameters(self):
        return [p for m in (self.next_head, self.pair_score, self.relation_score) for p in m.parameters()]

    def next_links(self, index: torch.Tensor) -> torch.Tensor:
        """[B, D, D] soft: [b, i, j] = some sentence says door j's key comes right after door i's."""
        pad = index >= self.features.shape[0]
        safe = index.clamp(max=self.features.shape[0] - 1)
        messages = torch.tanh(self.next_head(self.next_features[safe].float()))  # [B,U,132,M]
        attention = self.pair_score(messages).squeeze(-1).softmax(-1)  # [B,U,132]: which pair, which way
        summary = (attention[..., None] * messages).sum(2)  # the message of the attended pair [B,U,M]
        relation = torch.sigmoid(self.relation_score(summary)).squeeze(-1).masked_fill(pad, 0.0)  # [B,U]
        per_pair = (relation[..., None] * attention).amax(1)  # [B,132]
        out = per_pair.new_zeros(index.shape[0], DOOR_COUNT, DOOR_COUNT)
        out[:, self.next_rows, self.next_cols] = per_pair
        return out

    def think(self, index, query, steps, with_links=False, extra_facts=None, extra_mask=None):
        facts, links = self.messages()
        pad = index >= self.features.shape[0]
        z0 = self.phi(facts[index]).masked_fill(pad[:, :, None, None, None], -1e4).amax(1)
        if extra_facts is not None:
            extra = self.phi(extra_facts).masked_fill(~extra_mask[:, :, None, None], -1e4)
            z0 = torch.maximum(z0, extra)
        strength = torch.sigmoid(self.link_score(self.psi(links[index]))).squeeze(-1)
        strength = strength.masked_fill(pad[:, :, None], 0.0).amax(1)
        batch = index.shape[0]
        same = strength.new_zeros(batch, DOOR_COUNT, DOOR_COUNT)
        same[:, self.pair_rows, self.pair_cols] = (strength > 0.5).float()
        same = same + same.transpose(1, 2)
        if self.branch_enabled:
            soft = self.next_links(index)
            self.last_next = soft
            nxt = soft if self.training else (soft > 0.5).float()
            same = same * (1 - torch.maximum(nxt, nxt.transpose(1, 2)))  # "not both"
            forward, backward = nxt.transpose(1, 2), nxt  # [b, j, i]: j hears i (i before j); [b, i, j]: i hears j
        self.last_strength = strength
        width = z0.shape[-1]
        z = z0
        rows = torch.arange(batch, device=index.device)
        beliefs = self.out(z).squeeze(-1).softmax(-1)
        all_logits, all_doors, all_changes = [], [], []
        for _ in range(steps):
            incoming = (same[:, :, :, None, None] * z[:, None]).amax(2)
            if self.branch_enabled:
                after = torch.roll(z, shifts=1, dims=2)   # [e, k] = z[e, k-1]: the door after e has e's key + 1
                before = torch.roll(z, shifts=-1, dims=2)  # [e, k] = z[e, k+1]: the door before e has e's key - 1
                incoming = torch.maximum(incoming, (forward[:, :, :, None, None] * after[:, None]).amax(2))
                incoming = torch.maximum(incoming, (backward[:, :, :, None, None] * before[:, None]).amax(2))
            new_z = self.cell(torch.cat([incoming, z0], -1).reshape(-1, 2 * width), z.reshape(-1, width)).view_as(z)
            new_beliefs = self.out(new_z).squeeze(-1).softmax(-1)
            all_changes.append((new_beliefs - beliefs).abs().amax((1, 2)))
            z, beliefs = new_z, new_beliefs
            all_logits.append(self.out(z[rows, query]).squeeze(-1))
            all_doors.append(self.out(z).squeeze(-1))
        self.last_changes = torch.stack(all_changes)
        logits = torch.stack(all_logits)
        halts = torch.zeros_like(logits[..., 0])
        if with_links:
            return logits, halts, strength, torch.stack(all_doors)
        return logits, halts
