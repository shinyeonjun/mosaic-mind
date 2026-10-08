"""The shared blackboard of MK1: parts write to and read from named slots, never call each other.

Every slot has an address (what it is about, fixed by structure) and content (what is said about it,
often learned). Stage 1 showed that only designed slots throw information away (tone cues) and that a
free sentence vector does not learn; anchored messages (address by structure, content learned) worked,
and the board follows the same rule.

Slots (batched tensors; N = episodes in the batch, D = doors, K = keys):
  reading.index      [N, U]          sentence ids into the reading table (pad = S)
  reading.table      [S, H, 768]     frozen-reader features: S sentences x H hypotheses
  trust.door_logits  [N, 3, K]       the trust part's view of each base door (from speakers + memory)
  thinker.first      dict            the thinker's first pass: answer logits, door beliefs, links
  router.asked       [N, 3] bool     base doors the router asks the trust part about
  answer.logits      [N, K]          the final answer (모름 when no key passes 0.5)
Each write is logged with the writing part, so a run can be traced.
"""

from dataclasses import dataclass, field

import torch


@dataclass
class Blackboard:
    slots: dict = field(default_factory=dict)
    log: list = field(default_factory=list)

    def write(self, slot: str, value, writer: str) -> None:
        self.slots[slot] = value
        shape = tuple(value.shape) if torch.is_tensor(value) else type(value).__name__
        self.log.append((writer, slot, shape))

    def read(self, slot: str):
        if slot not in self.slots:
            raise KeyError(f"nothing written to {slot!r} yet (written: {sorted(self.slots)})")
        return self.slots[slot]

    def has(self, slot: str) -> bool:
        return slot in self.slots

    def trace(self) -> str:
        return "\n".join(f"{writer:10s} -> {slot:20s} {shape}" for writer, slot, shape in self.log)
