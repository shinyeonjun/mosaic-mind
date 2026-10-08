"""Minimal LoRA adapters for nn.Linear layers (no extra dependency).

Each wrapped layer computes base(x) + scale * B(A(dropout(x))). The base weight stays
frozen in bf16; A and B are trainable float32 matrices, and B starts at zero so the
wrapped model initially matches the base model exactly.
"""

import math

import torch
from torch import nn

# LFM2 linear layers: attention q/k/v/out, short-conv in/out, and MLP w1/w2/w3.
DEFAULT_TARGETS = ("q_proj", "k_proj", "v_proj", "out_proj", "in_proj", "w1", "w2", "w3")


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float):
        super().__init__()
        self.base = base
        self.scale = alpha / rank
        self.dropout = nn.Dropout(dropout)
        self.lora_a = nn.Parameter(torch.empty(rank, base.in_features, device=base.weight.device))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        update = self.dropout(x).float() @ self.lora_a.T @ self.lora_b.T
        return self.base(x) + (self.scale * update).to(x.dtype)


def apply_lora(model: nn.Module, rank: int, alpha: float, dropout: float,
               targets: tuple[str, ...] = DEFAULT_TARGETS) -> int:
    """Freeze `model` and wrap matching Linear layers; return the trainable parameter count."""
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    replacements = [
        (parent, name, child)
        for parent in model.modules()
        for name, child in parent.named_children()
        if isinstance(child, nn.Linear) and name in targets
    ]
    if not replacements:
        raise RuntimeError("no LoRA target layers found")
    for parent, name, child in replacements:
        setattr(parent, name, LoRALinear(child, rank, alpha, dropout))
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
        if name.endswith(("lora_a", "lora_b"))
    }


def load_lora(model: nn.Module, state: dict[str, torch.Tensor]) -> None:
    missing = set(state) - set(model.state_dict())
    if missing:
        raise RuntimeError(f"adapter has keys the model lacks: {sorted(missing)[:3]}")
    model.load_state_dict(state, strict=False)
