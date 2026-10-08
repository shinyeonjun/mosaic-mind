import torch
from torch.nn import functional as F

from cognitive_lab.core.gru import GRUCore
from cognitive_lab.core.rims import RIMsCore
from cognitive_lab.task import ACTION_COUNT, INPUT_SIZE, generate_episodes


def main() -> None:
    observations, labels, corrected, missing = generate_episodes(256, seed=42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    observations, labels = observations.to(device), labels.to(device)
    assert observations.shape == (256, 8, INPUT_SIZE)
    assert labels.min().item() >= 0 and labels.max().item() < ACTION_COUNT
    assert corrected.any() and missing.any()

    for model in (
        GRUCore(INPUT_SIZE, ACTION_COUNT),
        GRUCore(INPUT_SIZE, ACTION_COUNT, hidden_size=153),
        RIMsCore(INPUT_SIZE, ACTION_COUNT),
    ):
        model = model.to(device)
        logits = model(observations[:16])
        assert logits.shape == (16, ACTION_COUNT)
        F.cross_entropy(logits, labels[:16]).backward()
        gradients = [p.grad for p in model.parameters() if p.grad is not None]
        assert gradients and all(torch.isfinite(grad).all() for grad in gradients)
        if isinstance(model, RIMsCore):
            router_gradient = model.router[-1].weight.grad
            assert router_gradient is not None and router_gradient.abs().sum().item() > 0

    print(f"Self-check passed on {device}: task labels and all three model variants are valid.")


if __name__ == "__main__":
    main()

