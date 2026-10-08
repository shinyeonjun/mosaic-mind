from torch import Tensor, nn


class GRUCore(nn.Module):
    """Single-layer recurrent baseline: sequence in, final action logits out."""

    def __init__(self, input_size: int, action_count: int, hidden_size: int = 256):
        super().__init__()
        self.recurrent = nn.GRU(input_size, hidden_size, batch_first=True)
        self.policy = nn.Linear(hidden_size, action_count)

    def forward(self, observations: Tensor) -> Tensor:
        return self.policy(self.encode(observations))

    def encode(self, observations: Tensor) -> Tensor:
        sequence, _ = self.recurrent(observations)
        return sequence[:, -1]

