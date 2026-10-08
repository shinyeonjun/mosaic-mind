import torch
from torch import Tensor, nn


class RIMsCore(nn.Module):
    """Minimal RIMs-style core with independently routed GRU state modules.

    Only the selected module's state changes at each step. Candidate module
    updates are still computed densely; this model tests selective state update,
    not reduced FLOPs or the full RIMs/shared-workspace architecture.
    """

    def __init__(
        self,
        input_size: int,
        action_count: int,
        module_count: int = 4,
        module_size: int = 64,
        top_k: int = 1,
    ):
        super().__init__()
        if not 1 <= top_k <= module_count:
            raise ValueError("top_k must be between 1 and module_count")
        self.module_count = module_count
        self.module_size = module_size
        self.top_k = top_k
        state_size = module_count * module_size
        self.router = nn.Sequential(
            nn.Linear(input_size + state_size, module_size),
            nn.Tanh(),
            nn.Linear(module_size, module_count),
        )
        self.modules_ = nn.ModuleList(
            nn.GRUCell(input_size, module_size) for _ in range(module_count)
        )
        self.policy = nn.Linear(state_size, action_count)

    def forward(self, observations: Tensor) -> Tensor:
        return self.policy(self.encode(observations))

    def encode(self, observations: Tensor) -> Tensor:
        batch_size, sequence_length, _ = observations.shape
        state = observations.new_zeros(batch_size, self.module_count, self.module_size)

        for step in range(sequence_length):
            current = observations[:, step]
            routing_logits = self.router(torch.cat((current, state.flatten(1)), dim=-1))
            routing_probabilities = routing_logits.softmax(dim=-1)
            selected_indices = routing_probabilities.topk(self.top_k, dim=-1).indices
            hard_gates = torch.zeros_like(routing_probabilities).scatter_(-1, selected_indices, 1.0)
            gates = hard_gates - routing_probabilities.detach() + routing_probabilities

            candidates = torch.stack(
                [cell(current, state[:, index]) for index, cell in enumerate(self.modules_)],
                dim=1,
            )
            state = state + gates.unsqueeze(-1) * (candidates - state)

        return state.flatten(1)

