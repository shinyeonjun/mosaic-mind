"""Rule world v9: branches of any shape on the frozen stage-3 thinker, and a hypothesis search for the shape.

A branch carries keys along its own links, re-mapped by its shape: a 3x3 0/1 matrix P (one of the 6
ways to re-map 3 keys). A door after e hears P z[e] (slot k gets e's slot j where P[k, j] = 1); a door
before e hears P^T z[e]. Which links are the branch's is learned as in world8/branch.py version 2 (one
link per sentence: relation gate x pair attention), from frozen-reader features of every sentence
against relation-neutral hypotheses ("b's key is fixed by a's"). The pair scorer starts on the
reader's 4-d head row 0, which marks the mentioned pair for every relation (+1 vs -0.97); the
relation gate starts neutral (it must learn which sentences are its relation: the 4-d head does not
tell relations apart). A pair that is any branch's link is not a same link.

Hypothesis search (the system finds the shape): a fresh branch is trained on the study material for
each of the 6 shapes; the one with the best held-out score is kept (ties within 0.01: the shape that
moves fewer keys). Gradient descent from "pass unchanged" does not find the shape (probe: the matrix
stayed near identity), so the shape is chosen by trying, not by sliding.
"""

import itertools

import torch
from torch import nn

from cognitive_lab.world3 import chains
from cognitive_lab.world3.thinker import DOOR_COUNT, Thinker, episode_tensors, evaluate, step_loss
from cognitive_lab.world8.language import ORDERED

SHAPES = [torch.eye(3)[list(p)] for p in itertools.permutations(range(3))]  # row k = one-hot of the source slot


def moved(shape: torch.Tensor) -> int:
    return int((shape.argmax(1) != torch.arange(3)).sum())


class Branch(nn.Module):
    def __init__(self, shape: torch.Tensor, reader_head: dict, message: int = 8):
        super().__init__()
        self.register_buffer("shape", shape.float())
        self.head = nn.Linear(reader_head["weight"].shape[1], message)
        self.pair_score = nn.Linear(message, 1)
        self.relation_score = nn.Linear(message, 1)
        with torch.no_grad():
            self.head.weight.normal_(0, 0.01)
            self.head.bias.zero_()
            self.head.weight[:4] = reader_head["weight"]
            self.head.bias[:4] = reader_head["bias"]
            self.pair_score.weight.zero_()
            self.pair_score.bias.zero_()
            self.pair_score.weight[0, 0] = 4.0
            self.relation_score.weight.normal_(0, 0.01)
            self.relation_score.bias.zero_()

    def links(self, features: torch.Tensor, pad: torch.Tensor, rows, cols) -> torch.Tensor:
        """features [B,U,132,768] -> soft [B,D,D]: [b, i, j] = door j's key is door i's re-mapped."""
        messages = torch.tanh(self.head(features))
        attention = self.pair_score(messages).squeeze(-1).softmax(-1)
        summary = (attention[..., None] * messages).sum(2)
        relation = torch.sigmoid(self.relation_score(summary)).squeeze(-1).masked_fill(pad, 0.0)
        per_pair = (relation[..., None] * attention).amax(1)
        out = per_pair.new_zeros(features.shape[0], DOOR_COUNT, DOOR_COUNT)
        out[:, rows, cols] = per_pair
        return out


class MultiBranchThinker(Thinker):
    def __init__(self, features: torch.Tensor, relation_features: torch.Tensor, reader_head: dict):
        super().__init__(features, message="state")
        self.register_buffer("relation_features", relation_features, persistent=False)  # fp16
        self.reader_head = reader_head
        self.branches = nn.ModuleList()
        rows, cols = zip(*[(chains.DOORS.index(a), chains.DOORS.index(b)) for a, b in ORDERED])
        self.register_buffer("rel_rows", torch.tensor(rows), persistent=False)
        self.register_buffer("rel_cols", torch.tensor(cols), persistent=False)

    def add_branch(self, shape: torch.Tensor) -> Branch:
        branch = Branch(shape, self.reader_head).to(self.features.device)
        self.branches.append(branch)
        return branch

    def think(self, index, query, steps, with_links=False, extra_facts=None, extra_mask=None):
        facts, links = self.messages()
        pad = index >= self.features.shape[0]
        z0 = self.phi(facts[index]).masked_fill(pad[:, :, None, None, None], -1e4).amax(1)
        strength = torch.sigmoid(self.link_score(self.psi(links[index]))).squeeze(-1)
        strength = strength.masked_fill(pad[:, :, None], 0.0).amax(1)
        batch = index.shape[0]
        same = strength.new_zeros(batch, DOOR_COUNT, DOOR_COUNT)
        same[:, self.pair_rows, self.pair_cols] = (strength > 0.5).float()
        same = same + same.transpose(1, 2)
        channels = []
        if len(self.branches):
            safe = index.clamp(max=self.features.shape[0] - 1)
            features = self.relation_features[safe].float()
            for branch in self.branches:
                soft = branch.links(features, pad, self.rel_rows, self.rel_cols)
                hard = soft if self.training else (soft > 0.5).float()
                same = same * (1 - torch.maximum(hard, hard.transpose(1, 2)))
                channels.append((hard.transpose(1, 2), hard, branch.shape))
        width = z0.shape[-1]
        z = z0
        rows = torch.arange(batch, device=index.device)
        beliefs = self.out(z).squeeze(-1).softmax(-1)
        all_logits, all_doors, all_changes = [], [], []
        for _ in range(steps):
            incoming = (same[:, :, :, None, None] * z[:, None]).amax(2)
            for forward, backward, shape in channels:
                after = torch.einsum("kj,bejw->bekw", shape, z)
                before = torch.einsum("jk,bejw->bekw", shape, z)  # P^T
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


def train_last_branch(model: MultiBranchThinker, data: tuple, validation: tuple, seed: int, epochs: int = 30,
                      learning_rate: float = 2e-3, batch_size: int = 32) -> tuple[float, int]:
    """Train only the newest branch; keep its best epoch by validation score. Returns (score, epoch)."""
    torch.manual_seed(seed)
    branch = model.branches[-1]
    params = list(branch.parameters())
    optimizer = torch.optim.Adam(params, lr=learning_rate)
    index, query, target, _ = data
    model.eval()
    with torch.no_grad():
        best = (evaluate(model, validation, 16)["adaptive"], 0, {k: v.clone() for k, v in branch.state_dict().items()})
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(query), device=query.device)
        for start in range(0, len(order), batch_size):
            batch = order[start:start + batch_size]
            logits, _ = model.think(index[batch], query[batch], 8)
            loss = step_loss(logits, target[batch]).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            value = evaluate(model, validation, 16)["adaptive"]
        if value > best[0]:
            best = (value, epoch, {k: v.clone() for k, v in branch.state_dict().items()})
    branch.load_state_dict(best[2])
    return best[0], best[1]


def search_shape(model: MultiBranchThinker, data: tuple, validation: tuple, seed: int, epochs: int = 30) -> dict:
    """Try every shape as a new branch; keep the best (ties within 0.01: fewer keys moved)."""
    trials = []
    for n, shape in enumerate(SHAPES):
        model.add_branch(shape)
        score, epoch = train_last_branch(model, data, validation, seed, epochs)
        trials.append({"shape": shape.argmax(1).tolist(), "moved": moved(shape), "score": round(score, 4), "epoch": epoch,
                       "state": {k: v.detach().clone() for k, v in model.branches[-1].state_dict().items()}})
        del model.branches[-1]
    top = max(t["score"] for t in trials)
    chosen = min((t for t in trials if t["score"] >= top - 0.01), key=lambda t: (t["moved"], -t["score"]))
    branch = model.add_branch(torch.eye(3)[chosen["shape"]])
    branch.load_state_dict(chosen["state"])
    return {"chosen": chosen["shape"], "trials": [{k: v for k, v in t.items() if k != "state"} for t in trials]}
