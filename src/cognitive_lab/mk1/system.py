"""MK1 on the blackboard: one system whose parts talk only through board slots (design/mk1-integration.md).

A turn, for a batch of sessions (episodes in order; the world reveals each answer afterwards):
  reader   -> reading.trust, reading.board   live frozen-reader features for what was said / posted
  trust    -> trust.door_logits              per base door, from speakers + memory of who was right
  thinker  -> thinker.first                  first pass over the board alone
  router   -> router.asked                   which base doors to ask the trust part about
  adapter  -> facts.from_trust               the asked doors' conclusions as thinker fact messages
  thinker  -> answer.logits                  second pass with those facts (answer = fixed-point step)
Which doors get asked is the policy: "trust" (the trust part alone answers about the base door the
speakers were asked about: stage 1), "think" (no trust part),
"ask-all" (stage 4), "route" (stage 7, the router decides).

The parts are the saved ones, unchanged. Only their wiring is new: they get features from the board,
not from the precomputed tables they were trained with (those are never loaded here).
"""

import torch
from torch import nn

from cognitive_lab.mk1.board import Blackboard
from cognitive_lab.mk1.reader import READING_MEMORY, ReaderService
from cognitive_lab.world2.hedged_system import ConnectedSystem
from cognitive_lab.world2.hedged_system import checkpoint_path as trust_checkpoint
from cognitive_lab.world2.integrated import CHECKPOINT_DIR, MAX_UTTERANCES
from cognitive_lab.world3 import chains
from cognitive_lab.world3.features import hypotheses, hypothesis_text
from cognitive_lab.world3.thinker import MAX_UTTERANCES as BOARD_SLOTS
from cognitive_lab.world3.thinker import Thinker, fixed_point_stop
from cognitive_lab.world4.composite import BASE_DOORS
from cognitive_lab.world4.system import CompositeSystem
from cognitive_lab.world7.routing import COST, Router
from cognitive_lab.world.interface_anchored import hypothesis as fact_hypothesis

STEPS = 16
POLICIES = ("trust", "think", "ask-all", "route")
THINKER_HYPOTHESES = [hypothesis_text(h) for h in hypotheses()]


def composite_sessions(sessions: list[dict]) -> list[dict]:
    """World v4/v7 sessions as MK1 sees them: speakers talk, the board is posted, a door is asked about.
    `anchor` is revealed with the answer after each episode (feedback, as in v2)."""
    return [{"speakers": s["speakers"], "episodes": [
        {"utterances": e["utterances"], "board": e["board"], "query": e["composite"]["query"],
         "keys": e["entities"]["keys"], "answer": e["answer"], "anchor": e["composite"]["anchor"]}
        for e in s["episodes"]]} for s in sessions]


def chain_sessions(episodes: list[dict]) -> list[dict]:
    """World v3 episodes: a board and a question, nobody talking. 모름 answers have no key."""
    return [{"speakers": [], "episodes": [
        {"utterances": [], "board": e["utterances"], "query": e["query"], "keys": list(chains.KEYS),
         "answer": e["answer"], "anchor": None}]} for e in episodes]


class MK1(nn.Module):
    def __init__(self, device: torch.device, seed: int = 42, store=READING_MEMORY):
        super().__init__()
        self.device = device
        self.reader = ReaderService(device, store=store)
        saved = torch.load(trust_checkpoint(seed, 8), map_location="cpu")
        empty_head = {"weight": torch.zeros(4, 768), "bias": torch.zeros(4)}
        trust = ConnectedSystem(torch.zeros(1, 768), empty_head, saved["message_size"])  # features come from the board
        thinker = Thinker(torch.zeros(1, len(THINKER_HYPOTHESES), 768), message="state")
        self.composite = CompositeSystem(trust, thinker)  # trust part + thinker + adapter (stage 4)
        self.composite.load_state_dict(torch.load(CHECKPOINT_DIR / f"world-v4_grown_seed-{seed}.pt", map_location="cpu")["state"])
        self.router = Router()
        self.router.load_state_dict(torch.load(CHECKPOINT_DIR / f"world-v7_router_seed-{seed}.pt", map_location="cpu")["router"])
        self.to(device).eval()

    @property
    def trust(self) -> ConnectedSystem:
        return self.composite.trust

    @property
    def thinker(self) -> Thinker:
        return self.composite.thinker

    # --- reader: posts what it read ---------------------------------------------------------------
    def read(self, board: Blackboard, sessions: list[dict]) -> None:
        batch, episodes = len(sessions), len(sessions[0]["episodes"])
        slot = {}
        index = torch.full((batch, episodes, 3, MAX_UTTERANCES, 3), -1, dtype=torch.long)
        speakers = torch.full((batch, episodes, MAX_UTTERANCES), -1, dtype=torch.long)
        for b, session in enumerate(sessions):
            names = {name: i for i, name in enumerate(session["speakers"])}
            for t, episode in enumerate(session["episodes"]):
                for u, utterance in enumerate(episode["utterances"][:MAX_UTTERANCES]):
                    speakers[b, t, u] = names[utterance["source"]]
                    for d, door in enumerate(BASE_DOORS):
                        for k, key in enumerate(episode["keys"]):
                            pair = (utterance["text"], fact_hypothesis(door, key))
                            index[b, t, d, u, k] = slot.setdefault(pair, len(slot))
        if slot:
            features = self.reader.features(list(slot))
            board.write("reading.trust", {"features": features, "index": index.masked_fill(index < 0, len(slot)),
                                          "speakers": speakers}, "reader")
        sentences = list(dict.fromkeys(x for s in sessions for e in s["episodes"] for x in e["board"]))
        lookup = {x: i for i, x in enumerate(sentences)}
        posted = torch.full((batch, episodes, BOARD_SLOTS), len(sentences), dtype=torch.long)
        for b, session in enumerate(sessions):
            for t, episode in enumerate(session["episodes"]):
                posted[b, t, :len(episode["board"])] = torch.tensor([lookup[x] for x in episode["board"]], dtype=torch.long)
        board.write("reading.board", {"features": self.reader.table(sentences, THINKER_HYPOTHESES), "index": posted},
                    "reader")

    # --- trust part: speakers + memory -> each base door's key ------------------------------------
    def trust_turn(self, board: Blackboard, feedback: dict) -> None:
        reading = board.read("reading.trust")
        table = torch.tanh(self.trust.head(reading["features"].to(self.device)))
        table = torch.cat([table, table.new_zeros(1, table.shape[1])])
        index, speakers = reading["index"].to(self.device), reading["speakers"].to(self.device)
        judge = self.trust.judge
        batch, episodes = speakers.shape[:2]
        memory = judge.initial_memory(batch)
        rows = torch.arange(batch, device=self.device)
        out = []
        for t in range(episodes):  # same loop as CompositeSystem.trust_logits
            reads = [judge.read(table[index[:, t, d]], speakers[:, t]) for d in range(3)]
            out.append(torch.stack([judge.logits(h, spoke, memory) for h, spoke in reads], 1))
            h = torch.stack([h for h, _ in reads], 1)[rows, feedback["anchor"][:, t]]
            memory = judge.update(memory, h, reads[0][1], feedback["truths"][:, t])
        board.write("trust.door_logits", torch.stack(out, 1).flatten(0, 1), "trust")

    # --- thinker: board (+ facts from other parts) -> answer ---------------------------------------
    def think(self, board: Blackboard, query: torch.Tensor, facts: dict | None = None):
        reading = board.read("reading.board")
        kept = self.thinker.features
        self.thinker.features = reading["features"].to(self.device)
        try:
            extra = {} if facts is None else {"extra_facts": facts["messages"], "extra_mask": facts["mask"]}
            logits, _, strength, doors = self.thinker.think(reading["index"].flatten(0, 1).to(self.device), query, STEPS,
                                                            with_links=True, **extra)
        finally:
            self.thinker.features = kept
        stop = fixed_point_stop(self.thinker.last_changes)
        rows = torch.arange(len(query), device=self.device)
        return {"answer": logits[stop, rows], "doors": doors[stop, rows].softmax(-1), "strength": strength}

    # --- router: what is missing -> which base doors to ask -----------------------------------------
    def route(self, board: Blackboard, query: torch.Tensor) -> None:
        first = board.read("thinker.first")
        count, doors = len(query), len(chains.DOORS)
        adjacency = torch.zeros(count, doors, doors, device=self.device)
        adjacency[:, self.thinker.pair_rows, self.thinker.pair_cols] = (first["strength"] > 0.5).float()
        adjacency = ((adjacency + adjacency.transpose(1, 2)) > 0).float() + torch.eye(doors, device=self.device)
        reach = nn.functional.one_hot(query, doors).float()
        for _ in range(doors):
            reach = ((reach[:, None, :] @ adjacency).squeeze(1) > 0).float()
        features = torch.stack([reach[:, :3], first["doors"][:, :3].amax(-1),
                                first["answer"].softmax(-1).amax(-1, keepdim=True).expand(-1, 3)], -1)
        board.write("router.asked", self.router(features) > COST, "router")

    # --- adapter: trust conclusions for the asked doors -> thinker fact messages --------------------
    def transmit(self, board: Blackboard) -> None:
        q = board.read("trust.door_logits").softmax(-1)
        asked = board.read("router.asked")
        sure = q.amax(-1, keepdim=True) > 0.5
        crisp = torch.where(sure, nn.functional.one_hot(q.argmax(-1), 3).float(), torch.full_like(q, 1 / 3))
        messages = torch.tanh(self.composite.adapter(torch.stack([crisp, crisp.amax(-1, keepdim=True).expand_as(crisp)], -1)))
        extra = messages.new_zeros(len(q), len(chains.DOORS), 3, messages.shape[-1])
        mask = torch.zeros(len(q), len(chains.DOORS), dtype=torch.bool, device=self.device)
        extra[:, :3] = messages * asked[..., None, None]
        mask[:, :3] = asked
        board.write("facts.from_trust", {"messages": extra, "mask": mask}, "adapter")

    @torch.no_grad()
    def forward(self, sessions: list[dict], policy: str = "route") -> tuple[dict, Blackboard]:
        """Answer logits [N,3] (N = sessions x episodes, in order), asks [N], targets [N] (-1 = 모름)."""
        if policy not in POLICIES:
            raise ValueError(f"policy {policy!r} not in {POLICIES}")
        board = Blackboard()
        flat = [e for s in sessions for e in s["episodes"]]
        query = torch.tensor([chains.DOORS.index(e["query"]) for e in flat], device=self.device)
        targets = torch.tensor([e["keys"].index(e["answer"]) if e["answer"] in e["keys"] else -1 for e in flat],
                               device=self.device)
        self.read(board, sessions)
        if policy != "think":
            episodes = len(sessions[0]["episodes"])
            feedback = {"truths": targets.view(len(sessions), episodes),
                        "anchor": torch.tensor([BASE_DOORS.index(e["anchor"]) for e in flat],
                                               device=self.device).view(len(sessions), episodes)}
            self.trust_turn(board, feedback)
        if policy == "trust":  # the trust part alone answers about the speakers' door (the v2-H question)
            answer = board.read("trust.door_logits")[torch.arange(len(flat), device=self.device), feedback["anchor"].flatten()]
            board.write("answer.logits", answer, "trust")
            self.reader.save()
            return {"logits": answer, "asks": torch.zeros(len(flat), device=self.device), "targets": targets}, board
        if policy == "think":
            board.write("router.asked", torch.zeros(len(flat), 3, dtype=torch.bool, device=self.device), "rule")
        elif policy == "ask-all":
            board.write("router.asked", torch.ones(len(flat), 3, dtype=torch.bool, device=self.device), "rule")
        else:
            board.write("thinker.first", self.think(board, query), "thinker")
            self.route(board, query)
        asked = board.read("router.asked")
        facts = None
        if asked.any():
            self.transmit(board)
            facts = board.read("facts.from_trust")
        board.write("answer.logits", self.think(board, query, facts)["answer"], "thinker")
        self.reader.save()
        return {"logits": board.read("answer.logits"), "asks": asked.float().sum(1), "targets": targets}, board
