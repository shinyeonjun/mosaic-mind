"""MK1 on the blackboard: one system whose parts talk only through board slots (design/mk1-integration.md).

A turn, for a batch of sessions (episodes in order; the world reveals each answer afterwards):
  reader    -> reading.trust, reading.board  live frozen-reader features for what was said / posted
  curiosity -> curiosity.asked               (optional) which speaker to ask about the queried door
  reader    -> reading.answers               what the asked speakers answered (the world answers)
  trust     -> doors.conclusions, doors.available  per base door, from speakers + memory of who was right
  thinker   -> thinker.first                 first pass over the board alone
  router    -> router.asked                  which base doors to ask the trust part about
  adapter   -> facts.from_trust              the asked doors' conclusions as thinker fact messages
  thinker   -> answer.logits                 second pass with those facts (answer = fixed-point step)
Which doors get asked is the policy: "trust" (the trust part alone answers about the base door the
speakers were asked about: stage 1), "think" (no trust part), "ask-all" (stage 4), "route" (stage 7,
the router decides). `curious=True` adds stage 2: before the trust part answers, the curiosity head may
ask one speaker about the queried door (only a base door: speakers never talk about other doors).

Article sessions (stage D, mk1/articles.py): sources hand over passages; the reading specialist reads
them (reading.articles), the real-text trust part picks a candidate answer or 모름 (trust_real.logits).
Which path a session takes is decided by what it brings: passages from sources, or speech and a board.

Stage F (world v11, boards that can be wrong): "arbitrate" lets the arbiter (mk1/arbiter.py) weigh the
board's and the article's word on the article door and replaces that door's fact with its conclusion;
"article-only" replaces it with the article part's conclusion; "ask-available" merges both (max).

Stage G: "verify-*" decide per question whether to ask the article part about the article door although
the board may already say something. The router sees the arbiter's confidence before asking (which knows
how often the board has been right this session) in place of the thinker's raw belief. Asking changes what
the arbiter remembers, so questions are taken in order (verify_loop). "verify-hindsight" knows the answer
(an upper bound for deciding when to ask, not a policy).

Linked sessions (stage E, world10): a real article is attached to one base door; the real-text trust part's
conclusion, turned into a key through the board's key labels, is written as that door's conclusion
(article -> doors.conclusions), and the router and adapter use it exactly as they use the trust part's.

The router may only ask about a base door some part has something on (`doors.available`): a door nobody
spoke about in an episode, or a door no article is attached to. (Stage 7 never met such episodes.)

The parts are the saved ones, unchanged. Only their wiring is new: they get features from the board,
not from the precomputed tables they were trained with (those are never loaded here).
"""

import torch
from torch import nn

from cognitive_lab.mk1.arbiter import Arbiter
from cognitive_lab.mk1.arbiter import door_scores as door_score
from cognitive_lab.mk1.articles import ArticleReading, trust_inputs
from cognitive_lab.mk1.board import Blackboard
from cognitive_lab.mk1.reader import READING_MEMORY, ReaderService
from cognitive_lab.mk1.relevance import board_relevant, fillers
from cognitive_lab.world2 import asking
from cognitive_lab.world2.asking_system import COST as ASK_COST
from cognitive_lab.world2.asking_system import CuriosityHead
from cognitive_lab.world2.hedged import _queried
from cognitive_lab.world2.hedged_system import ConnectedSystem
from cognitive_lab.world2.hedged_system import checkpoint_path as trust_checkpoint
from cognitive_lab.world2.integrated import CHECKPOINT_DIR, MAX_UTTERANCES
from cognitive_lab.world3 import chains
from cognitive_lab.world3.features import hypotheses, hypothesis_text
from cognitive_lab.world3.thinker import MAX_UTTERANCES as BOARD_SLOTS
from cognitive_lab.world3.thinker import Thinker, fixed_point_stop
from cognitive_lab.world4.composite import BASE_DOORS
from cognitive_lab.world5 import klue
from cognitive_lab.world4.system import CompositeSystem
from cognitive_lab.world6 import trust as real_trust
from cognitive_lab.world7.routing import COST, Router
from cognitive_lab.world.interface_anchored import hypothesis as fact_hypothesis

STEPS = 16
POLICIES = ("trust", "think", "ask-all", "ask-available", "route", "article-only", "arbitrate",
            "verify-never", "verify-always", "verify-router", "verify-hindsight")
THINKER_HYPOTHESES = [hypothesis_text(h) for h in hypotheses()]


def _answers(session: dict, episode: dict) -> dict:
    """What each speaker would answer if asked about the speakers' door (v2-A rule: fixed per episode
    and speaker, None if they already spoke about it). MK1 only reads an answer if it asked."""
    out = {}
    for name in session["speakers"]:
        answer = asking.ask(session, episode, name)
        out[name] = None if answer is None else answer[0]["text"]
    return out


def speaker_sessions(sessions: list[dict]) -> list[dict]:
    """World v2-H/v2-A sessions: speakers talk about the base doors, the question names one; no board."""
    return [{"speakers": s["speakers"], "episodes": [
        {"utterances": e["utterances"], "board": [], "query": _queried(e), "keys": e["entities"]["keys"],
         "answer": e["answer"], "anchor": _queried(e), "answers": _answers(s, e)}
        for e in s["episodes"]]} for s in sessions]


def composite_sessions(sessions: list[dict]) -> list[dict]:
    """World v4/v7 sessions as MK1 sees them: speakers talk, the board is posted, a door is asked about.
    `anchor` is revealed with the answer after each episode (feedback, as in v2)."""
    return [{"speakers": s["speakers"], "episodes": [
        {"utterances": e["utterances"], "board": e["board"], "query": e["composite"]["query"],
         "keys": e["entities"]["keys"], "answer": e["answer"], "anchor": e["composite"]["anchor"],
         "answers": _answers(s, e), "tag": e["composite"].get("type", "v4")}  # tag: for scoring only, MK1 never reads it
        for e in s["episodes"]]} for s in sessions]


def chain_sessions(episodes: list[dict]) -> list[dict]:
    """World v3 episodes: a board and a question, nobody talking. 모름 answers have no key."""
    return [{"speakers": [], "episodes": [
        {"utterances": [], "board": e["utterances"], "query": e["query"], "keys": list(chains.KEYS),
         "answer": e["answer"], "anchor": None, "answers": {}, "tag": "board-only"}]} for e in episodes]


class MK1(nn.Module):
    def __init__(self, device: torch.device, seed: int = 42, store=READING_MEMORY, reading: str = "relevant",
                 precision: str = "fp32", articles_from_cache: bool = False, article_precision: str = "fp32"):
        super().__init__()
        self.device, self.reading = device, reading
        self.reader = ReaderService(device, store=store, precision=precision)
        # "relevant": read board sentences only against the hypotheses that matter, fillers elsewhere
        # (mk1/relevance.py: 37 of 102, scores unchanged);
        # "full": read every hypothesis, as the parts were trained.
        self.fillers = fillers(self.reader) if reading == "relevant" else None
        saved = torch.load(trust_checkpoint(seed, 8), map_location="cpu")
        empty_head = {"weight": torch.zeros(4, 768), "bias": torch.zeros(4)}
        trust = ConnectedSystem(torch.zeros(1, 768), empty_head, saved["message_size"])  # features come from the board
        thinker = Thinker(torch.zeros(1, len(THINKER_HYPOTHESES), 768), message="state")
        self.composite = CompositeSystem(trust, thinker)  # trust part + thinker + adapter (stage 4)
        self.composite.load_state_dict(torch.load(CHECKPOINT_DIR / f"world-v4_grown_seed-{seed}.pt", map_location="cpu")["state"])
        self.router = Router()
        self.router.load_state_dict(torch.load(CHECKPOINT_DIR / f"world-v7_router_seed-{seed}.pt", map_location="cpu")["router"])
        self.curiosity = CuriosityHead()
        self.curiosity.load_state_dict(torch.load(CHECKPOINT_DIR / f"world-v2a_curiosity_cost-0.2_seed-{seed}.pt",
                                                  map_location="cpu")["state"])
        self.trust_real = real_trust.TrustPart()
        self.trust_real.load_state_dict(torch.load(real_trust.CHECKPOINT_DIR / f"world-v6_trust_seed-{seed}.pt",
                                                   map_location="cpu")["state"])
        # Article readings: live fp32, or (wiring check) seeded from the bf16 v6 cache.
        self.articles = ArticleReading(device, seed_from_cache=articles_from_cache, precision=article_precision)
        self.verifier = None  # stage G: a router retrained for verifying, if one is loaded
        self.arbiter = Arbiter()
        arbiter_file = CHECKPOINT_DIR / f"mk1-arbiter_seed-{seed}.pt"
        if arbiter_file.exists():
            self.arbiter.load_state_dict(torch.load(arbiter_file, map_location="cpu")["state"])
        self.to(device).eval()

    @property
    def trust(self) -> ConnectedSystem:
        return self.composite.trust

    @property
    def thinker(self) -> Thinker:
        return self.composite.thinker

    # --- reader: posts what it read ---------------------------------------------------------------
    def read(self, board: Blackboard, sessions: list[dict], curious: bool) -> None:
        batch, episodes = len(sessions), len(sessions[0]["episodes"])
        speakers_count = max(len(s["speakers"]) for s in sessions)
        slot = {}

        def claim(text, keys):  # [3 doors, 3 keys] row ids for one sentence
            return [[slot.setdefault((text, fact_hypothesis(door, key)), len(slot)) for key in keys] for door in BASE_DOORS]

        index = torch.full((batch, episodes, 3, MAX_UTTERANCES, 3), -1, dtype=torch.long)
        speakers = torch.full((batch, episodes, MAX_UTTERANCES), -1, dtype=torch.long)
        answers = torch.full((batch, episodes, max(speakers_count, 1), 3, 3), -1, dtype=torch.long)
        has_answer = torch.zeros(batch, episodes, max(speakers_count, 1), dtype=torch.bool)
        for b, session in enumerate(sessions):
            names = {name: i for i, name in enumerate(session["speakers"])}
            for t, episode in enumerate(session["episodes"]):
                for u, utterance in enumerate(episode["utterances"][:MAX_UTTERANCES]):
                    speakers[b, t, u] = names[utterance["source"]]
                    index[b, t, :, u] = torch.tensor(claim(utterance["text"], episode["keys"]))
                if curious:  # the world's answers are only read for curiosity; MK1 uses one only if it asked
                    for name, text in episode["answers"].items():
                        if text is not None:
                            answers[b, t, names[name]] = torch.tensor(claim(text, episode["keys"]))
                            has_answer[b, t, names[name]] = True
        if slot:
            pad = len(slot)
            board.write("reading.trust", {"features": self.reader.features(list(slot)),
                                          "index": index.masked_fill(index < 0, pad), "speakers": speakers,
                                          "answers": answers.masked_fill(answers < 0, pad), "has_answer": has_answer},
                        "reader")
        sentences = list(dict.fromkeys(x for s in sessions for e in s["episodes"] for x in e["board"]))
        lookup = {x: i for i, x in enumerate(sentences)}
        posted = torch.full((batch, episodes, BOARD_SLOTS), len(sentences), dtype=torch.long)
        for b, session in enumerate(sessions):
            for t, episode in enumerate(session["episodes"]):
                if episode["board"]:
                    posted[b, t, :len(episode["board"])] = torch.tensor([lookup[x] for x in episode["board"]], dtype=torch.long)
        if not sentences:
            features = torch.zeros(0, len(THINKER_HYPOTHESES), 768)
        elif self.fillers is None:
            features = self.reader.table(sentences, THINKER_HYPOTHESES)
        else:
            features = self.fillers.expand(len(sentences), -1, -1).clone()
            wanted = [(i, j) for i, x in enumerate(sentences) for j in board_relevant(x)]
            rows = self.reader.features([(sentences[i], THINKER_HYPOTHESES[j]) for i, j in wanted])
            features[[i for i, _ in wanted], [j for _, j in wanted]] = rows
        board.write("reading.board", {"features": features, "index": posted}, "reader")

    # --- trust part (+ curiosity): speakers + memory -> each base door's key -------------------------
    def trust_turn(self, board: Blackboard, feedback: dict, query: torch.Tensor, curious: bool) -> None:
        count = query.numel()
        if not board.has("reading.trust"):  # nobody spoke anywhere in this batch
            board.write("doors.conclusions", torch.zeros(count, 3, 3, device=self.device), "trust")
            board.write("doors.available", torch.zeros(count, 3, dtype=torch.bool, device=self.device), "trust")
            board.write("curiosity.asked", torch.full((count,), -1, device=self.device), "curiosity")
            return
        reading = board.read("reading.trust")
        table = torch.tanh(self.trust.head(reading["features"].to(self.device)))
        table = torch.cat([table, table.new_zeros(1, table.shape[1])])
        index, speakers = reading["index"].to(self.device), reading["speakers"].to(self.device)
        answers, has_answer = reading["answers"].to(self.device), reading["has_answer"].to(self.device)
        judge = self.trust.judge
        batch, episodes = speakers.shape[:2]
        query = query.view(batch, episodes)
        memory = judge.initial_memory(batch)
        rows = torch.arange(batch, device=self.device)
        out, asked_speaker = [], []
        for t in range(episodes):  # the loop of CompositeSystem.trust_logits, plus asking (asking_system.run_policy)
            reads = [judge.read(table[index[:, t, d]], speakers[:, t]) for d in range(3)]
            h = torch.stack([h for h, _ in reads], 1)  # [B,3 doors,S,3 keys,W]
            spoke = reads[0][1]
            target = torch.full((batch,), -1, dtype=torch.long, device=self.device)
            if curious:
                base = query[:, t] < 3  # speakers can only be asked about a base door
                door = query[:, t].clamp(max=2)
                h_q = h[rows, door]
                before = judge.logits(h_q, spoke, memory)
                predicted = self.curiosity(CuriosityHead.features(memory, h_q, spoke, before))
                best_gain, best = predicted.max(-1)
                target = torch.where(base & (best_gain > ASK_COST), best, target)
                # The asked speaker's answer is one more sentence they said: one more reading step for
                # that speaker, for every base door (nothing new if they already spoke about it).
                pick = target.clamp(min=0)
                new = (target >= 0) & has_answer[rows, t, pick]
                for d in range(3):
                    x = judge.embed(table[answers[rows, t, pick, d]])  # [B,3 keys,W]
                    hd = h[rows, d, pick]
                    stepped = judge.reader(x.reshape(-1, x.shape[-1]), hd.reshape(-1, hd.shape[-1])).view_as(hd)
                    h[rows, d, pick] = torch.where(new[:, None, None], stepped, hd)
                spoke = spoke.clone()
                spoke[rows, pick] = torch.where(new, torch.ones_like(spoke[rows, pick]), spoke[rows, pick])
            asked_speaker.append(target)
            out.append(torch.stack([judge.logits(h[:, d], spoke, memory) for d in range(3)], 1))
            memory = judge.update(memory, h[rows, feedback["anchor"][:, t]], spoke, feedback["truths"][:, t])
        board.write("doors.conclusions", torch.stack(out, 1).flatten(0, 1), "trust")
        board.write("doors.available", (speakers >= 0).any(-1).flatten()[:, None].expand(-1, 3), "trust")
        board.write("curiosity.asked", torch.stack(asked_speaker, 1).flatten(), "curiosity")

    # --- thinker: board (+ facts from other parts) -> answer ---------------------------------------
    def think(self, board: Blackboard, query: torch.Tensor, facts: dict | None = None, replace: bool = False):
        reading = board.read("reading.board")
        kept = self.thinker.features
        self.thinker.features = reading["features"].to(self.device)
        try:
            extra = {} if facts is None else {"extra_facts": facts["messages"], "extra_mask": facts["mask"],
                                              "extra_replace": replace}
            logits, _, strength, doors = self.thinker.think(reading["index"].flatten(0, 1).to(self.device), query, STEPS,
                                                            with_links=True, **extra)
        finally:
            self.thinker.features = kept
        stop = fixed_point_stop(self.thinker.last_changes)
        rows = torch.arange(len(query), device=self.device)
        return {"answer": logits[stop, rows], "doors": doors[stop, rows].softmax(-1), "strength": strength}

    # --- router: what is missing -> which base doors to ask -----------------------------------------
    def router_features(self, first: dict, query: torch.Tensor) -> torch.Tensor:
        """Per base door [N,3,3]: [linked to the queried door, sureness of its key, sureness of the answer]."""
        count, doors = len(query), len(chains.DOORS)
        adjacency = torch.zeros(count, doors, doors, device=self.device)
        adjacency[:, self.thinker.pair_rows, self.thinker.pair_cols] = (first["strength"] > 0.5).float()
        adjacency = ((adjacency + adjacency.transpose(1, 2)) > 0).float() + torch.eye(doors, device=self.device)
        reach = nn.functional.one_hot(query, doors).float()
        for _ in range(doors):
            reach = ((reach[:, None, :] @ adjacency).squeeze(1) > 0).float()
        return torch.stack([reach[:, :3], first["doors"][:, :3].amax(-1),
                            first["answer"].softmax(-1).amax(-1, keepdim=True).expand(-1, 3)], -1)

    def route(self, board: Blackboard, query: torch.Tensor, gate: bool = True) -> None:
        asked = self.router(self.router_features(board.read("thinker.first"), query)) > COST
        if gate:
            asked = asked & board.read("doors.available")
        board.write("router.asked", asked, "router")

    # --- adapter: trust conclusions for the asked doors -> thinker fact messages --------------------
    def transmit(self, board: Blackboard, mask: torch.Tensor | None = None) -> None:
        q = board.read("doors.conclusions").softmax(-1)
        asked = board.read("router.asked") if mask is None else mask
        sure = q.amax(-1, keepdim=True) > 0.5
        crisp = torch.where(sure, nn.functional.one_hot(q.argmax(-1), 3).float(), torch.full_like(q, 1 / 3))
        messages = torch.tanh(self.composite.adapter(torch.stack([crisp, crisp.amax(-1, keepdim=True).expand_as(crisp)], -1)))
        extra = messages.new_zeros(len(q), len(chains.DOORS), 3, messages.shape[-1])
        mask = torch.zeros(len(q), len(chains.DOORS), dtype=torch.bool, device=self.device)
        extra[:, :3] = messages * asked[..., None, None]
        mask[:, :3] = asked
        board.write("facts.from_trust", {"messages": extra, "mask": mask}, "adapter")

    @torch.no_grad()
    def solve(self, sessions: list[dict], policy: str = "route", curious: bool = False,
              gate: bool = True) -> tuple[dict, Blackboard]:
        """One batch of sessions with the same number of episodes. Answer logits [N,3] (N = sessions x
        episodes, in order), router asks [N], speaker asks [N], targets [N] (-1 = 모름)."""
        if policy not in POLICIES:
            raise ValueError(f"policy {policy!r} not in {POLICIES}")
        board = Blackboard()
        flat = [e for s in sessions for e in s["episodes"]]
        count, episodes = len(flat), len(sessions[0]["episodes"])
        query = torch.tensor([chains.DOORS.index(e["query"]) for e in flat], device=self.device)
        targets = torch.tensor([e["keys"].index(e["answer"]) if e["answer"] in e["keys"] else -1 for e in flat],
                               device=self.device)
        self.read(board, sessions, curious)
        if policy != "think" and sessions[0].get("linked"):
            self.article_doors(board, sessions)
        elif policy != "think":
            anchors = [BASE_DOORS.index(e["anchor"]) if e["anchor"] is not None else 0 for e in flat]
            feedback = {"truths": targets.clamp(min=0).view(len(sessions), episodes),
                        "anchor": torch.tensor(anchors, device=self.device).view(len(sessions), episodes)}
            self.trust_turn(board, feedback, query, curious)
        speaker_asks = ((board.read("curiosity.asked") >= 0).float() if board.has("curiosity.asked")
                        else torch.zeros(count, device=self.device))
        if policy == "trust":  # the trust part alone answers about the speakers' door (the v2-H question)
            answer = board.read("doors.conclusions")[torch.arange(count, device=self.device), feedback["anchor"].flatten()]
            board.write("answer.logits", answer, "trust")
            self.reader.save()
            return {"logits": answer, "asks": torch.zeros(count, device=self.device), "speaker_asks": speaker_asks,
                    "targets": targets}, board
        if policy == "think":
            board.write("router.asked", torch.zeros(count, 3, dtype=torch.bool, device=self.device), "rule")
        elif policy == "ask-all":
            board.write("router.asked", torch.ones(count, 3, dtype=torch.bool, device=self.device), "rule")
        elif policy == "ask-available":  # every door some part has something on
            board.write("router.asked", board.read("doors.available").clone(), "rule")
        elif policy.startswith("verify-"):  # stage G: ask to verify, decided per question in order
            board.write("thinker.first", self.think(board, query), "thinker")
            self.verify_loop(board, sessions, query, policy.removeprefix("verify-"))
        elif policy in ("article-only", "arbitrate"):  # stage F: the article door's fact is replaced
            if policy == "arbitrate":
                board.write("thinker.first", self.think(board, query), "thinker")
                self.arbitrate(board, sessions)
            board.write("router.asked", board.read("doors.available").clone(), "rule")
        else:
            board.write("thinker.first", self.think(board, query), "thinker")
            self.route(board, query, gate)
        asked = board.read("router.asked")
        facts = None
        verify = policy.startswith("verify-")
        fact_doors = board.read("doors.available") if verify else asked  # verify: the arbiter's word always replaces
        if fact_doors.any():
            self.transmit(board, fact_doors)
            facts = board.read("facts.from_trust")
        replace = policy in ("article-only", "arbitrate") or verify
        board.write("answer.logits", self.think(board, query, facts, replace)["answer"], "thinker")
        self.reader.save()
        return {"logits": board.read("answer.logits"), "asks": asked.float().sum(1), "speaker_asks": speaker_asks,
                "targets": targets}, board

    def article_doors(self, board: Blackboard, sessions: list[dict]) -> None:
        """Linked sessions: the article part's conclusion about the article door, as a key (via the labels)."""
        articles = [[e["article"] for e in s["episodes"]] for s in sessions]
        pairs = [(a["question"], r["passage"]) for row in articles for a in row for r in a["reports"]]
        flat = iter(self.articles.read(pairs))
        readings = [[[next(flat) for _ in a["reports"]] for a in row] for row in articles]
        board.write("reading.articles", readings, "reader-qa")
        data, candidates = trust_inputs(articles, readings, real_trust.SLOTS, real_trust.world.SPEAKERS)
        choice = real_trust.choices(self.trust_real({k: v.to(self.device) for k, v in data.items()})).flatten().tolist()
        episodes = [e for s in sessions for e in s["episodes"]]
        conclusions = torch.zeros(len(episodes), 3, 3, device=self.device)  # uniform = 모름
        available = torch.zeros(len(episodes), 3, dtype=torch.bool, device=self.device)
        for n, (e, found, k) in enumerate(zip(episodes, (c for row in candidates for c in row), choice)):
            door = BASE_DOORS.index(e["article"]["door"])
            available[n, door] = True
            if k >= 0:
                labels = [klue.normalize(e["legend"][key]) for key in e["keys"]]
                if klue.normalize(found[k]) in labels:  # the chosen answer names a key
                    conclusions[n, door, labels.index(klue.normalize(found[k]))] = 20.0
        board.write("doors.conclusions", conclusions, "trust-real")
        board.write("doors.available", available, "trust-real")

    def channel_says(self, board: Blackboard, sessions: list[dict]) -> dict:
        """Per article door: what the board (thinker's first pass, if sure) and the article part (its
        conclusion) say, as one-hot keys [B,E,2,3] (zeros = nothing); truth [B,E] (revealed after each question)."""
        episodes = [e for s in sessions for e in s["episodes"]]
        rows = torch.arange(len(episodes), device=self.device)
        door = torch.tensor([BASE_DOORS.index(e["article"]["door"]) for e in episodes], device=self.device)
        belief = board.read("thinker.first")["doors"][rows, door]  # [N,3]
        board_says = nn.functional.one_hot(belief.argmax(-1), 3).float() * (belief.amax(-1, keepdim=True) > 0.5)
        article = board.read("doors.conclusions")[rows, door]
        article_says = nn.functional.one_hot(article.argmax(-1), 3).float() * (article.amax(-1, keepdim=True) > 0)
        shape = (len(sessions), len(sessions[0]["episodes"]))
        truth = torch.tensor([e["keys"].index(e["answer"]) for e in episodes], device=self.device)
        return {"says": torch.stack([board_says, article_says], 1).view(*shape, 2, 3), "truth": truth.view(shape),
                "door": door}

    def arbitrate(self, board: Blackboard, sessions: list[dict]) -> None:
        """The arbiter's conclusion about each article door replaces both channels' (as doors.conclusions)."""
        views = self.channel_says(board, sessions)
        logits = self.arbiter(views["says"], views["truth"]).flatten(0, 1)  # truth = feedback after each question
        board.write("arbiter.logits", logits, "arbiter")
        p = logits.softmax(-1)[:, :3]
        best, pick = p.max(-1)
        conclusions = torch.zeros(len(p), 3, 3, device=self.device)
        rows = torch.arange(len(p), device=self.device)
        conclusions[rows, views["door"]] = nn.functional.one_hot(pick, 3).float() * 20.0 * (best > 0.5)[:, None]
        board.write("doors.conclusions", conclusions, "arbiter")

    def verify_loop(self, board: Blackboard, sessions: list[dict], query: torch.Tensor, mode: str) -> None:
        """Per question, in order: the arbiter's view without asking -> decide whether to ask the article part
        -> the arbiter's conclusion from what was heard -> after the answer, remember what each channel said."""
        views = self.channel_says(board, sessions)
        says, truth, door = views["says"], views["truth"], views["door"]
        batch, episodes = truth.shape
        rows = torch.arange(len(query), device=self.device)
        features = self.router_features(board.read("thinker.first"), query)[rows, door].view(batch, episodes, 3)
        memory = self.arbiter.initial_memory(batch)
        logits_out, asked_out, features_out = [], [], []
        for t in range(episodes):
            said = says[:, t]
            board_only = said.clone()
            board_only[:, 1] = 0
            without, withit = self.arbiter.step(board_only, memory), self.arbiter.step(said, memory)
            f = features[:, t].clone()
            f[:, 1] = without.softmax(-1)[:, :3].amax(-1)  # how sure MK1 is of the door without asking
            features_out.append(f)
            if mode == "never":
                ask = torch.zeros(batch, dtype=torch.bool, device=self.device)
            elif mode == "always":
                ask = torch.ones(batch, dtype=torch.bool, device=self.device)
            elif mode == "router":
                ask = (self.verifier if self.verifier is not None else self.router)(f) > COST
            else:  # hindsight: ask when it helps by more than its cost (needs the answer)
                ask = door_score(withit, truth[:, t]) - door_score(without, truth[:, t]) > COST
            logits_out.append(torch.where(ask[:, None], withit, without))
            asked_out.append(ask)
            memory = self.arbiter.update(memory, torch.where(ask[:, None, None], said, board_only), truth[:, t])
        logits = torch.stack(logits_out, 1).flatten(0, 1)
        board.write("arbiter.logits", logits, "arbiter")
        board.write("verify.features", torch.stack(features_out, 1).flatten(0, 1), "router")
        p = logits.softmax(-1)[:, :3]
        best, pick = p.max(-1)
        conclusions = torch.zeros(len(p), 3, 3, device=self.device)
        conclusions[rows, door] = nn.functional.one_hot(pick, 3).float() * 20.0 * (best > 0.5)[:, None]
        board.write("doors.conclusions", conclusions, "arbiter")
        asked = torch.zeros(len(p), 3, dtype=torch.bool, device=self.device)
        asked[rows, door] = torch.stack(asked_out, 1).flatten()
        board.write("router.asked", asked, "router")

    @torch.no_grad()
    def channel_views(self, sessions: list[dict], chunk: int = 100) -> dict:
        """Arbiter training inputs for linked sessions, computed in chunks of sessions."""
        parts = []
        for start in range(0, len(sessions), chunk):
            group = sessions[start:start + chunk]
            board = Blackboard()
            query = torch.tensor([chains.DOORS.index(e["query"]) for s in group for e in s["episodes"]], device=self.device)
            self.read(board, group, False)
            self.article_doors(board, group)
            board.write("thinker.first", self.think(board, query), "thinker")
            parts.append(self.channel_says(board, group))
        self.reader.save()
        return {k: torch.cat([p[k] for p in parts]) for k in ("says", "truth")}

    @torch.no_grad()
    def solve_articles(self, sessions: list[dict]) -> tuple[dict, Blackboard]:
        """Article sessions (same number of questions each): answers (candidate text or 모름) per question."""
        board = Blackboard()
        episodes = [s["episodes"] for s in sessions]
        pairs = [(e["question"], r["passage"]) for row in episodes for e in row for r in e["reports"]]
        flat = iter(self.articles.read(pairs))
        readings = [[[next(flat) for _ in e["reports"]] for e in row] for row in episodes]
        board.write("reading.articles", readings, "reader-qa")
        data, candidates = trust_inputs(episodes, readings, real_trust.SLOTS, real_trust.world.SPEAKERS)
        logits = self.trust_real({k: v.to(self.device) for k, v in data.items()})  # right = feedback after each question
        board.write("trust_real.logits", logits, "trust-real")
        choice = real_trust.choices(logits).flatten().tolist()
        answers = [c[k] if k >= 0 else "모름" for c, k in zip((c for row in candidates for c in row), choice)]
        board.write("answer.text", answers, "trust-real")
        count = len(answers)
        zeros = torch.zeros(count, device=self.device)
        return {"answer": answers, "asks": zeros, "speaker_asks": zeros.clone()}, board

    def forward(self, sessions: list[dict], policy: str = "route", curious: bool = False, gate: bool = True):
        """Any mix of sessions: grouped (articles / rule-world sessions by episode count), solved, and returned in
        the original order. Every episode gets an answer text (a key, a candidate answer, or 모름); when every
        session is a rule-world one, the answer logits and targets are returned too."""
        groups: dict[tuple, list[int]] = {}
        for i, s in enumerate(sessions):
            kind = "articles" if "sources" in s and not s.get("linked") else "rule"
            groups.setdefault((kind, len(s["episodes"])), []).append(i)
        parts, boards, offsets, position = [], [], {}, 0
        for (kind, _), members in groups.items():
            chosen = [sessions[i] for i in members]
            if kind == "articles":
                out, board = self.solve_articles(chosen)
            else:
                out, board = self.solve(chosen, policy, curious, gate)
                keys = [e["keys"] for s in chosen for e in s["episodes"]]
                q = out["logits"].softmax(-1)
                best, pick = q.max(-1)
                out["answer"] = [k[i] if b > 0.5 else "모름" for k, i, b in zip(keys, pick.tolist(), best.tolist())]
            parts.append(out)
            boards.append(board)
            for i in members:  # where session i's episodes landed in the concatenated output
                offsets[i] = position
                position += len(sessions[i]["episodes"])
        index = torch.cat([torch.arange(offsets[i], offsets[i] + len(sessions[i]["episodes"]))
                           for i in range(len(sessions))]).to(self.device)
        shared = set.intersection(*(set(p) for p in parts))
        merged = {}
        for k in shared:
            if k == "answer":
                flat = [a for p in parts for a in p["answer"]]
                merged[k] = [flat[i] for i in index.tolist()]
            else:
                merged[k] = torch.cat([p[k] for p in parts])[index]
        return merged, boards


def episode_score(episode: dict, answer: str) -> float:
    """The world's score (+1 right, 0 for 모름 when an answer exists, -1 wrong; a right 모름 is +1)."""
    if episode.get("tag") == "article":
        return 0.0 if answer == "모름" else (1.0 if klue.is_right(answer, episode["answers"]) else -1.0)
    if answer == "모름":
        return 0.0 if episode["answer"] in episode["keys"] else 1.0
    return 1.0 if answer == episode["answer"] else -1.0
