"""Judges for rule world v2: the Bayes oracle and code-based learners.

Every agent reads a speaker's *final* statement about the queried door (a later
statement by the same speaker replaces an earlier one) and judges from those claims.
Claims come from gold events or from a language module's parsed events.
"""

from cognitive_lab.world.templates import UNKNOWN

PRIOR_ACCURACY = 0.6  # mean of the accuracy levels; used before any evidence


def final_claims(events: list[dict], door: str, keys: list[str]) -> list[tuple[str, int, bool]]:
    """(speaker, key index, polarity) for each speaker's latest statement about `door`."""
    latest: dict[str, tuple[int, bool]] = {}
    for event in events:
        if event["type"] == "chatter" or event["door"] != door or event["key"] not in keys:
            continue
        latest[event["source"]] = (keys.index(event["key"]), event["polarity"])
    return [(speaker, key, polarity) for speaker, (key, polarity) in latest.items()]


def posterior(claims: list[tuple[str, int, bool]], accuracy: dict[str, float]) -> list[float]:
    """Exact posterior over the true key under the world's statement model."""
    weights = [1.0, 1.0, 1.0]
    for speaker, key, polarity in claims:
        p = min(max(accuracy[speaker], 1e-3), 1 - 1e-3)
        for truth in range(3):
            if polarity:
                weights[truth] *= p if key == truth else (1 - p) / 2
            else:
                weights[truth] *= (1 - p) if key == truth else p / 2
    total = sum(weights)
    return [w / total for w in weights]


def decide(probabilities: list[float], key_names: list[str]) -> str:
    """Answer only when it beats abstaining in expectation (+1/-1 vs 0)."""
    best = max(range(3), key=lambda k: probabilities[k])
    return key_names[best] if probabilities[best] > 0.5 else UNKNOWN


class BaseJudge:
    """Reads claims from `events_of(episode)`; gold events unless a reader is plugged in."""

    gold_session = False

    def __init__(self, reader=None):
        self.reader = reader

    def events_of(self, episode: dict) -> tuple[list[dict], str]:
        if self.reader is None:
            return episode["gold_events"], episode["gold_question"]["door"]
        return self.reader(episode)

    def cost_report(self) -> dict:
        return {"reader": self.reader.cost_report() if self.reader is not None else "gold events"}

    def start_session(self, meta: dict) -> None:
        self.speakers = list(meta["speakers"])

    def claims(self, episode: dict) -> tuple[list[tuple[str, int, bool]], list[str]]:
        events, door = self.events_of(episode)
        keys = [k.removesuffix(" 열쇠") for k in episode["entities"]["keys"]]
        return final_claims(events, door, keys), episode["entities"]["keys"]

    def feedback(self, episode: dict, answer: str) -> None:
        pass


class NonLearningJudge(BaseJudge):
    """Judges that never learn ignore either kind of feedback."""

    def feedback_partial(self, episode: dict, answered: str | None, correct: bool | None) -> None:
        pass


class OracleJudge(NonLearningJudge):
    """O2: Bayes with the speakers' true (hidden) accuracies, including drift."""

    gold_session = True

    def start_session(self, meta: dict) -> None:
        super().start_session(meta)
        self.hidden = meta["hidden"]

    def act(self, episode: dict) -> str:
        accuracies = list(self.hidden["accuracies"])
        drift = self.hidden["drift"]
        if drift and episode["index"] >= drift["at"]:
            accuracies[drift["speaker"]] = drift["to"]
        claims, names = self.claims(episode)
        return decide(posterior(claims, dict(zip(self.speakers, accuracies))), names)


class MajorityJudge(NonLearningJudge):
    """B0: most positively named key among final claims; no learning."""

    def act(self, episode: dict) -> str:
        claims, names = self.claims(episode)
        votes = [0, 0, 0]
        for _, key, polarity in claims:
            if polarity:
                votes[key] += 1
        top = max(votes)
        winners = [k for k in range(3) if votes[k] == top]
        return names[winners[0]] if top > 0 and len(winners) == 1 else UNKNOWN


class EqualTrustJudge(NonLearningJudge):
    """B1: Bayes assuming every speaker has the population-mean accuracy; no learning."""

    def act(self, episode: dict) -> str:
        claims, names = self.claims(episode)
        return decide(posterior(claims, {s: PRIOR_ACCURACY for s in self.speakers}), names)


class LearnedTrustJudge(BaseJudge):
    """B2: Bayes with per-speaker accuracy estimated from feedback.

    Each speaker's accuracy has a Beta prior with mean PRIOR_ACCURACY and strength
    `prior_strength`. After feedback, each final claim about the queried door counts
    as correct or wrong. With `decay` < 1, older evidence fades (B2-forget).
    """

    def __init__(self, reader=None, prior_strength: float = 2.0, decay: float = 1.0):
        super().__init__(reader)
        self.prior_strength = prior_strength
        self.decay = decay

    def start_session(self, meta: dict) -> None:
        super().start_session(meta)
        self.correct = {s: 0.0 for s in self.speakers}
        self.wrong = {s: 0.0 for s in self.speakers}

    def estimate(self) -> dict[str, float]:
        a0 = PRIOR_ACCURACY * self.prior_strength
        b0 = (1 - PRIOR_ACCURACY) * self.prior_strength
        return {s: (a0 + self.correct[s]) / (a0 + b0 + self.correct[s] + self.wrong[s]) for s in self.speakers}

    def act(self, episode: dict) -> str:
        claims, names = self.claims(episode)
        self._pending = claims
        return decide(posterior(claims, self.estimate()), names)

    def feedback(self, episode: dict, answer: str) -> None:
        truth = episode["entities"]["keys"].index(answer)
        for s in self.speakers:
            self.correct[s] *= self.decay
            self.wrong[s] *= self.decay
        for speaker, key, polarity in self._pending:
            if speaker not in self.correct:
                continue
            right = (key == truth) if polarity else (key != truth)
            if right:
                self.correct[speaker] += 1
            else:
                self.wrong[speaker] += 1


class NLIReader:
    """Language module from rule world v1 (fine-tuned mDeBERTa), reused unchanged.

    Turns an episode's text into solver-format events. Speakers are passed through as
    metadata; the queried door is matched by name in the question.
    """

    def __init__(self, checkpoint, device: str):
        from cognitive_lab.world.nli_parser import DEFAULT_NLI_DIR, P2NLIAgent

        self.agent = P2NLIAgent(device, DEFAULT_NLI_DIR, checkpoint, abstain=False)

    def __call__(self, episode: dict) -> tuple[list[dict], str | None]:
        from cognitive_lab.world.parser_llm import parse_episode

        events, _, _ = parse_episode(self.agent.module, episode)
        asked = [d for d in episode["entities"]["doors"] if d in episode["question"]]
        return events, asked[0].removesuffix(" 문") if len(asked) == 1 else None

    def cost_report(self) -> dict:
        report = self.agent.cost_report()
        return {k: report[k] for k in ("language_module", "checkpoint", "language_module_calls",
                                        "language_module_cache_hits", "peak_vram_gib")}


AGENTS = {
    "o2": lambda args: OracleJudge(),
    "b0-majority": lambda args: MajorityJudge(),
    "b1-equal": lambda args: EqualTrustJudge(),
    "b2-learn": lambda args: LearnedTrustJudge(),
    "b2-forget": lambda args: LearnedTrustJudge(decay=0.9),
}
