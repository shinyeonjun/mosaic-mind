"""B3: the Bayes-optimal learner for rule world v2 (does not know accuracies, knows the world's prior).

Derivation (per speaker s, accuracy level l in L = {0.95, 0.75, 0.5, 0.2}, prior pi):

Statement likelihoods given the true key T:
    positive claim c:  P(c | T, l) = l          if c == T  else (1 - l) / 2
    negative claim n:  P(n | T, l) = l / 2      if n != T  else (1 - l)
Call a statement "right" if it matches the truth (c == T, or n != T). In every case
the likelihood is l * const if right and (1 - l) * const if wrong, with constants that
do not depend on l. Once feedback reveals T, the level posterior is therefore
    P(l | history_s) ∝ pi(l) * l^right_s * (1 - l)^wrong_s,
so (right, wrong) counts are sufficient statistics, and speakers stay independent
because each update uses only that speaker's own statements.

Prediction for the current episode marginalizes each speaker's level separately:
    P(T = k | claims, history) ∝ prod_s sum_l P(l | history_s) * P(claim_s | T = k, l).

Drift: before every episode each speaker's level is resampled from pi with hazard h,
    P_t(l) = (1 - h) * P_{t-1}(l) + h * pi(l),
which turns the count update into a forward filter. h = 0 recovers the static model.
The world's "not all four levels equal" constraint is ignored (a small approximation).

Decision: answer argmax_k P(T = k | ...) when it exceeds 0.5, else 모름. With scores
+1 / -1 / 0 this maximizes expected score: answering k is worth 2 q_k - 1.
"""

from cognitive_lab.world.templates import UNKNOWN
from cognitive_lab.world2.agents import BaseJudge, decide
from cognitive_lab.world2.generator import ACCURACY_LEVELS

PRIOR = [1 / len(ACCURACY_LEVELS)] * len(ACCURACY_LEVELS)


def claim_likelihood(key: int, polarity: bool, truth: int, level: float) -> float:
    if polarity:
        return level if key == truth else (1 - level) / 2
    return (1 - level) if key == truth else level / 2


class BayesLearnerJudge(BaseJudge):
    def __init__(self, reader=None, hazard: float = 0.0):
        super().__init__(reader)
        self.hazard = hazard

    def start_session(self, meta: dict) -> None:
        super().start_session(meta)
        self.belief = {s: list(PRIOR) for s in self.speakers}

    def predictive(self, claims: list[tuple[str, int, bool]]) -> list[float]:
        weights = [1.0, 1.0, 1.0]
        for speaker, key, polarity in claims:
            belief = self.belief.get(speaker)
            if belief is None:
                continue
            for truth in range(3):
                weights[truth] *= sum(
                    b * claim_likelihood(key, polarity, truth, level)
                    for b, level in zip(belief, ACCURACY_LEVELS)
                )
        total = sum(weights)
        return [w / total for w in weights]

    def act(self, episode: dict) -> str:
        if self.hazard:
            for speaker, belief in self.belief.items():
                self.belief[speaker] = [(1 - self.hazard) * b + self.hazard * p for b, p in zip(belief, PRIOR)]
        claims, names = self.claims(episode)
        self._pending = claims
        if not claims:
            return UNKNOWN
        return decide(self.predictive(claims), names)

    def feedback(self, episode: dict, answer: str) -> None:
        truth = episode["entities"]["keys"].index(answer)
        for speaker, key, polarity in self._pending:
            belief = self.belief.get(speaker)
            if belief is None:
                continue
            right = (key == truth) if polarity else (key != truth)
            updated = [b * (level if right else 1 - level) for b, level in zip(belief, ACCURACY_LEVELS)]
            total = sum(updated)
            self.belief[speaker] = [u / total for u in updated]

    def trust(self) -> dict[str, float]:
        """Posterior mean accuracy per speaker (for analysis)."""
        return {s: sum(b * l for b, l in zip(belief, ACCURACY_LEVELS)) for s, belief in self.belief.items()}


class InformedBayesJudge(BaseJudge):
    """B3-informed: B3 written by an engineer who anticipated two of rule world v2-M's deviations.

    - Separate level beliefs for each speaker's positive and negative statements
      (handles "polarity" speakers; costs evidence for everyone else).
    - A hazard applied before *every* episode (handles gradual "drifting" speakers).
    It still assumes independent speakers, so "copier" correlations are not modeled.
    """

    def __init__(self, reader=None, hazard: float = 0.05):
        super().__init__(reader)
        self.hazard = hazard

    def start_session(self, meta: dict) -> None:
        super().start_session(meta)
        self.belief = {(s, polarity): list(PRIOR) for s in self.speakers for polarity in (True, False)}

    def act(self, episode: dict) -> str:
        for key, belief in self.belief.items():
            self.belief[key] = [(1 - self.hazard) * b + self.hazard * p for b, p in zip(belief, PRIOR)]
        claims, names = self.claims(episode)
        self._pending = claims
        if not claims:
            return UNKNOWN
        weights = [1.0, 1.0, 1.0]
        for speaker, key, polarity in claims:
            belief = self.belief.get((speaker, polarity))
            if belief is None:
                continue
            for truth in range(3):
                weights[truth] *= sum(b * claim_likelihood(key, polarity, truth, level)
                                      for b, level in zip(belief, ACCURACY_LEVELS))
        total = sum(weights)
        return decide([w / total for w in weights], names)

    def feedback(self, episode: dict, answer: str) -> None:
        truth = episode["entities"]["keys"].index(answer)
        for speaker, key, polarity in self._pending:
            belief = self.belief.get((speaker, polarity))
            if belief is None:
                continue
            right = (key == truth) if polarity else (key != truth)
            updated = [b * (level if right else 1 - level) for b, level in zip(belief, ACCURACY_LEVELS)]
            total = sum(updated)
            self.belief[(speaker, polarity)] = [u / total for u in updated]

