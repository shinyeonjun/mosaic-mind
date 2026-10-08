"""Judges for partial feedback: after answering, the agent only hears right/wrong; 모름 hears nothing.

B4 (exact joint Bayes). With partial feedback the per-speaker count statistics of B3
are no longer sufficient: "wrong" leaves two possible truths, which couples speakers.
The world is small enough to track the exact joint posterior over all 4^4 = 256
accuracy-level assignments, under the generator's true process:
- prior: levels i.i.d. uniform, excluding the 4 all-equal assignments;
- drift: before episode DRIFT_AT, with probability DRIFT_SHARE one uniformly chosen
  speaker moves to a uniformly chosen level at least MIN_DRIFT away.
For each episode, L[z, k] = prod_s P(claim_s | T = k, level_s(z)). Updates:
    truth known (full feedback, or "right" on answer k):  b(z) *= L[z, T]
    "wrong" on answer k:                                    b(z) *= sum_{T != k} L[z, T]
    abstained (no feedback):                                b(z) *= sum_T L[z, T]
The last line matters: agreement patterns carry information even without feedback.
The prediction is q_k ∝ sum_z b(z) L[z, k].

Cost: the joint state space has 4^S entries for S speakers (256 at S=4, ~1M at S=10).
B4 refuses S > MAX_EXACT_SPEAKERS; drift is applied axis-by-axis on the [4]*S tensor
instead of as a 4^S x 4^S matrix.

B5 (assumed-density filtering) is the best *scalable* code approximation: it keeps one
level distribution per speaker and, after each episode, sets each speaker's belief to
the exact marginal of the joint update started from the factorized belief:
    r_{-s}(T) ∝ fb(T) * prod_{u != s} sum_l b_u(l) P(claim_u | T, l)
    b_s(l)   ∝ b_s(l) * sum_T r_{-s}(T) P(claim_s | T, l)
where fb(T) encodes the feedback (1[T = k] if right, 1[T != k] if wrong, 1 if abstained).
It is exact whenever the truth is revealed and costs O(S) per episode.

The *belief* is exact; the *policy* is not. Answering only when max q > 0.5 is optimal
for the current episode but ignores the information an answer would buy. The
`explore_until`/`explore_threshold` variant answers more readily early in a session.
Choosing that trade-off optimally is a POMDP; these variants are tuned heuristics.
"""

import numpy as np

from cognitive_lab.world.templates import UNKNOWN
from cognitive_lab.world2.agents import BaseJudge, PRIOR_ACCURACY
from cognitive_lab.world2.generator import ACCURACY_LEVELS, DRIFT_AT, DRIFT_SHARE, MIN_DRIFT

LEVELS = np.array(ACCURACY_LEVELS)
N_LEVELS = len(LEVELS)
MAX_EXACT_SPEAKERS = 10


def _move_matrix() -> np.ndarray:
    """M[i, j] = probability that a drifting speaker at level i moves to level j."""
    move = np.zeros((N_LEVELS, N_LEVELS))
    for i in range(N_LEVELS):
        options = [j for j in range(N_LEVELS) if abs(LEVELS[j] - LEVELS[i]) >= MIN_DRIFT]
        move[i, options] = 1 / len(options)
    return move


MOVE = _move_matrix()
_SPACE_CACHE: dict[int, tuple[np.ndarray, np.ndarray]] = {}


def joint_space(speakers: int) -> tuple[np.ndarray, np.ndarray]:
    """(level-index table [4^S, S], prior [4^S]) for S speakers; prior excludes all-equal states."""
    if speakers not in _SPACE_CACHE:
        grids = np.indices((N_LEVELS,) * speakers).reshape(speakers, -1).T  # [4^S, S]
        prior = np.ones(len(grids))
        prior[(grids == grids[:, :1]).all(axis=1)] = 0.0
        _SPACE_CACHE[speakers] = (grids.astype(np.int8), prior / prior.sum())
    return _SPACE_CACHE[speakers]


def apply_drift(belief: np.ndarray, speakers: int) -> np.ndarray:
    """With probability DRIFT_SHARE one uniformly chosen speaker moves level by MOVE."""
    tensor = belief.reshape((N_LEVELS,) * speakers)
    moved = sum(np.moveaxis(np.tensordot(tensor, MOVE, axes=([axis], [0])), -1, axis) for axis in range(speakers))
    return (1 - DRIFT_SHARE) * belief + DRIFT_SHARE / speakers * moved.reshape(-1)


def claim_table(key: int, polarity: bool) -> np.ndarray:
    """[N_LEVELS, 3]: P(claim | T = k, level)."""
    level = LEVELS[:, None]
    match = (np.arange(3) == key)[None, :]
    if polarity:
        return np.where(match, level, (1 - level) / 2)
    return np.where(match, 1 - level, level / 2)


def feedback_mask(answered: int | None, correct: bool | None) -> np.ndarray:
    if answered is None:
        return np.ones(3)
    mask = (np.arange(3) == answered).astype(float)
    return mask if correct else 1 - mask


class JointBayesJudge(BaseJudge):
    """B4: exact joint posterior over speaker accuracy levels; works with full or partial feedback."""

    def __init__(self, reader=None, explore_until: int = 0, explore_threshold: float = 0.5):
        super().__init__(reader)
        self.explore_until = explore_until
        self.explore_threshold = explore_threshold

    def start_session(self, meta: dict) -> None:
        super().start_session(meta)
        if len(self.speakers) > MAX_EXACT_SPEAKERS:
            raise RuntimeError(f"exact joint Bayes over 4^{len(self.speakers)} states is infeasible "
                               f"(limit {MAX_EXACT_SPEAKERS} speakers); use b5-adf or n2")
        self.index = {name: s for s, name in enumerate(self.speakers)}
        self.states, prior = joint_space(len(self.speakers))
        self.belief = prior.copy()

    def _likelihood(self, claims) -> np.ndarray:
        table = np.ones((len(self.states), 3))
        for speaker, key, polarity in claims:
            if speaker in self.index:
                table *= claim_table(key, polarity)[self.states[:, self.index[speaker]]]
        return table

    def predictive(self) -> np.ndarray:
        q = self.belief @ self._L
        return q / q.sum()

    def act(self, episode: dict) -> str:
        if episode["index"] == DRIFT_AT:
            self.belief = apply_drift(self.belief, len(self.speakers))
        claims, names = self.claims(episode)
        self._L = self._likelihood(claims)
        q = self.predictive()
        best = int(q.argmax())
        threshold = self.explore_threshold if episode["index"] < self.explore_until else 0.5
        return names[best] if claims and q[best] > threshold else UNKNOWN

    def _update(self, factor: np.ndarray) -> None:
        self.belief = self.belief * factor
        self.belief /= self.belief.sum()

    def feedback(self, episode: dict, answer: str) -> None:
        self._update(self._L[:, episode["entities"]["keys"].index(answer)])

    def feedback_partial(self, episode: dict, answered: str | None, correct: bool | None) -> None:
        if answered is None:
            self._update(self._L.sum(axis=1))
            return
        key = episode["entities"]["keys"].index(answered)
        if correct:
            self._update(self._L[:, key])
        else:
            self._update(self._L.sum(axis=1) - self._L[:, key])


class ADFJudge(BaseJudge):
    """B5: factorized per-speaker level beliefs with assumed-density filtering (scales as O(S))."""

    def __init__(self, reader=None, explore_until: int = 0, explore_threshold: float = 0.5):
        super().__init__(reader)
        self.explore_until = explore_until
        self.explore_threshold = explore_threshold

    def start_session(self, meta: dict) -> None:
        super().start_session(meta)
        self.belief = {s: np.full(N_LEVELS, 1 / N_LEVELS) for s in self.speakers}

    def _messages(self, claims) -> dict[str, np.ndarray]:
        """Per claiming speaker: [N_LEVELS, 3] table and its marginal message over T."""
        return {speaker: claim_table(key, polarity) for speaker, key, polarity in claims if speaker in self.belief}

    def act(self, episode: dict) -> str:
        if episode["index"] == DRIFT_AT:
            share = DRIFT_SHARE / len(self.speakers)
            self.belief = {s: (1 - share) * b + share * (b @ MOVE) for s, b in self.belief.items()}
        claims, names = self.claims(episode)
        self._tables = self._messages(claims)
        q = np.ones(3)
        for speaker, table in self._tables.items():
            q *= self.belief[speaker] @ table
        q /= q.sum()
        best = int(q.argmax())
        threshold = self.explore_threshold if episode["index"] < self.explore_until else 0.5
        return names[best] if self._tables and q[best] > threshold else UNKNOWN

    def _update(self, mask: np.ndarray) -> None:
        messages = {s: self.belief[s] @ table for s, table in self._tables.items()}
        updated = {}
        for speaker, table in self._tables.items():
            others = mask.copy()
            for other, message in messages.items():
                if other != speaker:
                    others = others * message
            belief = self.belief[speaker] * (table @ others)
            updated[speaker] = belief / belief.sum()
        self.belief.update(updated)

    def feedback(self, episode: dict, answer: str) -> None:
        truth = episode["entities"]["keys"].index(answer)
        self._update((np.arange(3) == truth).astype(float))

    def feedback_partial(self, episode: dict, answered: str | None, correct: bool | None) -> None:
        key = None if answered is None else episode["entities"]["keys"].index(answered)
        self._update(feedback_mask(key, correct))


class CountPartialJudge(BaseJudge):
    """B2-partial: per-speaker right/wrong counts using only what feedback reveals.

    "right" on answer k reveals T = k, so every claim is scored. "wrong" on k scores only
    claims about k (a positive claim on k was wrong, a negative claim on k was right).
    Abstaining reveals nothing. Prediction is Bayes with the plug-in mean accuracy.
    """

    def __init__(self, reader=None, prior_strength: float = 2.0, explore_until: int = 0,
                 explore_threshold: float = 0.5):
        super().__init__(reader)
        self.prior_strength = prior_strength
        self.explore_until = explore_until
        self.explore_threshold = explore_threshold

    def start_session(self, meta: dict) -> None:
        super().start_session(meta)
        self.counts = {s: [0.0, 0.0] for s in self.speakers}

    def estimate(self) -> dict[str, float]:
        a0, b0 = PRIOR_ACCURACY * self.prior_strength, (1 - PRIOR_ACCURACY) * self.prior_strength
        return {s: (a0 + r) / (a0 + b0 + r + w) for s, (r, w) in self.counts.items()}

    def act(self, episode: dict) -> str:
        from cognitive_lab.world2.agents import posterior

        claims, names = self.claims(episode)
        self._pending = claims
        q = posterior(claims, self.estimate())
        best = max(range(3), key=lambda k: q[k])
        threshold = self.explore_threshold if episode["index"] < self.explore_until else 0.5
        return names[best] if claims and q[best] > threshold else UNKNOWN

    def _score(self, speaker: str, right: bool) -> None:
        if speaker in self.counts:
            self.counts[speaker][0 if right else 1] += 1

    def feedback(self, episode: dict, answer: str) -> None:
        truth = episode["entities"]["keys"].index(answer)
        for speaker, key, polarity in self._pending:
            self._score(speaker, (key == truth) if polarity else (key != truth))

    def feedback_partial(self, episode: dict, answered: str | None, correct: bool | None) -> None:
        if answered is None:
            return
        key = episode["entities"]["keys"].index(answered)
        if correct:
            self.feedback(episode, answered)
            return
        for speaker, claimed, polarity in self._pending:
            if claimed == key:
                self._score(speaker, not polarity)
