"""Analysis helpers for rule world v2 (used by notebook/world2_learning_to_judge.ipynb).

All numbers come from saved results or deterministic replays; nothing here trains.
"""

import json
import math
import statistics
from pathlib import Path

import torch

from cognitive_lab.world2.bayes_learner import BayesLearnerJudge
from cognitive_lab.world2.generator import ACCURACY_LEVELS, accuracies_at
from cognitive_lab.world2.neural_judge import N2Model, checkpoint_path, session_tensors

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = PROJECT_ROOT / "artifacts" / "results"
DATA_DIR = PROJECT_ROOT / "artifacts" / "world-v2" / "data"
SEEDS = tuple(range(42, 48))
T_95_DF5 = 2.571  # two-sided 95% t quantile, 5 degrees of freedom (6 seeds)


def load_result(agent: str, seed: int, part: str = "test") -> dict:
    return json.loads((RESULTS_DIR / f"world-v2_{agent}_{part}_seed-{seed}.json").read_text(encoding="utf-8"))


def load_sessions(seed: int, part: str = "test") -> list[dict]:
    with (DATA_DIR / f"seed-{seed}" / f"{part}.jsonl").open(encoding="utf-8") as file:
        return [json.loads(line) for line in file]


def summary_table(agents: list[str], seeds=SEEDS) -> list[dict]:
    """Mean ± sd over seeds of overall, first/last 10 episodes, and post-drift scores."""
    rows = []
    for agent in agents:
        results = [load_result(agent, seed) for seed in seeds]
        metrics = {
            "overall": [r["overall"]["mean_score"] for r in results],
            "episodes_1_10": [r["by_episode_bucket"]["1-10"]["mean_score"] for r in results],
            "episodes_31_40": [r["by_episode_bucket"]["31-40"]["mean_score"] for r in results],
            "after_drift_10": [r["drift_sessions"]["first_10_after_change"]["mean_score"] for r in results],
        }
        rows.append({"agent": agent, **{k: (statistics.mean(v), statistics.stdev(v)) for k, v in metrics.items()}})
    return rows


def paired_difference(agent: str, reference: str, seeds=SEEDS) -> tuple[float, float, float]:
    """Mean seed-paired difference in overall score with a 95% t interval."""
    diffs = [load_result(agent, s)["overall"]["mean_score"] - load_result(reference, s)["overall"]["mean_score"]
             for s in seeds]
    mean = statistics.mean(diffs)
    half = T_95_DF5 * statistics.stdev(diffs) / math.sqrt(len(diffs))
    return mean, mean - half, mean + half


def learning_curve(agent: str, seeds=SEEDS) -> list[float]:
    """Mean score at each episode index (0..39), pooled over sessions and seeds."""
    totals, counts = [0.0] * 40, [0] * 40
    for seed in seeds:
        for record in load_result(agent, seed)["predictions"]:
            totals[record["index"]] += record["score"]
            counts[record["index"]] += 1
    return [t / c for t, c in zip(totals, counts)]


def load_n2(seed: int, memory: bool = True) -> N2Model:
    saved = torch.load(checkpoint_path(seed, memory), map_location="cpu")
    model = N2Model(saved["hidden"], saved["memory"])
    model.load_state_dict(saved["state"])
    return model.eval()


def n2_probabilities(model: N2Model, sessions: list[dict]) -> tuple[torch.Tensor, torch.Tensor]:
    kinds, keys, truths = session_tensors(sessions)
    with torch.no_grad():
        return model(kinds, keys, truths).softmax(-1), truths


def b3_probabilities(sessions: list[dict], hazard: float = 0.01) -> torch.Tensor:
    """Replay B3 on gold claims and return its predictive distribution per episode."""
    judge = BayesLearnerJudge(hazard=hazard)
    out = torch.zeros(len(sessions), len(sessions[0]["episodes"]), 3)
    for b, session in enumerate(sessions):
        judge.start_session({"speakers": session["speakers"]})
        for t, episode in enumerate(session["episodes"]):
            judge.act(episode)
            claims = judge._pending
            out[b, t] = torch.tensor(judge.predictive(claims) if claims else [1 / 3] * 3)
            judge.feedback(episode, episode["answer"])
    return out


def log_loss(probabilities: torch.Tensor, truths: torch.Tensor) -> float:
    picked = probabilities.gather(-1, truths.unsqueeze(-1)).clamp_min(1e-9)
    return float(-picked.log().mean())


def reliability(probabilities: torch.Tensor, truths: torch.Tensor, bins: int = 10) -> list[tuple[float, float, int]]:
    """(mean confidence, accuracy, count) per confidence bin of the top choice."""
    confidence, choice = probabilities.max(-1)
    correct = (choice == truths).float()
    rows = []
    edges = torch.linspace(1 / 3, 1.0, bins + 1)
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (confidence >= low) & (confidence < high if high < 1 else confidence <= high)
        if mask.any():
            rows.append((float(confidence[mask].mean()), float(correct[mask].mean()), int(mask.sum())))
    return rows


def trust_trajectories(session: dict, model: N2Model, hazard: float = 0.01) -> dict:
    """True accuracy, B3 posterior-mean accuracy, and N2's implied accuracy per speaker over time.

    N2's implied accuracy inverts the Bayes form of its positive-claim evidence:
    exp(pos_match - pos_other) = p / ((1 - p) / 2)  =>  p = r / (2 + r).
    """
    kinds, keys, truths = session_tensors([session])
    speakers = session["speakers"]
    judge = BayesLearnerJudge(hazard=hazard)
    judge.start_session({"speakers": speakers})
    state = model.initial_state(1, len(speakers))
    true, b3, n2 = [], [], []
    with torch.no_grad():
        for t, episode in enumerate(session["episodes"]):
            true.append(accuracies_at(session, t))
            if hazard:
                judge.belief = {s: [(1 - hazard) * b + hazard / len(ACCURACY_LEVELS) for b in bel]
                                for s, bel in judge.belief.items()}
            b3.append([judge.trust()[s] for s in speakers])
            pos_match, pos_other, _, _ = model.evidence(state)[0].unbind(-1)
            ratio = (pos_match - pos_other).exp()
            n2.append((ratio / (2 + ratio)).tolist())
            claims, _ = judge.claims(episode)
            judge._pending = claims
            judge.feedback(episode, episode["answer"])
            state = model.update(state, kinds[:, t], keys[:, t], truths[:, t])
    return {"speakers": speakers, "true": true, "b3": b3, "n2": n2, "drift": session["hidden"]["drift"]}
