"""Rule world v6 sessions and the non-learned judges (design/world-v6-real-trust.md).

A session: 40 different real KLUE questions (a question may recur in other sessions), 4 sources with hidden accuracies (v2 levels, not all equal).
Per question 1-3 sources report (weights 0.2 / 0.4 / 0.4). A source passes on the true passage with
probability = its accuracy, else the covered (false) passage; what it reports is the reading
specialist's answer on that version (cached in reports.py). Each question therefore has at most
two distinct candidate answers: the reading of the true version and of the false one. False
reports agree with each other (the same planted answer), like a rumour.

Score: answer a candidate -> +1 if right, -1 if wrong; 모름 -> 0 (every question has an answer).
After each question the right answer is revealed.

python -m cognitive_lab.world6.world      # judges that do not learn from data
"""

import random

import torch

from cognitive_lab.world2.generator import ACCURACY_LEVELS
from cognitive_lab.world5 import klue
from cognitive_lab.world6.reports import load

SESSION_LENGTH, SPEAKERS = 40, 4
TALKER_WEIGHTS = {1: 0.2, 2: 0.4, 3: 0.4}
SESSIONS = {"train": 2000, "validation": 40, "test": 100}


def generate(part: str, seed: int, count: int | None = None) -> list[dict]:
    reports = [r for r in load().values() if r["part"] == part]
    reports.sort(key=lambda r: r["guid"])
    rng = random.Random(f"world-v6|{part}|{seed}")
    sessions = []
    for n in range(count or SESSIONS[part]):
        while True:
            accuracies = [rng.choice(ACCURACY_LEVELS) for _ in range(SPEAKERS)]
            if len(set(accuracies)) > 1:
                break
        episodes = []
        picked = rng.sample(reports, SESSION_LENGTH)  # no question twice within a session
        for t in range(SESSION_LENGTH):
            record = picked[t]
            count_ = rng.choices(list(TALKER_WEIGHTS), weights=list(TALKER_WEIGHTS.values()))[0]
            talkers = rng.sample(range(SPEAKERS), count_)
            said = []
            for s in talkers:
                truthful = rng.random() < accuracies[s]
                said.append({"speaker": s, "truthful": truthful,
                             **(record["true_report"] if truthful else record["false_report"])})
            candidates = []
            for report in said:
                if not any(klue.normalize(report["answer"]) == klue.normalize(c) for c in candidates):
                    candidates.append(report["answer"])
            for report in said:
                report["candidate"] = next(i for i, c in enumerate(candidates)
                                           if klue.normalize(c) == klue.normalize(report["answer"]))
            right = [klue.is_right(c, record["answers"]) for c in candidates]
            episodes.append({"guid": record["guid"], "index": t, "reports": said, "candidates": candidates,
                             "right": right, "true_reading": record["true_report"]["answer"]})
        sessions.append({"session_id": f"v6-{part}-s{seed}-{n:04d}", "accuracies": accuracies, "episodes": episodes})
    return sessions


def score(choice: int, episode: dict) -> int:
    """choice: candidate index, or -1 for 모름."""
    return 0 if choice < 0 else (1 if episode["right"][choice] else -1)


# ---- judges that do not learn from data ------------------------------------------------

def majority(episode: dict, gap: int) -> int:
    """Most-reported candidate if it leads the next by at least `gap` reports, else 모름."""
    votes = [0] * len(episode["candidates"])
    for report in episode["reports"]:
        votes[report["candidate"]] += 1
    order = sorted(range(len(votes)), key=lambda k: -votes[k])
    lead = votes[order[0]] - (votes[order[1]] if len(order) > 1 else 0)
    return order[0] if lead >= gap else -1


def confident(episode: dict, threshold: float) -> int:
    """No trust: the candidate whose best report has the highest reading margin (span - no-answer
    score), if that margin exceeds the threshold, else 모름."""
    margin = [-1e9] * len(episode["candidates"])
    for report in episode["reports"]:
        margin[report["candidate"]] = max(margin[report["candidate"]], report["span_score"] - report["null_score"])
    best = max(range(len(margin)), key=lambda k: margin[k])
    return best if margin[best] > threshold else -1


def run_confident(sessions: list[dict], threshold: float) -> float:
    episodes = [e for s in sessions for e in s["episodes"]]
    return round(sum(score(confident(e, threshold), e) for e in episodes) / len(episodes), 4)


def tune_confident(sessions: list[dict]) -> float:
    margins = sorted({r["span_score"] - r["null_score"] for s in sessions for e in s["episodes"] for r in e["reports"]})
    grid = margins[::max(1, len(margins) // 200)] + [1e9]
    return max(grid, key=lambda t: (run_confident(sessions, t), t))


def reading_rates(part: str = "train") -> tuple[float, float]:
    """P(reading is right | true version), P(reading is right | false version), from data."""
    reports = [r for r in load().values() if r["part"] == part]
    true_right = sum(klue.is_right(r["true_report"]["answer"], r["answers"]) for r in reports) / len(reports)
    false_right = sum(klue.is_right(r["false_report"]["answer"], r["answers"]) for r in reports) / len(reports)
    return true_right, false_right


def candidate_posterior(episode: dict, accuracy_of, rates: tuple[float, float]) -> list[float]:
    """P(candidate k is right). Hypotheses: which candidate is the true-version reading (or that
    the two versions read the same). Then right with the reading rates."""
    true_right, false_right = rates
    k = len(episode["candidates"])
    if k == 1:  # every source said the same thing: it is the true reading if anyone was truthful
        p_any_true = 1 - torch.tensor([1 - accuracy_of(r["speaker"]) for r in episode["reports"]]).prod().item()
        return [p_any_true * true_right + (1 - p_any_true) * false_right]
    weights = []
    for h in range(k):  # h = index of the true-version reading
        w = 1.0
        for r in episode["reports"]:
            a = accuracy_of(r["speaker"])
            w *= a if r["candidate"] == h else 1 - a
        weights.append(w)
    total = sum(weights) or 1.0
    p_true = [w / total for w in weights]
    return [p_true[c] * true_right + (1 - p_true[c]) * false_right for c in range(k)]


def decide(p: list[float]) -> int:
    best = max(range(len(p)), key=p.__getitem__)
    return best if p[best] > 0.5 else -1


def run_bayes(sessions: list[dict], learn: bool, rates: tuple[float, float]) -> float:
    """Known accuracies (learn=False) or accuracies learned from feedback (learn=True, a level
    posterior per source updated with P(the source saw the true version | feedback))."""
    prior = [1 / len(ACCURACY_LEVELS)] * len(ACCURACY_LEVELS)
    true_right, false_right = rates
    total = 0
    count = 0
    for session in sessions:
        belief = [list(prior) for _ in range(SPEAKERS)]
        mean = lambda s: sum(b * l for b, l in zip(belief[s], ACCURACY_LEVELS))
        accuracy_of = mean if learn else (lambda s: session["accuracies"][s])
        for episode in session["episodes"]:
            p = candidate_posterior(episode, accuracy_of, rates)
            total += score(decide(p), episode)
            count += 1
            if not learn:
                continue
            # P(candidate h is the true reading | which candidates were right)
            k = len(episode["candidates"])
            if k == 1:
                continue  # all agree: says little about who was truthful
            weights = []
            for h in range(k):
                w = 1.0
                for r in episode["reports"]:
                    w *= mean(r["speaker"]) if r["candidate"] == h else 1 - mean(r["speaker"])
                for c in range(k):
                    rate = true_right if c == h else false_right
                    w *= rate if episode["right"][c] else 1 - rate
                weights.append(w)
            z = sum(weights) or 1.0
            for r in episode["reports"]:
                p_truthful = weights[r["candidate"]] / z
                s = r["speaker"]
                updated = [b * (l * p_truthful + (1 - l) * (1 - p_truthful)) for b, l in zip(belief[s], ACCURACY_LEVELS)]
                belief[s] = [u / sum(updated) for u in updated]
    return round(total / count, 4)


def run_majority(sessions: list[dict], gap: int) -> float:
    episodes = [e for s in sessions for e in s["episodes"]]
    return round(sum(score(majority(e, gap), e) for e in episodes) / len(episodes), 4)


def main() -> None:
    rates = reading_rates("train")
    print("reading rates (true version right, false version right):", [round(x, 3) for x in rates])
    validation = generate("validation", 0)
    gaps = {g: run_majority(validation, g) for g in (0, 1, 2, 3, 99)}
    gap = max(gaps, key=lambda g: gaps[g])
    print("majority on validation by gap:", gaps, "-> gap", gap)
    threshold = tune_confident(validation)
    print(f"reading confidence threshold (validation): {threshold:.3f} -> {run_confident(validation, threshold)}")
    for seed in (42, 43, 44):
        test = generate("test", seed)
        print(f"seed {seed}: majority {run_majority(test, gap)} | reading confidence {run_confident(test, threshold)} | "
              f"formula learner {run_bayes(test, True, rates)} | "
              f"knows accuracies {run_bayes(test, False, rates)} | always 모름 0.0")


if __name__ == "__main__":
    main()
