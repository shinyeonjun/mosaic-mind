"""Session runner for rule world v2: answer, score, then give feedback, episode by episode.

Every episode is also scored by the oracle (O2) on the same input so each agent's
regret (oracle score minus agent score) is measured on identical episodes.
"""

import statistics
import time

from cognitive_lab.world.scoreboard import score_answer
from cognitive_lab.world2.agents import OracleJudge
from cognitive_lab.world2.generator import DRIFT_AT, SESSION_LENGTH

BUCKETS = ((0, 10), (10, 20), (20, 30), (30, 40))
HIDDEN_EPISODE_KEYS = {"answer", "split_keys", "hidden_other_truths"}


def visible(episode: dict, gold_input: bool) -> dict:
    hidden = HIDDEN_EPISODE_KEYS | (set() if gold_input else {"gold_events", "gold_question"})
    return {k: v for k, v in episode.items() if k not in hidden}


def _mean(values: list[float]) -> float | None:
    return round(statistics.mean(values), 4) if values else None


def run_sessions(agent, sessions: list[dict], gold_input: bool, feedback: str = "full") -> dict:
    """feedback="full": the true key after every episode. feedback="partial": only whether
    an answer was right; abstaining (모름) gives no feedback."""
    if feedback not in ("full", "partial"):
        raise ValueError(f"unknown feedback mode: {feedback}")
    from cognitive_lab.world2.misspecified import OracleMJudge

    records = []
    latencies = []
    for session in sessions:
        public = {"session_id": session["session_id"], "speakers": session["speakers"]}
        full = {**public, "hidden": session["hidden"]}
        agent.start_session(full if getattr(agent, "gold_session", False) else public)
        oracle = OracleMJudge() if session["hidden"].get("variant") == "misspecified" else OracleJudge()
        oracle.start_session(full)
        drift = session["hidden"]["drift"]
        for episode in session["episodes"]:
            seen = visible(episode, gold_input)
            started = time.perf_counter()
            prediction = agent.act(seen)
            latencies.append(time.perf_counter() - started)
            oracle_prediction = oracle.act(episode)
            answer = episode["answer"]
            records.append({
                "episode_id": episode["episode_id"],
                "index": episode["index"],
                "case": episode["split_keys"]["case"],
                "drift_session": drift is not None,
                "answer": answer,
                "prediction": prediction,
                "score": score_answer(prediction, answer),
                "oracle_score": score_answer(oracle_prediction, answer),
            })
            if feedback == "full":
                agent.feedback(seen, answer)
            elif prediction == "모름":
                agent.feedback_partial(seen, None, None)
            else:
                agent.feedback_partial(seen, prediction, prediction == answer)

    def summary(rows: list[dict]) -> dict:
        answered = [r for r in rows if r["prediction"] != "모름"]
        return {
            "episodes": len(rows),
            "mean_score": _mean([r["score"] for r in rows]),
            "oracle_mean_score": _mean([r["oracle_score"] for r in rows]),
            "regret": _mean([r["oracle_score"] - r["score"] for r in rows]),
            "abstain_rate": round(1 - len(answered) / len(rows), 4) if rows else None,
            "answered_accuracy": _mean([float(r["score"] == 1) for r in answered]),
        }

    latencies.sort()
    by_bucket = {
        f"{start + 1}-{end}": summary([r for r in records if start <= r["index"] < end]) for start, end in BUCKETS
    }
    drift_rows = [r for r in records if r["drift_session"]]
    result = {
        "sessions": len(sessions),
        "episodes_per_session": SESSION_LENGTH,
        "gold_input": gold_input,
        "feedback": feedback,
        "overall": summary(records),
        "by_episode_bucket": by_bucket,
        "drift_sessions": {
            "before_change": summary([r for r in drift_rows if r["index"] < DRIFT_AT]),
            "first_10_after_change": summary([r for r in drift_rows if DRIFT_AT <= r["index"] < DRIFT_AT + 10]),
            "last_10": summary([r for r in drift_rows if r["index"] >= DRIFT_AT + 10]),
        },
        "by_case": {
            case: summary([r for r in records if r["case"] == case])
            for case in sorted({r["case"] for r in records})
        },
        "latency_seconds": {"p50": round(latencies[len(latencies) // 2], 6)},
        "predictions": records,
    }
    if hasattr(agent, "cost_report"):
        result["cost"] = agent.cost_report()
    return result
