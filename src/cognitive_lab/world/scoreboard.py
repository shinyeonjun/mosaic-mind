"""Calibrated scoring and a uniform runner for any `episode -> answer` agent."""

import statistics
import time
from collections import Counter
from typing import Callable

from cognitive_lab.world.solver import solve
from cognitive_lab.world.templates import UNKNOWN, key_name

Agent = Callable[[dict], str]


def score_answer(prediction: str, answer: str) -> int:
    """+1 correct (including a correct UNKNOWN), 0 for UNKNOWN when an answer exists, -1 wrong."""
    if prediction == answer:
        return 1
    return 0 if prediction == UNKNOWN else -1


def visible_episode(episode: dict, gold_input: bool) -> dict:
    """Hide the answer always; hide gold events unless the agent runs in the gold-input condition."""
    hidden = {"answer", "split_keys", "tags"} | (
        set() if gold_input else {"gold_events", "gold_question"}
    )
    return {key: value for key, value in episode.items() if key not in hidden}


def run_agent(agent: Agent, episodes: list[dict], gold_input: bool) -> dict:
    scores, latencies = [], []
    by_case: dict[str, list[tuple[int, bool]]] = {}
    unknown_when_answer = guessed_when_unknown = 0
    answer_count = unknown_count = 0
    predictions = Counter()
    records = []

    for episode in episodes:
        started = time.perf_counter()
        prediction = agent(visible_episode(episode, gold_input))
        latencies.append(time.perf_counter() - started)
        answer = episode["answer"]
        score = score_answer(prediction, answer)
        scores.append(score)
        predictions[prediction] += 1
        records.append({
            "episode_id": episode["episode_id"],
            "case": episode["split_keys"]["case"],
            "answer": answer,
            "prediction": prediction,
            "score": score,
        })
        by_case.setdefault(episode["split_keys"]["case"], []).append((score, prediction == answer))
        if answer == UNKNOWN:
            unknown_count += 1
            guessed_when_unknown += prediction != UNKNOWN
        else:
            answer_count += 1
            unknown_when_answer += prediction == UNKNOWN

    latencies.sort()
    result = {
        "episodes": len(episodes),
        "gold_input": gold_input,
        "mean_score": round(statistics.mean(scores), 4),
        "accuracy": round(sum(s == 1 for s in scores) / len(scores), 4),
        "unknown_when_answer_exists_rate": round(unknown_when_answer / max(1, answer_count), 4),
        "guess_when_unknown_rate": round(guessed_when_unknown / max(1, unknown_count), 4),
        "by_case": {
            case: {
                "episodes": len(rows),
                "mean_score": round(statistics.mean(s for s, _ in rows), 4),
                "accuracy": round(sum(c for _, c in rows) / len(rows), 4),
            }
            for case, rows in sorted(by_case.items())
        },
        "latency_seconds": {
            "p50": round(latencies[len(latencies) // 2], 6),
            "p95": round(latencies[max(0, int(len(latencies) * 0.95) - 1)], 6),
        },
        "prediction_counts": dict(predictions),
        "predictions": records,
    }
    if hasattr(agent, "cost_report"):
        result["cost"] = agent.cost_report()
    return result


def solver_agent(episode: dict) -> str:
    """S1: exact solver over gold events; requires the gold-input condition."""
    keys = [name.removesuffix(" 열쇠") for name in episode["entities"]["keys"]]
    door = episode["gold_question"]["door"]
    key = solve(episode["gold_events"], door, keys)
    return UNKNOWN if key == UNKNOWN else key_name(key)


def naive_reader_agent(episode: dict) -> str:
    """Reference heuristic: last key stated positively about the queried door, ignoring trust."""
    door = episode["gold_question"]["door"]
    keys = [e["key"] for e in episode["gold_events"]
            if e["type"] != "chatter" and e["door"] == door and e["polarity"]]
    return key_name(keys[-1]) if keys else UNKNOWN


def _llm_factory(mode: str, lora: bool = False):
    def build(args) -> tuple[Agent, bool]:
        from cognitive_lab.world.finetune import adapter_path
        from cognitive_lab.world.llm_agent import LLMAgent

        adapter = (args.adapter or adapter_path(args.split, args.seed)) if lora else None
        if adapter is not None and not adapter.exists():
            raise SystemExit(f"missing LoRA adapter {adapter}; run `python -m cognitive_lab.world.finetune` first")
        return LLMAgent(mode, args.device, args.model_dir, args.fewshot_examples, adapter), False

    return build


def _rule_pipeline(args) -> tuple[Agent, bool]:
    from cognitive_lab.world.rule_parser import rule_pipeline_agent

    return rule_pipeline_agent, False


def _p2_factory(abstain: bool):
    def build(args) -> tuple[Agent, bool]:
        from cognitive_lab.world.parser_llm import P2Agent, parser_path

        adapter = args.adapter or parser_path(args.split, args.seed)
        if not adapter.exists():
            raise SystemExit(f"missing P2 parser {adapter}; run `python -m cognitive_lab.world.parser_llm` first")
        return P2Agent(args.device, args.model_dir, adapter, abstain), False

    return build


def _p2_nli_factory(trained: bool, abstain: bool = False):
    def build(args) -> tuple[Agent, bool]:
        from cognitive_lab.world.nli_parser import DEFAULT_NLI_DIR, P2NLIAgent, nli_path

        checkpoint = None
        if trained:
            checkpoint = args.adapter or nli_path(args.split, args.seed)
            if not checkpoint.exists():
                raise SystemExit(f"missing NLI checkpoint {checkpoint}; run `python -m cognitive_lab.world.nli_parser` first")
        return P2NLIAgent(args.device, DEFAULT_NLI_DIR, checkpoint, abstain), False

    return build


# name -> factory(args) -> (agent, receives gold events)
AGENTS = {
    "s1": lambda args: (solver_agent, True),
    "naive_reader": lambda args: (naive_reader_agent, True),
    "p1-rule": _rule_pipeline,
    "l1-choice": _llm_factory("choice"),
    "l1-generate": _llm_factory("generate"),
    "l1-lora-choice": _llm_factory("choice", lora=True),
    "l1-lora-generate": _llm_factory("generate", lora=True),
    "p2-lora": _p2_factory(abstain=False),
    "p2-lora-abstain": _p2_factory(abstain=True),
    "p2-nli-zeroshot": _p2_nli_factory(trained=False),
    "p2-nli": _p2_nli_factory(trained=True),
    "p2-nli-abstain": _p2_nli_factory(trained=True, abstain=True),
}
