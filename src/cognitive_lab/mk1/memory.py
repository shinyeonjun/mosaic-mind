"""MK1 long-term memory: what the world confirmed stays known across sessions (design/mk1-integration.md, H).

Keyed by the question (normalized text); the value is a candidate answer the world confirmed as right after
the question. Unconfirmed conclusions of MK1 itself are never written. Sessions are taken in order, a day
(several sessions, solved in parallel) at a time; what was learned during a day is written at its end.

python -m cognitive_lab.mk1.memory      # stage H: build in the v6 stream, reuse in v11
"""

import json
import time

import torch

from cognitive_lab.world5 import klue


class LongTermMemory:
    def __init__(self):
        self.facts: dict[str, str] = {}
        self.pending: list[tuple[str, str]] = []
        self.frozen = False

    @staticmethod
    def key(question: str) -> str:
        return klue.normalize(question)

    def recall(self, question: str) -> str | None:
        return self.facts.get(self.key(question))

    def confirm(self, question: str, answer: str) -> None:
        """Feedback: the world said `answer` was right for `question`. Written at the end of the day."""
        if not self.frozen:
            self.pending.append((self.key(question), answer))

    def end_of_day(self) -> int:
        new = 0
        for key, answer in self.pending:
            new += key not in self.facts
            self.facts[key] = answer
        self.pending = []
        return new


def stream(model, sessions: list[dict], day: int = 10, **options) -> tuple[list[str], list[int]]:
    """Solve sessions in order, `day` sessions at a time, closing each day in the memory (if any)."""
    answers, learned = [], []
    for start in range(0, len(sessions), day):
        out, _ = model(sessions[start:start + day], **options)
        answers += out["answer"]
        if model.memory is not None:
            learned.append(model.memory.end_of_day())
    return answers, learned


def main() -> None:
    from cognitive_lab.mk1.arbiter import BOARD_ACCURACIES
    from cognitive_lab.mk1.articles import article_sessions
    from cognitive_lab.mk1.integration import bootstrap
    from cognitive_lab.mk1.system import MK1, episode_score
    from cognitive_lab.world2.integrated import RESULTS_DIR
    from cognitive_lab.world6 import world
    from cognitive_lab.world7.routing import COST
    from cognitive_lab.world10 import world as w10

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    started = time.perf_counter()
    model = MK1(device)
    sessions = article_sessions(world.generate("test", 42), "test")
    episodes = [e for s in sessions for e in s["episodes"]]
    session_of = torch.tensor([i for i, s in enumerate(sessions) for _ in s["episodes"]])
    result = {"stage": "MK1-H", "v6_stream": {}}
    scores = {}
    for name in ("memory-off", "memory-on"):
        model.memory = LongTermMemory() if name == "memory-on" else None
        model.recalls = []
        answers, learned = stream(model, sessions)
        scores[name] = torch.tensor([episode_score(e, a) for e, a in zip(episodes, answers)])
        by_day = [round(scores[name][(session_of >= d) & (session_of < d + 10)].mean().item(), 4) for d in range(0, 100, 10)]
        result["v6_stream"][name] = {"score": round(scores[name].mean().item(), 4), "by_day": by_day}
        if name == "memory-on":
            recalled = [(e, a) for e, a, r in zip(episodes, answers, model.recalls) if r]
            result["v6_stream"][name].update({
                "facts_learned_per_day": learned, "facts_known_at_end": len(model.memory.facts),
                "recalled_share": round(len(recalled) / len(episodes), 4),
                "recalled_wrong_share": round(sum(not klue.is_right(a, e["answers"]) for e, a in recalled) / max(1, len(recalled)), 4)})
        print(name, json.dumps(result["v6_stream"][name], ensure_ascii=False), flush=True)
    on, off = scores["memory-on"], scores["memory-off"]
    first, last = session_of < 10, session_of >= 90

    def growth(s):
        g = torch.Generator().manual_seed(0)
        d = torch.stack([s[last][torch.randint(int(last.sum()), (int(last.sum()),), generator=g)].mean()
                         - s[first][torch.randint(int(first.sum()), (int(first.sum()),), generator=g)].mean() for _ in range(2000)])
        return [round(d.quantile(0.025).item(), 4), round(d.quantile(0.975).item(), 4)]

    result["v6_stream"]["on_minus_off_ci95"] = bootstrap(on - off)
    result["v6_stream"]["growth_last10_minus_first10_ci95"] = {"memory-on": growth(on), "memory-off": growth(off)}

    # Cross-task: the memory built above, frozen, in fresh v11 sessions (seed 43) as the arbiter's third channel.
    memory = model.memory
    memory.frozen = True
    linked = w10.generate("test", 43, board_accuracies=BOARD_ACCURACIES)
    l_episodes = [e for s in linked for e in s["episodes"]]
    nets = {}
    for name, mem in (("arbiter-2-channels", None), ("arbiter-3-channels (memory)", memory)):
        model.memory = mem
        out, _ = model(linked, "arbitrate")
        raw = torch.tensor([episode_score(e, a) for e, a in zip(l_episodes, out["answer"])])
        nets[name] = raw - COST * out["asks"].cpu()
    known = torch.tensor([memory.recall(e["article"]["question"]) is not None for e in l_episodes])
    two, three = nets["arbiter-2-channels"], nets["arbiter-3-channels (memory)"]
    result["v11_cross_task"] = {"questions_known": round(known.float().mean().item(), 4),
                                **{k: round(v.mean().item(), 4) for k, v in nets.items()},
                                "known": {k: round(v[known].mean().item(), 4) for k, v in nets.items()},
                                "unknown": {k: round(v[~known].mean().item(), 4) for k, v in nets.items()},
                                "three_minus_two_ci95": bootstrap(three - two)}
    print("v11", json.dumps(result["v11_cross_task"], ensure_ascii=False), flush=True)
    v6, v11 = result["v6_stream"], result["v11_cross_task"]
    result["criteria"] = {
        "1a_memory_helps": v6["on_minus_off_ci95"][0] > 0,
        "1b_grows_over_days": v6["growth_last10_minus_first10_ci95"]["memory-on"][0] > 0,
        "2_carries_to_another_task": v11["three_minus_two_ci95"][0] > 0,
        "3_no_harm_when_unknown": abs(v11["unknown"]["arbiter-3-channels (memory)"] - v11["unknown"]["arbiter-2-channels"]) <= 0.01,
        "4_no_wrong_memories": v6["memory-on"]["recalled_wrong_share"] <= 0.01,
    }
    result["graduated"] = all(result["criteria"].values())
    result["seconds"] = round(time.perf_counter() - started, 1)
    path = RESULTS_DIR / "mk1-h_long_term_memory.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("criteria", "graduated")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
