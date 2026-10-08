"""MK1 against one language model on the same problems with the same information (design/mk1-versus.md).

Families (600 test problems each, fixed samples):
  chains   - rule world v3: a board of "same key" links and key facts; longer chains than in training, 모름 cases
  articles - rule world v6: 1-3 sources hand over the true or a covered article; who lies is learned from feedback
  doors    - rule world v10: a real article tells which key opens a door, a board chain carries it on
The language model (LFM2.5-1.2B-Instruct) sees the board sentences, the articles and the question as text. For
the session families it also gets the session's record so far: each earlier question's revealed answer and, per
source, whether that source's article contained it (cleaner than the feedback MK1 gets: MK1 only learns which of
its own readings was right). Answers, in three ways for chains and doors (the language model is credited with its
best way per family, chosen on these very problems: a choice that favours it, not MK1): the option with the highest
total log-probability ("sum"; short options such as 모름 are favoured), the highest mean log-probability per token
("mean"), or a written reply whose earliest option counts ("write"; a reply naming no option counts as 모름).
Articles: a written short answer. Scored the same way for both: +1 right, 0 모름 (a right 모름 is +1), -1 wrong;
for articles also leniently ("contains": right when the reply contains a gold answer), for both systems.

python -m cognitive_lab.mk1.versus --system mk1
python -m cognitive_lab.mk1.versus --system llm            # zero-shot (J1)
python -m cognitive_lab.mk1.versus --system llm --adapter artifacts/checkpoints/mk1-versus-lora.pt   # trained (J2)
"""

import argparse
import json
import random
import time
from pathlib import Path

import torch

from cognitive_lab.world2.integrated import RESULTS_DIR
from cognitive_lab.world3 import chains as v3
from cognitive_lab.world5 import klue

UNKNOWN = "모름"
SAMPLE = {"chains": 600, "articles": 15, "doors": 15}  # chains: problems; the others: sessions of 40

CHAIN_RULES = ("문과 열쇠 퍼즐이다. 안내판 문장은 모두 참이다. '같은 열쇠' 문장은 두 문이 같은 열쇠로 열린다는 뜻이다. "
               "질문한 문의 열쇠를 안내판만으로 알 수 없으면 '모름'이 정답이다.")
ARTICLE_RULES = ("여러 출처가 같은 질문에 대해 기사를 건넸다. 어떤 출처는 정답을 다른 말로 바꿔 놓은 거짓 기사를 준다. "
                 "출처마다 정확도가 다르고, 지난 질문 기록에서 누가 정답을 담은 기사를 줬는지 볼 수 있다. "
                 "기사에서 정답을 짧게(몇 단어) 그대로 답하라. 확실하지 않으면 '모름'이라고만 답하라.")
DOOR_RULES = ("문과 열쇠 퍼즐이다. 안내판 문장은 모두 참이다. '같은 열쇠' 문장은 두 문이 같은 열쇠로 열린다는 뜻이다. "
              "열쇠마다 이름표가 붙어 있다. 기사 문 하나의 열쇠는 기사 질문의 정답이 적힌 이름표의 열쇠다. "
              "여러 출처가 그 기사 질문에 대해 기사를 건넸고, 어떤 출처는 거짓 기사를 준다. 지난 질문 기록에서 누가 정답을 담은 기사를 줬는지 볼 수 있다. "
              "질문한 문의 열쇠를 고르고, 알 수 없으면 '모름'을 고르라.")


def problems() -> dict:
    from cognitive_lab.mk1.articles import article_sessions
    from cognitive_lab.mk1.system import chain_sessions
    from cognitive_lab.world6 import world as v6
    from cognitive_lab.world10 import world as v10

    episodes = v3.generate("test", 45)[:SAMPLE["chains"]]
    return {"chains": chain_sessions(episodes),
            "articles": article_sessions(v6.generate("test", 42, SAMPLE["articles"]), "test"),
            "doors": v10.generate("test", 42, SAMPLE["doors"])}


def contains(passage: str, answers: list[str]) -> bool:
    text = klue.normalize(passage)
    return any(klue.normalize(a) in text for a in answers if klue.normalize(a))


def record_lines(history: list[tuple[str, list[dict], list[str]]]) -> list[str]:
    """Earlier questions of the session: revealed answer and, per source, whether its article contained it."""
    lines = []
    for n, (question, reports, answers) in enumerate(history, 1):
        verdicts = ", ".join(f"출처{r['source'] + 1}: {'정답 담김' if contains(r['passage'], answers) else '정답 없음'}"
                             for r in reports)
        lines.append(f"{n}. 정답 '{answers[0]}' — {verdicts}")
    return lines


def chain_prompt(e: dict, options: list[str]) -> list[dict]:
    board = "\n".join(f"- {x}" for x in e["board"])
    return [{"role": "system", "content": CHAIN_RULES},
            {"role": "user", "content": f"안내판:\n{board}\n\n질문: {e['query']}은 무슨 열쇠로 열어?\n보기: {', '.join(options)}"}]


def article_prompt(e: dict, history: list) -> list[dict]:
    record = "\n".join(record_lines(history)) or "(아직 없음)"
    articles = "\n\n".join(f"[출처{r['source'] + 1}의 기사]\n{r['passage']}" for r in e["reports"])
    return [{"role": "system", "content": ARTICLE_RULES},
            {"role": "user", "content": f"지난 질문 기록:\n{record}\n\n{articles}\n\n질문: {e['question']}"}]


def door_prompt(e: dict, history: list, options: list[str]) -> list[dict]:
    record = "\n".join(record_lines(history)) or "(아직 없음)"
    board = "\n".join(f"- {x}" for x in e["board"])
    legend = ", ".join(f"{k} = '{v}'" for k, v in e["legend"].items())
    a = e["article"]
    articles = "\n\n".join(f"[출처{r['source'] + 1}의 기사]\n{r['passage']}" for r in a["reports"])
    return [{"role": "system", "content": DOOR_RULES},
            {"role": "user", "content": f"지난 질문 기록:\n{record}\n\n안내판:\n{board}\n\n열쇠 이름표: {legend}\n"
                                        f"기사 문: {a['door']} (기사 질문: {a['question']})\n\n{articles}\n\n"
                                        f"질문: {e['query']}은 무슨 열쇠로 열어?\n보기: {', '.join(options)}"}]


def options_for(e: dict, key: str) -> list[str]:
    options = list(e["keys"]) + [UNKNOWN]
    random.Random(key).shuffle(options)
    return options


class LanguageModel:
    def __init__(self, device: torch.device, adapter: Path | None = None):
        from cognitive_lab.world.llm_agent import DEFAULT_MODEL_DIR, load_model

        self.device = device
        self.tokenizer, self.model = load_model(DEFAULT_MODEL_DIR, device)
        if adapter is not None:
            from cognitive_lab.world.lora import apply_lora, load_lora

            checkpoint = torch.load(adapter, map_location="cpu")
            apply_lora(self.model, checkpoint["lora"]["rank"], checkpoint["lora"]["alpha"], 0.0)
            load_lora(self.model, checkpoint["state"])
            self.model.to(device)
        self.model.eval()
        self.tokens = 0

    @torch.inference_mode()
    def choose(self, messages: list[dict], options: list[str]) -> dict:
        """{"sum": option, "mean": option}: highest total / mean-per-token log-probability."""
        prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        prompt_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        total, mean = [], []
        for option in options:
            continuation = self.tokenizer(option + "<|im_end|>", add_special_tokens=False)["input_ids"]
            ids = torch.tensor([prompt_ids + continuation], device=self.device)
            logits = self.model(input_ids=ids, logits_to_keep=len(continuation) + 1).logits
            log_probs = logits[0, :-1].float().log_softmax(-1)
            value = log_probs.gather(-1, torch.tensor(continuation, device=self.device)[:, None]).sum().item()
            total.append(value)
            mean.append(value / len(continuation))
            self.tokens += ids.shape[1]
        return {"sum": options[max(range(len(options)), key=lambda i: total[i])],
                "mean": options[max(range(len(options)), key=lambda i: mean[i])]}

    def write_choice(self, messages: list[dict], options: list[str]) -> str:
        reply = self.generate(messages, raw=True)
        found = [(reply.find(o), o) for o in options if o in reply]
        return min(found)[1] if found else UNKNOWN

    @torch.inference_mode()
    def generate(self, messages: list[dict], raw: bool = False) -> str:
        encoded = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True,
                                                     return_dict=True, return_tensors="pt").to(self.device)
        output = self.model.generate(**encoded, do_sample=False, max_new_tokens=24, pad_token_id=self.tokenizer.pad_token_id,
                                     use_cache=True)  # training turns the config cache off for checkpointing
        self.tokens += output.shape[1]
        text = self.tokenizer.decode(output[0, encoded["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        if raw:
            return text
        text = text.splitlines()[0].strip() if text else ""
        return UNKNOWN if not text or UNKNOWN in text else text


def llm_answers(model: LanguageModel, family: str, sessions: list[dict]) -> dict[str, list[str]]:
    """Answers per way of answering (see the module docstring)."""
    out: dict[str, list[str]] = {}
    count, started = 0, time.perf_counter()
    for s in sessions:
        history = []
        for e in s["episodes"]:
            if family == "articles":
                out.setdefault("write", []).append(model.generate(article_prompt(e, history)))
                history.append((e["question"], e["reports"], e["answers"]))
            else:
                if family == "chains":
                    options = options_for(e, e["query"] + str(count))
                    messages = chain_prompt(e, options)
                else:
                    options = options_for(e, e["article"]["question"])
                    messages = door_prompt(e, history, options)
                    a = e["article"]
                    history.append((a["question"], a["reports"], a["answers"]))
                for way, answer in model.choose(messages, options).items():
                    out.setdefault(way, []).append(answer)
                out.setdefault("write", []).append(model.write_choice(messages, options))
            count += 1
            if count % 100 == 0:
                print(f"  {family}: {count} problems ({time.perf_counter() - started:.0f}s)", flush=True)
    return out


def contains_score(episode: dict, answer: str) -> float:
    """Lenient article scoring: right when the reply contains a gold answer."""
    if answer == UNKNOWN:
        return 0.0
    return 1.0 if any(klue.normalize(g) and klue.normalize(g) in klue.normalize(answer) for g in episode["answers"]) else -1.0


def mk1_answers(device: torch.device, family: str, sessions: list[dict]) -> list[str]:
    from cognitive_lab.mk1.system import MK1

    model = MK1(device)
    policy = {"chains": "think", "articles": "route", "doors": "route"}[family]
    return model(sessions, policy)[0]["answer"]


def main() -> None:
    from cognitive_lab.mk1.system import episode_score

    parser = argparse.ArgumentParser(description="MK1 vs one language model on the same problems")
    parser.add_argument("--system", choices=("mk1", "llm"), required=True)
    parser.add_argument("--adapter", type=Path, default=None)
    parser.add_argument("--families", nargs="+", default=list(SAMPLE))
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    name = args.system if args.adapter is None else f"llm-{args.adapter.stem}"
    data = problems()
    model = LanguageModel(device, args.adapter) if args.system == "llm" else None
    result = {"system": name, "families": {}}
    for family in args.families:
        sessions = data[family]
        episodes = [e for s in sessions for e in s["episodes"]]
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        ways = llm_answers(model, family, sessions) if model else {"mk1": mk1_answers(device, family, sessions)}
        seconds = time.perf_counter() - started
        entry = {"problems": len(episodes), "seconds": round(seconds, 1),
                 "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2) if device.type == "cuda" else None, "ways": {}}
        for way, answers in ways.items():
            scores = [episode_score(e, a) for e, a in zip(episodes, answers)]
            entry["ways"][way] = {"score": round(sum(scores) / len(scores), 4), "right": round(sum(x == 1 for x in scores) / len(scores), 4),
                                  "wrong": round(sum(x == -1 for x in scores) / len(scores), 4), "answers": answers, "scores": scores}
            if family == "articles":
                lenient = [contains_score(e, a) for e, a in zip(episodes, answers)]
                entry["ways"][way]["contains_score"] = round(sum(lenient) / len(lenient), 4)
        best = max(entry["ways"], key=lambda w: entry["ways"][w]["score"])
        entry["best_way"] = best
        result["families"][family] = entry
        print(name, family, {w: {k: v for k, v in x.items() if k not in ("answers", "scores")} for w, x in entry["ways"].items()},
              f"best={best}", f"{seconds:.0f}s", flush=True)
    path = RESULTS_DIR / f"mk1-versus_{name}.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
