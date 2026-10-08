"""L1: a small instruction-tuned LLM answering alone from the episode text.

Two answer modes share one prompt:
- choice: score each allowed answer by its log-probability as the assistant reply
  and pick the highest. No format failures; measures what the model prefers.
- generate: greedy decoding, then take the earliest allowed answer in the text.
  Replies with no allowed answer count as invalid (scored as a wrong answer).

The agent never receives gold events, only the world rules, utterances, and question.
Answer options are shuffled per episode (seeded by episode ID) so a preference for
the first listed option cannot masquerade as an answer. Optional few-shot examples
are prior user/assistant turns built from training episodes.
"""

import random
import time
from collections import Counter
from pathlib import Path

import torch

from cognitive_lab.world.templates import UNKNOWN

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODEL_DIR = PROJECT_ROOT / "model" / "LFM2.5-1.2B-Instruct"
INVALID = "<무효 응답>"
END_OF_TURN = "<|im_end|>"

SYSTEM_PROMPT = (
    "너는 문과 열쇠 퍼즐의 답을 고르는 판단기다.\n"
    "세계 규칙: {rules}\n"
    "답은 다음 보기 중 하나를 그대로 한 줄로만 쓴다: {options}\n"
    "설명이나 다른 말은 쓰지 않는다."
)


def shuffled_options(episode: dict) -> list[str]:
    options = list(episode["entities"]["keys"]) + [UNKNOWN]
    random.Random(episode["episode_id"]).shuffle(options)
    return options


def _user_turn(episode: dict, options: list[str]) -> str:
    lines = [
        f"{index}. [{utterance['source']}] {utterance['text']}"
        for index, utterance in enumerate(episode["utterances"], 1)
    ]
    return (
        "대화:\n" + "\n".join(lines)
        + f"\n\n질문: {episode['question']}\n보기: {', '.join(options)}"
    )


def build_messages(episode: dict, examples: list[dict] | None = None) -> tuple[list[dict], list[str]]:
    options = shuffled_options(episode)
    system = SYSTEM_PROMPT.format(rules=episode["world_rules"], options=", ".join(options))
    messages = [{"role": "system", "content": system}]
    for example in examples or []:
        messages.append({"role": "user", "content": _user_turn(example, shuffled_options(example))})
        messages.append({"role": "assistant", "content": example["answer"]})
    messages.append({"role": "user", "content": _user_turn(episode, options)})
    return messages, options


def select_examples(train: list[dict], shots: int) -> list[dict]:
    """Pick the first training episode of each case type in turn, cycling until `shots`."""
    by_case: dict[str, list[dict]] = {}
    for episode in train:
        by_case.setdefault(episode["split_keys"]["case"], []).append(episode)
    cases = sorted(by_case)
    chosen = []
    round_index = 0
    while len(chosen) < shots:
        for case in cases:
            if len(chosen) < shots and round_index < len(by_case[case]):
                chosen.append(by_case[case][round_index])
        round_index += 1
    return chosen


def parse_answer(text: str, options: list[str]) -> str:
    """Return the allowed answer that appears earliest in `text`, or INVALID."""
    found = [(text.find(option), option) for option in options if option in text]
    return min(found)[1] if found else INVALID


def load_model(model_dir: Path, device: torch.device) -> tuple:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16)
    return tokenizer, model.to(device)


class LLMAgent:
    def __init__(
        self,
        mode: str,
        device: str,
        model_dir: Path = DEFAULT_MODEL_DIR,
        examples: list[dict] | None = None,
        adapter: Path | None = None,
        loaded: tuple | None = None,
    ):
        """`adapter` is a LoRA checkpoint from `world.finetune`; `loaded` reuses an
        already loaded (tokenizer, model) pair, e.g. for validation during training."""
        if mode not in ("choice", "generate"):
            raise ValueError(f"unknown mode: {mode}")
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but PyTorch cannot access a CUDA device")
        self.mode = mode
        self.device = torch.device(device)
        if self.device.type == "cuda" and loaded is None:
            torch.cuda.reset_peak_memory_stats(self.device)
        started = time.perf_counter()
        if loaded is not None:
            self.tokenizer, self.model = loaded
        else:
            self.tokenizer, self.model = load_model(model_dir, self.device)
        self.adapter = None
        if adapter is not None:
            from cognitive_lab.world.lora import apply_lora, load_lora

            checkpoint = torch.load(adapter, map_location="cpu")
            settings = checkpoint["lora"]
            apply_lora(self.model, settings["rank"], settings["alpha"], 0.0)
            load_lora(self.model, checkpoint["state"])
            self.model.to(self.device)
            self.adapter = str(adapter)
        self.model.eval()
        self.load_seconds = round(time.perf_counter() - started, 3)
        self.model_dir = Path(model_dir)
        self.calls = 0
        self.generated_tokens = 0
        self.invalid = 0
        self.raw_samples: list[dict] = []
        self.examples = examples or []
        self.chosen_positions: Counter = Counter()

    def __call__(self, episode: dict) -> str:
        messages, options = build_messages(episode, self.examples)
        with torch.inference_mode():
            if self.mode == "choice":
                answer, detail = self._choose(messages, options)
            else:
                answer, detail = self._generate(messages, options)
        if answer == INVALID:
            self.invalid += 1
        else:
            self.chosen_positions[options.index(answer)] += 1
        if len(self.raw_samples) < 20:
            self.raw_samples.append(
                {"episode_id": episode["episode_id"], "options": options, "answer": answer, **detail}
            )
        return answer

    def _generate(self, messages: list[dict], options: list[str]) -> tuple[str, dict]:
        encoded = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
        ).to(self.device)
        output = self.model.generate(
            **encoded,
            do_sample=False,
            max_new_tokens=16,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        new_tokens = output[0, encoded["input_ids"].shape[1]:]
        self.calls += 1
        self.generated_tokens += int(new_tokens.numel())
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        return parse_answer(text, options), {"raw_reply": text}

    def _choose(self, messages: list[dict], options: list[str]) -> tuple[str, dict]:
        prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        prompt_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        continuations = [
            self.tokenizer(option + END_OF_TURN, add_special_tokens=False)["input_ids"]
            for option in options
        ]
        # One unpadded pass per option, keeping logits only for the answer tokens:
        # full-sequence logits over the vocabulary ran out of memory with few-shot prompts.
        scores = []
        for continuation in continuations:
            input_ids = torch.tensor([prompt_ids + continuation], device=self.device)
            logits = self.model(input_ids=input_ids, logits_to_keep=len(continuation) + 1).logits
            log_probs = logits[0, :-1].float().log_softmax(dim=-1)
            targets = torch.tensor(continuation, device=self.device)
            scores.append(log_probs.gather(-1, targets.unsqueeze(-1)).sum().item())
            self.calls += 1
        best = max(range(len(options)), key=lambda index: scores[index])
        return options[best], {"option_log_probs": dict(zip(options, [round(s, 3) for s in scores]))}

    def cost_report(self) -> dict:
        peak = (
            round(torch.cuda.max_memory_allocated(self.device) / 2**30, 3)
            if self.device.type == "cuda"
            else None
        )
        return {
            "model": self.model_dir.name,
            "mode": self.mode,
            "device": str(self.device),
            "decoding": "greedy, max 16 new tokens" if self.mode == "generate" else "option log-probability",
            "load_seconds": self.load_seconds,
            "model_calls": self.calls,
            "generated_tokens": self.generated_tokens,
            "invalid_replies": self.invalid,
            "adapter": self.adapter,
            "few_shot_examples": [example["episode_id"] for example in self.examples],
            "chosen_option_position_counts": dict(sorted(self.chosen_positions.items())),
            "peak_vram_gib": peak,
            "raw_samples": self.raw_samples,
        }
