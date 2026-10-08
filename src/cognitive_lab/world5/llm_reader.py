"""LFM2.5-1.2B on KLUE-MRC (frozen): the language-model-alone baseline and the answer candidates.

Two prompts, greedy decoding, cached per question (resumable: one JSON line per question):
  abstain : "answer with words from the passage; if the passage has no answer, write only 모름"
            -> the language-model-alone baseline
  answer  : the same without the 모름 option -> one candidate per question for the MK1 judge
Each record keeps the reply, the parsed answer, and token log-probabilities of the reply.

python -m cognitive_lab.world5.llm_reader --mode abstain --parts test
python -m cognitive_lab.world5.llm_reader --mode answer --parts train validation test
"""

import argparse
import json
import time

import torch

from cognitive_lab.world.llm_agent import DEFAULT_MODEL_DIR, load_model
from cognitive_lab.world5 import klue

CACHE_DIR = klue.PROJECT_ROOT / "artifacts" / "cache"
SYSTEM = ("지문을 읽고 질문의 답을 지문에서 찾아 쓰는 과제입니다. 답은 지문에 나온 표현 그대로, 단어나 짧은 구(보통 1~6단어)로만 "
          "쓰세요. 문장으로 쓰거나 설명하지 마세요.")
ABSTAIN = " 지문에서 답을 찾을 수 없으면 '모름'이라고만 쓰세요."
# Hand-written examples (not from KLUE) that show the answer format; the last shows 모름.
SHOTS = (
    ("서울시는 내년부터 청년 1만 명에게 월 50만 원의 활동비를 지급한다고 3일 밝혔다. 대상은 만 19세에서 34세 사이의 미취업자다.",
     "서울시가 활동비를 지급하는 청년은 몇 명인가?", "1만 명"),
    ("이 탑은 1931년 프랑스 건축가 르네 마르탱이 설계했으며, 높이는 87미터에 이른다.",
     "이 탑을 설계한 건축가의 국적은?", "프랑스"),
    ("회사는 신제품 판매량이 전년보다 20% 늘었다고 발표했지만, 구체적인 매출액은 공개하지 않았다.",
     "신제품의 매출액은 얼마인가?", "모름"),
)
MAX_NEW_TOKENS = 32


def prompt(question: dict, mode: str) -> list[dict]:
    """System instruction + format examples (the 모름 example only in abstain mode) + the question.
    (A bare instruction made the model answer with whole sentences; found on validation.)"""
    messages = [{"role": "system", "content": SYSTEM + (ABSTAIN if mode == "abstain" else "")}]
    for context, query, answer in SHOTS:
        if answer == klue.UNKNOWN and mode != "abstain":
            continue
        messages += [{"role": "user", "content": f"지문: {context}\n질문: {query}"},
                     {"role": "assistant", "content": answer}]
    messages.append({"role": "user", "content": f"지문: {question['context']}\n질문: {question['question']}"})
    return messages


def parse(reply: str, mode: str) -> str:
    line = reply.strip().split("\n")[0].strip()
    line = line.removeprefix("답:").strip().strip("\"'“”‘’「」『』.,·").strip()
    if mode == "abstain" and (line.startswith(klue.UNKNOWN) or line == ""):
        return klue.UNKNOWN
    return line


def cache_path(mode: str) -> "klue.Path":
    return CACHE_DIR / f"world-v5_llm-{mode}.jsonl"


def load_cache(mode: str) -> dict[str, dict]:
    path = cache_path(mode)
    if not path.exists():
        return {}
    return {r["guid"]: r for r in map(json.loads, path.read_text(encoding="utf-8").splitlines()) if r}


@torch.no_grad()
def run(mode: str, parts: list[str], device: torch.device) -> None:
    done = load_cache(mode)
    todo = [q for part in parts for q in klue.load(part) if q["guid"] not in done]
    print(f"{mode}: {len(done)} cached, {len(todo)} to go", flush=True)
    if not todo:
        return
    tokenizer, model = load_model(DEFAULT_MODEL_DIR, device)
    model.eval()
    started = time.perf_counter()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with cache_path(mode).open("a", encoding="utf-8") as out:
        for n, question in enumerate(todo, 1):
            encoded = tokenizer.apply_chat_template(prompt(question, mode), add_generation_prompt=True,
                                                    tokenize=True, return_dict=True, return_tensors="pt").to(device)
            generated = model.generate(**encoded, do_sample=False, max_new_tokens=MAX_NEW_TOKENS,
                                       pad_token_id=tokenizer.pad_token_id, output_scores=True,
                                       return_dict_in_generate=True)
            tokens = generated.sequences[0, encoded["input_ids"].shape[1]:]
            logprobs = [torch.log_softmax(step[0].float(), -1)[token].item() for step, token in zip(generated.scores, tokens)]
            reply = tokenizer.decode(tokens, skip_special_tokens=True)
            record = {"guid": question["guid"], "reply": reply, "answer": parse(reply, mode),
                      "token_logprobs": [round(x, 4) for x in logprobs], "prompt_tokens": int(encoded["input_ids"].shape[1])}
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            if n % 100 == 0:
                out.flush()
                rate = n / (time.perf_counter() - started)
                print(f"{n}/{len(todo)} ({rate:.2f}/s, about {(len(todo) - n) / rate / 60:.0f} min left)", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="LFM2.5-1.2B answers on KLUE-MRC (cached)")
    parser.add_argument("--mode", choices=("abstain", "answer"), required=True)
    parser.add_argument("--parts", nargs="+", default=["test"])
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    device = torch.device("cuda" if args.device != "cpu" and torch.cuda.is_available() else "cpu")
    run(args.mode, args.parts, device)


if __name__ == "__main__":
    main()
