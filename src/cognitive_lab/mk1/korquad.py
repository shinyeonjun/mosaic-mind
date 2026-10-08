"""External exam: KorQuAD 1.0 (Korean reading), taken once, like a real test (design/mk1-versus.md).

Rules (decided before opening the exam): nothing is trained or tuned on KorQuAD. MK1's reading specialist is
the KLUE-trained one, unchanged, fp32; the language model (LFM2.5-1.2B-Instruct) is used as it is (zero-shot), or
with the J2 adapter trained only on MK1's own training worlds. Before scoring, passages that also occur in the
reading specialist's KLUE training data are flagged (contamination check), and scores are given for both groups.
Scoring: exact match and character-level F1 after normalization (a re-implementation in the spirit of the
official KorQuAD/SQuAD script: lowercase, punctuation and spaces removed), plus the project's `is_right`.
KorQuAD 1.0 questions always have an answer; 모름 counts as wrong.

python -m cognitive_lab.mk1.korquad --system mk1
python -m cognitive_lab.mk1.korquad --system llm
"""

import argparse
import json
import re
import time
from collections import Counter
from pathlib import Path

import torch

from cognitive_lab.world2.integrated import RESULTS_DIR
from cognitive_lab.world5 import klue

EXAM = Path(__file__).resolve().parents[3] / "data" / "korquad-v1" / "KorQuAD_v1.0_dev.json"
PROMPT = "다음 글을 읽고 질문의 답을 글에서 찾아 짧게(몇 단어) 그대로 답하라. 설명은 쓰지 않는다."


def load_exam() -> list[dict]:
    data = json.loads(EXAM.read_text(encoding="utf-8"))["data"]
    return [{"id": qa["id"], "context": p["context"], "question": qa["question"], "answers": [a["text"] for a in qa["answers"]]}
            for article in data for p in article["paragraphs"] for qa in p["qas"]]


def normalize(text: str) -> str:
    return re.sub(r"[\s\W_]+", "", text.lower())


def exact(prediction: str, golds: list[str]) -> float:
    return float(any(normalize(prediction) == normalize(g) for g in golds))


def char_f1(prediction: str, golds: list[str]) -> float:
    best = 0.0
    p = Counter(normalize(prediction))
    for g in golds:
        gc = Counter(normalize(g))
        common = sum((p & gc).values())
        if common:
            precision, recall = common / sum(p.values()), common / sum(gc.values())
            best = max(best, 2 * precision * recall / (precision + recall))
    return best


def contamination(exam: list[dict]) -> list[bool]:
    """Passages the reading specialist saw while training (KLUE-MRC train, the part it was trained on)."""
    from cognitive_lab.world5.reader_qa import reader_training_rows

    seen = {normalize(r["context"]) for r in reader_training_rows()}
    return [normalize(q["context"]) in seen for q in exam]


def mk1_answers(exam: list[dict], device: torch.device) -> list[str]:
    from cognitive_lab.world5.reader_qa import Reader

    reader = Reader(device)  # fp32, KLUE-trained, unchanged
    answers = []
    for start in range(0, len(exam), 64):
        chunk = exam[start:start + 64]
        answers += [r["answer"] for r in reader.read_many([(q["question"], q["context"]) for q in chunk])]
    return answers


def llm_answers(exam: list[dict], device: torch.device, adapter: Path | None, no_unknown: bool = False) -> list[str]:
    """`no_unknown`: the model may not write 모름 (KorQuAD questions all have answers; the reading specialist is
    likewise run without its no-answer option), so both systems answer every question."""
    from cognitive_lab.mk1.versus import LanguageModel

    model = LanguageModel(device, adapter)
    if no_unknown:
        banned = [ids for ids in {tuple(model.tokenizer(w, add_special_tokens=False)["input_ids"]) for w in ("모름", " 모름")}]
        model.bad_words_ids = [list(ids) for ids in banned]
    answers, started = [], time.perf_counter()
    for q in exam:
        answers.append(model.generate([{"role": "system", "content": PROMPT},
                                       {"role": "user", "content": f"글:\n{q['context']}\n\n질문: {q['question']}"}]))
        if len(answers) % 250 == 0:
            print(f"  {len(answers)}/{len(exam)} ({time.perf_counter() - started:.0f}s)", flush=True)
    return answers


def main() -> None:
    parser = argparse.ArgumentParser(description="KorQuAD 1.0 as an external exam")
    parser.add_argument("--system", choices=("mk1", "llm"), required=True)
    parser.add_argument("--adapter", type=Path, default=None)
    parser.add_argument("--no-unknown", action="store_true", help="the language model may not answer 모름")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    exam = load_exam()
    seen = contamination(exam)
    started = time.perf_counter()
    answers = mk1_answers(exam, device) if args.system == "mk1" else llm_answers(exam, device, args.adapter, args.no_unknown)
    seconds = time.perf_counter() - started
    name = (args.system if args.adapter is None else f"llm-{args.adapter.stem}") + ("-no-unknown" if args.no_unknown else "")
    rows = [(exact(a, q["answers"]), char_f1(a, q["answers"]), float(klue.is_right(a, q["answers"])), s)
            for q, a, s in zip(exam, answers, seen, strict=True)]

    def summary(selected):
        n = len(selected)
        return {"questions": n, "exact_match": round(100 * sum(r[0] for r in selected) / n, 2),
                "char_f1": round(100 * sum(r[1] for r in selected) / n, 2), "is_right": round(100 * sum(r[2] for r in selected) / n, 2)}

    result = {"exam": "KorQuAD 1.0 dev (taken once)", "system": name, "all": summary(rows),
              "passage_seen_in_reader_training": summary([r for r in rows if r[3]]) if any(seen) else None,
              "passage_unseen": summary([r for r in rows if not r[3]]),
              "seconds": round(seconds, 1), "questions_per_second": round(len(exam) / seconds, 2), "answers": answers}
    path = RESULTS_DIR / f"exam-korquad1_{name}.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "answers"}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
