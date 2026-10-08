"""MK1 on real Korean text (KLUE-MRC): the language model reads, a frozen checker verifies, a small
learned judge decides whether to answer or say 모름 (design/world-v5-real-korean.md).

Per question, the language model's candidate (answer mode, cached by llm_reader) becomes:
  - language-model features: mean / min / first-token log-probability, token count, character
    length, whether the candidate occurs in the passage
  - checker features (mDeBERTa-v3-base-mnli-xnli, frozen, fp32, cached): premise = the passage
    sentence holding the candidate and its neighbors (<= 400 characters; the passage opening when the
    candidate is not found), hypothesis = "질문: {q} 답: {a}". The pooled 768-d feature goes
    through a learned connection head (768 -> 8), and the NLI head's 3 logits are kept.
Judge: MLP -> P(right), P(wrong, an answer exists), P(no answer in the passage).
Answer iff its expected score beats 모름: P(right) - P(wrong) - P(none) > P(none).

Baselines: always 모름; the language model alone (abstain prompt); the language model's own
confidence (answer iff mean log-probability > a threshold tuned on validation).

python -m cognitive_lab.world5.mk1 --seed 42
"""

import argparse
import json
import random
import re
import time

import torch
from torch import nn

from cognitive_lab.world5 import klue
from cognitive_lab.world5.llm_reader import load_cache

CACHE_DIR = klue.PROJECT_ROOT / "artifacts" / "cache"
RESULTS_DIR = klue.PROJECT_ROOT / "artifacts" / "results"
NLI_DIR = klue.PROJECT_ROOT / "model" / "mDeBERTa-v3-base-mnli-xnli"
PREMISE_CHARS = 400
CLASSES = ("right", "wrong", "none")


def sentences(context: str) -> list[tuple[int, int]]:
    """Character spans of sentences (split after . ? ! and the sentence-final 다.)."""
    spans, start = [], 0
    for match in re.finditer(r"[.?!](?=\s|$)", context):
        spans.append((start, match.end()))
        start = match.end()
    if start < len(context):
        spans.append((start, len(context)))
    return spans


def locate(context: str, candidate: str) -> int:
    """Start of the candidate in the passage (exact, then ignoring spaces), or -1."""
    position = context.find(candidate)
    if position >= 0 or not candidate.strip():
        return position
    squeezed = re.sub(r"\s+", "", candidate)
    index = [i for i, c in enumerate(context) if not c.isspace()]
    position = "".join(context[i] for i in index).find(squeezed)
    return index[position] if position >= 0 else -1


def premise(context: str, candidate: str) -> tuple[str, bool]:
    position = locate(context, candidate)
    if position < 0:
        return context[:PREMISE_CHARS], False
    spans = sentences(context)
    k = next((i for i, (a, b) in enumerate(spans) if a <= position < b), 0)
    lo, hi = spans[max(0, k - 1)][0], spans[min(len(spans) - 1, k + 1)][1]
    text = context[lo:hi]
    if len(text) > PREMISE_CHARS:  # keep the candidate's own sentence, trimmed around it
        a, b = spans[k]
        text = context[max(a, position - PREMISE_CHARS // 2):min(b, position + PREMISE_CHARS // 2)]
    return text.strip(), True


def hypothesis(question: str, candidate: str) -> str:
    return f"질문: {question} 답: {candidate}"


def llm_features(record: dict, found: bool) -> list[float]:
    logprobs = record["token_logprobs"] or [0.0]
    return [sum(logprobs) / len(logprobs), min(logprobs), logprobs[0], len(logprobs) / 32,
            min(len(record["answer"]), 60) / 60, float(found)]


@torch.no_grad()
def checker_features(pairs: list[tuple[str, str]], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Pooled 768-d features and NLI logits [N,3] from the frozen checker, fp32."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(NLI_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(NLI_DIR, dtype=torch.float32).to(device).eval()
    pooled, logits = [], []
    for start in range(0, len(pairs), 64):
        chunk = pairs[start:start + 64]
        batch = tokenizer([p for p, _ in chunk], [h for _, h in chunk], return_tensors="pt", padding=True,
                          truncation="only_first", max_length=256).to(device)
        out = model(**batch, output_hidden_states=True)
        mask = batch["attention_mask"].unsqueeze(-1).float()
        pooled.append(((out.hidden_states[-1] * mask).sum(1) / mask.sum(1)).cpu())
        logits.append(out.logits.float().cpu())
    return torch.cat(pooled), torch.cat(logits)


def build(part: str, device: torch.device) -> dict:
    """Cached tensors for one split: checker features, NLI logits, LM features, labels, candidates."""
    path = CACHE_DIR / f"world-v5_mk1-features_{part}.pt"
    if path.exists():
        return torch.load(path)
    questions, answers = klue.load(part), load_cache("answer")
    missing = [q["guid"] for q in questions if q["guid"] not in answers]
    if missing:
        raise RuntimeError(f"{len(missing)} {part} questions have no cached candidate; run llm_reader --mode answer")
    pairs, lm, labels, candidates = [], [], [], []
    for q in questions:
        record = answers[q["guid"]]
        text, found = premise(q["context"], record["answer"])
        pairs.append((text, hypothesis(q["question"], record["answer"])))
        lm.append(llm_features(record, found))
        candidates.append(record["answer"])
        labels.append(2 if q["impossible"] else (0 if klue.is_right(record["answer"], q["answers"]) else 1))
    pooled, nli = checker_features(pairs, device)
    data = {"pooled": pooled, "nli": nli, "lm": torch.tensor(lm), "labels": torch.tensor(labels),
            "candidates": candidates, "guids": [q["guid"] for q in questions]}
    torch.save(data, path)
    return data


class Judge(nn.Module):
    def __init__(self, use_checker: bool = True, size: int = 8, width: int = 64):
        super().__init__()
        self.use_checker = use_checker
        self.head = nn.Linear(768, size)
        inputs = 6 + (size + 3 if use_checker else 0)
        self.net = nn.Sequential(nn.Linear(inputs, width), nn.Tanh(), nn.Linear(width, width), nn.Tanh(),
                                 nn.Linear(width, 3))

    def forward(self, data: dict) -> torch.Tensor:
        parts = [data["lm"]]
        if self.use_checker:
            parts += [torch.tanh(self.head(data["pooled"])), data["nli"]]
        return self.net(torch.cat(parts, -1))


def decide(probabilities: torch.Tensor) -> torch.Tensor:
    """True where answering beats 모름 in expected score."""
    right, wrong, none = probabilities.unbind(-1)
    return right - wrong - none > none


def scores_of(answer: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """+1 / -1 / 0 per question when answering (answer=True) or saying 모름."""
    answered = torch.where(labels == 0, 1.0, -1.0)
    abstained = torch.where(labels == 2, 1.0, 0.0)
    return torch.where(answer, answered, abstained)


def train_judge(train: dict, validation: dict, seed: int, use_checker: bool, epochs: int = 200) -> tuple[Judge, int]:
    torch.manual_seed(seed)
    model = Judge(use_checker)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best = (-9.0, 0, None)
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(train["labels"]))
        for start in range(0, len(order), 128):
            batch = {k: v[order[start:start + 128]] for k, v in train.items() if torch.is_tensor(v)}
            loss = nn.functional.cross_entropy(model(batch), batch["labels"])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            value = scores_of(decide(model(validation).softmax(-1)), validation["labels"]).mean().item()
        if value > best[0]:
            best = (value, epoch, {k: v.clone() for k, v in model.state_dict().items()})
    model.load_state_dict(best[2])
    return model.eval(), best[1]


def bootstrap(difference: torch.Tensor, seed: int, draws: int = 10000) -> list[float]:
    generator = torch.Generator().manual_seed(seed)
    index = torch.randint(len(difference), (draws, len(difference)), generator=generator)
    means = difference[index].mean(1).sort().values
    return [round(means[int(0.025 * draws)].item(), 4), round(means[int(0.975 * draws)].item(), 4)]


def main() -> None:
    parser = argparse.ArgumentParser(description="MK1 judge on KLUE-MRC")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    device = torch.device("cuda" if args.device != "cpu" and torch.cuda.is_available() else "cpu")
    random.seed(args.seed)
    started = time.perf_counter()
    data = {part: build(part, device) for part in ("train", "validation", "test")}
    labels = data["test"]["labels"]

    # Baselines
    abstain_cache = load_cache("abstain")
    test_questions = {q["guid"]: q for q in klue.load("test")}
    lm_alone = torch.tensor([klue.score(abstain_cache[g]["answer"], test_questions[g]) for g in data["test"]["guids"]],
                            dtype=torch.float)
    always_unknown = scores_of(torch.zeros_like(labels, dtype=torch.bool), labels)
    thresholds = data["validation"]["lm"][:, 0].unique()
    tuned = max(thresholds.tolist(), key=lambda t: (scores_of(data["validation"]["lm"][:, 0] > t,
                                                               data["validation"]["labels"]).mean().item(), -t))
    confidence = scores_of(data["test"]["lm"][:, 0] > tuned, labels)

    judges = {}
    for name, use_checker in (("mk1", True), ("mk1-no-checker", False)):
        model, epoch = train_judge(data["train"], data["validation"], args.seed, use_checker)
        with torch.no_grad():
            answer = decide(model(data["test"]).softmax(-1))
        judges[name] = (scores_of(answer, labels), answer, epoch)

    mk1 = judges["mk1"][0]
    strongest = max((always_unknown, confidence), key=lambda s: s.mean().item())

    def summary(s: torch.Tensor, answer: torch.Tensor | None = None) -> dict:
        out = {"score": round(s.mean().item(), 4)}
        if answer is not None:
            out["answer_rate"] = round(answer.float().mean().item(), 4)
            out["made_up_on_no_answer"] = round(answer[labels == 2].float().mean().item(), 4)
        return out

    result = {"world": "v5-klue-mrc", "seed": args.seed, "test_questions": len(labels),
              "label_counts": {c: int((labels == i).sum()) for i, c in enumerate(CLASSES)},
              "always_unknown": summary(always_unknown),
              "lm_alone": {**summary(lm_alone),
                           "answer_rate": round(sum(abstain_cache[g]["answer"] != klue.UNKNOWN
                                                    for g in data["test"]["guids"]) / len(labels), 4)},
              "lm_confidence": {**summary(confidence, data["test"]["lm"][:, 0] > tuned), "threshold": round(tuned, 4)},
              "mk1": {**summary(mk1, judges["mk1"][1]), "selected_epoch": judges["mk1"][2]},
              "mk1_no_checker": {**summary(judges["mk1-no-checker"][0], judges["mk1-no-checker"][1]),
                                 "selected_epoch": judges["mk1-no-checker"][2]},
              "oracle_filter": round(scores_of(labels == 0, labels).mean().item(), 4),
              "mk1_minus_strongest_baseline": {"mean": round((mk1 - strongest).mean().item(), 4),
                                               "bootstrap_95": bootstrap(mk1 - strongest, args.seed)},
              "mk1_minus_no_checker": {"mean": round((mk1 - judges["mk1-no-checker"][0]).mean().item(), 4),
                                       "bootstrap_95": bootstrap(mk1 - judges["mk1-no-checker"][0], args.seed)},
              "trainable_parameters": sum(p.numel() for p in Judge(True).parameters()),
              "seconds": round(time.perf_counter() - started, 1)}
    out = RESULTS_DIR / f"world-v5_mk1_seed-{args.seed}.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
