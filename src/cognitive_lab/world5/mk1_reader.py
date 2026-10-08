"""MK1 v2 on KLUE-MRC: reading specialist (frozen) + NLI checker (frozen) + small judge.

Per question the reading specialist (reader_qa predictions) gives an answer span, its span
score, and the no-answer score. The checker gives NLI features on (passage sentence holding the
answer, "질문: q 답: a"), as in MK1 v1. The judge (connection head 768 -> 8 + MLP) predicts
P(right), P(wrong, an answer exists), P(no answer) and answers iff that beats 모름 in expected
score. Baselines: always 모름, the language model alone (abstain prompt), MK1 v1, and the
reading specialist alone (answer iff span score - no-answer score > a threshold tuned on
validation).

python -m cognitive_lab.world5.mk1_reader --seed 42
"""

import argparse
import json
import time

import torch
from torch import nn

from cognitive_lab.world5 import klue
from cognitive_lab.world5.llm_reader import load_cache
from cognitive_lab.world5.mk1 import (CACHE_DIR, RESULTS_DIR, bootstrap, checker_features, decide, hypothesis,
                                      premise, scores_of)
from cognitive_lab.world5.reader_qa import CACHE as READER_CACHE
from cognitive_lab.world5.reader_qa import load_predictions


def reader_features(record: dict, found: bool) -> list[float]:
    span, null = record["span_score"], record["null_score"]
    return [span / 10, null / 10, (span - null) / 10, min(len(record["answer"]), 60) / 60, float(found)]


def build(part: str, device: torch.device) -> dict:
    path = CACHE_DIR / f"world-v5_mk1-reader-features_{part}.pt"
    if path.exists() and path.stat().st_mtime > READER_CACHE.stat().st_mtime:
        return torch.load(path)
    questions, predictions = klue.load(part), load_predictions()
    pairs, features, labels = [], [], []
    for q in questions:
        record = predictions[q["guid"]]
        text, found = premise(q["context"], record["answer"])
        pairs.append((text, hypothesis(q["question"], record["answer"])))
        features.append(reader_features(record, found))
        labels.append(2 if q["impossible"] else (0 if klue.is_right(record["answer"], q["answers"]) else 1))
    pooled, nli = checker_features(pairs, device)
    data = {"pooled": pooled, "nli": nli, "reader": torch.tensor(features), "labels": torch.tensor(labels),
            "guids": [q["guid"] for q in questions]}
    torch.save(data, path)
    return data


class Judge(nn.Module):
    def __init__(self, size: int = 8, width: int = 64):
        super().__init__()
        self.head = nn.Linear(768, size)
        self.net = nn.Sequential(nn.Linear(5 + size + 3, width), nn.Tanh(), nn.Linear(width, width), nn.Tanh(),
                                 nn.Linear(width, 3))

    def forward(self, data: dict) -> torch.Tensor:
        return self.net(torch.cat([data["reader"], torch.tanh(self.head(data["pooled"])), data["nli"]], -1))


def train_judge(train: dict, validation: dict, seed: int, epochs: int = 200) -> tuple[Judge, int]:
    torch.manual_seed(seed)
    model = Judge()
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


def main() -> None:
    parser = argparse.ArgumentParser(description="MK1 v2: reading specialist + checker + judge on KLUE-MRC")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    device = torch.device("cuda" if args.device != "cpu" and torch.cuda.is_available() else "cpu")
    started = time.perf_counter()
    data = {part: build(part, device) for part in ("train", "validation", "test")}
    labels = data["test"]["labels"]
    questions = {q["guid"]: q for q in klue.load("test")}

    abstain_cache = load_cache("abstain")
    lm_alone = torch.tensor([klue.score(abstain_cache[g]["answer"], questions[g]) for g in data["test"]["guids"]],
                            dtype=torch.float)
    v1_path = RESULTS_DIR / f"world-v5_mk1_seed-{args.seed}.json"
    v1_score = json.loads(v1_path.read_text(encoding="utf-8"))["mk1"]["score"] if v1_path.exists() else None
    always_unknown = scores_of(torch.zeros_like(labels, dtype=torch.bool), labels)
    margin_val, margin_test = data["validation"]["reader"][:, 2], data["test"]["reader"][:, 2]
    tuned = max(margin_val.unique().tolist() + [-1e9],
                key=lambda t: (scores_of(margin_val > t, data["validation"]["labels"]).mean().item(), t))
    reader_alone_answer = margin_test > tuned
    reader_alone = scores_of(reader_alone_answer, labels)

    model, epoch = train_judge(data["train"], data["validation"], args.seed)
    with torch.no_grad():
        answer = decide(model(data["test"]).softmax(-1))
    mk1 = scores_of(answer, labels)

    def summary(s, a=None):
        out = {"score": round(s.mean().item(), 4)}
        if a is not None:
            out.update(answer_rate=round(a.float().mean().item(), 4),
                       made_up_on_no_answer=round(a[labels == 2].float().mean().item(), 4),
                       right_when_answering=round((labels[a] == 0).float().mean().item(), 4) if a.any() else None)
        return out

    result = {"world": "v5-klue-mrc", "experiment": "mk1-reader", "seed": args.seed,
              "label_counts": {"right": int((labels == 0).sum()), "wrong": int((labels == 1).sum()),
                               "none": int((labels == 2).sum())},
              "always_unknown": summary(always_unknown), "lm_alone": summary(lm_alone), "mk1_v1": v1_score,
              "reader_alone": {**summary(reader_alone, reader_alone_answer), "threshold": round(tuned, 4)},
              "mk1_v2": {**summary(mk1, answer), "selected_epoch": epoch},
              "oracle_filter": round(scores_of(labels == 0, labels).mean().item(), 4),
              "mk1_minus_lm_alone": {"mean": round((mk1 - lm_alone).mean().item(), 4),
                                     "bootstrap_95": bootstrap(mk1 - lm_alone, args.seed)},
              "mk1_minus_reader_alone": {"mean": round((mk1 - reader_alone).mean().item(), 4),
                                         "bootstrap_95": bootstrap(mk1 - reader_alone, args.seed)},
              "trainable_parameters": sum(p.numel() for p in model.parameters()),
              "seconds": round(time.perf_counter() - started, 1)}
    out = RESULTS_DIR / f"world-v5_mk1-reader_seed-{args.seed}.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
