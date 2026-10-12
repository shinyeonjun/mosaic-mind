"""Step 1 of the antiques path (design/antiques-rethink-2026-10-12.md, 7): the NLI antique (mDeBERTa-v3-base-mnli-xnli,
used as downloaded, no training) as a checking satellite for document questions, recorded next to the existing
signals so no language model has to run again (the speaker's reply and the reader's answer are already recorded).

For every recorded question (calibration data and exam dev), each document is the premise and the hypothesis is
  'The answer to the question "<query>" is <answer>.'
  nli_reply_entail    the largest P(entailment) over documents for the speaker's reply (0 when it rejected)
  nli_reply_contra    the largest P(contradiction) over documents for the speaker's reply (0 when it rejected)
  nli_reader_entail   the largest P(entailment) for the reading satellite's best span (0 when it is empty)
  nli_seconds         time spent on this question
Weights in fp16 (as the other satellites since 2026-10-12). Written to a side file keyed by question id.

python -m cognitive_lab.mk1v3.nli_features
"""

import json
import time

import torch

from cognitive_lab.mk1v3.signals import OUT as CALIB, ROOT, hotpot_items, musique_train_items, squad_items

SIDE = ROOT / "artifacts" / "results" / "mk1v3-nli-features.jsonl"
NLI_DIR = ROOT / "model" / "mDeBERTa-v3-base-mnli-xnli"
FEATURES = ["nli_reply_entail", "nli_reply_contra", "nli_reader_entail"]


class Checker:
    def __init__(self, device: torch.device, dtype=torch.float16):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(NLI_DIR)
        self.model = AutoModelForSequenceClassification.from_pretrained(NLI_DIR, dtype=torch.float32).to(device=device, dtype=dtype).eval()
        labels = {v.lower(): int(k) for k, v in self.model.config.id2label.items()}
        self.entail, self.contra = labels["entailment"], labels["contradiction"]

    @torch.no_grad()
    def check(self, docs: list[str], query: str, answer: str, batch: int = 16) -> tuple[float, float]:
        """Largest P(entailment) and P(contradiction) of the answer statement over the documents."""
        if not answer.strip():
            return 0.0, 0.0
        hyp = f'The answer to the question "{query}" is {answer.strip()}.'
        e, c = 0.0, 0.0
        for s in range(0, len(docs), batch):
            chunk = docs[s:s + batch]
            x = self.tokenizer(chunk, [hyp] * len(chunk), truncation="only_first", max_length=512, padding=True,
                               return_tensors="pt").to(self.device)
            p = self.model(**x).logits.float().softmax(-1)
            e, c = max(e, float(p[:, self.entail].max())), max(c, float(p[:, self.contra].max()))
        return e, c


def features(checker: Checker, item: dict, row: dict) -> dict:
    t = time.perf_counter()
    rejected = bool(row["rejected"])
    re, rc = (0.0, 0.0) if rejected else checker.check(item["docs"], item["query"], row["reply"][:300])
    de, _ = checker.check(item["docs"], item["query"], row.get("reader_answer") or "")
    return {"nli_reply_entail": re, "nli_reply_contra": rc, "nli_reader_entail": de,
            "nli_seconds": round(time.perf_counter() - t, 3)}


def main() -> None:
    from cognitive_lab.mk1v3.exam_check import OUT as EXAM, items as exam_items

    items = {it["id"]: it for it in squad_items(300) + hotpot_items(300) + musique_train_items(400) + exam_items()}
    rows = [json.loads(l) for path in (CALIB, EXAM) for l in path.read_text(encoding="utf-8").splitlines()]
    done = {json.loads(l)["id"] for l in SIDE.read_text(encoding="utf-8").splitlines()} if SIDE.exists() else set()
    checker = Checker(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    with SIDE.open("a", encoding="utf-8") as log:
        for n, r in enumerate(x for x in rows if x["id"] not in done):
            log.write(json.dumps({"id": r["id"], **features(checker, items[r["id"]], r)}) + "\n")
            log.flush()
            if n % 200 == 0:
                print(n, flush=True)
    print("Saved:", SIDE)


if __name__ == "__main__":
    main()
