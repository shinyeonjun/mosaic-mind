"""v3: per-document reading features, added next to the recorder's signals (the recorder kept only the best document).

For every recorded question (calibration data and exam dev), the English reading satellite reads each document again:
  docs_positive        documents whose best span beats their no-answer score
  margin_second        the second-best margin
  docs_agree_reply     documents whose span appears in the speaker's reply
  docs_agree_best      documents whose span equals the best document's span
Written to a side file keyed by question id; calibrate.py merges it when present.

python -m cognitive_lab.mk1v3.reader_features
"""

import json

import torch

from cognitive_lab.mk1v3.signals import OUT as CALIB, ROOT, hotpot_items, musique_train_items, squad_items

SIDE = ROOT / "artifacts" / "results" / "mk1v3-reader-features.jsonl"


def features(reads: list[dict], reply: str) -> dict:
    margins = sorted((r["span_score"] - r["null_score"] for r in reads), reverse=True)
    best = max(reads, key=lambda r: r["span_score"] - r["null_score"])["answer"].lower().strip()
    spans = [r["answer"].lower().strip() for r in reads if r["span_score"] > r["null_score"] and r["answer"].strip()]
    return {"docs_positive": float(sum(m > 0 for m in margins)), "margin_second": margins[1] if len(margins) > 1 else -10.0,
            "docs_agree_reply": float(sum(s in reply.lower() for s in spans)),
            "docs_agree_best": float(sum(s == best for s in spans))}


def main() -> None:
    from cognitive_lab.mk1v2.reader_en import CHECKPOINT as READER_EN
    from cognitive_lab.mk1v3.exam_check import OUT as EXAM, items as exam_items
    from cognitive_lab.world5.reader_qa import Reader

    items = {it["id"]: it for it in squad_items(300) + hotpot_items(300) + musique_train_items(400) + exam_items()}
    rows = [json.loads(l) for path in (CALIB, EXAM) for l in path.read_text(encoding="utf-8").splitlines()]
    done = {json.loads(l)["id"] for l in SIDE.read_text(encoding="utf-8").splitlines()} if SIDE.exists() else set()
    reader = Reader(torch.device("cuda" if torch.cuda.is_available() else "cpu"), checkpoint=READER_EN)
    with SIDE.open("a", encoding="utf-8") as log:
        for n, r in enumerate(x for x in rows if x["id"] not in done):
            it = items[r["id"]]
            reads = reader.read_many([(it["query"], d) for d in it["docs"]])
            log.write(json.dumps({"id": r["id"], **features(reads, r["reply"])}) + "\n")
            if n % 200 == 0:
                print(n, flush=True)
    print("Saved:", SIDE)


if __name__ == "__main__":
    main()
