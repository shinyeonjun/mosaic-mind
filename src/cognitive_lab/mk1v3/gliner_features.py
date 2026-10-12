"""Step 1 of the antiques path (design/antiques-rethink-2026-10-12.md, 7): the GLiNER antique (gliner2.5-multi-v1, as
downloaded, run locally in fp16) as an answer-type satellite for document questions, recorded next to the existing
signals so no language model has to run again.

The expected answer type comes from the question's first words (a fixed rule, no learning):
  who / whom / whose -> person     when / what year / which year / what date -> date
  where -> location                how many / how much -> number              otherwise: no type
Features (0 when the question has no type, the speaker rejected, or the answer is empty):
  gl_typed            1 if the question has an expected type
  gl_reply_type       1 if GLiNER finds an entity of the expected type in the speaker's reply
  gl_reply_in_docs    1 if such an entity also appears word for word in some document
  gl_reader_type      1 if GLiNER finds an entity of the expected type in the reading satellite's best span
  gl_seconds          time spent on this question

python -m cognitive_lab.mk1v3.gliner_features
"""

import json
import re
import time

from cognitive_lab.mk1v3.signals import OUT as CALIB, ROOT, hotpot_items, musique_train_items, squad_items

SIDE = ROOT / "artifacts" / "results" / "mk1v3-gliner-features.jsonl"
GLINER_DIR = ROOT / "model" / "gliner2.5-multi-v1"
FEATURES = ["gl_typed", "gl_reply_type", "gl_reply_in_docs", "gl_reader_type"]
RULES = [(r"^(who|whom|whose)\b", "person"), (r"^(when|what year|which year|what date|in what year|in which year)\b", "date"),
         (r"^(where|in what country|in which country|in what city|in which city)\b", "location"),
         (r"^(how many|how much)\b", "number")]


def expected_type(query: str) -> str | None:
    q = query.strip().lower()
    return next((t for pattern, t in RULES if re.search(pattern, q)), None)


class Typer:
    def __init__(self):
        from gliner2 import AutoExtractor

        self.model = AutoExtractor.from_pretrained(str(GLINER_DIR), map_location="cuda", quantize=True)

    def entities(self, text: str, kind: str) -> list[str]:
        if not text.strip():
            return []
        return self.model.extract_entities(text, [kind])["entities"].get(kind, [])


def features(typer: Typer, item: dict, row: dict) -> dict:
    t = time.perf_counter()
    kind = expected_type(item["query"])
    out = {f: 0.0 for f in FEATURES}
    if kind is not None:
        out["gl_typed"] = 1.0
        if not row["rejected"]:
            found = typer.entities(row["reply"][:300], kind)
            out["gl_reply_type"] = float(bool(found))
            docs = " ".join(item["docs"]).lower()
            out["gl_reply_in_docs"] = float(any(e.lower() in docs for e in found))
        out["gl_reader_type"] = float(bool(typer.entities(row.get("reader_answer") or "", kind)))
    out["gl_seconds"] = round(time.perf_counter() - t, 3)
    return out


def main() -> None:
    from cognitive_lab.mk1v3.exam_check import OUT as EXAM, items as exam_items

    items = {it["id"]: it for it in squad_items(300) + hotpot_items(300) + musique_train_items(400) + exam_items()}
    rows = [json.loads(l) for path in (CALIB, EXAM) for l in path.read_text(encoding="utf-8").splitlines()]
    done = {json.loads(l)["id"] for l in SIDE.read_text(encoding="utf-8").splitlines()} if SIDE.exists() else set()
    typer = Typer()
    with SIDE.open("a", encoding="utf-8") as log:
        for n, r in enumerate(x for x in rows if x["id"] not in done):
            log.write(json.dumps({"id": r["id"], **features(typer, items[r["id"]], r)}) + "\n")
            log.flush()
            if n % 200 == 0:
                print(n, flush=True)
    print("Saved:", SIDE)


if __name__ == "__main__":
    main()
