"""v3 step 2, transfer check: the calibrators learned on SQuAD/HotpotQA, applied unchanged to the exams' development
splits (RGB, MuSiQue), against v2's hand-wired flows with their dev-tuned thresholds. Sealed parts are not read.

python -m cognitive_lab.mk1v3.exam_check            (records signals once, then scores)
"""

import json
import statistics

import numpy as np

from cognitive_lab.mk1v3.calibrate import FEATURES, WEIGHTS, Logistic, auroc, decide, ece, head_rows, matrix
from cognitive_lab.mk1v3.signals import ROOT

OUT = ROOT / "artifacts" / "results" / "mk1v3-exam-dev-signals.jsonl"
RGB_SETTINGS = ["noise-0.0", "noise-0.4", "noise-0.8", "rejection", "integration"]


def items() -> list[dict]:
    from cognitive_lab.mk1v2 import musique, rgb

    out = []
    for setting in RGB_SETTINGS:
        for it in rgb.exam(setting, "dev"):
            out.append({"id": f"rgb|{setting}|{it['id']}", "source": "rgb", "query": it["query"], "docs": it["docs"],
                        "answers": [], "truth": it["answer"], "answerable": setting != "rejection"})
    for it in musique.exam("dev"):
        out.append({"id": f"musique|{it['id']}", "source": "musique", "query": it["query"], "docs": it["docs"],
                    "answers": [a for a in it["answers"] if a], "answerable": it["answerable"]})
    return out


def rejudge(row: dict, item: dict) -> dict:
    if item["source"] == "rgb":  # RGB's own checker (answer groups, all needed for integration)
        from cognitive_lab.mk1v2 import rgb

        row["right"] = rgb.judge(row["reply"], item["truth"])["right"]
    return row


def main() -> None:
    import torch

    from cognitive_lab.mk1v2.llm_service import LLMService
    from cognitive_lab.mk1v3.signals import Recorder

    its = items()
    by_id = {it["id"]: it for it in its}
    done = {json.loads(l)["id"] for l in OUT.read_text(encoding="utf-8").splitlines()} if OUT.exists() else set()
    todo = [it for it in its if it["id"] not in done]
    if todo:
        print(f"recording {len(todo)} dev questions", flush=True)
        with LLMService("e2b") as llm, OUT.open("a", encoding="utf-8") as log:
            rec = Recorder(llm, torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            for n, it in enumerate(todo, 1):
                log.write(json.dumps(rec.record(it), ensure_ascii=False) + "\n")
                log.flush()
                if n % 50 == 0:
                    print(f"  {n}/{len(todo)}", flush=True)
    rows = [rejudge(json.loads(l), by_id[json.loads(l)["id"]]) for l in OUT.read_text(encoding="utf-8").splitlines()]
    for r in rows:
        r["rejected"] = float(r["rejected"])
    saved = json.loads(WEIGHTS.read_text(encoding="utf-8"))
    models = {h: Logistic.from_dict(saved["models"][h]) for h in FEATURES}
    out = {}
    for source in ("rgb", "musique"):
        rs = [r for r in rows if r["source"] == source]
        res = {"questions": len(rs)}
        for head in FEATURES:
            hr = head_rows(rs, head)
            X, y = matrix(hr, head)
            p = models[head].predict(X)
            res[head] = {"n": len(hr), "auroc": round(auroc(p, y), 3), "ece": round(ece(p, y), 3)}

        def net(r, choice):
            if choice == "reject":
                return 0 if r["answerable"] else 1
            return (1 if r["right"] else -1) if r["answerable"] else -1

        res["net_speaker_alone"] = round(statistics.mean(net(r, "reject" if r["rejected"] else "answer") for r in rs), 4)
        res["net_v3_calibrated"] = round(statistics.mean(net(r, decide(r, models)) for r in rs), 4)
        out[source] = res
    out["note"] = ("v2 hand-wired dev nets for reference: RGB 0.700 over six testbeds incl. counterfactual (not comparable "
                   "one to one; here five testbeds, simple net), MuSiQue 0.560 (same items, same net).")
    print(json.dumps(out, indent=1))
    (ROOT / "artifacts" / "results" / "mk1v3-exam-dev-check.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
