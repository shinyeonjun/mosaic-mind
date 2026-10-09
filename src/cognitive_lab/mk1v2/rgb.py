"""External exam: RGB (Chen et al., AAAI 2024), English, re-implemented from the authors' evaluator.

Data: github.com/chen700564/RGB (CC BY-NC-SA 4.0, non-commercial). Testbeds used:
  en_refine  noise robustness (accuracy at noise rates 0-0.8) and negative rejection (noise rate 1)
  en_fact    counterfactual robustness (documents carry a wrong answer): error detection (ED), correction (CR)
  en_int     information integration (the answer needs facts from several documents)
Protocol (as `evalue.py`): 5 passages per question, positives and negatives drawn with random.seed(2333) per
question; right when every answer group has an alias inside the reply (case-insensitive); "insufficient
information" in the reply = a rejection, "factual errors" = an error detection. The authors' system prompt and
instruction. Temperature 0 for every system here (the authors used 0.7 for ChatGPT), so results are reproducible.

Rules (design/audit-2026-10-08.md, the exam discipline): RGB has no training split and is never trained on. A fixed
10% of the questions (seed "rgb-split") is the development set for engineering; the other 90% is sealed and is only
read with --final, once.

python -m cognitive_lab.mk1v2.rgb --system e2b --set dev
"""

import argparse
import json
import math
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data" / "rgb"
RESULTS = ROOT / "artifacts" / "results"
PASSAGES = 5
SYSTEM = ("You are an accurate and reliable AI assistant that can answer questions with the help of external documents. "
          "Please note that external documents may contain noisy or factually incorrect information. If the information in "
          "the document contains the correct answer, you will give an accurate answer. If the information in the document does "
          "not contain the answer, you will generate ’I can not answer the question because of the insufficient information "
          "in documents.‘. If there are inconsistencies with the facts in some of the documents, please generate the "
          "response 'There are factual errors in the provided documents.' and provide the correct answer.")
INSTRUCTION = "Document:\n{DOCS} \n\nQuestion:\n{QUERY}"
SETTINGS = {  # testbed name -> (file, noise rate)
    "noise-0.0": ("en_refine", 0.0), "noise-0.4": ("en_refine", 0.4), "noise-0.8": ("en_refine", 0.8),
    "rejection": ("en_refine", 1.0), "counterfactual": ("en_fact", 0.0), "integration": ("en_int", 0.0),
}


def load(name: str, part: str) -> list[dict]:
    rows = [json.loads(line) for line in (DATA / f"{name}.json").read_text(encoding="utf-8").splitlines() if line.strip()]
    ids = sorted(r["id"] for r in rows)
    random.Random(f"rgb-split|{name}").shuffle(ids)
    dev = set(ids[:max(1, len(ids) // 10)])
    if part == "dev":
        return [r for r in rows if r["id"] in dev]
    if part == "final":
        return [r for r in rows if r["id"] not in dev]
    raise ValueError(part)


def documents(instance: dict, noise_rate: float, filename: str) -> list[str]:
    """`processdata` of the authors' evaluator (correct_rate 0), with random.seed(2333) set by the caller."""
    neg_num = math.ceil(PASSAGES * noise_rate)
    pos_num = PASSAGES - neg_num
    if "_int" in filename:
        for group in instance["positive"]:
            random.shuffle(group)
        docs = [group[0] for group in instance["positive"]]
        if len(docs) < pos_num:
            longest = max(len(group) for group in instance["positive"])
            for i in range(1, longest):
                for group in instance["positive"]:
                    if len(group) > i:
                        docs.append(group[i])
                        if len(docs) == pos_num:
                            break
                if len(docs) == pos_num:
                    break
        neg_num = PASSAGES - len(docs)
        if neg_num > 0:
            docs += instance["negative"][:neg_num]
    elif "_fact" in filename:
        indexes = list(range(len(instance["positive"])))
        selected = random.sample(indexes, min(len(indexes), pos_num))
        docs = [instance["positive_wrong"][i] for i in selected]
        if neg_num > 0:
            docs += instance["negative"][:neg_num]
    else:
        if noise_rate == 1:
            neg_num, pos_num = PASSAGES, 0
        elif neg_num > len(instance["negative"]):
            neg_num = len(instance["negative"])
            pos_num = PASSAGES - neg_num
        elif pos_num > len(instance["positive"]):
            pos_num = len(instance["positive"])
            neg_num = PASSAGES - pos_num
        docs = instance["positive"][:pos_num] + instance["negative"][:neg_num]
    random.shuffle(docs)
    return docs


def check(prediction: str, truth) -> list[int]:
    prediction = prediction.lower()
    truth = truth if isinstance(truth, list) else [truth]
    labels = []
    for group in truth:
        if isinstance(group, list):
            labels.append(int(any(alias.lower() in prediction for alias in group)))
        else:
            labels.append(int(group.lower() in prediction))
    return labels


def judge(prediction: str, truth) -> dict:
    rejected = "insufficient information" in prediction
    labels = [-1] if rejected else check(prediction, truth)
    return {"rejected": rejected, "right": (not rejected) and 0 not in labels and 1 in labels,
            "error_detected": "factual errors" in prediction}


def exam(setting: str, part: str) -> list[dict]:
    """Questions with their 5 documents, exactly as the evaluator draws them."""
    filename, noise = SETTINGS[setting]
    items = []
    for instance in load(filename, part):
        random.seed(2333)
        items.append({"id": instance["id"], "query": instance["query"], "answer": instance["answer"],
                      "docs": documents(instance, noise, filename), "setting": setting})
    return items


def score(setting: str, judged: list[dict]) -> dict:
    n = len(judged)
    out = {"questions": n}
    if setting == "rejection":
        out["rejection_rate"] = round(sum(j["rejected"] for j in judged) / n, 4)
    else:
        out["accuracy"] = round(sum(j["right"] for j in judged) / n, 4)
    if setting == "counterfactual":
        detected = [j for j in judged if j["error_detected"]]
        out["error_detection_rate"] = round(len(detected) / n, 4)
        out["error_correction_rate"] = round(sum(j["right"] for j in detected) / len(detected), 4) if detected else 0.0
    return out


def run_llm(name: str, part: str, settings: list[str]) -> dict:
    from cognitive_lab.mk1v2.llm_service import LLMService

    result = {"system": f"gemma-4-{name} alone", "part": part, "settings": {}}
    with LLMService(name) as llm:
        for setting in settings:
            judged = []
            for item in exam(setting, part):
                reply = llm.chat([{"role": "system", "content": SYSTEM},
                                  {"role": "user", "content": INSTRUCTION.format(DOCS="\n".join(item["docs"]), QUERY=item["query"])}],
                                 max_tokens=200)
                judged.append({"id": item["id"], "reply": reply, **judge(reply, item["answer"])})
            result["settings"][setting] = {**score(setting, judged), "replies": judged}
            print(name, setting, score(setting, judged), flush=True)
        result["cost"] = llm.cost()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="RGB as an external exam")
    parser.add_argument("--system", required=True, help="e2b | e4b (a language model alone) | mk1")
    parser.add_argument("--set", choices=("dev", "final"), default="dev")
    parser.add_argument("--final", action="store_true", help="required to read the sealed 90%%")
    parser.add_argument("--settings", nargs="+", default=list(SETTINGS))
    args = parser.parse_args()
    if args.set == "final" and not args.final:
        raise SystemExit("the final set is sealed: pass --final, once, when the system is frozen")
    if args.system in ("e2b", "e4b"):
        result = run_llm(args.system, args.set, args.settings)
    else:
        from cognitive_lab.mk1v2.system import run_mk1

        result = run_mk1(args.set, args.settings)
    path = RESULTS / f"exam-rgb_{args.system}_{args.set}.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({s: {k: v for k, v in r.items() if k != "replies"} for s, r in result["settings"].items()}, indent=1))
    print("Saved:", path)


if __name__ == "__main__":
    main()
