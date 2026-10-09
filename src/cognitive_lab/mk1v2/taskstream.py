"""External exam: skill acquisition on a stream of new tasks (design/mk1-v2-spec.md, 10).

Six public text-classification tasks arrive one after another. In each task, texts come one at a time; the system
answers, then the true label is revealed (feedback). No retraining step: whatever is learned is learned in the stream.
Two conditions: "named" (real label names, prior knowledge helps) and "symbolic" (labels renamed A, B, C, ... per
task: only learning from feedback helps; Chollet's skill-acquisition efficiency).

Data (data/taskstream/, research datasets on Hugging Face): SST-2 validation, AG News test, TREC-QC test (coarse
labels), Emotion validation, SUBJ test, TweetEval sentiment test. Per task, disjoint samples: 100 development texts
and 200 sealed final texts (seed "taskstream|<task>"); the task order is fixed (seed "taskstream-order").
Metric: online accuracy (mean over the stream), per condition. Every system answers every text (no abstention).

Systems:
  gemma-e2b / gemma-e4b  the label set, the 16 most recent feedback examples of this task, the text -> a label
                         (the first label named in the reply; none named counts as wrong). k = 16 fixed in advance.
  mk1                    grows a new part per task: a prototype classifier on frozen e5-small embeddings, updated
                         with every feedback; in the named condition also the NLI satellite's zero-shot guess; the
                         counting arbiter weighs the channels by how often each has been right in this task.

python -m cognitive_lab.mk1v2.taskstream --system mk1 --set dev
"""

import argparse
import json
import random
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data" / "taskstream"
RESULTS = ROOT / "artifacts" / "results"
SIZES = {"dev": 100, "final": 200}
K_EXAMPLES = 16
TASKS = {
    "sst2": ("sst2_validation.parquet", "sentence", "label", ["negative", "positive"]),
    "ag_news": ("ag_news_test.parquet", "text", "label", ["World", "Sports", "Business", "Sci/Tech"]),
    "trec": ("trec_test.jsonl", "text", "label_coarse", ["description", "entity", "abbreviation", "human", "numeric", "location"]),
    "emotion": ("emotion_validation.parquet", "text", "label", ["sadness", "joy", "love", "anger", "fear", "surprise"]),
    "subj": ("subj_test.jsonl", "text", "label", ["objective", "subjective"]),
    "tweet_sentiment": ("tweet_sentiment_test.parquet", "text", "label", ["negative", "neutral", "positive"]),
}
SYMBOLS = "ABCDEF"


def _rows(task: str) -> list[dict]:
    file, text_key, label_key, _ = TASKS[task]
    path = DATA / file
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq

        raw = pq.read_table(path).to_pylist()
    else:
        raw = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [{"text": r[text_key], "label": int(r[label_key])} for r in raw]


def stream(part: str, condition: str) -> list[dict]:
    """Tasks in a fixed order; per task the texts of this part with their label names for the condition."""
    order = sorted(TASKS)
    random.Random("taskstream-order").shuffle(order)
    out = []
    for task in order:
        rows = _rows(task)
        random.Random(f"taskstream|{task}").shuffle(rows)
        dev, final = rows[:SIZES["dev"]], rows[SIZES["dev"]:SIZES["dev"] + SIZES["final"]]
        names = TASKS[task][3] if condition == "named" else list(SYMBOLS[:len(TASKS[task][3])])
        out.append({"task": task, "labels": names, "items": dev if part == "dev" else final})
    return out


def run_gemma(name: str, part: str) -> dict:
    from cognitive_lab.mk1v2.llm_service import LLMService

    result = {"system": f"gemma-4-{name} (in-context, k={K_EXAMPLES})", "part": part, "conditions": {}}
    with LLMService(name) as llm:
        for condition in ("named", "symbolic"):
            tasks = []
            for task in stream(part, condition):
                labels, seen, right, replies = task["labels"], [], [], []
                for item in task["items"]:
                    shots = "\n".join(f"Text: {t}\nLabel: {labels[y]}" for t, y in seen[-K_EXAMPLES:])
                    prompt = (f"Classify the text into exactly one of these labels: {', '.join(labels)}.\n"
                              f"{'Labelled examples:' + chr(10) + shots + chr(10) if shots else ''}"
                              f"Text: {item['text']}\nAnswer with the label only.")
                    reply = llm.chat([{"role": "user", "content": prompt}], max_tokens=8).strip()
                    guess = parse(reply, labels, condition)
                    guess = -1 if guess is None else guess
                    right.append(int(guess == item["label"]))
                    replies.append(reply)
                    seen.append((item["text"], item["label"]))
                tasks.append({"task": task["task"], "accuracy": round(statistics.mean(right), 4), "right": right, "replies": replies})
                print(name, condition, task["task"], tasks[-1]["accuracy"], flush=True)
            result["conditions"][condition] = {"accuracy": round(statistics.mean(t["accuracy"] for t in tasks), 4), "tasks": tasks}
        result["cost"] = llm.cost()
    return result


class Embedder:
    def __init__(self, device):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch, self.device = torch, device
        directory = ROOT / "model" / "multilingual-e5-small"
        self.tokenizer = AutoTokenizer.from_pretrained(directory)
        self.model = AutoModel.from_pretrained(directory).to(device).eval()

    def __call__(self, texts: list[str]):
        torch = self.torch
        with torch.no_grad():
            x = self.tokenizer(["query: " + t for t in texts], padding=True, truncation=True, max_length=128, return_tensors="pt").to(self.device)
            h = self.model(**x).last_hidden_state
            e = (h * x["attention_mask"][..., None]).sum(1) / x["attention_mask"].sum(1, keepdim=True)
            return torch.nn.functional.normalize(e, dim=-1).cpu()


class ZeroShot:
    """The NLI satellite's guess: which "This text is about <label>." hypothesis is most entailed."""

    def __init__(self, device):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch, self.device = torch, device
        directory = ROOT / "model" / "mDeBERTa-v3-base-mnli-xnli"
        self.tokenizer = AutoTokenizer.from_pretrained(directory)
        self.model = AutoModelForSequenceClassification.from_pretrained(directory, dtype=torch.float32).to(device).eval()
        self.entail = {v.lower(): int(k) for k, v in self.model.config.id2label.items()}["entailment"]

    def __call__(self, text: str, labels: list[str]) -> int:
        torch = self.torch
        with torch.no_grad():
            x = self.tokenizer([text] * len(labels), [f"This text is about {l}." for l in labels], truncation=True,
                               max_length=256, padding=True, return_tensors="pt").to(self.device)
            return int(self.model(**x).logits.float().softmax(-1)[:, self.entail].argmax())


def parse(reply: str, labels: list[str], condition: str) -> int | None:
    """The first label named in a reply (named), or the label letter it starts with (symbolic)."""
    if condition == "named":
        found = [(reply.lower().find(l.lower()), i) for i, l in enumerate(labels) if l.lower() in reply.lower()]
    else:
        found = [(0, i) for i, l in enumerate(labels) if reply.strip(" .:'\"").upper().startswith(l)]
    return min(found)[1] if found else None


def run_mk1(part: str, prior_strength: float = 2.0, channels: tuple = ("proto", "speaker")) -> dict:
    """Channels weighed by per-channel counts in this task (Beta(prior_strength / 2, prior_strength / 2) start, as the
    counting arbiter):
      proto    the part grown for this task: prototypes of frozen e5-small embeddings, updated with every feedback
      speaker  Gemma 4 E2B's guess (its replies from the gemma-e2b run of this part: the same call, made once)
      zero     the NLI satellite's zero-shot guess (named only; on dev it never helped, so it is off in the frozen MK1)
    """
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embed = Embedder(device)
    zero = ZeroShot(device) if "zero" in channels else None
    speaker = None
    if "speaker" in channels:
        speaker = json.loads((RESULTS / f"exam-taskstream_e2b_{part}.json").read_text(encoding="utf-8"))["conditions"]
    result = {"system": f"MK1 ({' + '.join(channels)} channels, counting arbiter)", "part": part, "conditions": {}}
    for condition in ("named", "symbolic"):
        tasks = []
        for task in stream(part, condition):
            replies = {t["task"]: t["replies"] for t in speaker[condition]["tasks"]}[task["task"]] if speaker else None
            labels, n = task["labels"], len(task["labels"])
            vectors = embed([it["text"] for it in task["items"]])
            sums, counts = torch.zeros(n, vectors.shape[1]), torch.zeros(n)
            hits = {c: [0.0, 0.0] for c in channels}  # right, spoke
            right = []
            for t, (v, item) in enumerate(zip(vectors, task["items"])):
                said = {}
                if "proto" in channels and counts.sum() > 0:
                    seen = counts > 0
                    sims = torch.where(seen, (sums / counts.clamp(min=1)[:, None]) @ v, torch.full((n,), -9.0))
                    said["proto"] = int(sims.argmax())
                if speaker is not None and (g := parse(replies[t], labels, condition)) is not None:
                    said["speaker"] = g
                if zero is not None and condition == "named":
                    said["zero"] = zero(item["text"], labels)
                log_p = torch.zeros(n)
                for channel, k in said.items():
                    r, s = hits[channel]
                    acc = (r + prior_strength * 0.5) / (s + prior_strength)
                    like = torch.full((n,), max(1e-6, (1 - acc) / (n - 1)))
                    like[k] = max(1e-6, acc)
                    log_p += like.log()
                guess = int(log_p.argmax()) if said else 0
                right.append(int(guess == item["label"]))
                for channel, k in said.items():  # feedback: the true label is revealed
                    hits[channel][0] += float(k == item["label"])
                    hits[channel][1] += 1
                sums[item["label"]] += v
                counts[item["label"]] += 1
            tasks.append({"task": task["task"], "accuracy": round(statistics.mean(right), 4), "right": right})
            print("mk1", condition, task["task"], tasks[-1]["accuracy"], flush=True)
        result["conditions"][condition] = {"accuracy": round(statistics.mean(t["accuracy"] for t in tasks), 4), "tasks": tasks}
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Skill acquisition on a stream of new tasks")
    parser.add_argument("--system", required=True, help="e2b | e4b | mk1")
    parser.add_argument("--set", choices=("dev", "final"), default="dev")
    parser.add_argument("--final", action="store_true", help="required to read the sealed final texts")
    args = parser.parse_args()
    if args.set == "final" and not args.final:
        raise SystemExit("the final set is sealed: pass --final, once, when the system is frozen")
    result = run_mk1(args.set) if args.system == "mk1" else run_gemma(args.system, args.set)
    path = RESULTS / f"exam-taskstream_{args.system}_{args.set}.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({c: v["accuracy"] for c, v in result["conditions"].items()}))
    print("Saved:", path)


if __name__ == "__main__":
    main()
