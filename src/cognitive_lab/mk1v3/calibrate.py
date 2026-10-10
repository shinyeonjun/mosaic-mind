"""v3 step 2: calibrators that turn satellite signals into probabilities (design/processing-flow-v3-2026-10-10.md).

Two heads, small L2 logistic regressions on the recorder's features (mk1v3/signals.py):
  right       P(the speaker's reply is right | it answered)
  answerable  P(the question can be answered from the documents)
Decision under the net score (+1 right / 0 abstain / -1 wrong, rejecting an unanswerable question +1):
  answer  ->  expected 2 * P(right) - 1        reject  ->  expected P(not answerable)
  answer iff 2 * P(right) - 1 > 1 - P(answerable). No tuned thresholds: the probabilities carry the decision.
Learned on calibration data only (SQuAD 2.0 dev, HotpotQA validation); exams are used only to check transfer.

python -m cognitive_lab.mk1v3.calibrate
"""

import json
import random

import numpy as np

ROOT = __import__("pathlib").Path(__file__).resolve().parents[3]
SIGNALS = ROOT / "artifacts" / "results" / "mk1v3-calibration-signals.jsonl"
WEIGHTS = ROOT / "artifacts" / "checkpoints" / "mk1v3-calibrator.json"
FEATURES = {
    "right": ["lp_mean", "lp_min", "lp_first", "gap_mean", "length", "sample_same_choice", "sample_overlap",
              "reader_margin", "reply_has_reader", "chain1", "chain2"],
    "answerable": ["rejected", "lp_mean", "lp_min", "sample_same_choice", "reader_margin", "chain1", "chain2"],
}


class Logistic:
    """L2 logistic regression by Newton steps, inputs standardised with the training mean and spread."""

    def __init__(self, l2: float = 1.0):
        self.l2 = l2

    def fit(self, X, y):
        X, y = np.asarray(X, float), np.asarray(y, float)
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-6
        A = np.hstack([(X - self.mu) / self.sd, np.ones((len(X), 1))])
        w = np.zeros(A.shape[1])
        reg = np.eye(A.shape[1]) * self.l2
        reg[-1, -1] = 0
        for _ in range(50):
            p = 1 / (1 + np.exp(-A @ w))
            w -= np.linalg.solve(A.T @ (A * (p * (1 - p))[:, None]) + reg, A.T @ (p - y) + reg @ w)
        self.w = w
        return self

    def predict(self, X):
        A = np.hstack([(np.asarray(X, float) - self.mu) / self.sd, np.ones((len(X), 1))])
        return 1 / (1 + np.exp(-A @ self.w))

    def to_dict(self):
        return {"mu": self.mu.tolist(), "sd": self.sd.tolist(), "w": self.w.tolist()}

    @classmethod
    def from_dict(cls, d):
        m = cls()
        m.mu, m.sd, m.w = np.array(d["mu"]), np.array(d["sd"]), np.array(d["w"])
        return m


def matrix(rows: list[dict], head: str):
    X = [[float(r[f]) for f in FEATURES[head]] for r in rows]
    y = [float(r["right"] if head == "right" else r["answerable"]) for r in rows]
    return np.array(X), np.array(y)


def head_rows(rows: list[dict], head: str) -> list[dict]:
    return [r for r in rows if not r["rejected"]] if head == "right" else rows


def auroc(p, y) -> float:
    p, y = np.asarray(p), np.asarray(y)
    pos, neg = p[y == 1], p[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    return float((pos[:, None] > neg[None]).mean() + 0.5 * (pos[:, None] == neg[None]).mean())


def ece(p, y, bins: int = 10) -> float:
    p, y = np.asarray(p), np.asarray(y)
    b = np.clip((p * bins).astype(int), 0, bins - 1)
    return float(sum(abs(p[b == k].mean() - y[b == k].mean()) * (b == k).mean() for k in range(bins) if (b == k).any()))


def crossval(rows: list[dict], head: str, folds: int = 5) -> np.ndarray:
    rows = head_rows(rows, head)
    X, y = matrix(rows, head)
    idx = list(range(len(rows)))
    random.Random(0).shuffle(idx)
    pred = np.zeros(len(rows))
    for k in range(folds):
        test = idx[k::folds]
        train = sorted(set(idx) - set(test))
        pred[test] = Logistic().fit(X[train], y[train]).predict(X[test])
    return pred


def decide(r: dict, models: dict) -> str:
    """'answer' or 'reject' for one recorded question."""
    p_ans = models["answerable"].predict(matrix([r], "answerable")[0])[0]
    if r["rejected"]:
        return "reject"  # the speaker gave no answer to keep (rescue by another satellite is a later step)
    p_right = models["right"].predict(matrix([r], "right")[0])[0]
    return "answer" if 2 * p_right - 1 > 1 - p_ans else "reject"


def net(r: dict, choice: str) -> int:
    if choice == "reject":
        return 0 if r["answerable"] else 1
    if not r["answerable"]:
        return -1
    return 1 if r["right"] else -1


def report(rows: list[dict]) -> dict:
    out = {"questions": len(rows), "by_source": {}}
    for head in FEATURES:
        hr = head_rows(rows, head)
        y = matrix(hr, head)[1]
        p = crossval(rows, head)
        out[head] = {"n": len(hr), "positive": int(y.sum()), "auroc_cv": round(auroc(p, y), 3), "ece_cv": round(ece(p, y), 3),
                     "single": {f: round(max(auroc(matrix(hr, head)[0][:, i], y), 1 - auroc(matrix(hr, head)[0][:, i], y)), 3)
                                for i, f in enumerate(FEATURES[head])}}
    # transfer between sources: learn on one, test on the other
    for train_src, test_src in (("squad", "hotpot"), ("hotpot", "squad")):
        tr = [r for r in rows if r["source"] == train_src]
        te = [r for r in rows if r["source"] == test_src]
        res = {}
        for head in FEATURES:
            m = Logistic().fit(*matrix(head_rows(tr, head), head))
            Xt, yt = matrix(head_rows(te, head), head)
            p = m.predict(Xt)
            res[head] = {"auroc": round(auroc(p, yt), 3), "ece": round(ece(p, yt), 3)}
        models = {h: Logistic().fit(*matrix(head_rows(tr, h), h)) for h in FEATURES}
        res["net_speaker_alone"] = round(float(np.mean([net(r, "reject" if r["rejected"] else "answer") for r in te])), 4)
        res["net_calibrated"] = round(float(np.mean([net(r, decide(r, models)) for r in te])), 4)
        out["by_source"][f"{train_src} -> {test_src}"] = res
    return out


def main() -> None:
    rows = [json.loads(l) for l in SIGNALS.read_text(encoding="utf-8").splitlines()]
    for r in rows:
        r["rejected"] = float(r["rejected"])
    out = report(rows)
    print(json.dumps(out, indent=1))
    models = {h: Logistic().fit(*matrix(head_rows(rows, h), h)).to_dict() for h in FEATURES}
    WEIGHTS.write_text(json.dumps({"features": FEATURES, "models": models}), encoding="utf-8")
    (ROOT / "artifacts" / "results" / "mk1v3-calibrator-report.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("Saved:", WEIGHTS)


if __name__ == "__main__":
    main()
