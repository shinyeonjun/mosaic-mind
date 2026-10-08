"""Objectivity audit of everything so far (design/audit-2026-10-08.md).

Measures what each claim rests on:
  A. leakage: test articles the reading specialist saw while training (KLUE train and validation share passages)
  B. rule worlds: test sentences that also occur in training
  C. a choice made by looking at test scores (reading only relevant hypotheses), re-checked on fresh seeds
  D. key MK1 results again with three part seeds (42/43/44, test worlds of the same seed) and with
     session-level (cluster) bootstrap: questions in one session share sources and memory, so a bootstrap
     over questions makes intervals too narrow
  E. the arbiter on board accuracies it never saw in training, against the Bayes ideal

python -m cognitive_lab.mk1.audit
"""

import json
import time

import torch

from cognitive_lab.world2.integrated import RESULTS_DIR

SEEDS = (42, 43, 44)


def cluster_bootstrap(values: torch.Tensor, session: torch.Tensor, seed: int = 0, rounds: int = 2000) -> list[float]:
    """95% interval of the mean, resampling whole sessions."""
    values, session = values.cpu().float(), session.cpu()
    ids = session.unique()
    sums = torch.zeros(len(ids)).index_add_(0, torch.searchsorted(ids, session), values)
    counts = torch.zeros(len(ids)).index_add_(0, torch.searchsorted(ids, session), torch.ones_like(values))
    g = torch.Generator().manual_seed(seed)
    pick = torch.randint(len(ids), (rounds, len(ids)), generator=g)
    means = (sums[pick].sum(1) / counts[pick].sum(1)).sort().values
    return [round(means[int(0.025 * rounds)].item(), 4), round(means[int(0.975 * rounds)].item(), 4)]


def episode_bootstrap(values: torch.Tensor, seed: int = 0, rounds: int = 2000) -> list[float]:
    values = values.cpu().float()
    g = torch.Generator().manual_seed(seed)
    means = values[torch.randint(len(values), (rounds, len(values)), generator=g)].mean(1).sort().values
    return [round(means[int(0.025 * rounds)].item(), 4), round(means[int(0.975 * rounds)].item(), 4)]


def session_ids(sessions: list[dict]) -> torch.Tensor:
    return torch.tensor([i for i, s in enumerate(sessions) for _ in s["episodes"]])


def leakage() -> dict:
    from cognitive_lab.world5 import klue
    from cognitive_lab.world5.reader_qa import load_predictions, reader_training_rows

    seen_passages = {r["context"] for r in reader_training_rows()}
    raw_train, raw_validation = klue._read("train"), klue._read("validation")
    out = {"klue_train_validation_shared_passages": len({r["context"] for r in raw_train} & {r["context"] for r in raw_validation})}
    predictions = load_predictions()
    for part in ("train", "validation", "test"):
        questions = klue.load(part)
        out[f"{part}_questions_with_passage_seen_in_reader_training"] = round(sum(q["context"] in seen_passages for q in questions) / len(questions), 4)
    answerable = [q for q in klue.load("test") if not q["impossible"]]
    for name, keep in (("seen_passage", True), ("unseen_passage", False)):
        chosen = [q for q in answerable if (q["context"] in seen_passages) == keep]
        out[f"reader_right_rate_{name}"] = {"rate": round(sum(klue.is_right(predictions[q["guid"]]["answer"], q["answers"]) for q in chosen) / len(chosen), 4),
                                            "questions": len(chosen)}
    return out, seen_passages


def template_overlap() -> dict:
    from cognitive_lab.world2 import hedged
    from cognitive_lab.world3 import chains

    train = {x for e in chains.generate("train", 45) for x in e["utterances"]}
    test = [x for e in chains.generate("test", 45) for x in e["utterances"]]
    out = {"v3_test_sentences_seen_in_training": round(sum(x in train for x in test) / len(test), 4)}
    train = {u["text"] for s in hedged.generate_part("train", 42) for e in s["episodes"] for u in e["utterances"]}
    test = [u["text"] for s in hedged.generate_part("test", 42) for e in s["episodes"] for u in e["utterances"]]
    out["v2h_test_utterances_seen_in_training"] = round(sum(x in train for x in test) / len(test), 4)
    return out


def relevance_fresh(device) -> dict:
    """Reading only relevant hypotheses was chosen while looking at graduation (test) scores. Fresh seeds:
    every answer must be the same as with full reading."""
    from cognitive_lab.mk1.system import MK1, chain_sessions, composite_sessions
    from cognitive_lab.world3 import chains
    from cognitive_lab.world7.routing import generate

    full, relevant = MK1(device, reading="full"), MK1(device, reading="relevant")
    out = {}
    for name, sessions, policy in (("v3 test seed 46", chain_sessions(chains.generate("test", 46)), "think"),
                                   ("v7 test seed 43", composite_sessions(generate("test", 43)), "route")):
        a, b = full(sessions, policy)[0]["answer"], relevant(sessions, policy)[0]["answer"]
        out[name] = {"answers": len(a), "same_answer": round(sum(x == y for x, y in zip(a, b)) / len(a), 4)}
    full.reader.save()
    relevant.reader.save()
    return out


def per_seed(device, seed: int, seen_passages: set) -> dict:
    from cognitive_lab.mk1.arbiter import BOARD_ACCURACIES
    from cognitive_lab.mk1.articles import article_sessions
    from cognitive_lab.mk1.memory import LongTermMemory, stream
    from cognitive_lab.mk1.stage_d import net_scores
    from cognitive_lab.mk1.system import MK1, composite_sessions, episode_score
    from cognitive_lab.world5 import klue
    from cognitive_lab.world6 import world
    from cognitive_lab.world7.routing import generate
    from cognitive_lab.world10 import world as w10

    model = MK1(device, seed=seed)
    out = {}
    # C: asking speakers (curiosity) inside the whole system, T questions.
    raw7 = generate("test", seed)
    sessions = composite_sessions(raw7)
    t = torch.tensor([e["tag"] == "T" for s in sessions for e in s["episodes"]])
    sid = session_ids(sessions)
    off = net_scores(sessions, model(sessions, "route", curious=False)[0])
    on = net_scores(sessions, model(sessions, "route", curious=True)[0])
    out["C_curiosity_gain_on_T"] = {"gain": round((on - off)[t].mean().item(), 4),
                                    "ci_questions": episode_bootstrap((on - off)[t]), "ci_sessions": cluster_bootstrap((on - off)[t], sid[t])}
    # D2: live reading of real articles, split by passages the reading specialist saw in training.
    raw6 = world.generate("test", seed)
    articles = article_sessions(raw6, "test")
    live = net_scores(articles, model(articles)[0])
    contexts = {q["question"]: q["context"] for q in klue.load("test")}
    seen = torch.tensor([contexts[e["question"]] in seen_passages for s in articles for e in s["episodes"]])
    out["D_articles_live"] = {"score": round(live.mean().item(), 4), "seen_passage": round(live[seen].mean().item(), 4),
                              "unseen_passage": round(live[~seen].mean().item(), 4), "seen_share": round(seen.float().mean().item(), 4)}
    # E: the stage-7 router on article doors, unseen combination AC.
    linked = w10.generate("test", seed)
    e_eps = [e for s in linked for e in s["episodes"]]
    ac = torch.tensor([e["tag"] == "AC" for e in e_eps])
    route = net_scores(linked, model(linked, "route")[0])
    ask = net_scores(linked, model(linked, "ask-available")[0])
    never = net_scores(linked, model(linked, "think")[0])
    needs = torch.tensor([e["tag"] in ("A", "AC") for e in e_eps])
    oracle = torch.where(needs, ask, never)
    out["E_router_vs_oracle_AC"] = {"router": round(route[ac].mean().item(), 4), "oracle": round(oracle[ac].mean().item(), 4)}
    # F: arbiter vs merging on boards that can be wrong.
    v11 = w10.generate("test", seed, board_accuracies=BOARD_ACCURACIES)
    sid11 = session_ids(v11)
    arb, merge = net_scores(v11, model(v11, "arbitrate")[0]), net_scores(v11, model(v11, "ask-available")[0])
    out["F_arbiter_minus_merge"] = {"diff": round((arb - merge).mean().item(), 4), "ci_questions": episode_bootstrap(arb - merge),
                                    "ci_sessions": cluster_bootstrap(arb - merge, sid11)}
    # H: long-term memory over the article stream, then frozen into fresh article-door sessions.
    episodes = [e for s in articles for e in s["episodes"]]
    sid6 = session_ids(articles)
    model.memory = None
    off_answers, _ = stream(model, articles)
    model.memory = LongTermMemory()
    on_answers, _ = stream(model, articles)
    off6 = torch.tensor([episode_score(e, a) for e, a in zip(episodes, off_answers)])
    on6 = torch.tensor([episode_score(e, a) for e, a in zip(episodes, on_answers)])
    out["H_memory_gain"] = {"gain": round((on6 - off6).mean().item(), 4), "ci_questions": episode_bootstrap(on6 - off6),
                            "ci_sessions": cluster_bootstrap(on6 - off6, sid6)}
    memory = model.memory
    memory.frozen = True
    fresh = w10.generate("test", seed + 100, board_accuracies=BOARD_ACCURACIES)
    sidf = session_ids(fresh)
    model.memory = None
    two = net_scores(fresh, model(fresh, "arbitrate")[0])
    model.memory = memory
    three = net_scores(fresh, model(fresh, "arbitrate")[0])
    model.memory = None
    out["H_transfer_gain"] = {"gain": round((three - two).mean().item(), 4), "ci_questions": episode_bootstrap(three - two),
                              "ci_sessions": cluster_bootstrap(three - two, sidf)}
    model.articles.reader = None
    return out


def arbiter_unseen_accuracies(device) -> dict:
    """The arbiter trained on board accuracies 1.0 / 0.75 / 0.5, tested on 0.9 and 0.25."""
    from cognitive_lab.mk1.arbiter import door_scores
    from cognitive_lab.mk1.system import MK1
    from cognitive_lab.world10 import world as w10

    model = MK1(device)
    sessions = w10.generate("test", 42, board_accuracies=(0.9, 0.25))
    views = model.channel_views(sessions)
    says, truth = views["says"], views["truth"]
    accuracy = torch.tensor([s["board_accuracy"] for s in sessions], device=device)[:, None].expand_as(truth)
    a_spoke = says[:, :, 1].amax(-1) > 0
    q = 0.65  # article part accuracy when it concludes (training data, stage F)
    prior = torch.tensor((1.0, 0.75, 0.5), device=device)  # the Bayes judge knows only the training levels

    def bayes():
        batch, episodes = truth.shape
        post = torch.ones(batch, 3, device=device) / 3
        out = []
        for t in range(episodes):
            bs, a_s = says[:, t, 0], says[:, t, 1]
            rb = (post * prior).sum(-1, keepdim=True)
            lik = torch.where(bs.amax(-1, keepdim=True) > 0, torch.where(bs > 0, rb, (1 - rb) / 2), torch.ones_like(bs))
            lik = lik * torch.where(a_s.amax(-1, keepdim=True) > 0, torch.where(a_s > 0, q, (1 - q) / 2), torch.ones_like(a_s))
            p = lik / lik.sum(-1, keepdim=True)
            out.append(torch.cat([p, torch.zeros(batch, 1, device=device)], -1).clamp(min=1e-9).log())
            right = bs.gather(-1, truth[:, t, None]).squeeze(-1)
            spoke = bs.amax(-1, keepdim=True) > 0
            post = post * torch.where(spoke, torch.where(right[:, None] > 0, prior, 1 - prior), torch.ones_like(post))
            post = post / post.sum(-1, keepdim=True)
        return torch.stack(out, 1)

    with torch.no_grad():
        arb = door_scores(model.arbiter(says, truth), truth)
    bay = door_scores(bayes(), truth)
    board_only = door_scores(torch.cat([says[:, :, 0] * 20, torch.zeros_like(truth)[..., None]], -1), truth)
    article_only = door_scores(torch.cat([says[:, :, 1] * 20, torch.zeros_like(truth)[..., None]], -1), truth)
    out = {}
    for r in (0.9, 0.25):
        m = accuracy == r
        out[str(r)] = {"arbiter": round(arb[m].mean().item(), 4), "bayes_with_training_levels": round(bay[m].mean().item(), 4),
                       "board_only": round(board_only[m].mean().item(), 4), "article_only": round(article_only[m].mean().item(), 4)}
    out["article_spoke_share"] = round(a_spoke.float().mean().item(), 4)
    return out


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    started = time.perf_counter()
    result = {"audit": "2026-10-08"}
    result["A_leakage"], seen_passages = leakage()
    print("A", json.dumps(result["A_leakage"], ensure_ascii=False), flush=True)
    result["B_rule_world_sentences"] = template_overlap()
    print("B", json.dumps(result["B_rule_world_sentences"], ensure_ascii=False), flush=True)
    result["C_relevance_on_fresh_seeds"] = relevance_fresh(device)
    print("C", json.dumps(result["C_relevance_on_fresh_seeds"], ensure_ascii=False), flush=True)
    result["D_seeds"] = {}
    for seed in SEEDS:
        result["D_seeds"][seed] = per_seed(device, seed, seen_passages)
        print("D", seed, json.dumps(result["D_seeds"][seed], ensure_ascii=False), flush=True)
        torch.cuda.empty_cache()
    result["E_arbiter_unseen_accuracies"] = arbiter_unseen_accuracies(device)
    print("E", json.dumps(result["E_arbiter_unseen_accuracies"], ensure_ascii=False), flush=True)
    result["seconds"] = round(time.perf_counter() - started, 1)
    path = RESULTS_DIR / "mk1-audit.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved: {path} ({result['seconds']:.0f}s)")


if __name__ == "__main__":
    main()
