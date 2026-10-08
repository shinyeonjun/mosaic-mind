"""Rule world v2-A: the tone-cue world (v2-H) where the judge may ask one question per episode.

After hearing the utterances, the judge may ask one speaker about the queried door, for a cost
c that is subtracted from the episode score. A speaker who has not yet spoken about that door
answers with a fresh statement drawn exactly like a v2 statement: assert (0.8) or negate (0.2)
with the speaker's current accuracy, a v1 template, and a v2-H tone cue. A speaker who already
spoke about it says nothing new (the cost is still paid). The answer is random but fixed per
(episode, speaker), so counterfactual training targets and the test are reproducible.

The ideal comparison is the Bayes learner with cues (it learns accuracies from feedback and
knows the cue probabilities) choosing by myopic value of information:
    VOI(s) = E_answer[ max(2 max_k q'_k - 1, 0) ] - max(2 max_k q_k - 1, 0),   ask argmax if > c.

python -m cognitive_lab.world2.asking     # ideal policies for several costs
"""

import random

from cognitive_lab.world.templates import ENTITY_SETS, TEMPLATES, door_name, key_name
from cognitive_lab.world2 import hedged
from cognitive_lab.world2.bayes_learner import PRIOR, claim_likelihood
from cognitive_lab.world2.generator import ACCURACY_LEVELS, accuracies_at

ASK_KINDS = {"assert": 0.8, "negate": 0.2}  # v2 chains: assert 0.65 + corrected 0.15, negate 0.20


def asked_door_label(episode: dict) -> str:
    return episode["gold_question"]["door"]


def has_spoken(episode: dict, speaker: str) -> bool:
    door = asked_door_label(episode)
    return any(e["type"] != "chatter" and e["source"] == speaker and e["door"] == door
               for e in episode["gold_events"])


def ask(session: dict, episode: dict, speaker: str) -> tuple[dict, dict] | None:
    """(utterance, gold event) answering a question to `speaker`, or None if nothing new."""
    if has_spoken(episode, speaker):
        return None
    rng = random.Random(f"world-v2-A|ask|{episode['episode_id']}|{speaker}")
    doors, keys = ENTITY_SETS["base"]["doors"], ENTITY_SETS["base"]["keys"]
    accuracy = accuracies_at(session, episode["index"])[session["speakers"].index(speaker)]
    truth = episode["entities"]["keys"].index(episode["answer"])
    door = doors.index(asked_door_label(episode))
    kind = rng.choices(list(ASK_KINDS), weights=list(ASK_KINDS.values()))[0]
    negative = kind == "negate"
    wrong = [k for k in range(3) if k != truth]
    if negative:
        key = rng.choice(wrong) if rng.random() < accuracy else truth
    else:
        key = truth if rng.random() < accuracy else rng.choice(wrong)
    right = (key != truth) if negative else (key == truth)
    cue = rng.choices(hedged.CUES, weights=[(hedged.CUE_GIVEN_RIGHT if right else hedged.CUE_GIVEN_WRONG)[c]
                                             for c in hedged.CUES])[0]
    _, template = rng.choice(TEMPLATES[kind])
    text = rng.choice(hedged.CUE_PREFIXES[cue]) + template.format(D=door_name(doors[door]), K=key_name(keys[key]))
    event = {"step": len(episode["utterances"]), "type": kind, "door": doors[door], "key": keys[key],
             "polarity": not negative, "source": speaker, "cue": cue, "asked": True}
    return {"source": speaker, "text": text}, event


def with_answer(episode: dict, answer: tuple[dict, dict] | None) -> dict:
    """The episode as heard after the question (unchanged if nothing new was said)."""
    if answer is None:
        return episode
    utterance, event = answer
    return {**episode, "utterances": episode["utterances"] + [utterance],
            "gold_events": episode["gold_events"] + [event]}


# ---- ideal judge: Bayes learner with cues ------------------------------------------------

def _claim_weight(claim, truth: int, belief: list[float]) -> float:
    _, key, polarity, cue = claim
    right = (key == truth) if polarity else (key != truth)
    return hedged.cue_factor(cue, right) * sum(b * claim_likelihood(key, polarity, truth, l)
                                               for b, l in zip(belief, ACCURACY_LEVELS))


def posterior(claims, beliefs: dict) -> list[float]:
    weights = [1.0, 1.0, 1.0]
    for claim in claims:
        for truth in range(3):
            weights[truth] *= _claim_weight(claim, truth, beliefs[claim[0]])
    total = sum(weights)
    return [w / total for w in weights]


def value(q: list[float]) -> float:
    return max(2 * max(q) - 1, 0.0)


def voi(q: list[float], belief: list[float]) -> float:
    """Expected gain in decision value from one fresh statement by a speaker with this level belief."""
    expected = 0.0
    for polarity, p_kind in ((True, ASK_KINDS["assert"]), (False, ASK_KINDS["negate"])):
        for key in range(3):
            for cue in hedged.CUES:
                joint = [q[t] * p_kind * _claim_weight((None, key, polarity, cue), t, belief) for t in range(3)]
                mass = sum(joint)
                if mass > 0:
                    expected += mass * value([j / mass for j in joint])
    return expected - value(q)


def _score(q: list[float], truth: int) -> int:
    best = max(range(3), key=q.__getitem__)
    return 0 if q[best] <= 0.5 else (1 if best == truth else -1)


def ideal_policies(sessions: list[dict], cost: float, hazard: float = 0.01) -> dict:
    """Net score (answer score minus cost) and ask rate for: never ask, the hand rule (ask a
    random silent speaker when it would abstain), and the VOI policy."""
    policies = ("never", "hand-rule", "voi")
    net = {p: 0.0 for p in policies}
    asks = {p: 0 for p in policies}
    count = 0
    for session in sessions:
        beliefs = {p: {s: list(PRIOR) for s in session["speakers"]} for p in policies}
        for episode in session["episodes"]:
            truth = episode["entities"]["keys"].index(episode["answer"])
            silent = [s for s in session["speakers"] if not has_spoken(episode, s)]
            for policy in policies:
                belief = beliefs[policy]
                for s in belief:
                    belief[s] = [(1 - hazard) * b + hazard * p for b, p in zip(belief[s], PRIOR)]
                claims = hedged.final_claims(episode)
                q = posterior(claims, belief)
                target = None
                if policy == "hand-rule" and max(q) <= 0.5 and silent:
                    target = random.Random(f"hand|{episode['episode_id']}").choice(silent)
                elif policy == "voi" and silent:
                    gains = {s: voi(q, belief[s]) for s in silent}
                    best = max(gains, key=gains.get)
                    target = best if gains[best] > cost else None
                heard = with_answer(episode, ask(session, episode, target)) if target else episode
                if target:
                    asks[policy] += 1
                    claims = hedged.final_claims(heard)
                    q = posterior(claims, belief)
                net[policy] += _score(q, truth) - (cost if target else 0.0)
                for speaker, key, polarity, _ in claims:
                    right = (key == truth) if polarity else (key != truth)
                    updated = [b * (l if right else 1 - l) for b, l in zip(belief[speaker], ACCURACY_LEVELS)]
                    belief[speaker] = [u / sum(updated) for u in updated]
            count += 1
    return {p: {"net": round(net[p] / count, 4), "ask_rate": round(asks[p] / count, 3)} for p in policies}


def main() -> None:
    sessions = hedged.generate_part("test", 42)
    for cost in (0.05, 0.1, 0.15, 0.2, 0.3):
        print(f"cost {cost}: {ideal_policies(sessions, cost)}", flush=True)


if __name__ == "__main__":
    main()
