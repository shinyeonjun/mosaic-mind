# One controller for MK1: literature on routing, deferral, combining, metareasoning, bandits

Scope: evidence 2018-2026 (plus classic foundations), focused on what a laptop-scale controller over
satellites {extractive reader, NLI verifier, hop selector + multi-hop reader, online prototype classifier,
ARC program search, Gemma E2B} can actually use. Numbers marked (mem) are from memory and were not
re-verified in this session; everything else was checked against the abstract/listing.

---

## 1. Cascades and routing between models of different cost

| Work | Year/venue | Mechanism | Escalation signal | Key number |
|---|---|---|---|---|
| FrugalGPT (Chen, Zaharia, Zou) https://arxiv.org/abs/2305.05176 | 2023 arXiv / TMLR 2024 | Sequential cascade of APIs; stop when a scorer accepts | Small learned scorer (DistilBERT) on (query, answer) -> P(answer correct); thresholds per stage chosen on dev | Matches GPT-4 with up to 98% cost cut, or +4% acc at same cost (mem) |
| AutoMix (Madaan et al.) https://arxiv.org/abs/2310.12963 | NeurIPS 2024 | Small LM answers, then few-shot self-verifies against context; POMDP "meta-verifier" maps noisy verifier scores to route | Self-verification score, de-noised by POMDP over observed verifier outputs | >50% compute cut at comparable performance; IBC up to 57-89% depending on version |
| MoT cascade (Yue et al.) https://arxiv.org/abs/2310.03094 | ICLR 2024 | Sample several answers from weak LM (CoT + Program-of-Thought); escalate only if they disagree | Answer consistency / vote margin across *different reasoning representations* | Same acc as GPT-4 alone at ~40% of its cost on 6 reasoning sets |
| Hybrid LLM (Ding et al.) https://arxiv.org/abs/2404.14618 | ICLR 2024 | Router (DeBERTa) predicts quality gap small-vs-large from query only | Learned router on query, trained on soft "small is good enough" labels | Up to 40% fewer large-model calls with no quality drop |
| RouteLLM (Ong et al.) https://arxiv.org/abs/2406.18665 | ICLR 2025 | Binary strong/weak router (matrix factorisation, BERT, causal-LM classifier, similarity-weighted ranking) from Chatbot Arena preferences + augmentation | Learned P(strong wins) from query only | >2x cost cut without quality loss; ~95% of GPT-4 MT-Bench with ~14% GPT-4 calls (mem); transfers to unseen model pairs |
| Zooter (Lu et al.) https://arxiv.org/abs/2311.08692 | NAACL 2024 | Distil reward-model scores of all candidates into a query->model router | Learned from reward-model "soft labels", query only | Beats best single model on average; best on ~44% of tasks (mem) |
| Tryage (Hari & Thomson) https://arxiv.org/abs/2308.11601 | 2023 arXiv | Router predicts each expert's downstream loss on the prompt | Predicted per-expert loss + user constraints | Picks optimal model 50.9% vs 23.6% for GPT-3.5 as router (mem) |
| RouterBench (Hu et al.) https://arxiv.org/abs/2403.12031 | 2024 arXiv / ICML-W | 405k precomputed outcomes, cost-quality frontier eval | — | Simple routers often barely beat best single model; big gap to oracle (lesson: routing gain is bounded by complementarity) |
| GraphRouter (Feng et al.) https://arxiv.org/abs/2410.03834 | ICLR 2025 | Heterogeneous graph (task, query, LLM nodes); edge prediction of effect/cost | GNN over query-task-model graph; new LLMs added as nodes | ≥12.3% over prior routers (mem); generalises to new LLMs without retrain |
| Universal routing (Jitkrittum et al.) https://arxiv.org/abs/2502.08773 | 2025 arXiv | Represent each model by its error vector on a small fixed prompt set; router is cluster-based | Cluster of query -> per-model error rate on that cluster | Handles models unseen at training — relevant when satellites are added |
| Confidence-cascade theory (Jitkrittum et al.) https://arxiv.org/abs/2307.02764 | NeurIPS 2023 | When is "defer if small model's max-prob < t" optimal? | — | Fails when the downstream model is a *specialist*, or under label noise / distribution shift; then a post-hoc learned deferral rule using both models' features wins |
| Token-level cascades (Gupta et al.) https://arxiv.org/abs/2404.10136 | ICLR 2024 | Deferral from quantiles of per-token log-probs, + learned post-hoc deferral | Token-logprob quantiles (not just mean) | Better cost-quality than sequence-level confidence |
| Cascade-aware training (CAT) https://arxiv.org/abs/2406.00060 | 2024 arXiv | Train the small model's loss to focus on examples the big model gets wrong / the small model can win | — | Better cascade tradeoff on 60+ SuperGLUE/WMT/FLAN tasks |
| CALM early exit (Schuster et al.) https://arxiv.org/abs/2207.07061 | NeurIPS 2022 | Per-token early exit with calibrated confidence via Learn-then-Test | Softmax gap / hidden-state saturation / learned classifier; thresholds with statistical guarantee | Up to ~3x speedup with guaranteed quality |

**Takeaways.** (a) Post-hoc signals computed *after* a cheap model answers (verifier score, agreement across
samples or across *representations*) beat query-only routers when the cheap pass is affordable; query-only
routers win when the cheap pass itself is expensive. (b) Gains vs single strong model are mostly *cost*,
not accuracy; accuracy gains appear only when models are complementary (RouterBench oracle gap). (c) Plain
max-probability deferral breaks for specialists — exactly MK1's case (ARC solver, extractive reader).

**Applicability to MK1.** Everything runs locally, so "cost" = latency/VRAM swaps, and MK1 can afford to
run cheap satellites first and decide afterwards (AutoMix/MoT style). The MoT idea maps directly: agreement
between *different* satellites (extractive span vs LLM answer vs NLI-entailed claim) is a better escalation
signal than any one confidence. Use the Jitkrittum lesson: the deferral rule must take features from
*both* the satellite and the context (task type, satellite's own score), not a single global threshold.
Universal-routing's "model = error vector on a probe set" is a cheap way to plug in new satellites.

---

## 2. Learning to defer, selective prediction, calibrated P(correct)

**Foundations.**
- Selective classification, SGR (Geifman & El-Yaniv, NeurIPS 2017) https://arxiv.org/abs/1705.08500 — pick a
  threshold on any confidence score so that risk <= r* with prob 1-delta (binomial bound on a held-out set).
  Needs only labelled calibration data and a ranking score. SelectiveNet (ICML 2019) trains the rejector jointly.
- Learning to defer (Madras, Pitassi, Zemel, NeurIPS 2018) https://arxiv.org/abs/1711.06664 — model learns
  when to hand off to a downstream decision maker whose errors it accounts for.
- Consistent L2D surrogate (Mozannar & Sontag, ICML 2020) https://arxiv.org/abs/2006.01862 — softmax over
  K classes + "defer" with cross-entropy surrogate; Bayes-consistent. Calibrated L2D with one-vs-all
  (Verma & Nalisnick, ICML 2022) https://arxiv.org/abs/2202.03673 gives calibrated P(expert correct).
  Multi-expert L2D (Verma, Barrenechea, Nalisnick, AISTATS 2023) https://arxiv.org/abs/2210.16955.
  Two-stage L2D for *fixed, pre-trained* predictors (Mao, Mohri, Zhong, 2023-24) https://arxiv.org/abs/2310.14774
  — learn only the deferral function over frozen experts. **This is MK1's setting** (frozen satellites).
- Conformal risk control (Angelopoulos et al., ICLR 2024) https://arxiv.org/abs/2208.02814 and Learn-then-Test
  https://arxiv.org/abs/2110.01052 — choose thresholds controlling any monotone loss (e.g., wrong-answer rate)
  with finite-sample guarantee; ~a few hundred calibration points suffice. Conformal LM (Quach et al., ICLR 2024)
  https://arxiv.org/abs/2306.10193; conformal factuality (Mohri & Hashimoto, ICML 2024) https://arxiv.org/abs/2402.10978.
- Selective QA under domain shift (Kamath, Jia, Liang, ACL 2020) https://arxiv.org/abs/2006.09462 — a small
  random-forest *calibrator* on hand features (softmax probs of top spans, span length, input length) trained
  partly on out-of-domain data beats the QA model's own softmax for abstention under shift. Directly
  analogous to MK1's extractive reader.

**LLM confidence estimators — what holds for small models.**
- P(True)/P(IK) (Kadavath et al., 2022) https://arxiv.org/abs/2207.05221 — calibration and self-evaluation
  *improve with scale*; small models are poor at it. Expect weak P(IK) from a 2B speaker.
- Verbalized confidence (Tian et al. 2023 https://arxiv.org/abs/2305.14975; Xiong et al. ICLR 2024
  https://arxiv.org/abs/2306.13063) — overconfident, clusters at 80-100%; works mainly for large RLHF models.
  Not reliable for E2B.
- Semantic entropy (Kuhn et al., ICLR 2023 https://arxiv.org/abs/2302.09664; Farquhar et al., Nature 2024
  https://www.nature.com/articles/s41586-024-07421-0) — sample ~5-10 answers, cluster by bidirectional
  entailment (an NLI model — MK1 already has one), entropy over clusters. Robust across sizes but 5-10x cost.
  Semantic-entropy probes (Kossen et al. 2024) https://arxiv.org/abs/2406.15927 predict it from one hidden state.
- LM-Polygraph benchmark https://arxiv.org/abs/2406.15627 — on 7B models, cheap white-box information scores
  (max sequence probability, mean token entropy) are competitive or best; results vary by task and metric.
- Fine-tuned uncertainty (Kapoor et al., NeurIPS 2024) https://arxiv.org/abs/2406.08391 — ~1k graded examples
  + small LoRA/probe gives far better calibration than verbalized or sampling methods, and transfers across
  tasks (mem for exact numbers). Supports "train a tiny correctness head on features" over prompting.

**Applicability to MK1.** Do not trust E2B's verbalized confidence or P(IK). Use (1) a two-stage L2D/
calibrator: a small logistic/GBM model per *satellite* predicting P(satellite correct | features) from
features you already have (reader span score & margin, NLI entail/contradict probs, hop selector score,
prototype cosine margin, program-search "found consistent program" flag, LLM token logprob quantiles,
cross-satellite agreement); (2) conformal / SGR thresholds on that P(correct) to fix a target error rate
for abstention, giving a guarantee instead of dev-tuned thresholds. Semantic entropy via MK1's own NLI
is the expensive fallback only when cheap signals are ambiguous.

---

## 3. Combining several imperfect sources

- **Dawid-Skene** (1979) — EM over per-source confusion matrices, no labels needed; weights sources by
  estimated accuracy. **Bayesian classifier combination** (Kim & Ghahramani, AISTATS 2012)
  http://proceedings.mlr.press/v22/kim12/kim12.pdf — Bayesian DS with priors, better with few items.
- **Snorkel / data programming** (Ratner et al., NeurIPS 2016 / VLDB 2017) https://arxiv.org/abs/1711.10160;
  **structure learning for correlated sources** (Bach et al., ICML 2017) https://arxiv.org/abs/1703.00854 —
  explicitly models dependencies; ignoring correlated labelling functions double-counts them.
- **Weighted majority / Hedge / multiplicative weights** (Littlestone & Warmuth 1994; Freund & Schapire 1997;
  Arora-Hazan-Kale survey 2012 https://theoryofcomputing.org/articles/v008a006/) — w_i <- w_i * exp(-eta*loss_i);
  regret O(sqrt(T log N)) vs best source in hindsight; needs only per-item feedback. Fully online.
- **Product of experts** (Hinton 2002) — multiply (log-add) calibrated probabilities; sharp but assumes
  conditional independence; overconfident when experts share errors.
- **Stacking** (Wolpert 1992) — meta-learner on base outputs, trained on out-of-fold predictions; the
  standard way to learn context-dependent trust. Needs labels (hundreds suffice for a logistic stacker).
- **Self-consistency** (Wang et al., ICLR 2023) https://arxiv.org/abs/2203.11171 — majority over sampled
  reasoning paths, +17.9% GSM8K on PaLM-540B; gains shrink for small models whose samples share errors.
- **Mixture-of-Agents** (Wang et al., 2024; ICLR 2025) https://arxiv.org/abs/2406.04692 — layered LLMs read
  each other's answers; 65.1% AlpacaEval 2.0 LC with open models vs 57.5% GPT-4o. **Rethinking MoA / Self-MoA**
  (Li et al., 2025) https://arxiv.org/abs/2502.00674 — mixing samples from the single best model beats mixing
  diverse weaker models (+6.6% AlpacaEval (mem)); quality of each source dominates diversity; adding weak
  sources hurts.
- **Correlated errors.** Goel et al., "Great Models Think Alike", ICML 2025 https://arxiv.org/abs/2502.04313 —
  CAPA metric (chance-adjusted error agreement); LLM errors grow more similar with capability; LLM judges
  favour similar models. Condorcet-style gains require (near-)independent errors; effective number of
  voters ~ N / (1 + (N-1)*rho).

**Applicability to MK1.** MK1's satellites are genuinely heterogeneous (span extraction, NLI, symbolic
search, kNN prototypes, generative LLM) — the best case for combination, but some share inputs (same
retrieved paragraphs, same bge embeddings), so measure pairwise error correlation (CAPA or simple phi on
dev) and down-weight correlated pairs. Practical recipe: a logistic stacker over calibrated satellite
P(correct)s + agreement features (this *is* the controller's trust model), updated online with Hedge-style
multiplicative weights per task family. Treat the ARC solver as a "veto/oracle" source (near-perfect
precision when it answers): when it answers, it overrides; when silent, it contributes nothing. Avoid MoA-style
"LLM reads everyone and rewrites" with a 2B speaker — Self-MoA shows weak aggregators/sources hurt.

---

## 4. Compute allocation and metareasoning

- **Value of computation** (Russell & Wefald, "Do the Right Thing" 1991; AIJ 1991) — compute iff expected
  improvement in decision utility > cost; myopic VOC approximation. **Selecting computations** (Hay, Russell,
  Tolpin, Shimony, UAI 2012) https://arxiv.org/abs/1207.5879 — bandit-style VOC for which simulation to run.
- **Resource-rational analysis** (Lieder & Griffiths, BBS 2020) https://doi.org/10.1017/S0140525X1900061X — optimal
  strategy given cost of computation; learned metareasoning policies.
- **Anytime algorithms / performance profiles** (Horvitz 1987; Zilberstein, AI Magazine 1996) — keep a
  best-so-far answer; stop using a profile P(quality | time spent, state). Monitoring is optimal stopping.
- **Adaptive computation**: ACT (Graves 2016) https://arxiv.org/abs/1603.08983, PonderNet (Banino et al. 2021)
  https://arxiv.org/abs/2107.05407 — learned halting probability with a compute penalty.
- **Rational Metareasoning for LLMs** (De Sabbata et al., NeurIPS-W 2024) https://arxiv.org/abs/2410.05563 —
  VOC-shaped reward + expert iteration; 23-45% fewer tokens at equal or better accuracy.
- **Adaptive-Consistency** (Aggarwal et al., EMNLP 2023) https://arxiv.org/abs/2305.11860 and early-stopping
  self-consistency (Li et al., ICLR 2024) https://arxiv.org/abs/2401.10480 — stop sampling when a Dirichlet/Beta
  posterior says the majority is settled; up to ~7.9x fewer samples with <0.1% acc drop (mem).
- **Test-time compute scaling**: Snell et al. (ICLR 2025) https://arxiv.org/abs/2408.03314 — compute-optimal
  allocation by difficulty is ~4x more efficient than best-of-N; test-time compute substitutes for a ~14x larger
  model on easy/medium problems but **not on hard ones**, where pretraining scale wins. Liu et al. 2025
  https://arxiv.org/abs/2502.06703 — with a good PRM, 1B beats 405B on MATH-500, 0.5B beats GPT-4o; optimal
  strategy depends on policy, verifier and difficulty. Caveat: math with strong verifiers only.

**Applicability to MK1.** The controller's "call another satellite / another hop / answer / abstain" is a
VOC decision: estimate expected gain in P(correct) from the next call (from the calibrators in Sec. 2) minus a
latency cost, take the action with highest myopic VOC, stop when all VOC < 0 — then abstain or answer by
comparing P(correct) to the conformal threshold. Snell's "hard problems don't benefit" explains MK1's
multi-hop loss: extra hops/samples won't rescue questions beyond the reader's competence, so the
controller should *detect* such cases and abstain cheaply rather than spend. Test-time search pays off where
MK1 has an exact verifier (ARC program search, NLI-checkable claims) — spend compute there.

---

## 5. Contextual bandits / online learning for model or tool selection

- **LinUCB** (Li et al., WWW 2010) https://arxiv.org/abs/1003.0146; **Thompson sampling for linear payoffs**
  (Agrawal & Goyal, ICML 2013) https://arxiv.org/abs/1209.3352. Regret ~ O(d sqrt(T)) (d = feature dim).
- **MixLLM** (Wang et al., NAACL 2025) https://arxiv.org/abs/2502.18482 — contextual-bandit router with
  query tags, quality and cost predictors, continual online updates; ~97% of GPT-4 quality at ~24% cost (mem).
- **PILOT** (Panda et al., 2025) https://arxiv.org/abs/2508.21141 — LinUCB in a shared query-LLM embedding
  space, warm-started offline from preference data, refined online; budget-aware (verify ID before citing).
- **BaRP** (2025) https://arxiv.org/abs/2510.07429 — routing from bandit feedback (only chosen model's score
  and cost seen); preference vector sets accuracy-cost tradeoff at test time; beats offline full-info routers.
- **Greedy LinUCB for sequential LLM selection** (2025) https://arxiv.org/abs/2506.17670 — multi-step,
  budget- and position-aware; no offline fine-tuning.
- **Dueling-feedback routing** (2025) https://arxiv.org/abs/2510.00841 — learns from pairwise preferences.

**How much feedback.** With d ≈ 20-50 handcrafted features and K ≈ 6 arms, linear bandits typically converge in
a few hundred to ~1-2k interactions; offline warm-start (full-information logs where every arm was run) cuts this
sharply — every routing paper above warm-starts. Exploration is the main cost; Thompson sampling with a
sensible prior explores less wastefully than epsilon-greedy.

**Applicability to MK1.** On a laptop MK1 can usually afford *full information* on training/dev items (run all
satellites, log every outcome), which turns routing into supervised learning — strictly better than bandits.
Bandits matter only for the online task stream where running everything is too slow or labels arrive only for
the chosen path. Use Thompson sampling over a Bayesian logistic / linear model on the same feature vector as the
calibrators, warm-started from full-information logs. MK1's existing win (online prototype learning, +0.12 over
E4B) is the natural feedback source.

---

## Ranked: 5 mechanisms most worth implementing in MK1's controller

1. **Per-satellite correctness calibrators (two-stage L2D / Kamath-style selective QA).** Small logistic/GBM
   heads predicting P(this satellite's answer is correct | cheap features + context), trained on full-information
   dev logs. Everything else (trust, stop, abstain, route) consumes these numbers; they replace hand-wired
   per-benchmark thresholds with one shared currency. Cheap (CPU), uses signals MK1 already produces,
   strongest evidence that a tiny external calibrator beats a small model's own confidence.
2. **Conformal / SGR abstention thresholds on that P(correct).** One target risk (e.g., ≤10% wrong among
   answered) yields thresholds with finite-sample guarantees from a few hundred calibration items — turns MK1's
   strongest result (abstention, RGB 41%->78%) into a principled, benchmark-agnostic rule and protects against
   dev overfitting. Recalibrate per task family; watch for shift (use mixed-domain calibration as Kamath).
3. **Agreement-based escalation with correlation-aware stacking.** Escalate/accept based on agreement between
   *different kinds* of satellites (MoT cascade: ~60% cost cut at equal accuracy), and fuse with a stacker whose
   weights are discounted by measured error correlation (Goel et al.; Bach et al.). ARC solver treated as a
   high-precision veto. Directly attacks "whom to trust" without letting the 2B LLM aggregate.
4. **Myopic value-of-computation stopping loop.** At each step pick the satellite/hop with the largest expected
   gain in max P(correct) minus latency cost; stop at VOC<0, then answer-or-abstain via (2). One loop replaces
   four hand-wired flows; plus Adaptive-Consistency-style posterior stopping when sampling the LLM. Gives
   explicit efficiency logging (fits the laptop-as-principle stance) and makes "abstain early on hopeless
   multi-hop" a learned behaviour instead of wasted compute.
5. **Thompson-sampling contextual bandit with Hedge-style online trust updates, warm-started offline.**
   For the online task stream: update calibrators/stacker weights from feedback (Hedge: regret O(sqrt(T log N)))
   and explore unfamiliar satellite paths with Thompson sampling. Lowest priority only because full-information
   offline logs already give most of the benefit; it is what makes the controller "learn from feedback" over time,
   matching MK1's demonstrated strength.

Not recommended now: query-only learned routers à la RouteLLM/GraphRouter (need large preference corpora,
target API cost, and MK1 can afford post-hoc signals); MoA with a 2B aggregator (Self-MoA evidence); verbalized
confidence / P(IK) from E2B (scale-dependent, unreliable); ACT/PonderNet-style end-to-end halting (needs
differentiable training of the whole system).
