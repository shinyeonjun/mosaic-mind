# Prior work: learning whom to trust from feedback, truth discovery, and meta-learned near-Bayes-optimal learners (incl. misspecification)

Scope note: the project being assessed is (1) a ~5k-param meta-learned, speaker-permutation-equivariant GRU judge that reaches within 0.004 of the Bayes-optimal learner on a 4-speaker, 40-episode trust task with +1/0/-1 scoring; (2) under partial (answer-only, abstain = no feedback) feedback, exact joint Bayes over 4^S level assignments is best, exploration adds ~+0.003 (n.s.), an ADF/factorized approximation matches and scales to S=20, and the network transfers zero-shot 4->20 speakers; (3) in a misspecified world (polarity-dependent accuracy, random-walk drift, copiers), the network trained on that world beats standard-model Bayes by +0.016 and partially-corrected Bayes by +0.005.

Items marked "[background, URL not re-verified this session]" are well-known papers cited from prior knowledge with their standard identifiers; the report writer should treat them as reliable but unchecked in this pass.

## Q1. Truth discovery and crowdsourcing label aggregation (Dawid-Skene, GLAD, surveys, copying detection, online/streaming, neural truth discovery)

### Takeaway
Jointly inferring source reliability and the hidden truth is a 45-year-old problem (Dawid-Skene 1979 onward) with Bayesian, online, time-varying-reliability, and source-copying variants all already published; the project's generative model (per-speaker symmetric accuracy, hidden key, copiers, drift) is a small special case of these. What is mostly absent in that literature is (a) the sequential decision layer with abstention and +1/0/-1 scoring and (b) a meta-learned (amortized) aggregator benchmarked against the exact Bayes-optimal learner; truth discovery methods are typically unsupervised EM/iterative schemes on batch data, not learners trained on many simulated sessions.

### Cited Findings
- Dawid & Skene (1979), "Maximum Likelihood Estimation of Observer Error-Rates Using the EM Algorithm", JRSS Series C (Applied Statistics) 28(1): introduced per-worker confusion matrices estimated with EM; described in later work as the method that "first introduced a method for estimating worker ability" — [Online Crowdsourcing, arXiv 1512.02393](https://arxiv.org/pdf/1512.02393); original [background, URL not re-verified]: https://doi.org/10.2307/2346806
- An online Dawid-Skene variant processes one item and its labels per step and converges, under non-restrictive conditions, to a stationary point of the marginal log-likelihood — [Online Crowdsourcing, arXiv 1512.02393](https://arxiv.org/pdf/1512.02393)
- Belief propagation / approximate message passing can attain Bayes-optimal label reconstruction in some parameter regions of the dense Dawid-Skene limit, with identified regions of sub-optimality — [Dense Limit of the Dawid-Skene Model..., arXiv 1803.04924](https://arxiv.org/pdf/1803.04924)
- GLAD: Whitehill, Ruvolo, Wu, Bergsma, Movellan (NeurIPS 2009), "Whose Vote Should Count More: Optimal Integration of Labels from Labelers of Unknown Expertise" — jointly models labeler expertise and item difficulty with EM [background, URL not re-verified]: https://papers.nips.cc/paper/2009/hash/f899139df5e1059396431415e770c6dd-Abstract.html
- Survey: Li, Gao, Meng, Li, Su, Zhao, Fan, Han, "A Survey on Truth Discovery", SIGKDD Explorations 17(2):1-16 (2015 issue; often cited as 2016) — truth discovery "integrates multi-source noisy information by estimating the reliability of each source" — [arXiv 1505.02463](https://arxiv.org/abs/1505.02463)
- Dong, Berti-Equille, Srivastava (2009), "Integrating conflicting data: the role of source dependence", PVLDB 2(1):550-561 — Bayesian iterative algorithm that infers copying/dependence between sources so that dependent false values are not over-counted — [PDF](https://lunadong.com/publication/dependence_vldb.pdf); companion "Sailing the Information Ocean with Awareness of Currents" — [arXiv 0909.1776](https://arxiv.org/abs/0909.1776)
- Crowdsourcing analogue of copier detection: "Identifying Dependent Annotators in Crowdsourcing" — [NSF PAR](https://par.nsf.gov/servlets/purl/10424925)
- Evolving truth / time-varying source reliability: Li et al., KDD 2015 ("On the discovery of evolving truth") — notes existing approaches are batch, cannot incrementally update truth and reliability, and that "observed source reliability changes over time" — [PDF](https://cse.buffalo.edu/~jing/doc/kdd15_evolve.pdf); Scalable Streaming Truth Discovery (SSTD) uses HMMs for evolving truth (summarized in search results) — [NSF PAR](https://par.nsf.gov/servlets/purl/10075153)
- Dynamic Bayesian classifier combination (DynIBCC): Simpson, Roberts, Psorakis, Smith, "Dynamic Bayesian Combination of Multiple Imperfect Classifiers" (2013, book chapter in "Decision Making and Imperfection") — variational Bayes over per-agent confusion matrices that change over time; applied to Galaxy Zoo Supernovae — [arXiv 1206.1831](https://arxiv.org/abs/1206.1831)
- Neural truth discovery: memory-network model learning source reliability and statement credibility ("Truth Discovery with Memory Network", Tsinghua Sci. & Tech. 2017) — [SciOpen](https://www.sciopen.com/article/10.23919/TST.2017.8195344); multi-layer NN for social-sensing truth discovery without assumed source-claim dependency distribution — [Notre Dame MASS'17 PDF](https://www3.nd.edu/%7Esslab/pdf/mass17.pdf); RBM + feed-forward hybrid ("Combining Restricted Boltzmann Machines with Neural Networks for Latent Truth Discovery") — [PapersWithCode](https://cs.paperswithcode.com/paper/combining-restricted-boltzmann-machines-with)
- Very recent LLM-agent analogues (closest in *framing* "learning whom to trust from feedback"):
  - Σ-Mem (Feng, Yang, Poria, arXiv July 30 2026): online reliability memory for LLM multi-agent systems with per-peer competence matrices and a global peer-relationship matrix, updated from post-decision +1/-1 correctness feedback with decay; outperforms majority vote and fixed peer selection; performance rises monotonically with feedback availability (5%-100%); does not compare to Bayes-optimal and uses hand-designed (not meta-learned) updates — [alphaXiv 2607.27958](https://www.alphaxiv.org/abs/2607.27958)
  - Epistemic Context Learning (ICLR 2026): "history-aware reference" — agents condition on peer interaction history to estimate peer reliability; Qwen3-4B with ECL outperforms a history-agnostic Qwen3-30B — [arXiv 2601.21742](https://arxiv.org/abs/2601.21742v1)

### Inferences
- Overlap with project component (3): copiers ~ Dong et al. 2009 source dependence; drift ~ evolving-truth/DynIBCC; polarity-dependent accuracy ~ full (asymmetric) Dawid-Skene confusion matrices vs. the "one-coin" symmetric model. So the *deviations themselves* are standard in this literature; the novelty is not the world model but the comparison "learned-from-data vs. Bayes from the simpler model."
- Overlap with component (2): ADF/factorized posteriors over worker reliability are standard practice (variational/EP worker models like DynIBCC); the S=20 scaling result is expected, not novel by itself.
- Σ-Mem and ECL show the "learn whom to trust from correctness feedback" framing is currently active (2026) in LLM multi-agent work, but neither benchmarks against an exact Bayes-optimal learner nor studies abstention-gated feedback; this makes the project's small, exactly-solvable testbed a plausible contribution as a reference benchmark.

### Gaps
- Did not find a paper that meta-trains (amortizes) a truth-discovery/label-aggregation model on simulated Dawid-Skene sessions and compares it to the exact Bayes posterior; a targeted search returned nothing, but absence of evidence from ~2 searches is weak.
- Did not verify whether online truth discovery papers handle "feedback only for the chosen answer" (they generally assume no ground truth at all, or full labels).

## Q2. Meta-learning approaching Bayes-optimal behavior (Ortega 2019, Mikulik 2020, follow-ups to 2026)

### Takeaway
Finding (1) — a memory-based meta-learner trained with log loss approaches the Bayes-optimal learner and its internal state tracks the Bayesian posterior — is a well-established result (Ortega et al. 2019 theory; Mikulik et al. 2020 empirical with state-machine simulation analysis; Genewein et al. 2023 for non-stationary/switching sources). The project's version replicates this in a new domain (multi-source trust with per-speaker equivariant memory); it is confirmatory, not novel in principle.

### Cited Findings
- Ortega et al. (DeepMind, arXiv 2019), "Meta-learning of Sequential Strategies": recasts memory-based meta-learning in a Bayesian framework; meta-learned strategies are near-optimal because they "amortize Bayes-filtered data," with adaptation implemented as memory dynamics = a state machine of sufficient statistics, turning sequential inference into regression — [arXiv 1905.03030](https://ar5iv.arxiv.org/html/1905.03030)
- Mikulik, Delétang, McGrath, Genewein, Martic, Legg, Ortega (NeurIPS 2020), "Meta-trained agents implement Bayes-optimal agents": on prediction and bandit tasks, meta-learned and Bayes-optimal agents behave similarly and share computational structure (one approximately simulates the other); Bayes-optimal agents are fixed points of meta-learning dynamics — [NeurIPS proceedings](https://proceedings.neurips.cc/paper/2020/hash/d902c3ce47124c66ce615d5ad9ba304f-Abstract.html), [arXiv 2010.11223](https://ar5iv.arxiv.org/html/2010.11223)
- Genewein et al. (ICML 2023), "Memory-Based Meta-Learning on Non-Stationary Distributions": Transformers, LSTMs and RNNs learn to approximate known Bayes-optimal algorithms for piecewise-stationary sources with unobserved switch points, behaving as if inferring switch points and per-segment parameters — [PMLR v202](https://proceedings.mlr.press/v202/genewein23a.html), [code](https://github.com/deepmind/nonstationary_mbml)
- Raventós, Paul, Chen, Ganguli (NeurIPS 2023): below a task-diversity threshold a pretrained transformer behaves like the Bayes estimator with the (non-diverse) pretraining prior; above it the transformer deviates from that Bayes estimator and generalizes like ridge regression — [arXiv 2306.15063](https://arxiv.org/abs/2306.15063)
- Müller, Hollmann, Hutter (arXiv 2024), "Bayes' Power for Explaining In-Context Learning Generalizations": argues NN training is better interpreted as approximating the true posterior of the data-generating process; also documents limits of what posteriors generalize to and where NNs fail to approximate them — [arXiv 2410.01565](https://arxiv.org/abs/2410.01565)
- Wakayama & Suzuki (arXiv 2025), "In-Context Learning Is Provably Bayesian Inference": decomposes ICL risk into a Bayes Gap (how close the trained model is to the Bayes-optimal in-context predictor) and posterior variance; posterior over task index concentrates exponentially in context length — [arXiv 2510.10981](https://arxiv.org/pdf/2510.10981)
- 2026 work probing what a "Bayes-filtered transformer believes" (decoding beliefs via predictive Monte Carlo) — [arXiv 2607.17060](https://arxiv.org/pdf/2607.17060); [arXiv 2602.04596](https://arxiv.org/pdf/2602.04596) (uncertainty decomposition for Bayes-filtered transformers). Only titles/abstract snippets seen.

### Inferences
- The project's "implied per-speaker trust tracks the Bayesian posterior mean" is the same kind of evidence Mikulik et al. 2020 give (state correspondence), at a smaller and more interpretable scale. Novelty here is the domain + the explicit structure (per-speaker shared-weight memory, key-equivariant head), not the phenomenon.
- The within-0.004 gap is a nice quantitative point but should be framed as a replication/extension of Ortega/Mikulik/Genewein, citing them.

### Gaps
- Did not find prior meta-learning-to-Bayes work specifically on multi-source reliability / truth discovery tasks; did not read the 2026 Bayes-filtered-transformer papers in full.

## Q3. Learned/amortized inference vs. misspecified model-based Bayes

### Takeaway
Two literatures disagree in tone: amortized simulation-based inference (SBI) papers stress that amortized networks are *more fragile* than MCMC when the *simulator* is misspecified (the network is trained on the wrong model); by contrast, model-based deep-learning work (e.g., KalmanNet) shows a learner trained on *real/true-world data* beats a filter derived from a mismatched model. Finding (3) belongs to the second category — and is essentially expected: a network trained on data from the true world vs. Bayes code from a wrong model. Its value lies in quantifying the gap and showing that partially hand-corrected Bayes (2/4 deviations) closes most of it (+0.016 -> +0.005).

### Cited Findings
- Amortized Bayesian inference methods "are highly sensitive to model misspecification when observed data fall outside the training distribution"; RVNP proposes variational error modelling to recover robust posteriors — [arXiv 2509.05724](https://arxiv.org/html/2509.05724v2)
- Schmitt, Bürkner, Köthe, Radev, "Detecting Model Misspecification in Amortized Bayesian Inference with Neural Networks" (GCPR 2023; extended arXiv 2024): amortized methods are more susceptible to estimation errors than non-amortized ones (e.g., MCMC) under misspecification; propose an unsupervised misspecification measure; robustness improves by injecting contaminants into simulations — [arXiv 2112.08866](https://arxiv.org/abs/2112.08866), [extended arXiv 2406.03154](https://arxiv.org/html/2406.03154v1)
- Self-consistency training improves amortized model comparison robustness under severe misspecification — [arXiv 2512.14308](https://arxiv.org/pdf/2512.14308)
- KalmanNet (Revach et al., IEEE Trans. Signal Processing 2022): RNN embedded in the Kalman filter flow learns the gain from data; "overcomes non-linearities and model mismatch, outperforming classic filtering methods operating with both mismatched and accurate domain knowledge" — [arXiv 2107.10043](https://arxiv.org/abs/2107.10043v3)
- Raventós et al. 2023 (above) is a case where deviating from the Bayes estimator under the training prior is what enables generalization to new tasks — [arXiv 2306.15063](https://arxiv.org/abs/2306.15063)

### Inferences
- The clean, novel-ish angle for (3) is the *ladder*: Bayes(standard model) < Bayes(2 of 4 deviations modeled) < meta-learned net trained on the true world, with CIs. This is a "price of unmodeled structure" measurement in a domain (source trust) where truth-discovery practitioners hand-craft dependence/drift models (Dong 2009, DynIBCC). Framed this way it complements KalmanNet-style results; framed as "learning beats wrong model" it is not novel.
- A reviewer will ask the fair-comparison question: does a Bayes learner with all 4 deviations modeled match the network? Without that upper reference, the result could be read as "under-specified baseline." Recommend adding it (feasible since the world is simulated).
- The SBI literature's caution is the converse risk: the network itself is misspecified if test world != training world; a test of the network trained on standard world evaluated on misspecified world (and vice versa) would connect to Schmitt et al.

### Gaps
- Found no paper making this exact comparison for truth discovery / source reliability.

## Q4. Permutation-equivariant/set architectures for multi-source aggregation; generalizing to more sources

### Takeaway
Set architectures (Deep Sets, Set Transformer) are standard, and crowdsourcing has "permutation" models — but those concern permutations of *classes/labels* (PICA) or of *worker ability orderings*, not worker-permutation-equivariant neural aggregators. The crowd-deep-learning literature commonly allocates a separate network/head per annotator and is described as non-scalable. A shared-weight per-source recurrent memory that transfers zero-shot from 4 to 20 sources appears uncommon in this specific domain, though the architectural idea itself is standard.

### Cited Findings
- PICA (Giancola, Paffenroth, Whitehill, HCOMP 2018): consensus model invariant to *class* permutations by each annotator via doubly-stochastic matrices — [AAAI OJS](https://ojs.aaai.org/index.php/HCOMP/article/view/13326), [PDF](https://users.wpi.edu/~jrwhitehill/GiancolaPaffenrothWhitehill2018.pdf)
- "A Permutation-based Model for Crowd Labeling: Optimal Estimation and Robustness" (Shah, Balakrishnan, Wainwright): generalizes Dawid-Skene via permutation-based (ordering) assumptions on worker abilities; WAN / OBI-WAN estimators — [arXiv 1606.09632](https://arxiv.org/abs/1606.09632)
- Crowdsourced deep learning methods often allocate a separate network per annotator, "inherently non-scalable," with experiments only up to 41 annotators — [Learning From Crowdsourced Noisy Labels, arXiv 2407.06902](https://arxiv.org/pdf/2407.06902) (statement as summarized in search result; not read in full)
- Deep Sets (Zaheer et al., NeurIPS 2017) [background, URL not re-verified]: https://arxiv.org/abs/1703.06114 ; Set Transformer (Lee et al., ICML 2019) [background]: https://arxiv.org/abs/1810.00825

### Inferences
- Novelty claim for the architecture should be modest: "applying shared-weight per-source memory (a DeepSets-style equivariant recurrent design) to sequential trust learning, with zero-shot transfer 4->20 sources and parity with an ADF Bayes baseline." The equivariance trick itself is not new.

### Gaps
- Did not find (in limited searching) a paper explicitly evaluating a source-permutation-equivariant neural aggregator's extrapolation to more sources than trained; may exist in multi-agent RL (e.g., mean-field / graph-network policies) — not searched.

## Q5. Value of exploration when abstaining gives no feedback (label-efficient prediction, apple tasting, partial monitoring)

### Takeaway
The setting "answer to get feedback, abstain to get none" is a known partial-monitoring structure (apple tasting / label-efficient prediction), where in the adversarial/minimax regime exploration is necessary and regret rates change. But in a Bayesian setting with a correct prior and an exact posterior, the incremental value of explicit exploration over myopic Bayes can be small; the project's ~+0.003 (n.s.) result is consistent with that, and is a modest empirical observation rather than a new theoretical result.

### Cited Findings
- Cesa-Bianchi, Lugosi, Stoltz, "Minimizing Regret with Label Efficient Prediction", IEEE Trans. Information Theory (2005): forecaster sees the outcome only when it asks; matching upper/lower bounds for fixed query budgets; Hannan consistency achievable with queries growing just faster than logarithmically — [HAL](https://hal.archives-ouvertes.fr/hal-00007537)
- Apple tasting: feedback is obtained only for one of two actions (tasting/discarding), making it closely related to label-efficient prediction — summarized in [Online Learning with Feedback Graphs: Beyond Bandits, arXiv 1502.07617](https://arxiv.org/pdf/1502.07617); original Helmbold, Littlestone, Long, "Apple Tasting", Information and Computation (2000) [background, URL not re-verified]
- Feedback graphs (Alon, Cesa-Bianchi, Dekel, Koren, COLT 2015) characterize regret by the observability structure of actions; abstention-without-feedback is a weakly-observable-type structure — [arXiv 1502.07617](https://arxiv.org/pdf/1502.07617)
- Adaptive selective sampling for online prediction with experts (NeurIPS 2023): label-efficient forecaster that queries adaptively — [NeurIPS 2023](https://neurips.cc/virtual/2023/poster/72851), [arXiv 2302.08397](https://arxiv.org/pdf/2302.08397)
- Mikulik et al. 2020 already showed meta-trained agents match Bayes-optimal agents on bandit tasks, i.e., where exploration is part of the Bayes-optimal policy — [arXiv 2010.11223](https://ar5iv.arxiv.org/html/2010.11223)

### Inferences
- In the project, the "optimal answer iff max posterior > 0.5" rule is myopic; the truly Bayes-optimal policy under partial feedback is a POMDP/Bayes-adaptive policy that may sometimes answer at posterior < 0.5 to buy information. The small +0.003 is evidence that, for 40-episode sessions with 4 reliability levels, the information value of forced answering is low. It would be more rigorous to compute (or bound) the Bayes-adaptive optimum (e.g., via Gittins-like reasoning or small-S dynamic programming) than to test heuristics.

### Gaps
- No source found quantifying the Bayesian value of information specifically in apple-tasting-style trust learning with abstention.

## Overall novelty assessment (synthesis for report writer)
- (1) Meta-learned judge ≈ Bayes-optimal, trust tracks posterior: **largely known phenomenon** (Ortega 2019; Mikulik 2020; Genewein 2023). New only as a domain instantiation with an interpretable equivariant per-speaker architecture.
- (2) Exact joint Bayes best; ADF matches & scales; exploration adds little; zero-shot 4->20: individual pieces are **standard** (variational/ADF worker models, DynIBCC; set-equivariance). The combination in a sequential, abstention-gated decision setting with an exact Bayes reference is a **reasonable benchmark contribution**, not a methodological novelty.
- (3) Net trained on misspecified world beats standard-model Bayes (+0.016) and partially-corrected Bayes (+0.005): **conceptually anticipated** by model-based-DL results (KalmanNet) and the truth-discovery literature that hand-models copying/drift (Dong 2009; Li 2015; Simpson 2013). The most defensible novelty is the quantified "ladder" in a fully controlled trust world; strength depends on adding a fully-specified Bayes upper reference.
- Closest single works by component: (1) Mikulik et al. 2020; (2) Simpson et al. 2013 DynIBCC + Cesa-Bianchi et al. 2005; (3) Dong et al. 2009 (copiers) + KalmanNet 2022 (learned > mismatched model); framing as "learning whom to trust from feedback": Σ-Mem 2026 and ECL (ICLR 2026).
