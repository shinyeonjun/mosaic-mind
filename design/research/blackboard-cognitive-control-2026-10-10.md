# Proven control mechanisms for a modular "satellites around a speaker" system (MK1)

Scope: what the control loop actually does, what was learned vs hand-coded, evidence, failure modes, and what transfers to MK1 (8GB VRAM, Gemma 4 E2B as speaker, small specialist satellites).
Confidence tags: [V] verified via search this session; [K] from background knowledge (well-known paper, figures from memory, check before quoting numbers).

---

## 1. Blackboard systems

### Hearsay-II (Erman, Hayes-Roth, Lesser, Reddy; ACM Computing Surveys 1980) [K]
https://dl.acm.org/doi/10.1145/356810.356816
- **Loop**: (1) knowledge sources (KSs) watch the blackboard (hierarchical levels: segment, syllable, word, phrase); a change triggers matching KS *preconditions*; (2) each triggered KS becomes a pending activation (KSAR); (3) a **focus-of-attention scheduler** rates every KSAR (credibility of its stimulus hypotheses, level, time region, expected gain toward a complete interpretation); (4) run the top-rated one; it writes/modifies hypotheses with credibility scores; (5) repeat until a full-utterance hypothesis with high enough credibility exists or resources run out.
- **Opportunistic control**: work islands of high confidence first and extend them (bottom-up and top-down mixed).
- **Learned vs hand-coded**: everything hand-coded, including the scheduler's rating function (the "priority" heuristics were hand-tuned).
- **Evidence**: 1976 ARPA SUR goal met (~90% sentence accuracy, 1011-word vocab, constrained grammar). But the much simpler HARPY (beam search over a compiled network) did better; the blackboard's generality cost speed.
- **Failure**: scheduler ratings were brittle and the main tuning burden; the scheduler was the system's bottleneck.

### HASP/SIAP and CRYSALIS (Nii et al. 1982; Engelmore & Terry 1979) [K]
- Event-driven control: KS activation by *event types* posted to the board, plus a control layer of strategy rules ("rule-based scheduler"). CRYSALIS (protein crystallography) used a separate set of strategy KSs to pick the next event to process.
- Lesson: control became an explicit knowledge source rather than a fixed rating formula.

### BB1 (Hayes-Roth, Artificial Intelligence 26, 1985) [K]
https://doi.org/10.1016/0004-3702(85)90063-3
- Two blackboards: **domain** and **control**. Control KSs write *focus decisions*, *strategies*, *policies* ("prefer KSs that extend high-credibility islands") onto the control board. Each cycle: (1) update agenda of triggered KSARs; (2) rate each KSAR against the current control-board foci/policies; (3) execute the best (domain or control KS); (4) loop. Control is reasoned about the same way as the problem.
- Learned: nothing (some later explanation/learning add-ons by Hayes-Roth's group; not mainstream).
- Evidence: demonstrated on PROTEAN (protein structure), Guardian (ICU monitoring). Flexible, but slow and authoring-heavy.

### Why blackboards faded [K]
1. **Control bottleneck**: one serial scheduler evaluating every pending KSAR each cycle; cost grows with KS count.
2. **Hand-tuned ratings**: no principled way to set priorities; performance hinged on them.
3. **Knowledge acquisition cost**: KS preconditions and hypothesis formats were bespoke.
4. **Better special-purpose alternatives** (HMMs/beam search for speech; statistical learning in general).
5. No learning loop: the system never improved its own scheduling from outcomes.

### Recent LLM revivals (2025-2026) [V]
- Han & Zhang, *Exploring Advanced LLM Multi-Agent Systems Based on Blackboard Architecture* (arXiv 2507.01701, Jul 2025). An LLM "control unit" picks which agents act from the board's contents each round; loop until consensus. Reports best average vs static/dynamic MAS baselines on commonsense/reasoning/math, with fewer tokens. https://arxiv.org/abs/2507.01701
- Salemi et al. (UMass + Google), *LLM-based Multi-Agent Blackboard System for Information Discovery in Data Science* (arXiv 2510.01285, v2 Jan 2026). Requests posted on the board; agents **self-select** whether to respond (no central controller needing capability knowledge). 13-57% relative gain over best baseline (incl. master-slave and RAG). https://arxiv.org/abs/2510.01285
- PatchBoard (arXiv 2605.29313, 2026): agents write schema-validated JSON-Patch edits to a typed state tree through a deterministic kernel. Reported 84.6% vs 30.8% (LangGraph) task success, unverified. https://arxiv.org/abs/2605.29313
- Counter-evidence: Cemri et al., *Why Do Multi-Agent LLM Systems Fail?* (NeurIPS 2025 D&B; MAST, 14 failure modes, 1600+ traces, 7 frameworks, kappa 0.88). Failures come mostly from **system design and inter-agent misalignment**, plus **weak task verification**; gains over single-agent or best-of-N are often minimal. https://arxiv.org/abs/2503.13657

**Applicability to MK1**: High. A *typed* blackboard (named slots, provenance, confidence, write log) is the cheapest way to let satellites see each other's outputs, and MK1 v1 already had one. Avoid the old failure by (a) making the scheduler a small *learned* scorer over (state features, module id) instead of hand ratings, (b) keeping the module count small (<10) so a full agenda rating per cycle costs nothing, (c) schema-validating writes (PatchBoard idea) so a bad module can't corrupt the board, and (d) treating verification as a first-class module (MAST's biggest gap). Do not use the small LLM as the control unit (see section 3).

---

## 2. Cognitive architectures' control loops

### Soar (Laird, Newell, Rosenbloom 1987; Laird 2012 *The Soar Cognitive Architecture*, MIT Press) [K]
https://soar.eecs.umich.edu/
- **Decision cycle**: input -> elaboration (all matching rules fire in parallel, propose operators and preferences) -> decision (preference semantics pick one operator) -> apply -> output. If preferences don't yield a unique choice (tie, conflict, no-change) -> **impasse** -> automatic **substate/subgoal** to resolve it (deliberate, look ahead, query memory). On resolution, **chunking** compiles the substate's reasoning into a new rule so the impasse doesn't recur.
- Soar 9 added **RL on numeric operator preferences** (Soar-RL), episodic and semantic memory.
- Learned: chunks (rules), RL operator values. Hand-coded: problem spaces, operator proposals.
- Evidence: TacAir-Soar (large real-time fielded simulation agent, ~8k rules, 1990s); many robot/game agents. Scaling problem: **utility problem** (too many chunks slow matching; Minton 1990 for EBL generally). Chunking learns speed-up, rarely new knowledge.
- **Applicability**: The impasse mechanism is the single most useful idea for MK1: "when the arbiter can't pick a clear answer (tie / low margin / conflict between modules), open a sub-step" = run an extra module or an extra hop, rather than looping blindly. Chunking analog: cache "for inputs like X, the route Y worked" in memory so the next similar input skips deliberation.

### ACT-R (Anderson et al., Psych Review 2004; Anderson 2007) [K]
http://act-r.psy.cmu.edu/
- **Loop**: buffers (goal, retrieval, visual, manual) hold one chunk each = a tiny typed blackboard. Each cycle (~50 ms), all productions matching the buffers form the conflict set; pick by **utility + noise**. Old form: expected gain E = P*G - C (P = probability of success, G = goal value, C = cost). Modern form: U_i(n) = U_i(n-1) + alpha [R_i(n) - U_i(n-1)] (Rescorla-Wagner/TD-like utility learning, Fu & Anderson 2006), with logistic noise s for exploration.
- Learned: production utilities (and production compilation; declarative activation via base-level learning). Hand-coded: productions themselves.
- Evidence: fits hundreds of human data sets (latency, error, fMRI mappings); utility learning reproduces probability-matching and skill acquisition. Not shown to scale as an AI system to open-ended tasks.
- **Applicability**: Very high and cheap. Use exactly this: each (situation-type, module) pair has a utility U updated by reward minus cost; select with softmax/noise. Costs (VRAM, latency) go in C explicitly, which fits the laptop-efficiency principle. This is a bandit; it needs no gradient training.

### LIDA (Franklin et al., IEEE TAMD 2014; Franklin & Patterson 2006) [K]
- **Cognitive cycle** (~10 Hz in theory): perceive -> local associations from memories -> **attention codelets** form coalitions -> coalitions compete for the **global workspace** -> winner broadcast to all modules -> procedural memory picks an action (via a behavior net) -> act; learning from each broadcast (perceptual, episodic, procedural).
- Evidence: conceptual, a few small demos (IDA for Navy billet assignment was the predecessor). No benchmark-scale evidence of learning or scale.
- **Applicability**: Use its *broadcast* idea (after each step, the winning content is visible to all modules) but not its machinery.

### CLARION (Sun 2002, 2016) [K]
- Two levels: implicit (neural, Q-learning) bottom; explicit (rules) top; **Rule-Extraction-Refinement** lifts successful implicit actions into rules; action chosen by combining both levels.
- Evidence: cognitive modeling fits (serial reaction time, minefield navigation). Not scaled.
- **Applicability**: The "implicit learned scorer + explicit cached rules extracted from successes" pairing maps onto learned router + rule cache.

### Which scaled/learned?
Only Soar (TacAir-Soar, Soar-RL) and ACT-R (utility learning, broadly validated against humans) have strong evidence; neither learns *new skills* from raw data. What survived everywhere: **a small typed working memory, a cycle that proposes-evaluates-selects one action, utility learning with noise, impasse-driven subgoaling.**
Review: Kotseruba & Tsotsos, *40 years of cognitive architectures* (AI Review 2020). https://doi.org/10.1007/s10462-018-9646-y

---

## 3. Modular neural / program composition

### Neural Module Networks (Andreas et al., CVPR 2016); N2NMN (Hu et al., ICCV 2017) [K]
https://arxiv.org/abs/1511.02799
- Parser produces a layout; modules (find, relate, and, describe) are assembled per question and trained jointly. N2NMN learned the layout policy with RL + expert-layout imitation.
- **Systematic generalization evidence**: Bahdanau et al., *Systematic Generalization: What Is Required and Can It Be Learned?* (ICLR 2019): NMNs with the *correct fixed tree layout* generalize systematically; **learned layouts and generic models do not**. https://arxiv.org/abs/1811.12889 . CLOSURE (Bahdanau et al. 2019) showed NMN-style models trained on CLEVR fail badly on novel combinations of known constructs. https://arxiv.org/abs/1912.05783
- **Lesson**: modularity helps only if the *composition* is right; learning the composition end-to-end is the hard, fragile part.

### Routing Networks (Rosenbaum, Klinger, Riemer, ICLR 2018); *Routing Networks and the Challenges of Modular and Compositional Computation* (Rosenbaum et al. 2019) [K]
https://arxiv.org/abs/1711.01239 ; https://arxiv.org/abs/1904.12774
- A router (multi-agent RL, one agent per task, WPL algorithm) picks a module per layer per input; modules and router trained together. Beat cross-stitch nets on multi-task CIFAR-100/MNIST with lower cost.
- **Failure modes (2019 paper)**: module collapse (router uses one module), module diversity loss, non-stationarity (router and modules co-adapt), exploration vs exploitation, credit assignment. Fix ideas: train modules first then router; per-task routers; entropy bonuses.
- **Applicability**: MK1's satellites are already trained and frozen, which **removes the main non-stationarity**; routing over frozen modules is a much easier contextual bandit.

### Modular meta-learning (Alet, Lozano-Perez, Kaelbling, CoRL 2018; BounceGrad) [K]
https://arxiv.org/abs/1806.10166
- Learn a library of modules; for a new task, *search* (simulated annealing) over compositions, keep modules fixed. Faster adaptation than MAML on small data. Applicability: "search over compositions with frozen modules" is exactly your ARC program-search satellite pattern, applicable to routing too.

### LLM-as-planner program composition [K]
- **VisProg** (Gupta & Kembhavi, CVPR 2023 best paper) and **ViperGPT** (Suris et al., ICCV 2023): large LLM writes a program calling vision modules; zero-shot competitive on GQA/NLVR/OK-VQA. Large planners (GPT-3/Codex). Errors dominated by *program errors* and module errors; no recovery loop.
- **HuggingGPT** (Shen et al., NeurIPS 2023): ChatGPT plans -> selects models by description -> executes -> summarizes. Failure: planning errors, cost, dependency on a very strong LLM.
- **Toolformer** (Schick et al., NeurIPS 2023): self-supervised learning of *when* to call tools (keep calls that reduce LM loss). Works at 6.7B; authors note tool-use benefit emerges only at ~775M+ and no chained tool use.
- **ReAct** (Yao et al., ICLR 2023), **Reflexion** (Shinn et al., NeurIPS 2023): interleave thought-action-observation; Reflexion adds verbal self-critique stored in memory. Strong with GPT-3.5/4; small models struggle with format and loop termination.
- **ToolkenGPT** (Hao et al., NeurIPS 2023): each tool is a learned *token embedding* for a frozen LLM; adding a tool = training one embedding from a few demos. https://arxiv.org/abs/2305.11554 . Applicability: cheap "new module" onboarding for the speaker if the LLM must trigger tools.
- **Compositionality gap / decomposition**: Press et al., *Measuring and Narrowing the Compositionality Gap* (Self-Ask, EMNLP Findings 2023) https://arxiv.org/abs/2210.03350 ; error propagation in QD+QA pipelines (GenDec, arXiv 2402.11166 [V]); a 2026 GEM paper finds smaller models (Llama 3.1 8B) decompose both explicit and implicit multi-hop questions poorly [V] https://aclanthology.org/2026.gem-main.11/ . This matches MK1's observation that E2B decomposition hurt multi-hop.

### LLM planner vs learned controller (synthesis)
- LLM planners: zero-shot flexibility, but quality tracks LLM size; error compounds per step (if step accuracy p, n-step chain ~ p^n); small planners also mis-format and fail to stop. MAST shows planning/coordination design is the top failure source.
- Learned controllers over frozen modules (routing as bandit/RL, ACT-R utilities): cheap, reliable, measurable, but need reward signal and can't invent new compositions.
- **Recommendation for MK1**: learned controller picks modules; the LLM only speaks (and maybe proposes candidate answers). Decomposition should be done by the *trained hop selector*, which already beats the LLM at it.

---

## 4. Verification loops and iterative refinement with small models

- **Self-consistency** (Wang et al., ICLR 2023) https://arxiv.org/abs/2203.11171 : sample k answers, majority vote. Robust gain across sizes; cost k x. Agreement rate is also a free confidence signal for abstention. [K]
- **Verifier-guided search**: Cobbe et al. 2021 (GSM8K): a 6B model + trained verifier beat a 175B fine-tuned model [K] https://arxiv.org/abs/2110.14168 . Lightman et al., *Let's Verify Step by Step* (ICLR 2024): process reward models > outcome RMs [K]. Snell et al., *Scaling LLM Test-Time Compute Optimally* (2024, ICLR 2025): compute-optimal test-time search lets a small model beat a 14x larger one on easy/medium problems, *not on hard ones*; best strategy depends on difficulty [K] https://arxiv.org/abs/2408.03314
- **Intrinsic self-correction fails**: Huang et al., ICLR 2024: without external feedback, revision often lowers accuracy; earlier gains used oracle stop signals; multi-agent debate doesn't beat self-consistency at equal samples [V] https://arxiv.org/abs/2310.01798
- **Small models need strong verifiers**: Zhang et al., SCORE (Findings ACL 2024): <=13B self-refiners gain with a strong (GPT-4) verifier deciding *when* to correct; weak self-verification is the bottleneck [V] https://arxiv.org/abs/2404.17140
- **Tool/external feedback works**: CRITIC (Gou et al., ICLR 2024): verification via tools (search, code) makes revision reliable [K] https://arxiv.org/abs/2305.11738 . Self-Refine (Madaan et al., NeurIPS 2023) works mainly with strong models [K].
- **Tiny recursive iteration**: HRM (Wang et al., 2025, 27M): ARC Prize re-test 32% ARC-AGI-1 semi-private (claim 41%), 2% ARC-AGI-2; ablations showed the **outer refinement loop** (iterate, predict, halt via learned Q-head) drove the gains, not the hierarchy; cross-task transfer limited, much performance from per-task training [V] https://arcprize.org/blog/hrm-analysis . TRM (Jolicoeur-Martineau, Samsung, arXiv 2510.04871, Oct 2025): 7M params, 2-layer net recursively refining a latent z and answer y with deep supervision; 45% ARC-AGI-1, 8% ARC-AGI-2, Sudoku-Extreme 87.4% (HRM 55%) [V] https://arxiv.org/abs/2510.04871
- **When looping helps vs hurts** (synthesis):
  - Helps when there's an **independent check** (NLI verifier, exact program execution on ARC train pairs, answer-span grounding, agreement across samples) and the loop changes *inputs* (new evidence, another hop, another module), not just re-asks the same model.
  - Helps for easy/medium items; on hard items extra compute mostly doesn't (Snell).
  - Hurts when the reviser is the same weak model with no feedback (Huang), when there's no stop rule (flips correct answers), and when each iteration compounds an error (decomposition chains).
  - Learned halting (HRM/TRM ACT-style Q-head, or simply calibrated confidence threshold) is the proven stop mechanism.
- **Applicability to MK1**: MK1 already has the right pieces: the NLI verifier and ARC train-pair execution are external, cheap verifiers. Loop = "verifier rejects -> gather more evidence / try next module / next program", never "ask E2B to fix itself". Use sample agreement + verifier score as the abstention signal.

---

## 5. Growth: adding modules without forgetting

- **Progressive Neural Networks** (Rusu et al., arXiv 2016): new column per task, frozen old columns, lateral connections. Zero forgetting, transfer via laterals; parameters grow linearly. [K] https://arxiv.org/abs/1606.04671
- **PackNet** (Mallya & Lazebnik, CVPR 2018): prune and freeze weights per task in one net; needs task id. [K]
- **LMC** (Ostapenko et al., NeurIPS 2021): each module has a local *structural* component (an autoencoder-like relevance scorer); composition from local scores, no task id; **adds a module when inputs are outliers** for all existing modules. Limitation: selection gets hard as modules grow to 30-100 tasks [V] https://arxiv.org/abs/2111.07736
- **Modular continual learning with search**: Veniat et al., MNTDP (ICLR 2021) — per new task, search over paths reusing old modules plus one new one; CTrL benchmark [K] https://arxiv.org/abs/2012.12631 . Mendez & Eaton, *Lifelong Learning of Compositional Structures* (ICLR 2021) — separate "assimilation" (find composition with frozen modules) from "accommodation" (add/adapt modules) [K] https://arxiv.org/abs/2007.07732
- **MoE growth**: Lifelong-MoE (Chen et al., ICML 2023) adds experts with distillation regularization [K]; MoE-Adapters for CL of vision-language models (Yu et al., CVPR 2024) — frozen CLIP + growing adapter experts + router, with a distribution-discriminative selector for unseen data [K] https://arxiv.org/abs/2403.11549 . Expert-library work for LLMs: Ostapenko et al., *Towards Modular LLMs by Building and Reusing a Library of LoRAs* (ICML 2024) with Arrow routing (zero-shot routing from LoRA weights' singular vectors, no router training) [K] https://arxiv.org/abs/2405.11157
- **How a controller learns to use a NEW module fast** (evidence-backed recipes):
  1. *Descriptor/embedding-based routing*: route by similarity between input and the module's prototype/key (LMC local scores, Arrow, ToolkenGPT embeddings, HuggingGPT descriptions). A new module only needs its own key; the router is not retrained.
  2. *Optimistic bandit prior*: give a new arm high initial utility/uncertainty (UCB/Thompson; ACT-R new production default utility) so it gets tried, then utility updates settle it.
  3. *Assimilation-first*: try existing compositions; only if all fail on a cluster of inputs (outlier/novelty detection), add a module (LMC, Mendez & Eaton).
- **Applicability to MK1**: Your per-task prototype classifier grown online already is LMC-like. Make every satellite expose a **competence key** (prototype of inputs it succeeded on + running success rate); router = similarity x Beta-posterior success - cost. New satellite = new key + optimistic prior; nothing else retrained. Frozen satellites mean zero forgetting by construction; the only thing that can "forget" is the router, so log router accuracy on an old-task replay set.

---

## (a) Recommended single processing loop for MK1

Shared state = typed blackboard: slots {input, input_type, candidates[(answer, source, evidence span, score)], evidence[], verifier_results[], route_history[], budget, confidence}. Every write is schema-checked and logged.

1. **Perceive/type**: cheap encoder (bge-large eye) embeds input; write `input_type` and features (has passage? grid? label set? question form). No LLM.
2. **Recall**: look up long-term memory (episodic cache of similar past inputs -> route that worked, outcome). If a high-confidence cached route exists, propose it first (chunking analog).
3. **Propose agenda**: every module whose precondition matches the board becomes a candidate action (Hearsay/BB1 triggering). Includes "answer now", "abstain".
4. **Select (learned)**: score each candidate = competence similarity x Beta posterior P(success | type, module) - lambda*cost (ACT-R PG-C, Thompson/UCB noise for exploration). Pick top-1 (or top-2 if cheap).
5. **Run & write**: module writes its output with confidence and provenance to the board (broadcast: all modules can now read it next cycle).
6. **Verify**: independent checker(s) on the current best candidate: NLI entailment vs evidence, span grounding, program check on ARC train pairs, k-sample agreement. Write scores.
7. **Arbitrate**: arbiter (counting/Beta + change-point, already built) combines candidate scores into calibrated confidence and a margin over the runner-up.
8. **Decide**: if confidence >= tau_answer and verified -> go to 10. If **impasse** (low margin, verifier conflict, missing slot e.g. bridge entity) and budget remains -> go to 3 with the impasse type as a feature (Soar subgoal: e.g. "need another hop", "need another reader"). If budget exhausted or confidence < tau_abstain -> abstain.
9. Loop guard: max N cycles (e.g. 4), never re-run the same module on unchanged inputs, never let the speaker revise without new evidence.
10. **Speak**: Gemma E2B verbalises the chosen answer constrained to the board's answer + evidence (extractive copy where possible).
11. **Learn**: when feedback arrives (label, exam-dev, or self-check outcome), update Beta/utility for each (type, module) used, store the episode (route, outcome) in memory, update competence keys. If a cluster of inputs fails for all modules (novelty/outlier stats), flag "grow a module" candidate.

Hand-coded: slot schema, preconditions, budget. Learned: selection utilities, arbiter calibration, halting thresholds (tune on dev only), memory.

## (b) Three biggest risks and how to measure them

1. **Control overhead beats hand-wiring by nothing (or hurts)** — the general loop is worse than the per-exam hand-wired flows (blackboard history; MAST "minimal gains"). Measure: on each existing exam's dev split, unified loop vs hand-wired flow accuracy, plus mean modules invoked and latency/VRAM per item. Pre-register: unified must be within -1 pt of hand-wired on every exam and better on at least one mixed-stream test where input type is not given.
2. **Router/selection collapse or mis-routing** (Rosenbaum failure modes; LMC selection degrades with many modules). Measure: routing confusion matrix vs oracle-best module per item (oracle computed by running all modules on dev), module usage entropy, regret = oracle accuracy - achieved accuracy; replay old-task routing accuracy after each new module is added (forgetting of the router).
3. **Looping without real verification / miscalibrated abstention** (Huang 2024; SCORE; error compounding). Measure: per-cycle flip analysis (right->wrong vs wrong->right counts per extra loop), accuracy vs cycles curve, calibration (ECE, risk-coverage / AURC curve for abstention), and verifier precision on items where it triggers a loop. Rule: a loop step stays only if wrong->right > right->wrong on dev.
