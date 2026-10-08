# Compositional generalization from structural inductive biases: prior work vs. the "reader + equivariant judge" project

Project being assessed (for comparison only): synthetic Korean rule world; (a) fine-tuned 1.2B LLM end-to-end fails on held-out (case type x source) combos and unseen entity names, while "reader -> facts -> separate judge" generalizes; (b) 2x2 ablation: with the same designed per-utterance facts, a generic transformer judge = 0.73 on held-out compositions, key-permutation-equivariant judge (shared GRU per key + DeepSets) = 0.93, plus source structure (shared per-source reader GRU, folded in trust order by a shared cell) = 1.00 on 4/4 seeds.

## Q1. Permutation/group equivariance and symmetry-based inductive biases for compositional generalization

### Takeaway
The idea that "compositional generalization = building the right symmetry into the architecture" is well established (Gordon et al. 2020; lexicon/symbol-invariance lines through 2026). Finding (b)'s direction — equivariance over interchangeable keys/entities beats a generic transformer on held-out combinations and unseen names — is therefore an expected, previously-argued result; what is new is only the specific setting (defeasible, trust-ordered, multi-source rule judgment) and the clean factorization of where the gain comes from.

### Cited Findings
- Gordon, Lopez-Paz, Baroni, Bouchacourt (ICLR 2020), "Permutation Equivariant Models for Compositional Generalization in Language": hypothesize that language compositionality is a form of group-equivariance and propose tools for building equivariant seq2seq models; on SCAN they show the equivariant architecture achieves the kind of compositional generalization required (permutations act on groups of interchangeable words, e.g. verbs / directions) — [ICLR virtual](https://iclr.cc/virtual/2020/poster/1592); [Meta AI page](https://ai.meta.com/research/publications/permutation-equivariant-models-for-compositional-generalization-in-language/); [arXiv 2003.05161](https://arxiv.org/abs/2003.05161)
  - Caveat: I could not machine-read the results table in this session. From memory (unverified here), equivariant seq2seq is near-perfect on SCAN "add jump" and high on "around right" but fails the length split, and the symmetry group must be specified by hand in advance. Check the paper's Table before quoting numbers.
- Akyürek & Andreas (ACL 2021), "Lexicon Learning for Few-Shot Sequence Modeling": many failures come from models not separating lexical phenomena from syntactic structure; adding a learned token-level lexical translation mechanism to decoders improves SCAN (jump, around right) and COGS — [ACL Anthology](https://aclanthology.org/2021.acl-long.382); [arXiv](https://arxiv.org/pdf/2106.03993)
- Işık & Li (arXiv, Jan 30 2026), "Names Don't Matter: Symbol-Invariant Transformer for Open-Vocabulary Learning": Transformer mechanism provably invariant to renaming of interchangeable tokens (parallel embedding streams per interchangeable token + aggregated attention); reports large gains on open-vocabulary tasks needing generalization to novel symbols — [arXiv 2601.23169](https://arxiv.org/abs/2601.23169v1)
- Fokoue et al. (IJCAI 2023), NIAGRA: name-invariant GNN formula representations for theorem proving; up to 10% in-domain and up to 28% better transfer to new domains than prior learned provers — [arXiv 2305.08676](https://arxiv.org/abs/2305.08676v1); [IJCAI](https://www.ijcai.org/proceedings/2023/359)
- Gontier et al. (NeurIPS 2020), "Measuring Systematic Generalization in Neural Proof Generation with Transformers": notes entity-name generalization problems when names are chosen arbitrarily — [arXiv 2009.14786](https://arxiv.org/pdf/2009.14786)
- Lake & Baroni (Nature, 2023), "Human-like systematic generalization through a meta-learning neural network" (MLC): an *alternative* to architectural priors — a standard transformer optimized via a stream of compositional tasks reaches human-like systematicity — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC10620072)

### Inferences
- The project's "shared GRU per key + DeepSets aggregation" is exactly the Gordon-style / DeepSets-style equivariance prior applied over fact keys (doors/keys) instead of over words. The finding that it fixes held-out compositions and unseen names is consistent with, not novel to, this literature.
- MLC is a useful counterpoint for the paper's discussion: generalization can also come from training distribution (meta-learning) rather than architecture; the project's 2x2 only tests architecture, not data-side alternatives.

### Gaps
- Exact SCAN numbers for Gordon et al. not verified in this session (PDF not parseable).
- Did not survey follow-ups like White & Cotterell (2022) "Equivariant transduction through invariant alignment" or LexSym (Akyürek & Andreas 2023) — known to exist from memory but not verified here.

## Q2. What SCAN/COGS/gSCAN/CFQ/SLOG and recent LLM studies show about generic seq2seq/transformers and fine-tuned pretrained LMs

### Takeaway
Generic transformers and fine-tuned pretrained LMs reach near-ceiling on lexical recombination (COGS) when tuned, but collapse on structural generalization (SLOG ~40.6%), and LLMs show a persistent "compositionality gap" and linearized-pattern-matching behavior. Finding (a) — a fine-tuned 1.2B LLM failing held-out (case x source) combinations and new names — is in line with this whole body of evidence.

### Cited Findings
- Li et al. (EMNLP 2023), "SLOG: A Structural Generalization Benchmark for Semantic Parsing": extends COGS with 17 structural generalization cases; "the generalization accuracy of Transformer models, including pretrained ones, only reaches 40.6%, while a structure-aware parser only achieves 70.8%", far from the near-perfect accuracy on COGS — [arXiv 2310.15040](https://arxiv.org/abs/2310.15040); [ACL Anthology](https://aclanthology.org/2023.emnlp-main.194)
- Yao & Koller (EMNLP 2022), "Structural generalization is hard for sequence-to-sequence models": a fine-tuned T5-base reaches ~83% overall on COGS, but seq2seq models fail structural generalization — [arXiv 2210.13050](https://arxiv.org/pdf/2210.13050)
- Csordás, Irie, Schmidhuber (EMNLP 2021), "The Devil is in the Detail": baseline transformers with default hyper-parameters fail dramatically; simple fixes (embedding scaling, early stopping on proper validation, relative positions, Universal Transformer) raise COGS 35%->81%, PCFG productivity 50%->85%, SCAN length to 100% (cutoff 26) — [ACL Anthology](https://aclanthology.org/2021.emnlp-main.49); [arXiv 2108.12284](https://arxiv.org/abs/2108.12284v1)
- Dziri et al. (NeurIPS 2023), "Faith and Fate: Limits of Transformers on Compositionality": on multiplication, logic grid puzzles and DP, transformer LLMs reduce multi-step compositional reasoning to linearized subgraph matching rather than systematic procedures; errors compound with complexity — [NeurIPS proceedings](https://proceedings.neurips.cc/paper_files/paper/2023/hash/deb3c28192f979302c157cb653c15e90-Abstract.html); [arXiv 2305.18654](https://arxiv.org/abs/2305.18654)
- Press et al. (Findings of EMNLP 2023), "Measuring and Narrowing the Compositionality Gap": as GPT-3 size grows, single-hop accuracy improves faster than multi-hop, so the compositionality gap does not shrink; self-ask decomposition helps — [arXiv 2210.03350](https://arxiv.org/abs/2210.03350v3)
- Ruis et al. (NeurIPS 2020), gSCAN: grounded extension of SCAN; discusses Gordon et al.'s equivariance approach as prior work — [arXiv 2003.05161 ar5iv rendering returned gSCAN content](https://ar5iv.labs.arxiv.org/html/2003.05161) (note: that ar5iv URL served the gSCAN paper in this session; treat as a mis-mapping and cite gSCAN directly)

### Inferences
- (a) mostly replicates a known phenomenon in a new domain (Korean, defeasible rules, small 1.2B model). Its value is as a motivating baseline, not as a novel claim. Reviewers will likely ask whether Csordás-style tuning tricks or larger LLMs close the gap.
- The project's held-out "case type x source" split is closest in spirit to SLOG/COGS-style systematic splits and to CLOSURE-style held-out combinations (Q4).

### Gaps
- CFQ (Keysers et al., ICLR 2020, MCD splits) and 2024–2026 evaluations of frontier LLMs on COGS/SLOG were not re-verified in this session.

## Q3. Neural module networks, neuro-symbolic, and "LLM reads, symbolic solver reasons"; do any ablate interface vs reasoner?

### Takeaway
The "neural parser + separate reasoner" decomposition is very well established (NMN, NS-VQA, Logic-LM, LINC, SatLM, PAL). But these systems use a *symbolic, hand-built* reasoner and compare whole pipelines against end-to-end LLMs; I found none that holds the intermediate representation fixed and swaps the *learned* reasoner's architecture (generic vs equivariant vs source-structured) in a 2x2 — which is the project's distinctive move.

### Cited Findings
- Pan et al. (Findings of EMNLP 2023), Logic-LM: LLM translates the problem into a symbolic language, a deterministic solver reasons, a result interpreter maps back; +39.2% over standard prompting and +18.4% over CoT across ProofWriter, PrOntoQA, FOLIO, LogicalDeduction, AR-LSAT; faithfulness guaranteed if formulation is correct — [arXiv 2305.12295](https://arxiv.org/pdf/2305.12295); [ACL Anthology](https://preview.aclanthology.org/moar-dois/2023.findings-emnlp.248)
- LINC (Olausson et al., EMNLP 2023): LLM as semantic parser into first-order logic, offloaded to an external theorem prover; SatLM (Ye et al., NeurIPS 2023): LLM writes declarative specifications solved by an automated solver — described in [Logic-LM paper](https://arxiv.org/pdf/2305.12295) and related surveys
- Bahdanau et al. (ICLR 2019), "Systematic Generalization: What Is Required and Can It Be Learned?" (SQOOP): modular models generalize much more systematically than generic ones, and generalization is "highly sensitive to the module layout"; end-to-end layout induction often learns layouts that do not generalize; suggests explicit priors/regularizers are needed — [arXiv 1811.12889](https://arxiv.org/abs/1811.12889)
- Bahdanau et al. (2019), CLOSURE: CLEVR models fail novel combinations of known constructs; even an NMN with ground-truth programs at test time generalizes badly; Vector-NMN module architecture improves it — [arXiv 1912.05783](https://arxiv.org/pdf/1912.05783)

### Inferences
- CLOSURE is the closest conceptual precedent to (b): it shows that a *correct interface (ground-truth programs)* is not sufficient — the module architecture determines systematic generalization. The project's "same designed facts, different judge -> 0.73 vs 0.93 vs 1.00" is the same message in a different domain, with a cleaner factorial design and an explicit symmetry/ordering-based explanation.
- Bahdanau 2019 (SQOOP) also matches (b) closely: structure (layout) of the reasoner, not just modularity, drives generalization.
- The novelty versus Logic-LM/LINC: those use symbolic solvers (generalization by construction); the project shows a *learned* neural judge can match that if given the right equivariance + trust-order fold, which is a different and narrower claim.

### Gaps
- NMN (Andreas et al. CVPR 2016), NS-VQA (Yi et al. NeurIPS 2018), PAL (Gao et al. ICML 2023) are well known but not re-verified this session.
- No paper found that runs an explicit 2x2 {interface format} x {reasoner architecture}; absence of evidence after limited search, not proof.

## Q4. Prior work disentangling representation/interface vs reasoner architecture; and defeasible / source-preference reasoning with learned modules

### Takeaway
Partial precedents exist for "architecture, not interface" (CLOSURE, SQOOP layout study, Mittal et al. on modularity confounds), and for source-preference defeasible reasoning (BoardgameQA). But I found no prior work combining (i) learned neural judges, (ii) source trust ordering / later-correction override, and (iii) held-out source x case combinations — this combination appears to be the project's genuine novel niche.

### Cited Findings
- Mittal, Bengio, Lajoie (NeurIPS 2022), "Is a Modular Architecture Enough?": asks whether gains of modular (MoE) systems come from real specialization or confounders such as ease of optimization; builds metrics for collapse/specialization; finds modular architectures can help OOD generalization but specialization is often not achieved — [arXiv 2206.02713](https://arxiv.org/pdf/2206.02713); [NeurIPS poster](https://neurips.cc/virtual/2022/poster/54964)
- Kazemi et al. (NeurIPS 2023 Datasets & Benchmarks), BoardgameQA: frames reasoning with contradictory information guided by *preferences over sources/rules* as defeasible reasoning; answers proved/disproved/unknown; fine-tuned LMs (with and without proofs) and few-shot LLMs perform poorly with conflicting information — [arXiv 2306.07934](https://arxiv.org/pdf/2306.07934)

### Inferences
- BoardgameQA is the closest task-level precedent for the project's "trust order + override" logic; it evaluates end-to-end LMs (supporting (a)) but does not propose a structured neural judge or held-out source x case splits.
- The project's "fold per-source states in trust order with a shared cell" is a learned analogue of a priority-ordered defeasible inference; I found no neural-module paper doing this.
- What remains novel (honest scope): (1) the 2x2 factorial design isolating judge inductive bias from message format with the same facts; (2) decomposing the judge's bias into two named symmetries (key-permutation equivariance and source-sharing + trust-ordered fold) and showing they are additive (0.73 -> 0.93 -> 1.00); (3) applying it to defeasible, source-reliability-ordered judgment with unseen source x case combos. What is NOT novel: that end-to-end fine-tuned LMs fail held-out compositions; that parser + separate reasoner generalizes; that equivariance/name-invariance helps unseen entities; that reasoner architecture rather than interface drives systematicity (CLOSURE/SQOOP).
- Weaknesses reviewers will flag relative to prior work: tiny synthetic world, 4 seeds, a single generic-transformer baseline (Csordás-style tuning could narrow the 0.73), and that the inductive bias is hand-specified to match the data generator's symmetries (same criticism leveled at Gordon et al.).

### Gaps
- Did not find neural work on source-reliability-ordered belief revision with unseen source-case generalization; literature on knowledge conflicts in RAG (context vs parametric) exists but was not surveyed here.
- Jarvis et al. (ICLR 2023) "On the specialization of neural modules" may be relevant to data-vs-architecture disentangling; not verified this session.
