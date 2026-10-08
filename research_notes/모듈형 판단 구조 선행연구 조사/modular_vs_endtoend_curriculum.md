# Prior work: modular / compound AI vs end-to-end LLMs, and curriculum-learning reproducibility

Project being positioned (for comparison): synthetic Korean rule world; LFM2.5-1.2B zero/few-shot near chance; LoRA fine-tune 0.95 in-distribution / 0.64 held-out compositions; parse-then-judge pipeline (LLM or 279M mDeBERTa NLI parser -> (door, key, polarity) facts -> symbolic judge) = 1.00 on all splits; curriculum +0.085 on 1 seed, -0.18 mean on 4 seeds (unstable).

Note on sourcing: items marked [abstract-level] were cited from the arXiv/venue record from prior knowledge and search snippets, not re-read in full in this session; numbers are only given where a page was actually read or a search snippet quoted them.

## Q1. Compound AI systems and orchestration: what evidence that orchestrated specialists beat a single LLM, and on what tasks?

### Takeaway
"Compound AI system" is now a mainstream framing (BAIR 2024), and there is solid evidence that composition beats a single model on (i) cost-quality trade-offs (routing/cascades), (ii) open-ended chat quality (Mixture-of-Agents), and (iii) tool/solver-dependent tasks. The project's idea of "orchestrate specialists instead of one LLM" is therefore not novel as a thesis; its contribution can only be a small, controlled, reproducible demonstration on held-out compositions.

### Cited Findings
- BAIR blog "The Shift from Models to Compound AI Systems" (Zaharia, Khattab, Chen, Davis, Miller, Potts, Zou, Carbin, Frankle, Rao, Ghodsi; 18 Feb 2024) defines a compound AI system as one "that tackles AI tasks using multiple interacting components, including multiple calls to models, retrievers, or external tools", argues SOTA increasingly comes from composing components rather than bigger models; cites AlphaCode 2 and AlphaGeometry (LLM + symbolic solver) as examples; Databricks figure: 60% of LLM apps use RAG, 30% multi-step chains — [BAIR blog](https://bair.berkeley.edu/blog/2024/02/18/compound-ai-systems/)
- Mixture-of-Agents (Junlin Wang, Jue Wang, Athiwaratkun, Ce Zhang, James Zou; June 2024, arXiv 2406.04692): layered LLM agents each refining previous layer's outputs; open-source-only Together MoA scored 65.1% on AlpacaEval 2.0 vs GPT-4o 57.5%; also SOTA on MT-Bench and FLASK — [arXiv](https://arxiv.org/html/2406.04692v1); [Together blog](https://together.ai/blog/together-moa). Note: this is chat-preference quality, not reasoning/generalization; components are all general LLMs, not specialists.
- FrugalGPT (Lingjiao Chen, Zaharia, Zou; arXiv 2305.05176, TMLR 2024): LLM cascade can match GPT-4 performance with up to 98% cost reduction or improve accuracy over GPT-4 by 4% at the same cost — [arXiv](https://arxiv.org/abs/2305.05176); [TMLR record](https://mlanthology.org/tmlr/2024/chen2024tmlr-frugalgpt)
- RouteLLM (UC Berkeley/LMSYS, ICLR 2025): reported >85% cost reduction on MT-Bench while keeping 95% of GPT-4 performance, routing ~14% of queries to the strong model — via secondary summary [neuraltrust blog](https://neuraltrust.ai/blog/llm-model-routing) (primary: arXiv 2406.18665, not read this session)
- [abstract-level] HuggingGPT (Shen et al., 2023, NeurIPS 2023): ChatGPT as controller that plans and dispatches sub-tasks to specialist Hugging Face models — the closest "LLM as orchestrator of specialist models" precedent — [arXiv 2303.17580](https://arxiv.org/abs/2303.17580)
- [abstract-level] Toolformer (Schick et al., 2023, NeurIPS 2023): LM self-supervises when to call APIs (calculator, QA, search, etc.) — [arXiv 2302.04761](https://arxiv.org/abs/2302.04761)
- [abstract-level] ReAct (Yao et al., ICLR 2023): interleaves reasoning traces with tool actions — [arXiv 2210.03629](https://arxiv.org/abs/2210.03629)
- [abstract-level] DSPy (Khattab et al., 2023; ICLR 2024): programming model that compiles/optimizes multi-call LM pipelines — [arXiv 2310.03714](https://arxiv.org/abs/2310.03714)

### Inferences
- In this literature the "orchestrator" is almost always a large LLM and the "specialists" are tools or other models; the project's inversion (small LLM / NLI model as *reader only*, decision made by deterministic code) is closer to the neuro-symbolic parse-then-solve line (Q2) than to HuggingGPT/MoA.
- Evidence that compound systems win is strongest where a component provides something the LLM fundamentally lacks (exact execution, retrieval, search). The project's judge is exactly such a component, so a 1.00 score is expected rather than surprising; reviewers will ask how much of the result is "the judge is the ground-truth rule".

### Gaps
- Did not read RouteLLM primary paper; numbers come from a secondary blog.
- No controlled study found that compares "brain-region-like" specialist modules vs one LLM on held-out compositional splits in a matched-parameter setting.

## Q2. LLM as semantic parser + symbolic solver vs end-to-end LLM reasoning: generalization benefits and failure modes

### Takeaway
This is the project's closest prior work. LINC, Logic-LM and SatLM all show that "LLM translates, solver decides" beats CoT, and Logic-LM shows the gap widens with reasoning depth (a form of OOD robustness). Their documented failure mode is the parser: translation errors / non-executable formalizations, especially on realistic text (FOLIO, AR-LSAT). The project reproduces this pattern in miniature; its distinctive bits are (a) a tiny, quickly fine-tuned NLI cross-encoder as the parser, (b) Korean, (c) explicit held-out-composition splits contrasted with LoRA end-to-end.

### Cited Findings
- LINC (Olausson, Gu, Lipkin, Zhang, Solar-Lezama, Tenenbaum, Levy; EMNLP 2023): "the LLM acts as a semantic parser, translating premises and conclusions from natural language to expressions in first-order logic", offloaded to a theorem prover. StarCoder+ (15.5B) with LINC beat GPT-3.5 and GPT-4 with CoT by 38% and 10% on ProofWriter — [ACL Anthology](https://aclanthology.org/2023.emnlp-main.313); [arXiv 2310.15164](https://arxiv.org/abs/2310.15164)
- LINC vs CoT: "although both methods on average succeed roughly equally often on this dataset [FOLIO-type], they exhibit distinct and complementary failure modes" — [arXiv 2310.15164](https://arxiv.org/abs/2310.15164)
- Logic-LM (Pan, Albalak, Wang, Wang; Findings of EMNLP 2023): LLM formulates problem in task-specific symbolic language -> deterministic solver -> interpreter; plus self-refinement using solver error messages. Average +39.2% over standard prompting and +18.4% over CoT across ProofWriter, PrOntoQA, FOLIO, LogicalDeduction, AR-LSAT — [arXiv 2305.12295](https://arxiv.org/abs/2305.12295v1); [ACL Anthology](https://preview.aclanthology.org/moar-dois/2023.findings-emnlp.248)
- Logic-LM depth robustness: on ProofWriter, advantage over CoT grows from +7.1% at depth 0 to +39.4% at depth 5 — [ar5iv full text](https://ar5iv.labs.arxiv.org/html/2305.12295)
- Logic-LM parser failure: executable rate of generated formalizations — PrOntoQA 99.4%/100% (GPT-3.5/GPT-4), ProofWriter 87.3%/99.0%, FOLIO 66.7%/79.9%, LogicalDeduction 100%, AR-LSAT 11.3%/32.6%; self-refinement adds ~5 points on average; error types: predicate inconsistency, semantic misinterpretation (e.g. "below" vs "above"), FOL grammar violations (free variables) — [ar5iv full text](https://ar5iv.labs.arxiv.org/html/2305.12295)
- SatLM (Xi Ye, Qiaochu Chen, Isil Dillig, Greg Durrett; NeurIPS 2023): LLM writes a declarative specification, SMT solver derives the answer; "declarative specification is closer to the problem description" so parsing is easier; outperforms program-aided LMs by 23% on a hard GSM subset; SOTA on LSAT and BoardgameQA — [arXiv 2305.09656](https://arxiv.org/abs/2305.09656v3); [NeurIPS page](https://neurips.cc/virtual/2023/poster/71537)
- [abstract-level] PAL (Gao et al., ICML 2023) and Program-of-Thoughts (Chen et al., TMLR 2023): LLM writes Python, interpreter computes the answer — [PAL arXiv 2211.10435](https://arxiv.org/abs/2211.10435); [PoT arXiv 2211.12588](https://arxiv.org/abs/2211.12588)
- [abstract-level] Faithful CoT (Lyu et al., IJCNLP-AACL 2023): translation to symbolic chain + deterministic solver so the explanation is faithful by construction — [arXiv 2301.13379](https://arxiv.org/abs/2301.13379)
- End-to-end limits: "Faith and Fate" (Dziri et al., NeurIPS 2023) finds transformer LLMs (incl. fine-tuned) solve compositional tasks (multiplication, logic grid puzzles, DP) by "linearized subgraph matching" and performance decays rapidly with complexity — directly analogous to the project's 0.95 ID / 0.64 held-out gap — [NeurIPS proceedings](https://proceedings.neurips.cc/paper_files/paper/2023/hash/deb3c28192f979302c157cb653c15e90-Abstract.html); [arXiv 2305.18654](https://arxiv.org/abs/2305.18654v1)
- Compositional semantic parsing: SCAN/COGS solved via least-to-most prompting (14 exemplars) and Compositional Program Generation (perfect SCAN/COGS with 14/22 examples); a 2024 survey covers compositional generalization in LLM semantic parsing — [Drozdov et al. ICLR 2023, arXiv 2209.15003](https://arxiv.org/pdf/2209.15003); [Survey arXiv 2404.13074](https://arxiv.org/pdf/2404.13074); [CPG](https://cs.paperswithcode.com/paper/compositional-program-generation-for)

### Inferences
- The project's main result (parse + code judge = 1.00 on all splits; LoRA end-to-end drops to 0.64 OOD) is the same qualitative claim as Logic-LM/LINC, in a much smaller and cleaner setting. Novelty is low as a claim, moderate as a controlled replication with a sub-300M parser in Korean.
- The 1.00 likely reflects that the synthetic language is templated, so parsing is near-trivial (cf. PrOntoQA 99-100% executable vs FOLIO 67-80%). Prior work predicts the pipeline will break first at the parser when utterances become paraphrased/naturalistic; the project should test this (paraphrase split, unseen vocabulary, negation/scope variations) and report parser errors separately from judge errors.
- The parse schema (door, key, polarity) and judge are hand-designed for the world; LINC/Logic-LM use general FOL/SMT. Reviewers will view a task-specific schema as "solving the task by design" unless the schema is shown to transfer or be learned.

### Gaps
- Did not find a paper that specifically uses a small NLI cross-encoder (not a generative LLM) as the semantic parser feeding a symbolic solver; may exist under "NLI-based information extraction" — not confirmed.
- Did not verify PAL/PoT/Faithful-CoT numbers in this session.

## Q3. Small task-specific models (e.g., NLI cross-encoders) vs LLMs on structured extraction/reasoning; cost/latency

### Takeaway
There is consistent evidence that fine-tuned small encoders (RoBERTa/DeBERTa class) beat zero-shot prompted large LLMs on classification-style tasks; the project's mDeBERTa result is consistent with this, not novel. Routing/cascade papers quantify cost savings for using smaller models where possible.

### Cited Findings
- "Fine-Tuned 'Small' LLMs (Still) Significantly Outperform Zero-Shot Generative AI Models in Text Classification" (Bucher & Martini, arXiv 2406.08660, 2024): fine-tuned RoBERTa/DeBERTa-V3/ELECTRA/XLNet consistently beat zero-shot ChatGPT/GPT-3.5/GPT-4/Claude Opus across sentiment, stance, emotion, party-position tasks — [arXiv](https://arxiv.org/abs/2406.08660v1); [HTML](https://arxiv.org/html/2406.08660v2)
- "Small Language Models are Good Too: An Empirical Study of Zero-Shot Classification" (arXiv 2404.11122, 2024) — [arXiv](https://arxiv.org/html/2404.11122v1)
- Cost: FrugalGPT up to 98% cost reduction at GPT-4-level accuracy — [arXiv](https://arxiv.org/abs/2305.05176)

### Inferences
- The project's "279M NLI model fine-tuned ~1 min matches a 1.2B LLM as parser" fits this literature. A useful addition would be measured latency/throughput and parameter counts per component, which is the axis where the modular design genuinely wins.

### Gaps
- No direct study found of NLI cross-encoders as fact extractors for symbolic reasoning, nor Korean-specific small-vs-large comparisons for structured extraction.

## Q4. Cognitive-architecture-inspired modular AI (positioning only)

### Takeaway
The "brain regions" motivation has well-known antecedents; cite them for framing but do not claim the project implements them.

### Cited Findings
- Goyal, Didolkar, Lamb, ... Bengio, "Coordination Among Neural Modules Through a Shared Global Workspace" (ICLR 2022 oral): specialist modules compete for a bandwidth-limited shared workspace; capacity limits encourage specialization and compositionality — [arXiv 2103.01197](https://arxiv.org/abs/2103.01197v2); [ICLR](https://www.iclr.cc/virtual/2022/oral/6383)
- VanRullen & Kanai, "Deep Learning and the Global Workspace Theory" (2020/2021) — [alphaxiv 2012.10390](https://alphaxiv.org/abs/2012.10390)
- LeCun, "A Path Towards Autonomous Machine Intelligence" (2022 position paper): modules for perception, world model, cost, actor, short-term memory, and a "configurator" doing executive control by configuring the other modules for the task — [Meta AI blog](https://ai.meta.com/blog/yann-lecun-advances-in-ai-research/); [summary](https://mishig-jepawiki.hf.space/wiki/papers/lecun-position-paper)

### Inferences
- The project's modules (reader / memory / judge) map loosely onto LeCun's perception / memory / cost-actor; but those works use learned, end-to-end-trained modules, while the project's judge is hand-coded. Frame as "neuro-symbolic modularity inspired by" rather than "brain-like architecture".

### Gaps
- Classic cognitive architectures (SOAR, ACT-R) and recent "LLM + cognitive architecture" papers (e.g., CoALA, Sumers et al. 2023) were not searched in this session.

## Q5. Curriculum learning: effects and reproducibility

### Takeaway
The project's curriculum result (gain on one seed, reversal over 4 seeds) matches the literature: careful studies find ordering adds little over random in standard settings, and a 2026 study on deductive reasoning post-training finds no robust advantage. The seed instability is itself the expected finding and a good illustration of why multi-seed reporting is required.

### Cited Findings
- Wu, Dyer, Neyshabur, "When Do Curricula Work?" (ICLR 2021 oral): thousands of orderings; "curricula have only marginal benefits, and randomly ordered samples perform as well or better than curricula and anti-curricula", benefits attributed to dynamic dataset size; curriculum (not anti-curriculum) helps only with limited training-time budget or noisy labels — [arXiv 2012.03107](https://arxiv.org/abs/2012.03107v2); [Google Research](https://research.google/pubs/when-do-curricula-work/)
- Mordig, Opedal, Weiyang Liu, Schölkopf, "Rethinking Easy-to-Hard: Limits of Curriculum Learning in Post-Training for Deductive Reasoning" (arXiv 2603.27226, Mar 2026): "across multiple model families and curriculum schedules, we find no robust advantage in difficulty-based sequencing over standard random sampling in either accuracy or response length", for both SFT and RL; "the specific ordering of training examples plays a negligible role in achieving compositional generalization" — [arXiv](https://arxiv.org/abs/2603.27226). This is the single closest prior result to the project's curriculum finding.
- Positive-leaning LLM study: "Strategic Data Ordering: Enhancing LLM Performance through Curriculum Learning" (arXiv 2405.07490, 2024) reports only slight gains for Mistral-7B/Gemma-7B — [alphaxiv](https://alphaxiv.org/abs/2405.07490)
- [abstract-level] Surveys: Soviany, Ionescu, Rota, Sebe, "Curriculum Learning: A Survey" (IJCV 2022) — [arXiv 2101.10382](https://arxiv.org/abs/2101.10382); Wang, Chen, Zhu, "A Survey on Curriculum Learning" (TPAMI 2021) — [arXiv 2010.13166](https://arxiv.org/abs/2010.13166)
- Bouthillier et al., "Accounting for Variance in Machine Learning Benchmarks" (MLSys 2021): variance from data sampling, parameter initialization and hyperparameter choice "impact markedly" benchmark conclusions; recommend multiple trials randomizing these sources — [arXiv 2103.03098](https://arxiv.org/abs/2103.03098v1); [MLSys](https://proceedings.mlsys.org/paper_files/paper/2021/hash/0184b0cd3cfb185989f858a1d9f5c1eb-Abstract.html)
- [abstract-level] Henderson et al., "Deep Reinforcement Learning that Matters" (AAAI 2018): different random seeds can produce statistically different-looking learning curves; argues for multi-seed significance testing — [arXiv 1709.06560](https://arxiv.org/abs/1709.06560)

### Inferences
- Report the curriculum result as a negative/unstable result with all seeds, mean ± std (or CI), and note that n=4 is too few to claim a harmful effect either; the honest claim is "no reliable benefit; high variance", consistent with Wu et al. 2021 and Mordig et al. 2026.
- The same multi-seed standard should be applied to the headline LoRA numbers (0.95 / 0.64); otherwise the OOD gap claim inherits the same criticism.

### Gaps
- Did not verify which training-budget regime the project is in; Wu et al. predict curriculum helps mainly under short budgets/noisy labels, which could be tested.

## Q6. Korean-language benchmarks for compositional generalization or rule reasoning

### Takeaway
I found no Korean SCAN/COGS-style compositional-generalization benchmark and no Korean synthetic rule-reasoning dataset (ProofWriter/PrOntoQA analogue). Existing Korean benchmarks target knowledge, culture, commonsense, or general NLU. This is the project's clearest gap to occupy, provided the dataset is released with documented splits.

### Cited Findings
- KoBEST (2022): Korean BoolQ, COPA, WiC, HellaSwag, SentiNeg — general NLU/commonsense, not compositional splits — [arXiv 2204.04541](https://arxiv.org/pdf/2204.04541)
- HAE-RAE Bench (LREC-COLING 2024): Korean vocabulary, history, general knowledge, reading comprehension; explicitly not focused on logical reasoning — [arXiv 2309.02706](https://arxiv.org/pdf/2309.02706); [ACL Anthology](https://aclanthology.org/2024.lrec-main.704/)
- CLIcK (2024): 1,995 questions on Korean cultural and linguistic knowledge from native exams — [arXiv 2403.06412](https://arxiv.org/pdf/2403.06412)
- HRMCR "Multi-Step Reasoning in Korean and the Emergent Mirage" (arXiv 2501.05712, 2025): template/algorithm-generated multi-step reasoning with Korean cultural knowledge — closest synthetic Korean reasoning set, but culture-grounded, not rule-composition — [ar5iv](https://ar5iv.labs.arxiv.org/html/2501.05712)
- Korean generative commonsense dataset (Findings of NAACL 2022, "A Dog Is Passing Over The Jet?") notes Korean PLMs struggle with compositional, commonsense generation — [ACL Anthology](https://preview.aclanthology.org/watermark/2022.findings-naacl.172)
- KMMLU: 35,030 expert-level MCQs across 45 subjects — [review](https://liner.com/ko/review/kmmlu-measuring-massive-multitask-language-understanding-in-korean)
- K-BrowseComp (arXiv 2606.02404, 2026): Korean web-browsing agent benchmark with multi-hop/multi-constraint items — [opentrain summary](https://www.opentrain.ai/papers/k-browsecomp-a-web-browsing-agent-benchmark-grounded-in-korean-contexts--arxiv-2606.02404/)
- Morphological compositional generalization in LLMs (NAACL 2025, arXiv 2410.12656) studies agglutinative Turkish/Finnish, not Korean; LLMs lack systematicity on novel roots — [arXiv](https://arxiv.org/abs/2410.12656v4)
- Korean-language reasoning evaluation papers on KoreaScience (e.g., MC-LARC, Korean commonsense evaluation) appear in search but were not read — [KoreaScience](https://www.koreascience.kr/article/CFKO202306643316072.pdf)

### Inferences
- A released Korean synthetic rule world with ID / held-out-composition splits would be a modest but genuine contribution; the relevant comparators are PrOntoQA/ProofWriter (English synthetic) and SCAN/COGS (English compositional).

### Gaps
- DBpia/KCI not searched directly (access limits); Korean-venue (KCC/HCLT) papers on Korean logical NLI or rule reasoning may exist and should be checked before claiming "first".

## Positioning advice (synthesis, flagged as inference)
- Do not claim novelty for "orchestration beats one LLM" or "parse + symbolic solver generalizes better" — both are established (BAIR 2024; LINC, Logic-LM, SatLM 2023; Faith and Fate 2023).
- Defensible contributions: (1) a fully controlled, small-scale replication showing the ID vs held-out-composition gap for LoRA end-to-end vs parse+judge, with a sub-300M NLI parser fine-tuned in ~1 minute; (2) a Korean synthetic compositional rule benchmark (apparent gap); (3) a multi-seed negative result on curriculum consistent with Wu et al. 2021 / Mordig et al. 2026.
- Must-do to be credible: multi-seed for all headline numbers; a harder parsing split (paraphrase, unseen lexicon, nested negation) so the pipeline is not trivially perfect; error attribution (parser vs judge); compare to an LLM-as-parser with a general formalism (FOL/Python) rather than only a hand-designed schema; report cost/latency.
