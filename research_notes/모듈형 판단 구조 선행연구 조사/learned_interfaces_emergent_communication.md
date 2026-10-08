# Learned interfaces between neural modules, emergent communication, and entity-anchored representations: prior work vs. the Korean door/key project

Project under assessment (for comparison only):
(a) A pooled single-vector message per utterance (continuous 16-d, or discrete 3 slots x 8 symbols via ST Gumbel-softmax) between an mDeBERTa encoder and a neural judge, trained end-to-end from final answers only, failed completely. Loss stayed at label-prior entropy, the discrete messages collapsed to one string, and probes found no door/key information. A curriculum did not help.
(b) Fixing the message's referent by structure fixed this. The encoder cross-encodes (utterance, hypothesis "door X opens with key k") for every candidate pair and emits a 4-d vector per pair, with no NLI labels. The content emerged: a linear probe recovered entailed/contradicted/unrelated at 99-100% (3 of 4 seeds), and the system generalized to unseen phrasings and entity names.

## Q1. Emergent communication: why learned messages fail or collapse, bottlenecks, discrete vs continuous, Gumbel-softmax, compositionality

### Takeaway
Failure (a) matches a well-documented failure mode. Without help, end-to-end trained sender/receiver pairs often settle in a "no communication" equilibrium where the receiver ignores the channel and predicts the prior. Discrete channels add pressure to minimize entropy. The fixes in the literature are auxiliary shaping losses, population or architectural pressure, and restricting the channel; none of them is "anchor the message to a candidate referent". Compositionality and generalization are only weakly linked. The field mostly studies Lewis/referential games with synthetic attribute-value inputs, not pretrained text encoders reading natural language.

### Cited Findings
- **Foundational surveys.** Lazaridou & Baroni (2020), "Emergent Multi-Agent Communication in the Deep Learning Era" (arXiv 2006.02419), review language emergence among deep agents. They note that deep RL widened the range of simulations and that new analysis methods are needed to characterize emergent languages — [arXiv](https://arxiv.org/abs/2006.02419v2); [ar5iv](https://ar5iv.arxiv.org/html/2006.02419)
- **Newer reviews:**
  - Boldt & Mortensen, "A Review of the Applications of Deep Learning-Based Emergent Communication" (TMLR 2024, arXiv 2407.03302) — [arXiv](https://arxiv.org/pdf/2407.03302); [mlanthology](https://mlanthology.org/tmlr/2024/boldt2024tmlr-review)
  - "Emergent language: a survey and taxonomy" (Autonomous Agents and Multi-Agent Systems, 2025) — [paperity](https://paperity.org/p/363760864/emergent-language-a-survey-and-taxonomy)
  - "The Five Ws of Multi-Agent Communication" (arXiv 2602.11583, 2026) — [arXiv](https://arxiv.org/pdf/2602.11583)
  - "Deep Learning Agents and the Emergence of Compositional Languages" (JAIR, 2026) — [City Univ. open access](https://openaccess.city.ac.uk/id/eprint/37660/1/17302_pub.pdf). (Seen only in search snippets; author list not verified.)
- **Collapse and no-communication equilibrium.** Eccles et al., "Biases for Emergent Communication in Multi-agent Reinforcement Learning" (NeurIPS 2019, arXiv 1912.05676):
  - There is an equilibrium where the speaker emits random symbols and the listener's policy is independent of the message. Empirically, agents often fail to use the channel at all.
  - Once the channel is used meaningfully, agents reach at least a local optimum.
  - Fix: auxiliary "positive signalling" loss (diverse messages in different situations) and "positive listening" loss (the listener acts differently for different messages).
  - Emergent communication methods are highly sensitive to hyperparameters.
  - [arXiv](https://arxiv.org/pdf/1912.05676)
- **Positive signalling vs. positive listening.** Lowe, Foerster, Boureau, Pineau & Dauphin, "On the Pitfalls of Measuring Emergent Communication" (AAMAS 2019, arXiv 1903.05168):
  - Most metrics measure positive signalling, not positive listening.
  - Agents can signal without the listener using the signal.
  - Proposes causal influence of communication (CIC) — [arXiv](https://arxiv.org/pdf/1903.05168)
- **Unconstrained channels are effective but not interpretable.** Kottur, Moura, Lee & Batra, "Natural Language Does Not Emerge 'Naturally' in Multi-Agent Dialog" (EMNLP 2017, Best Paper):
  - Invented languages reach near-perfect task reward but are "decidedly not interpretable or compositional".
  - Human-like, compositional languages appear only when restrictions on how agents may communicate are increased, such as vocabulary size and memoryless listeners.
  - [ACL Anthology](https://aclanthology.org/D17-1321/); [arXiv](https://arxiv.org/abs/1706.08502)
- **Gumbel-softmax vs. REINFORCE.** Havrylov & Titov, "Emergence of Language with Multi-agent Games: Learning to Communicate with Sequences of Symbols" (NeurIPS 2017, arXiv 1705.11192):
  - The straight-through Gumbel-softmax relaxation converges much faster than REINFORCE and yields more effective protocols.
  - The protocols show some compositionality and variability.
  - [arXiv](https://arxiv.org/abs/1705.11192v1)
- **Entropy-minimization pressure.** Kharitonov, Chaabouni, Bouchacourt & Baroni, "Entropy Minimization in Emergent Languages" (ICML 2020):
  - Under common training procedures, the mutual information between the sender's input and the message is pushed to the minimum that still allows success.
  - This pressure grows as the channel becomes more discrete. Discreteness also increases robustness to overfitting and adversarial attacks.
  - [PMLR](https://proceedings.mlr.press/v119/kharitonov20a.html)
- **Compositionality is not needed for generalization.** Chaabouni, Kharitonov, Bouchacourt, Dupoux & Baroni, "Compositionality and Generalization in Emergent Languages" (ACL 2020):
  1. With large enough input spaces, emergent languages can refer to novel composite concepts.
  2. Compositionality and generalization are not correlated.
  3. Compositionality helps transmission to new learners.
  4. Compositionality "does not arise from simple generalization pressure".
  - [ACL Anthology](https://aclanthology.org/2020.acl-main.407)
- **Overfitting decomposition.** Rita, Tallec, Michel, Grill, Pietquin, Dupoux & Strub, "Emergent Communication: Generalization and Overfitting in Lewis Games" (NeurIPS 2022):
  - The Lewis-game loss splits into a co-adaptation loss and an information loss.
  - Controlling overfitting on co-adaptation recovers more compositional, more general languages.
  - [NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2022/hash/093b08a7ad6e6dd8d34b9cc86bb5f07c-Abstract.html)
- **Population heterogeneity.** Rita et al., "On the role of population heterogeneity in emergent communication" (ICLR 2022, arXiv 2204.12982): heterogeneity in training speed makes larger populations develop more stable, structured languages — [ar5iv](https://ar5iv.labs.arxiv.org/html/2204.12982)
- **Measuring compositionality against an oracle.** Andreas, "Measuring Compositionality in Representation Learning" (ICLR 2019) proposes Tree Reconstruction Error: how well representations are approximated by composing inferred primitives from an oracle analysis. This is conceptually close to the project's "linear probe recovers a human-designed schema" test — [arXiv](https://arxiv.org/abs/1902.07181)

### Inferences
- Failure mode (a) matches the "no-communication equilibrium" (Eccles et al. 2019) almost exactly:
  - Loss at label-prior entropy means the judge ignores the message, i.e., no positive listening.
  - Collapse to a single discrete string fits the entropy-minimization pressure on discrete channels (Kharitonov et al. 2020) once the receiver gives no gradient signal for diversity.
  - So (a) is a replication of a known phenomenon in a new setting (pretrained multilingual text encoder, Korean, multi-utterance aggregation). It is not a new finding.
- The project did not try the standard remedies for (a): positive signalling/listening losses (Eccles 2019), channel restrictions (Kottur 2017), or population/reset methods (Rita 2022). A reviewer would ask whether these rescue (a). Any claim that "pooled messages cannot work" should be scoped to "did not work without these known remedies".
- In emergent-communication terms, (b) removes the referential burden from the channel. The sender no longer has to encode *which* door/key; the structure says which pair a message is about, and only a low-entropy relation (3-4 values) must emerge. This is close to Kottur's "restrict the channel" lesson, but done by indexing messages by candidate referent rather than shrinking the vocabulary. I found no emergent-communication paper that frames the fix this way.

### Gaps
- Not verified: whether Chaabouni et al., "Emergent Communication at Scale" (ICLR 2022), or Mu & Goodman, "Emergent Communication of Generalizations" (NeurIPS 2021), discuss referent-indexed messages. Both are relevant but were not fetched.
- I found no 2024-2026 primary study that specifically uses a pretrained transformer text encoder as sender in an emergent-communication setup with collapse analysis. Only the surveys above were located.

## Q2. Learned interfaces between modules in modular / compositional systems

### Takeaway
Modular systems with learned inter-module messages generalize systematically only with strong structural priors, such as a correct layout, bandwidth limits, or competition for a workspace. Even then, the intermediate outputs often do not mean what their designers intended unless auxiliary supervision is added. The project's (b) result is the reverse case: structure alone, with no auxiliary supervision on the message, produced an intermediate output that matches the intended schema.

### Cited Findings
- **Layout matters more than modularity.** Bahdanau et al., "Systematic Generalization: What Is Required and Can It Be Learned?" (ICLR 2019):
  - Modular models generalize much more systematically, but this is "highly sensitive to the module layout".
  - End-to-end methods often learn inappropriate layouts or parametrizations.
  - Systematic generalization "may require explicit regularizers or priors in addition to modularity".
  - [arXiv](https://arxiv.org/abs/1811.12889)
- **Intermediate outputs are not faithful by default.** Subramanian, Bogin, Gupta, Wolfson, Singh, Berant & Gardner, "Obtaining Faithful Interpretations from Compositional Neural Networks" (ACL 2020):
  - On NLVR2 and DROP, neural module networks trained end-to-end produce intermediate outputs that differ from what each module is meant to compute.
  - Faithfulness improves with auxiliary supervision plus specific module-architecture choices.
  - [arXiv](https://arxiv.org/pdf/2005.00724)
- **Bandwidth-limited shared workspace.** Goyal, Didolkar, Lamb, Badola, Ke, Rahaman, Binas, Blundell, Mozer & Bengio, "Coordination Among Neural Modules Through a Shared Global Workspace" (ICLR 2022 oral): specialist modules communicate through a bandwidth-limited workspace and compete for access; the capacity limit encourages specialization and compositionality — [arXiv](https://arxiv.org/abs/2103.01197v2); [OpenReview](https://openreview.net/forum?id=XzTtHjgPDsT)

### Inferences
- The (b) result fits Bahdanau et al.'s thesis that structural priors (layout) are what make modular systems generalize, rather than modularity alone. Here the "layout" is the per-(utterance, hypothesis) factorization.
- It goes against the usual observation of Subramanian et al. (2020) that intermediate outputs drift from intended semantics. In (b), the interface ended up linearly aligned with the intended entailed/contradicted/unrelated schema without auxiliary supervision. Plausible reasons:
  - The 4-d per-pair channel is very narrow.
  - The judge's reasoning (find the pair that is entailed and not contradicted) leaves little room for alternative encodings.
  - mDeBERTa may already carry entailment-like features from pretraining.
- This alignment, with 1 of 4 seeds deviating, should be reported as an empirical observation, not a guarantee. It is in tension with documented unfaithfulness, which makes it interesting.
- RIMs (Goyal et al. 2019, arXiv 1909.10893) is a relevant foundation for sparse inter-module communication via attention. It was not fetched in this session, so no specific claims about it are made here.

### Gaps
- Not fetched: Andreas et al. 2016 "Neural Module Networks", Hu et al. 2017 "End-to-End Module Networks", and RIMs 2019. Their specifics (e.g., whether layout supervision was needed) are not cited here.
- I found no paper that directly compares "single pooled message" vs. "per-candidate-referent message" interfaces between two neural modules under answer-only supervision.

## Q3. Slot-based / object-centric / entity-anchored representations and systematic generalization

### Takeaway
There is solid evidence that factoring representations into per-object slots enables generalization to unseen combinations, both empirically (Slot Attention) and theoretically under structural assumptions (Wiedemer et al. 2024). The project's per-(door, key) cross-encoding is a form of entity-anchored or argument-anchored representation in text, with slots fixed by enumeration instead of learned by attention. Its generalization to unseen entity names fits the theory: the content of the representation is about the relation, while entity identity is handled by structure.

### Cited Findings
- **Slot Attention.** Locatello, Weissenborn, Unterthiner, Mahendran, Heigold, Uszkoreit, Dosovitskiy & Kipf, "Object-Centric Learning with Slot Attention" (NeurIPS 2020):
  - Slots are exchangeable and bind to objects through competitive attention.
  - They enable "generalization to unseen compositions" in object discovery and property prediction.
  - [NeurIPS](https://papers.nips.cc/paper/2020/hash/8511df98c02ab60aea1b2356c013bc0f-Abstract.html)
- **Provable compositional generalization.** Wiedemer, Brady, Panfilov, Juhos, Bethge & Brendel, "Provable Compositional Generalization for Object-Centric Learning" (ICLR 2024 oral):
  - Autoencoders with structural assumptions on the decoder (compositional/additive), plus encoder-decoder consistency, provably learn object-centric representations that generalize compositionally.
  - They note that the common belief "object-centric means compositional generalization" previously lacked a principled basis.
  - [ICLR proceedings](https://proceedings.iclr.cc/paper_files/paper/2024/hash/57c89126d60c209f48d0e6395c766bb3-Abstract-Conference.html); [arXiv html](https://arxiv.org/html/2310.05327v2)

### Inferences
- **Close analogue:** a Relation Network (Santoro et al. 2017, arXiv 1706.01427, not fetched) applies a shared function g(o_i, o_j) to every object pair and aggregates. The project's encoder f(utterance, hypothesis(door, key)), applied to every candidate pair with shared weights and aggregated by a judge, is structurally a text-based relation network. Its "objects" are hypothesis-instantiated candidate pairs, not perceptual slots.
- **Difference from Slot Attention:** in the project, slots are given by symbolic enumeration of candidate (door, key) pairs and instantiated as natural-language hypotheses. Binding is not learned. This is closer to "query-conditioned" encoding, as in cross-encoder retrieval or reading comprehension where the question is concatenated to the passage, than to unsupervised object discovery.
- **Why unseen names generalize:** the entity name appears verbatim in both premise and hypothesis, so the cross-encoder only needs a string-matching and relation function. This is the textual counterpart of Wiedemer et al.'s point that structural factorization buys compositional generalization.

### Gaps
- I did not locate a paper that explicitly calls hypothesis-templated cross-encoding over enumerated entity pairs "entity-anchored messages" for an inter-module interface. The term appears to be the project's own.
- Recurrent Entity Networks (Henaff et al., ICLR 2017) and pointer/argument-anchored representations were not fetched; no specific claims made.

## Q4. NLI-style (premise, hypothesis) cross-encoders as a general reading interface, especially without NLI labels; probing for a human schema

### Takeaway
Using entailment as a universal interface is well established (zero-shot classification, EFL, Multee for multi-hop QA). In those systems, however, the entailment function is pretrained or fine-tuned on NLI labels, or the task is recast with explicit entailment labels. The project's distinctive element is that the NLI-shaped interface had no NLI labels: the 3-way entailed/contradicted/unrelated structure emerged only from downstream answers, and a probe confirmed it. I found no paper that reports this exact result.

### Cited Findings
- **Entailment as Few-Shot Learner (EFL).** Wang, Fang, Khabsa, Mao & Ma (arXiv 2104.14690, 2021):
  - NLP tasks are reformulated as entailment, and the model is fine-tuned with as few as 8 examples.
  - +12% over few-shot SOTA on 18 tasks, competitive with roughly 500x larger models.
  - Relies on entailment-formatted supervision and NLI-style intermediate training.
  - [arXiv](https://arxiv.org/abs/2104.14690)
- **Universal NLI classifiers.** "Building Efficient Universal Classifiers with Natural Language Inference" (Laurer et al., arXiv 2312.17543, 2023): label hypotheses are verified for entailment by NLI-trained models — [arXiv](https://arxiv.org/html/2312.17543v2)
- **Multee.** Trivedi, Kwon, Khot, Sabharwal & Balasubramanian, "Repurposing Entailment for Multi-Hop Question Answering Tasks" (NAACL 2019):
  - QA is reduced to entailment.
  - A local module scores sentence importance and a global module aggregates.
  - Both use entailment functions pretrained on SNLI/MultiNLI.
  - [ACL Anthology](https://aclanthology.org/N19-1302/); [arXiv](https://arxiv.org/pdf/1904.09380)
- **Latent-variable weak supervision for evidence and verification.** Examples: weakly supervised fact-checking that jointly extracts evidence sentences and verifies claims without sentence annotations; Min et al., "A Discrete Hard EM Approach for Weakly Supervised Question Answering" (EMNLP 2019), which treats the solution as a latent discrete variable over a precomputed candidate set — [ACL Anthology D19-1284](https://preview.aclanthology.org/setup/D19-1284); [Princeton PDF](https://www.cs.princeton.edu/~danqic/papers/emnlp2019.pdf)
- **Linear-probe recovery of a human schema.** Andreas (ICLR 2019, TRE) formalizes measuring how well learned representations match an oracle compositional analysis. This is methodologically close to the project's probe — [arXiv](https://arxiv.org/abs/1902.07181)

### Inferences
- **Closest overlap with (b):** Multee and EFL both use (text, hypothesis) entailment as the interface between reading and a downstream decision. The differences:
  - The project uses no NLI-labeled data or NLI-pretrained head.
  - The hypotheses are instantiated over enumerated candidate (entity, entity) pairs and aggregated by a learned judge.
  - The 3-way NLI schema is found post hoc by probing, not imposed.
- **Confound:** mDeBERTa-v3 base checkpoints are widely used for NLI after further fine-tuning, but the raw pretrained encoder is not NLI-trained. If the project used a raw checkpoint, the "no NLI labels" claim holds. If it used an XNLI-fine-tuned checkpoint, the claim weakens substantially. The project should state the exact checkpoint. (Inference; not verified for the project.)
- Emergence of entailed vs. contradicted vs. unrelated is functionally forced if the judge's correct answer requires distinguishing "door X opens with k" from "door X does not open with k" and from irrelevant utterances. The novelty is that the human NLI schema is the solution gradient descent finds when the interface is shaped like NLI. This parallels latent semantic parsing, where a well-designed program space makes the right latent structure the easiest one to find (see Q5).

### Gaps
- I found no paper that trains a (premise, hypothesis) cross-encoder from scratch or from a non-NLI checkpoint purely via a downstream answer loss and then shows by probing that its outputs recover the 3-way NLI relation. Absence from my searches is not proof of absence. A targeted search of "latent entailment", "weakly supervised NLI", and "neural natural logic" (2021-2026) is recommended.
- Not fetched: work on emergent entailment in end-to-end trained verifiers, e.g., FEVER-style latent evidence models.

## Q5. Latent-variable semantic parsing / learning from denotations as an analogue

### Takeaway
Weakly supervised semantic parsing learned structured intermediate representations (logical forms) from answers only. It worked because the space of latent structures was highly constrained (a grammar or executable formalism with a deterministic executor). Its main failure mode was spurious programs that reach the right answer for the wrong reason. Project (b) is a neural analogue: a constrained, enumerated hypothesis space plus a deterministic-ish aggregation by the judge makes the intended latent relation learnable from answers.

### Cited Findings
- **DCS from question-answer pairs.** Liang, Jordan & Klein, "Learning Dependency-Based Compositional Semantics" (ACL 2011): learns to map questions to answers via latent logical forms induced from question-answer pairs, using a new formalism (DCS) — [ACL Anthology P11-1060](https://aclanthology.org/P11-1060/); [arXiv](https://arxiv.org/pdf/1109.6841)
- **Scaling to Freebase.** Berant, Chou, Frostig & Liang, "Semantic Parsing on Freebase from Question-Answer Pairs" (EMNLP 2013): trains a semantic parser at Freebase scale from question-answer pairs instead of annotated logical forms — [ACL Anthology D13-1160](https://aclanthology.org/D13-1160/)
- **Spurious programs.** Guu, Pasupat, Liu & Liang, "From Language to Programs: Bridging Reinforcement Learning and Maximum Marginal Likelihood" (ACL 2017):
  - With indirect supervision, the central problem is spurious programs that accidentally produce the correct result.
  - Randomized beam search combines MML's systematic search with RL's exploration to guard against them.
  - [arXiv](https://arxiv.org/pdf/1704.07926)
- **Hard EM over a precomputed candidate set.** Min et al. (EMNLP 2019) cast weakly supervised QA as discrete latent-variable learning over a task-specific candidate set containing one correct option — [Princeton PDF](https://www.cs.princeton.edu/~danqic/papers/emnlp2019.pdf)

### Inferences
- **The analogy is close:** in (b), the candidate (door, key) hypotheses play the role of a candidate program set, the judge plays the role of an executor, and the per-pair 4-d vector is a soft latent "denotation-relevant" label. In both, the structure of the latent space carries most of the inductive bias.
- **Spurious solutions:** the 1 of 4 seeds in (b) whose probe did not reach 99-100% is the neural counterpart of a spurious program, a latent encoding that solves training answers but does not align with the intended schema. The project should report it in those terms and check whether that seed's test generalization was also perfect.
- **Difference from semantic parsing:** the latent space is not discrete symbolic programs. It is a continuous vector per candidate, and the schema was not pre-specified as a type system. This makes the project closer to "neural latent variable over a structured index set" than to classical semantic parsing.

### Gaps
- Not fetched: Clarke et al. 2010 ("Driving semantic parsing from the world's response"), Pasupat & Liang 2015/2016, and neural program induction work (e.g., Neural Symbolic Machines, Liang et al. 2017). They would strengthen this section but no claims are made about them here.

## Overall novelty assessment (synthesis across questions)

### Takeaway
(a) is a replication of a known failure: the no-communication equilibrium and discrete-channel collapse. (b) combines known ingredients: an entailment-as-interface format, relation-network-style per-pair factorization, and learning latent structure from denotations. The combination is uncommon and I found no prior paper that reports it: a pretrained multilingual encoder emitting per-candidate-referent NLI-shaped messages to a learned judge, trained only from final answers, with the human 3-way entailment schema recovered by linear probing and zero-shot transfer to unseen entity names and phrasings.

### Cited Findings
- Known failure of unaided learned messages: Eccles et al. 2019 — [arXiv](https://arxiv.org/pdf/1912.05676); Lowe et al. 2019 — [arXiv](https://arxiv.org/pdf/1903.05168); Kharitonov et al. 2020 — [PMLR](https://proceedings.mlr.press/v119/kharitonov20a.html)
- Restricting the channel improves interpretability: Kottur et al. 2017 — [ACL](https://aclanthology.org/D17-1321/)
- Structural priors (layout) are needed for systematic generalization: Bahdanau et al. 2019 — [arXiv](https://arxiv.org/abs/1811.12889)
- Intermediate module outputs are usually unfaithful without auxiliary supervision: Subramanian et al. 2020 — [arXiv](https://arxiv.org/pdf/2005.00724)
- Entailment as a universal interface, with NLI supervision: Wang et al. 2021 — [arXiv](https://arxiv.org/abs/2104.14690); Trivedi et al. 2019 — [ACL](https://aclanthology.org/N19-1302/)
- Answer-only supervision works when the latent space is structured, with spurious solutions as the risk: Liang et al. 2011 — [ACL](https://aclanthology.org/P11-1060/); Guu et al. 2017 — [arXiv](https://arxiv.org/pdf/1704.07926)

### Inferences
- **Defensible novelty claims:**
  1. A controlled contrast between pooled messages and referent-anchored messages under identical answer-only supervision, with the anchored version succeeding where the pooled one fails completely.
  2. Emergence of a human NLI-like 3-way schema in an interface with no NLI labels, verified by a linear probe.
  3. Perfect generalization to unseen entity names and phrasings.
  4. A Korean-language and mDeBERTa setting.
- **Claims to avoid:**
  - That pooled messages "cannot" work, since standard remedies were untested.
  - That the schema is "discovered" independently of the encoder's pretraining; the checkpoint matters.
  - Generality beyond a small synthetic door/key domain with few candidates. Enumerating all candidate pairs scales quadratically and is the same cost known from cross-encoder vs. bi-encoder trade-offs.
- **Baselines a reviewer would expect:**
  - Pooled message plus positive-signalling/listening losses.
  - Pooled message with a larger or continuous channel and no bottleneck.
  - Bi-encoder (separate utterance and hypothesis encodings) vs. cross-encoder, to isolate whether cross-attention or referent indexing matters.
  - An NLI-fine-tuned encoder as an upper bound.

### Gaps
- The novelty statement rests on the absence of a matching paper in ~20 targeted searches. A dedicated search on Semantic Scholar/Google Scholar for "latent entailment" + "answer supervision" + "probing", and for 2024-2026 work on "interfaces between pretrained modules" (e.g., model stitching, learned communication between LLM agents via embeddings), is recommended before claiming priority.
