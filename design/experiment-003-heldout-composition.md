# Experiment 003: held-out target-rule compositions

## Question

Does the RIMs-style core retain its correction-accuracy advantage over the parameter-matched GRU when the complete target-rule composition was absent from training?

## Hypothesis

On the test split, RIMs will have higher correction accuracy than `gru_parameter_matched` across the six seeds used in Experiment 002. The primary result is the paired per-seed difference in `corrected_accuracy`.

## One changed factor

Experiment 003 changes the split unit from individual random episodes to a target-rule template:

`(target door, previous key, final key)`

Ordinary examples have the same previous and final key. Corrected examples have different keys. The 27 templates are divided into fixed, disjoint sets:

| Split | Ordinary templates | Corrected templates | Total |
|---|---:|---:|---:|
| Train | 3 | 12 | 15 |
| Validation | 3 | 3 | 6 |
| Test | 3 | 3 | 6 |

Every split contains all three final-key labels. Template assignment stays fixed across seeds. Each split keeps the existing 8-step sequence, 15% missing-rule rate, 40% correction rate among rule-present episodes, and default episode counts.

## Models and evaluation

Keep all three Experiment 002 model configurations. Treat `rims` versus `gru_parameter_matched` as the primary comparison; `gru_state_matched` remains a capacity control. Use validation accuracy for checkpoint selection and evaluate the held-out test templates once at the end. Report per-seed and mean ± standard deviation for overall accuracy, correction accuracy, and missing-rule accuracy.

## Interpretation limit

This split holds out complete ordered target-rule templates. It does not hold out entire worlds, and distractor events may still contain the same door-key pairs. A positive result supports generalization across these target-rule compositions in this synthetic task; it does not establish generalization to natural language or real-world tasks.
