# Experiment 004: rotating held-out compositions

## Question

Does RIMs retain an advantage on correction rules that were not present during training, across more than one hand-picked test split?

## Design

Use three deterministic folds over the 27 target-rule templates `(target door, previous key, final key)`. Each fold has disjoint train, validation, and test templates:

| Split | Ordinary | Corrected | Total templates |
|---|---:|---:|---:|
| Train | 3 | 9 | 12 |
| Validation | 3 | 3 | 6 |
| Test | 3 | 6 | 9 |

Each split includes all three final-key labels. Every one of the 27 templates appears in the test set exactly once across the three folds. Within each fold, the sets remain disjoint. Missing-rule episodes retain the existing 15% rate; corrections remain 40% of rule-present episodes. Sequence length, episode counts, architectures, optimizer, and epoch limit stay the same.

## Training and reporting

Select checkpoints by validation `corrected_accuracy`, the primary metric. Keep the three Experiment 002 model configurations and the six seeds 42–47. Test summaries include accuracy for every target-rule template, including an example count, as well as overall, correction, and missing-rule accuracy.

The full run produces 18 result files: one per fold and seed. Compare RIM with the parameter-matched GRU within each fold/seed pair, then summarize results across folds and seeds. Treat folds as planned coverage of the 27 templates, not as independent task domains.

## Interpretation limit

This tests generalization across target-rule templates in the current synthetic task. Distractor facts and input encoding remain synthetic, so results do not establish generalization to language or real-world tasks.
