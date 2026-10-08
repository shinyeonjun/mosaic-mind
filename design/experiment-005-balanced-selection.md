# Experiment 005: balanced checkpoint selection

## Question

Can checkpoint selection preserve both correction-rule generalization and abstention when no rule is present?

## Change

Keep Experiment 004's three composition folds, data sizes, architectures, optimizer, epochs, and seeds 42–47. Change only the validation checkpoint-selection score:

```text
balanced_accuracy = (corrected_accuracy + missing_rule_accuracy) / 2
```

The two component metrics receive equal weight regardless of their example counts. Report both components and overall test accuracy; do not use the balanced score as a replacement for those results.

## Run plan

First run fold 2, seed 42 with epoch metrics and compare it with the matching Experiment 004 run. If the result is valid, run all three folds with seeds 42–47. Experiment 005 writes separate result and checkpoint filenames.

## Interpretation limit

This tests a checkpoint-selection tradeoff on the current synthetic task. It does not change the training loss or establish a general advantage for either model architecture.
