# Experiment 007: generalization to unseen rule compositions

## Question

Can a reward-trained query policy choose useful probes for a rule composition it never encountered during training, when each of that rule's component operations has appeared in other combinations?

## Hypothesis

**H1 (primary):** each learned controller achieves higher mean task reward on held-out rule compositions than the random-query baseline. The information-gain policy is an exact-catalog algorithmic reference, not a learned model.

**H2 (exploratory):** compare RIMs with a parameter-matched GRU using paired per-seed test-reward differences. No architecture winner is assumed.

## Environment

A rule is a composition of one event scope and one reducer:

- Scope: all events, events for the target door, or events for other doors.
- Reducer: first, second, last, minimum, maximum, or most frequent key. A mode tie selects the smallest key; `second` returns `ABSTAIN` when fewer than two events match.

This gives 18 candidate rules. Probes retain the existing three ordered events and 216 possible actions. Final cases retain the existing 4–8-event generator. The agent receives only probe outcomes and the remaining candidate mask; the final case is hidden until probing ends.

Before training, enumerate every probe against every rule and require all 18 rules to have distinct outcome signatures. This prevents an impossible rule-identification benchmark.

## Split and controls

Index scopes by `s = 0..2` and reducers by `r = 0..5`; assign each composition to fold `(s + r) mod 3`. Each fold holds out six compositions and trains on the other 12. Every scope and reducer still occurs in training, and across three folds each composition is tested exactly once. Validation episodes use training compositions only.

Compare no-probe, random, fixed-order, exact information-gain, GRU-256, parameter-matched GRU, and RIMs policies. All use the same 18-rule candidate catalog, posterior update, test episodes, and final answer decoder. The learned policies use the existing reward-only REINFORCE objective and 0.05 probe cost. Set the matched GRU width to 206 for the expanded input and record exact parameter counts.

First run fold 0, seed 42 as a pilot. If its catalog, tensor-shape, and output checks pass, run folds 0–2 and seeds 42–47. Aggregate the three held-out folds within each seed; report a paired 95% t interval over the six seed clusters (df=5), alongside each fold result.

## Measurements

- **Primary:** mean task reward by probe budget, `accuracy - 0.05 × mean_probes`.
- **Secondary:** accuracy, mean probes, rule-identification rate, and per-composition accuracy.
- **Comparisons:** learned policy minus random; RIMs minus parameter-matched GRU; each learned policy minus information gain.
- Record every seed/fold result. Keep accuracy and query cost visible so reward gains cannot hide extra probing.

## Interpretation limit

This tests transfer among 18 explicitly enumerated rules built from familiar components. The candidate catalog and exact posterior update still include the test rules. Success is evidence for recombining known operations in this small simulator; it is not evidence that the agent invents rules outside the catalog, understands natural language, or explores real-world tasks.

## Run

Pilot:

```cmd
python -m cognitive_lab.active.exp007.train --fold 0 --seed 42 --device cuda
```

Full set after reviewing the pilot:

```cmd
for /L %F in (0,1,2) do @for %S in (42,43,44,45,46,47) do @python -m cognitive_lab.active.exp007.train --fold %F --seed %S --device cuda
```

Results and checkpoints use `experiment-007` filenames and do not replace earlier experiments.
