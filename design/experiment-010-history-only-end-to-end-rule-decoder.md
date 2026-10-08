# Experiment 010: decode and apply rules from history only

## Question

Can a recurrent agent infer the semantic parts of a hidden rule from probe history, stop without an oracle posterior, and apply its inferred rule to a held-out composed task?

## Hypotheses

- **H1:** A factorized decoder for scope and reducer transfers to held-out rule compositions and approaches the history-only query policy with its exact-posterior answer decoder.
- **H0:** Removing exact-posterior answer decoding and oracle termination causes a clear loss, even when query choices use history only.

## Controlled comparison

- Keep Experiment 009's folds, six seeds, data sizes, architectures, optimizer, learning rate, budgets, and held-out task cases.
- Reuse Experiment 009's history-only controller, exact-posterior answer-decoder, and information-gain references.
- Jointly train a query/STOP head and two rule-decoding heads from the recurrent state: one predicts scope (3 values), the other predicts reducer (6 values).
- Train only on hidden rules from the training fold. The teacher supplies information-gain queries and STOP labels when its exact training posterior is a singleton; scope and reducer labels supervise every observed history.
- At evaluation, pass only prior probe identities, outcomes, and budgets to the agent. The agent may STOP or query until budget is exhausted. It composes its predicted scope and reducer, then applies that rule to the held-out task case.
- Exact posterior state remains inside the simulator for generating feedback and metrics. It never controls agent stopping or answer decoding.

## Measurements

- **Primary:** paired budget-3 task-reward difference versus the matched Experiment 009 history-only query policy with exact-posterior answer decoding.
- **Mechanism:** exact composed-rule accuracy, scope accuracy, reducer accuracy, teacher-action agreement, STOP rate, and information-gain-optimal query rate.
- **Secondary:** answer accuracy, mean probes, and held-out-composition accuracy at budgets 1 and 2.
- Pool the three held-out folds within each seed and compute two-sided 95% t intervals across six seed clusters (df=5). Treat these as exploratory.

## Limits

The agent has no exact posterior input, oracle stop condition, or posterior-based answer decoder. However, the teacher and simulator retain the fixed 18-rule grammar, and the decoder knows all scope and reducer primitives. This tests transfer to unseen combinations of known primitives, not open-world discovery or invention of new primitives.

## Run from the project root in an activated venv

Pilot:

```cmd
python -m cognitive_lab.active.exp010.train --fold 0 --seed 42 --device cuda
```

After reviewing the pilot, run the remaining 17 jobs and aggregate:

```cmd
for %S in (43 44 45 46 47) do @python -m cognitive_lab.active.exp010.train --fold 0 --seed %S --device cuda
for /L %F in (1,1,2) do @for %S in (42 43 44 45 46 47) do @python -m cognitive_lab.active.exp010.train --fold %F --seed %S --device cuda
python -m cognitive_lab.active.exp010.aggregate
```

Results and checkpoints use `experiment-010` filenames.
