# Experiment 009: choose queries from history only

## Question

Can the recurrent query policy imitate useful information-gain decisions on held-out rule compositions when it no longer receives the exact posterior mask?

## Hypotheses

- **H1:** history-only policies retain high information-gain-optimal query rates and task reward close to the posterior-input imitators from Experiment 008.
- **H0:** removing the posterior mask causes a clear loss in query quality or task reward, suggesting the recurrent state does not reconstruct enough of the belief state.
- This is a state-input ablation. It does not test open-world rule discovery or a learned final-answer decoder.

## Controlled comparison

- Keep Experiment 008's folds, seeds, architectures, training examples, optimizer, learning rate, budgets, checkpoint selection, and test cases.
- Train with the same exact information-gain labels, generated from training-fold hidden rules only.
- The controller's posterior-mask input is replaced with zeros. It receives only the sequence of previous probe identities, observed outcomes, and remaining/initial budget.
- Compare each history-only policy with the same architecture's posterior-input imitator, REINFORCE policy, and exact information-gain reference.
- Verify the information-gain reference reproduces Experiment 008 before training each run.

## Measurements

- **Primary:** paired budget-3 task-reward difference versus the posterior-input imitator, using six seed clusters (pool the three folds within each seed; two-sided 95% t interval, df=5).
- **Mechanism:** exact teacher-action agreement and information-gain-optimal rate by budget.
- **Secondary:** accuracy, mean probes, task reward at budgets 1 and 2, and per-composition accuracy.

## Limits

The simulator still maintains exact posterior state for filtering and for ending a round when one rule remains. The final answer decoder also uses that posterior. The teacher and evaluation world retain the fixed 18-rule catalog. Thus this tests history-based query selection, not a fully independent reasoner or open-world rule discovery.

## Run from the project root in an activated venv

Pilot:

```cmd
python -m cognitive_lab.active.exp009.train --fold 0 --seed 42 --device cuda
```

After reviewing the pilot, run the remaining 17 jobs and aggregate:

```cmd
for %S in (43 44 45 46 47) do @python -m cognitive_lab.active.exp009.train --fold 0 --seed %S --device cuda
for /L %F in (1,1,2) do @for %S in (42 43 44 45 46 47) do @python -m cognitive_lab.active.exp009.train --fold %F --seed %S --device cuda
python -m cognitive_lab.active.exp009.aggregate
```

Results and checkpoints use `experiment-009` filenames.
