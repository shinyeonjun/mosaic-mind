# Experiment 008: imitate an information-gain teacher

## Question

Does replacing reward-only REINFORCE with supervised teacher imitation improve held-out-composition query policies, or does the remaining gap persist because of the controller/state representation?

## Hypotheses

- **H1:** imitation raises held-out task reward versus the same controller trained by REINFORCE in Experiment 007.
- **H2 (mechanism):** imitation improves exact teacher-action agreement and information-gain-optimal query rate.
- **Interpretation:** if query agreement rises but reward does not, entropy reduction is an imperfect proxy for the task reward. If neither rises, the controller or state representation may limit transfer; that result alone does not identify which one.

## Controlled comparison

- Reuse all Experiment 007 folds, seeds, architectures, train/validation/test sizes, optimizer, learning rate, probe cost, and checkpoint-selection metric.
- For each training batch, sample the same balanced training-rule schedule and random budgets as Experiment 007. Generate expert trajectories by repeatedly selecting the exact maximum-information-gain probe and applying its deterministic outcome.
- Accumulate cross-entropy over all query states in each episode batch, then take one optimizer step, matching Experiment 007's updates per epoch.
- Train on cross-entropy labels for those probes. Generate teacher trajectories only from rules in that fold's training split; validation and test rules never appear as hidden teacher episodes.
- Keep the same fixed 18-rule catalog and exact posterior update used in Experiment 007. The test cases are regenerated with the same fold/seed and checked against Experiment 007's saved information-gain metrics before training starts.
- Compare each imitation policy with its same-architecture REINFORCE policy and with the information-gain reference on paired test episodes.

## Measurements

- **Primary:** task-reward change at probe budget 3, pooled across each seed's three held-out folds. Report paired two-sided 95% t intervals across six seed clusters (df=5).
- **Secondary:** accuracy, mean probes, task reward at budgets 1 and 2, exact teacher-action agreement, information-gain-optimal rate, and per-composition accuracy.
- A teacher-action match uses the teacher's deterministic tie break. The information-gain-optimal rate treats equal-information probes as equally good.

## Limits

The teacher optimizes expected reduction in uncertainty over the fixed rule catalog, not task reward directly. Improvement would show that this teacher signal transfers better than reward-only REINFORCE in this simulator; it would not establish open-world rule invention or general reasoning.

## Run from the project root in an activated venv

Pilot:

```cmd
python -m cognitive_lab.active.exp008.train --fold 0 --seed 42 --device cuda
```

After reviewing the pilot, run the remaining 17 jobs and aggregate:

```cmd
for %S in (43 44 45 46 47) do @python -m cognitive_lab.active.exp008.train --fold 0 --seed %S --device cuda
for /L %F in (1,1,2) do @for %S in (42 43 44 45 46 47) do @python -m cognitive_lab.active.exp008.train --fold %F --seed %S --device cuda
python -m cognitive_lab.active.exp008.aggregate
```

Results and checkpoints use `experiment-008` filenames and leave earlier experiments intact.
