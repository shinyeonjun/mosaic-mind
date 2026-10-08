#!/usr/bin/env bash
# Stage 3 recipe (design/world-v3-thinking.md): grow the thinker in two stages.
#   1) links + propagation (all links on at start, pruned by the link cost)
#   2) links frozen; every door's answer is trained too; stopping is the fixed-point rule
# usage: scripts/world3_grow.sh SEED [--test]
set -e
seed=$1; shift
py=venv/Scripts/python.exe
$py -m cognitive_lab.world3.thinker --seed "$seed" --message state --all-doors 0 --epochs 10 --tag=-links
$py -m cognitive_lab.world3.thinker --seed "$seed" --message state --all-doors 1.0 --epochs 30 \
    --init "world-v3_thinker-links_seed-$seed.pt" --freeze-links --tag=-grown "$@"
