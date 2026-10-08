@echo off
setlocal enabledelayedexpansion
rem Overnight run: staged growth (read -> judge -> tune together) vs joint training from scratch,
rem rule world v1 composition split, seeds 46-51. Results go to artifacts\results, checks and
rem progress to artifacts\results\overnight_staged_vs_joint.log. About 70 minutes per seed.
rem Usage (from any folder):  D:\AI_great\scripts\overnight_staged_vs_joint.cmd

cd /d D:\AI_great
set PY=venv\Scripts\python.exe
set PYTHONIOENCODING=utf-8
set LOG=artifacts\results\overnight_staged_vs_joint.log
set RUN=%PY% -m cognitive_lab.world.interface_anchored --system anchored-learned --encoder nli --split composition --device cuda
set CHECK=%PY% -m cognitive_lab.world.interface_checks --judge-kind source-fold

echo ==== start %date% %time% >> %LOG%
for %%S in (46 47 48 49 50 51) do (
    echo [seed %%S] stage 1 read !time! >> %LOG%
    %RUN% --seed %%S --epochs 4 >> %LOG% 2>&1

    echo [seed %%S] stage 2 judge !time! >> %LOG%
    %RUN% --seed %%S --judge source-fold --init-encoder artifacts\checkpoints\world-v1_reader-anchored-learned_composition_seed-%%S.pt --freeze-encoder >> %LOG% 2>&1
    %CHECK% --stem anchored-learned-sourcefold-staged-frozen --seed %%S --reader world-v1_reader-anchored-learned_composition_seed-%%S.pt --frozen >> %LOG% 2>&1

    echo [seed %%S] stage 3 tune together !time! >> %LOG%
    %RUN% --seed %%S --judge source-fold --init-encoder artifacts\checkpoints\world-v1_reader-anchored-learned_composition_seed-%%S.pt --init-judge artifacts\checkpoints\world-v1_judge-anchored-learned-sourcefold-staged-frozen_composition_seed-%%S.pt --encoder-lr 1e-5 --judge-lr 3e-4 --epochs 3 >> %LOG% 2>&1
    %CHECK% --stem anchored-learned-sourcefold-staged-tuned --seed %%S --reader world-v1_reader-anchored-learned-sourcefold-staged-tuned_composition_seed-%%S.pt >> %LOG% 2>&1

    echo [seed %%S] control joint from scratch !time! >> %LOG%
    %RUN% --seed %%S --judge source-fold >> %LOG% 2>&1
)
echo ==== done %date% !time! >> %LOG%
echo Finished. See %LOG%
