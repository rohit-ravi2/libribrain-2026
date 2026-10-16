#!/bin/bash
# CPU prep (cache -> T5 targets -> memmap), then GPU: smoke (1% of an epoch) and the main T5-1 run.
cd ~/projects/libribrain-2026; PY=~/venvs/libribrain/bin/python
while pgrep -f "python scripts/build_cache.py" > /dev/null; do sleep 30; done
NT=8 nice $PY scripts/text_targets.py cpu > logs/text_targets.log 2>&1
nice $PY scripts/make_memmap.py > logs/make_memmap.log 2>&1
~/bin/notify "LibriBrain: T5 targets + memmap ready ($(tail -1 logs/make_memmap.log))" "LibriBrain"
scripts/gpu_run.sh t5 bash -c "$PY scripts/train_t5.py --tag t5smoke --epochs 1 --max_steps 60 > logs/t5smoke.log 2>&1 && $PY scripts/train_t5.py --tag t5_s0 > logs/t5_s0.log 2>&1"
~/bin/notify "LibriBrain T5-1 done: $(grep SUMMARY logs/t5_s0.log | cut -c1-140)" "LibriBrain"
