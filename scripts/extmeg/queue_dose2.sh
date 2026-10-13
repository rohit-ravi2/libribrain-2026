#!/bin/bash
# EXT-1 reduced: ext20 -> ext58 (control = deep_c, already on disk). Runs behind vr1 via gpu_run.
cd ~/projects/libribrain-2026; PY=~/venvs/libribrain/bin/python
COMMON="--win 3s --ctx 1 --rel 0 --t5 1.0 --epochs 8 --seed 0 --deep 1 --broad_frac 0.0 --ctx_layers 4 --ext_frac 0.2"
$PY scripts/train_v3_ext.py --tag ext20 $COMMON --ext_subjects 20 > logs/ext20.log 2>&1
$PY scripts/train_v3_ext.py --tag ext58 $COMMON --ext_subjects 58 > logs/ext58.log 2>&1
$PY scripts/broad_protocol.py deep_c ext20 ext58 | tee logs/extmeg_dose_protocol.txt
~/bin/notify "EXT-1 dose (vs deep_c): $(cut -c1-120 logs/extmeg_dose_protocol.txt | tr '\n' ' ')" "LibriBrain extmeg"
