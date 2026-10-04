#!/bin/bash
# Exact MEG-XL Broad baseline reproduction on NYU Torch HPC (needs >=80 GB GPU; README requirement).
# PREPARED, NOT RUN. Rohit's decision (RUN_STATE.md open decision 1). Site specifics marked CHECK.
#SBATCH --job-name=megxl-broad
#SBATCH --gres=gpu:1                 # CHECK: Torch H100/H200 gres name, e.g. gpu:h100:1
#SBATCH --constraint=h100|h200       # CHECK
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=%x-%j.out
set -euo pipefail
WORK=${SCRATCH:-$HOME}/libribrain-2026          # CHECK: Torch scratch path
cd $WORK
# One-time setup (run interactively first):
#   git clone https://github.com/neural-processing-lab/MEG-XL && cd MEG-XL
#   python3.12 -m venv ~/venvs/megxl && ~/venvs/megxl/bin/pip install -r requirements.txt
#   ~/venvs/megxl/bin/huggingface-cli download pnpl/MEG-XL meg-xl-med-v2.ckpt --local-dir ckpt
#   rsync the LibriBrain100 Sherlock1 sub-1..32 h5/tsv (≈20 GB) from the workstation:
#     rsync -av rohit@100.83.95.39:/mnt/ssd4tb/data/LibriBrain100/Sherlock1/ data/LibriBrain100/Sherlock1/
source ~/venvs/megxl/bin/activate
cd MEG-XL
python -m brainstorm.evaluate_criss_cross_word_classification \
    --config-name=eval_criss_cross_word_classification_libribrain100_multisub_train \
    model.criss_cross_checkpoint=$WORK/ckpt/meg-xl-med-v2.ckpt   # CHECK: data root override key in the config
