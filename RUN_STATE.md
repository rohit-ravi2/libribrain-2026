# RUN_STATE.md — live state (update on every launch and completion)

## Dates (verified 2026-10-04 via `kaggle competitions list`: deadline 2026-10-16 12:00 UTC = Oct 15 AoE)
| Date | Event | Days left from 2026-10-04 |
|---|---|---|
| Oct 13 | internal code freeze | 9 |
| Oct 14 | final submission (charter) | 10 |
| Oct 16 12:00 UTC | Kaggle close / write-up due (Oct 15 AoE) | 11 |

## Kaggle status (2026-10-04)
- `userHasEntered: False` → **Rohit must click Join Competition / accept rules on kaggle.com for the Broad track** before any submission. ListSubmissions returns 403 until he does.
- Submissions: 0 lifetime, 10/day remaining.
- Public LB: 34 teams, top 0.729 (contaminated, see RULES.md §4). Snapshot from 2026-10-04 in `ref/lb_broad_20261004/`.

## Open decisions (Rohit's)
0. **GPU priority (2026-10-04 19:05 EDT, urgent notify sent):** the meta-emg queue holds ~/projects/.gpu_lock since 16:07, with ~5 h+ projected left. LibriBrain's queue_next waits behind it. Options: give LibriBrain priority, or HPC (scripts/hpc/).
1. Exact MEG-XL reproduction on NYU Torch HPC vs the approximate path on the 4060 Ti → proceeding approximate; HPC job script to be prepared at `scripts/hpc/`.
2. Whether to email the organisers about TTA → draft at `drafts/email_tta.md`, NOT sent.
3. Join the Kaggle Broad competition (identity-bound click).

## PENDING (updated 2026-10-13 06:40 EDT) — GPU queue empty
- **Best LB:** Broad **0.6555** = `submissions/final6_broad_ens3.csv` (3 seeds); Deep **0.6535** = `submissions/final6_deep_ens5.csv` (5 seeds). Recipe: subj0-only training, 3 s sentence-epoch windows, 4-layer sentence transformer (no timing), T5 SigLIP + CE, ce readout τ0.5, no rerank, identity input layers for Broad subjects.
- **Rohit must SELECT these two files as final on the Kaggle pages before Oct 16 12:00 UTC.** Kaggle auto-picks best public otherwise, but select explicitly.
- Maps closed (all local, split v2 / subj0 blocks): broad_frac 0.5/0.2/0.1/0 → 0.452/0.528/0.594/0.632; ctx 2/4/6; epochs 8/12/24; 5 vs 3 seeds (Broad noise, Deep +0.03). LM rerank: helps weak models only (retired).
- Bet C (extmeg): LPP preprocessing status in logs/extmeg_preprocess_lpp_listen.log; next step = loader + third stream (docs/extmeg/PLAN.md §5). Given that Broad-subject data hurt transfer, this is a hypothesis test for after the deadline.
- Code review package (if asked): scripts/train_v3.py, predict_holdout_v3.py, src/{cont,models,v3lib,common}.py, scripts/build_cont.py, scripts/text_targets.py; checkpoints runs/final6_s{0..4}_ep7.pt.
