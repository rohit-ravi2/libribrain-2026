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

## PENDING (updated 2026-10-12 ~13:30 EDT) — nothing running; GPU lock free
- **Best LB file: submissions/v3_final2_lmrerank_b1.csv = 0.4344** (FINAL-2 five-model ensemble + within-sentence trigram rerank). **Rohit must select it as the final submission on the Kaggle page before Oct 16 12:00 UTC** (second slot: v3_final2_ens5.csv, 0.4284).
- Kaggle: 8 lifetime submissions; Deep track entered, no submissions.
- Tests closed 10-12: LM rerank +0.006 (kept); MV1 acoustic targets +0.005 (not detected); FS100 −0.052 (worse). FINAL-4 queue (scripts/queue_final4.sh) prepared but NOT launched — would only matter if MV1 were detected.
- Open for Rohit: timing-feature admissibility (default NO, unused); Deep-track calibration file (unused); code/checkpoints for the organisers' review are at HEAD (scripts/train_v3.py, predict_holdout_v3.py, lm_rerank_holdout.py, src/).
