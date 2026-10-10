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

## PENDING (updated 2026-10-10 13:15 EDT)
- **queue_final3** RUNNING: final_v3d1 seeds 4,5,6 (all data, 8 ep) → submissions/v3_final3_ens8.csv → submit (FINAL-3, pred LB 0.43–0.445).
- DONE 10-10: clean ladder v3a→v3d1 (local deploy 0.463/0.455, ens 0.475); V3-E map (no lever); adaptation (not detected); LB FINAL-1 0.4108, FINAL-2 **0.4284**. Kaggle: 6 lifetime submissions; 8 remaining today.
- Decisions for Rohit: (1) is within-sentence word TIMING admissible? (default NO; tainted runs v3c/v3d at 0.33/0.51 local kept only as a record); (2) Deep-track calibration submission (never okayed); (3) organiser email (unsent; moot now that TTA/sentence context are ruled in-rules).
- Freeze Oct 13; Kaggle close Oct 16 12:00 UTC. Code review requires the submitted code: scripts/train_v3.py, predict_holdout_v3.py, src/cont.py, src/v3lib.py, src/models.py (+ cache builders).
