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

## PENDING (updated 2026-10-14 14:30 UTC)
- **Deadline:** Kaggle close **2026-10-16 12:00 UTC**; new-submission hard stop ~10:00 UTC Oct 16.
- **Best LB:** Deep **0.70767** `submissions/drift1_deep_bn_balT3.csv` (FINAL-6c ×10 + per-subject BN recalibration on own holdout rows + Sinkhorn T3). Broad **0.69213** `submissions/final6_broad_ens10_balT2.csv` (BN-1 Broad version pending). **Rohit must SELECT on Kaggle.**
- **In flight:** BN-1 Broad predict (`logs/bn1_broad.log`, holds the GPU lock under its bash pid) → chain `scripts/queue_resume5.sh` (log `logs/queue_resume5.log`): submit BN-1 → ABS-1 stage 2 (`logs/abs_eval.log`) → INF-1b slim on deep_c (`logs/infer_map_b.log`) → SPB-1 train `--pure_runs 1` (`logs/spb1.log`) + AdaBN eval (`logs/infer_map_spb1.log`) → INF-2 self-training/TENT map (`logs/tta_map.log`) → DW-1 width (`logs/dw1.log`). All pre-registered in PREDICTIONS.md.
- **Decision rules:** SPB-1 ≥ +0.02 (AdaBN numbers) → retrain 10 finals with `--pure_runs 1`, submit with `--bnrecal 1`. INF-2 ≥ +0.015 → add to predictor. ABS-1 fused ≥ +0.02 → holdout implementation.
- INF-1 full run was stopped part-way (partial rows in `logs/infer_map_partial.log`) to give BN-1 the GPU.
