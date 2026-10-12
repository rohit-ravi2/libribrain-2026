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

## PENDING (updated 2026-10-13 00:15 EDT)
- **Best LB:** Broad **0.6555** (submissions/final6_broad_ens3.csv); Deep **0.6535** (submissions/final6_deep_ens5.csv). Rerank retired (hurts strong models). FINAL-6b 5-seed Broad 0.6486 (noise).
- **GPU queue:** deep1c (deep_g ctx6, deep_h 12 ep) → final6b (seeds 3,4 → 5-model ensembles, self-submits) → deep1d (deep_i 24 ep). Order among waiters may vary.
- Tomorrow: FINAL-7 at the best epochs/depth for both tracks; select final files on Kaggle (Rohit) before Oct 16 12:00 UTC.
- Bet C: LPP preprocessing overnight (watcher). Reassess: Broad-subject data HURT transfer, so adding LPP subjects is now a hypothesis test, not an expected gain.
