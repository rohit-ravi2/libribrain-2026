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

## PENDING (updated 2026-10-12 21:10 EDT)
- **GPU queue (in lock order):** deep1b (deep_f running) → deep1c (deep_g, deep_h) and final6 (both wait; order may flip). final6 = subj0-only recipe ×3 seeds → Broad raw + Broad rerank + Deep files, self-submits, records in RUNS.md, urgent notify.
- **Best LB so far:** Broad 0.4403 (submissions/v3_final2_lmvocab_b1.csv); Deep 0.5662 (submissions/deep_v3_final2_ens5.csv). FINAL-6 predicted Broad 0.55–0.60, Deep 0.62–0.68.
- **Bet C:** LPP download (rclone) → watcher launches scripts/extmeg/preprocess_lpp_listen.py overnight (logs/extmeg_preprocess_lpp_listen.log). Next step after that: loader + third training stream (docs/extmeg/PLAN.md §5). Given deep_c, the extmeg bet needs rethinking: Broad subjects' short recordings HURT transfer, so 58 LPP subjects × 1.6 h may too; the test is the 0/20/58-subject curve.
- Rohit actions: select final files on both Kaggle pages before Oct 16 12:00 UTC (wait for FINAL-6 scores); Donders DUA for Armeni/MOUS if Bet C continues.
