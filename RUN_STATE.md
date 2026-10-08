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

## PENDING (updated 2026-10-04 ~18:20 EDT)
- **queue_next** QUEUED behind the meta-emg GPU lock (pid 2104347, since 16:07): `scripts/queue_next.sh` =
  1. final_e1 ×3 seeds (all data, 1 ep, τ from local E1) → `submissions/final_e1_ens3.csv` (submission #2)
  2. b6 (B0-init trunk + subj0-teacher KD)
  3. b5a/b5c/b5b regularisation map
  Logs logs/final_e1_s*.log, logs/b6.log, logs/b5*.log.
- **submit_b0** watcher: auto-submits submissions/b0_broad.csv once Rohit joins (log logs/submit_b0.log).
- DONE today: b0, p1a, p1b (CPU), c1 (CPU), b1, b2a, b2b, b4, e1 ×3 + ens. See RUNS.md.
- **download_audio** (started 08:02 10-08): ~11 GB wav → /mnt/ssd4tb/data/LibriBrain100_audio, log logs/download_audio.log
- **b7 queue** (10-08): speech-target B7 s0/s1 + matched b5a_s1; logs logs/b7_s*.log, logs/b5a_s1.log
