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

## PENDING (updated 2026-10-10 08:10 EDT — BUILD arc, target 0.40 local)
- **build_cont** (CPU, 6 proc): scripts/build_cont.py → /mnt/ssd4tb/data/libribrain_cache/cont/<base>.npy (306×T, 50 Hz fp16, FIR anti-alias) + <base>_ev.npz (all words). Log logs/build_cont.log
- **text_targets** (GPU): fixed tokenizer crash; scripts/text_targets.py cuda → runs_t5/<base>.npy. Log logs/text_targets.log
- **queue_v3** RUNNING (GPU lock, started 10-10 ~08:55 EDT): v3a → v3b → v3c → v3d, 4 ep × 1500 steps each (~15 min/run). Logs logs/v3?.log, results runs/v3?.json
- build_cont DONE (226 runs, 11 GB), text_targets DONE (226 runs, 3.9 GB, 53k unique sentences).
- DONE 10-08 after session OOM: v2smoke 0.298, v2ref_s0 0.284, v2ref_s1 0.288 (deploy; null 0.234). Not yet in RUNS.md → recorded below.
