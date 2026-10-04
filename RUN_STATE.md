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
1. Exact MEG-XL reproduction on NYU Torch HPC vs the approximate path on the 4060 Ti → proceeding approximate; HPC job script to be prepared at `scripts/hpc/`.
2. Whether to email the organisers about TTA → draft at `drafts/email_tta.md`, NOT sent.
3. Join the Kaggle Broad competition (identity-bound click).

## PENDING
- **b0** DONE 10:48 EDT → runs/b0.pt (subj0-B 0.498, eval-B 0.211).
- **predict_b0** DONE 11:31 → submissions/b0_broad.csv (37,439 rows, valid).
- **day1 GPU queue** RUNNING since 13:24 EDT (lock held): p1a DONE (composite 0.289); b1 smoke DONE; b1 full running → logs/b1.log.
- **b2 queue** QUEUED behind day1: `scripts/queue_b2.sh` = b2a (val-selected, per-subject A) → b2b (subject-agnostic trunk). ~30 min each.
- **c1** (CPU) running: scripts/control_within_linear.py → runs/c1_within_linear.json.
- **submit_b0** watcher (PID 2013402): polls every 10 min; submits submissions/b0_broad.csv once Rohit joins the Broad comp; log logs/submit_b0.log
