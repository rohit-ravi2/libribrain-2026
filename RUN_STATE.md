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

## PENDING (updated 2026-10-13 15:30 EDT)
- **Best LB:** Broad **0.6913** (`submissions/final6_broad_ens3_balT2.csv`), Deep **0.6639** (`submissions/final6_deep_ens5_balT3.csv`). Both = subj0-only recipe + per-subject Sinkhorn class balance (T 2–3). **Rohit must select these on Kaggle before Oct 16 12:00 UTC.**
- **GPU queue:** Bet C dose-response (ext0 control → ext20 → ext58, scripts/extmeg/queue_dose.sh, ~75–90 min each under load) → vr1 (vocab-run oversampling test, scripts/queue_vr1.sh). Results: logs/ext*.log, logs/vr1.log, logs/vr1_diag.log; `scripts/broad_protocol.py ext0 ext20 ext58 vr1`.
- Today's findings: balance re-rank +0.03/+0.04 local, +0.036/+0.010 LB; prototype pooling NOT DETECTED (all spaces); read-sentence proxy (held-out TIMIT) 0.705 = prose → sentence form is not the holdout's difficulty; vocab-only runs locally 0.57 vs 0.65 → VR-1 tests oversampling them.
- Next decisions: if VR-1 lifts the vocab-run subset ≥ 0.04 with overall intact → retrain finals (both tracks) with --vocab_runs 0.3, balance, submit. If ext20/ext58 ≥ +0.03 → multi-subject pretraining reopens (post-deadline scale-up).
