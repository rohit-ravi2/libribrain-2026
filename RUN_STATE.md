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

## PENDING (updated 2026-10-14 14:20 UTC, session resume after the usage cap + reboot)
- **Deadline:** Kaggle close **2026-10-16 12:00 UTC**; new-submission hard stop ~10:00 UTC Oct 16.
- **Best LB (fallbacks, Rohit to SELECT):** Broad **0.69213** `submissions/final6_broad_ens10_balT2.csv` (alt ens3_balT2 0.69134); Deep **0.66390** `submissions/final6_deep_ens10_balT3.csv` (ens5_balT3 ties; files differ, 92% top-1 agreement).
- **Nothing running** (machine rebooted; GPU lock free). FINAL-6c, VR-1, ext20 all recorded. ext58 never ran (post-deadline, Rohit's call).
- **Next (approved Oct 13 plan, untested items):** (c) inference-only: frozen-trunk adaptation at 0.66, Deep TTA / normalisation recalibration, small-β prior after balance; (a) analysis-by-synthesis gate (generative word-response model on subj0, fused with decoder, local Q3/Q4); Deep encoder-width map (subj0-only). Each pre-registered below in PREDICTIONS.md.
