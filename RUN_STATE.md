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

## PENDING (updated 2026-10-15 3:30 PM ET)
- **Deadline:** Kaggle closes 2026-10-16 8:00 AM ET (12:00 UTC). Suggested last submission ~6:00 AM ET.
- **Public bests (auto-selected if nothing is chosen):** Broad 0.70958 `slp2_broad_k20.csv`; Deep 0.71611 `slp3_deep_bn_k20.csv`.
- **Held finals (NOT submitted; Rohit's gate = local > 0.8 on both, unmet on Broad):** Broad `fita1_broad_ens5_k20.csv` (labelled per-subject A + SLP + Sinkhorn); Deep `final7_deep_bn_k20.csv` / `final7_deep_k20.csv` (width × 5 seeds ± AdaBN + SLP + Sinkhorn).
- Nothing running. Screens closed today: see PREDICTIONS.md (FITA-2 promoted; width promoted for Deep; T5-3B, SPB, DTW, Riemannian, state gating, FIR taps, MEG-XL, neural fields, ENS-T, ANAT priors not promoted).
