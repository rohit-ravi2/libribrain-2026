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

## PENDING (updated 2026-10-14 15:40 UTC)
- **Deadline:** Kaggle close **2026-10-16 12:00 UTC**; new-submission hard stop ~10:00 UTC Oct 16.
- **Best LB:** Deep **0.70767** `submissions/drift1_deep_bn_balT3.csv` (= `bnD_deep_a1.0_balT3.csv`; FINAL-6c ×10 + full per-subject AdaBN + Sinkhorn T3). Broad **0.69213** `submissions/final6_broad_ens10_balT2.csv`. **Rohit must SELECT on Kaggle.**
- **GPU:** held by sister meta-emg 'E1 full' (30 ep, from ~14:40 UTC, ETA ~18:50 UTC). Urgent notify sent asking Rohit whether to pause it; no reply, so lock respected. Evaluations run on CPU (`LB_DEV=cpu`).
- **In flight (CPU):** BN-α Broad LB arm: `submissions/bna05_broad_ens3*.csv` (ens3 + partial AdaBN α 0.5, dropout off) builds on CPU, self-submits; log `logs/bna05_broad.log`; compare against ens3_balT2 0.69134.
- **Queued (GPU, `scripts/queue_gpu7.sh`, log `logs/queue_gpu7.log`):** INF-1b (BN α 0.5 + labelled fitA for SEEN) → SPB-1 train (`--pure_runs 1`) → BN-α maps on spb1 and deep_c (Q3+Q4) → DW-1.
- **Findings today:** Deep discount = BN statistics on the new session (DRIFT-1 +0.044); full recal hurts new Broad brains on LB (−0.011) but partial α 0.5 gains +0.028 locally (6/7); Deep prefers α 1 (α 0.5 −0.029 LB). ABS-1 linear encoding-model fusion not detected.
- **If BN-α Broad LB ≥ +0.01:** rebuild with all 10 models at α 0.5 (+ fitA for subjects 1–32 if INF-1b passes) on GPU, submit.
