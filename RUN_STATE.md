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

## PENDING (updated 2026-10-14 17:58 UTC)
- **Deadline:** Kaggle close **2026-10-16 12:00 UTC**; new-submission hard stop ~10:00 UTC Oct 16.
- **Best LB (Rohit must SELECT):** Broad **0.70958** `submissions/slp2_broad_k20.csv` (FINAL-6c ens10 → self-learned within-subject HMM prior κ 20 → Sinkhorn T2). Deep **0.71611** `submissions/slp3_deep_bn_k20.csv` (ens10 + per-subject AdaBN → HMM κ 20 → Sinkhorn T3). Recipes: `predict_holdout_v3.py [--bnrecal 1]` → `hmm_prior.py --kappa 20 [--track deep --T 3]`.
- **Submissions today:** Broad 10/10 used, Deep 9/10 used. They reset at 00:00 UTC.
- **GPU:** sister meta-emg 'E1 full' until ~20:25 UTC. Then `scripts/queue_gpu9.sh` (log `logs/queue_gpu9.log`): SLP-ST Deep/Broad files → T5-3B targets → TX-1 → LSO-1 (+PX α map) → INF-1b → SPB-LSO (+PX α map) → DW-1.
- **Ready for tomorrow's slots:** `submissions/slpt_broad_k3_{50,10}.csv` (trigram SLP), SLP-ST files when built.
- **Closed today (LB):** BN recal on Broad (α1 −0.011, α0.5 −0.002); DIV-1 Deep diversity (−0.008); SLP end/bal_em/vocab-base options (all worse); ABS-1 linear encoding model (local, not detected).
- **Next CPU idea:** trigram-backoff SLP (states = word pairs, κ3 backoff to the κ20 bigram), LB-tested tomorrow.
