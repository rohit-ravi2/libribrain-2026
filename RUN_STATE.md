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

## STATE (updated 2026-10-20 8:55 PM ET; leakage paper Figure 1 added) — competition closed, post-competition phase
- Kaggle closed 2026-10-16 8:00 AM ET. Final public: **Broad 0.72230** (`zsens_fita3_k20_iso0_T1.75.csv`, 8th public), **Deep 0.72929** (`final7_deep_bn_isoB0bn_k20_isog085.csv`). Private scores not yet released (code review). All 35 Broad / 26 Deep submissions in `results/lb_history/`.
- Unsubmitted artefacts: `submissions/zsens12_broad_raw.csv` (12-donor zero-shot ensemble, finished 10-16 7:25 AM, never scored); `docs/home_stretch/files/zsens_fita3_k20_flatconf0.3_iso0_T1.75.csv` (confident-row flattening, sim +0.0036, no slot left).
- Post-competition plan (Rohit, 10-16): public repo (README/license/repro done, not pushed) -> blog (`docs/writeups/blog_post.md`, draft, two [ROHIT] gaps) -> website page -> leakage/evaluation paper -> main ML paper -> philosophy paper -> neuroscience note -> JOSS toolkit -> Kaggle write-ups for all competitions.
- 2026-10-21: repo PUBLIC at github.com/rohit-ravi2/libribrain-2026 (history as-is; no vendor names in current files; public commits carry no trailer). Website page LIVE: rohitravi.com/projects/libribrain-2026 (personalwebsite b7c2d1d, Vercel). PNPL contact: not yet (Rohit).
- 2026-10-21: blog series LIVE: rohitravi.com/posts/libribrain/ (A "The margin above 0.63", B "Arguing with the stopping points"; site 7b57f1c). Next: leakage paper seeded CIs (GPU, needs his go).
