# REPORT.md — LibriBrain 2026 Broad track (DRAFT, living document; frozen Oct 13)

All numbers trace to `runs/*.json`, produced by committed scripts. Predictions and outcomes are in PREDICTIONS.md; the run table is in RUNS.md.

## Validation design
- **Split (`src/split.py`):** disjoint in subject AND stimulus.
  - Eval subjects 3/6/9/12; selection (VAL) subjects 1/5/10, all from the 100% group.
  - Block B = the last 25% of the shared Sherlock1 ses-11/12 stimulus clock. It is excluded from training for everyone, subj0 included, with a 1 s guard.
  - Few-shot proxies: a subject's own words before the 25/50/75% points (≈10 / 20 / 30 min).
- **Composite:** holdout-row-weighted, (12·s100 + 10·s50 + 10·s25 + 7·s0)/39.
- **Leakage audit (`results/leakage_audit.md`): all PASS.**
  - subject axis;
  - stimulus axis (≥1.43 s gap);
  - text axis (0/1643 block-B 8-grams in training text);
  - no holdout timing/shuffle metadata in code.
- **Local floor:** prior ranking scores **0.263**, not 0.20, because ~38 of 50 words occur in block B. Seed spread is ±0.015–0.02, so only differences ≳0.03 are resolvable.
- **Known gap to holdout:** the holdout has 3,587 isolated-word windows with no local analogue. Local numbers are a sentence-context proxy.

## What moved the held-out-subject metric (as of 2026-10-04)
| Structure | Composite | Zero-shot | Note |
|---|---|---|---|
| B0 subj0 decoder, unaligned | 0.211 | 0.211 | subj0 itself 0.498 on a new stimulus; raw sensor transfer is nil |
| P1b ridge sensor map onto subj0 (no labels) | ≤0.262 | 0.233 | reaches the floor at best |
| C1 within-subject linear, own data only | — | — | s100 0.288 (positive control) |
| P1a per-subject A, frozen B0, CE + KD to subj0 paired window | 0.289 | 0.266 | first above-floor transfer |
| B1 multi-subject trunk, selected on subj0 | 0.237 | 0.220 | selection misaligned; the trunk specialises after ~1 epoch |
| B2a multi-subject trunk, selected on VAL subjects | **0.310** | **0.311** | VAL picks epoch 0 |
| B2b subject-agnostic trunk | 0.301 | 0.282 | per-subject layers in training help zero-shot |
| B4 + cross-subject InfoNCE (β=0.5) | 0.299 | 0.295 | not detected at β=0.5 |
| E1 3-seed ensemble, 1-epoch trunks | — | 0.293 | +0.015 over the mean single |

## Negative / instrument results kept in
- Sensor-space transfer without alignment: nil (0.211, below the floor).
- Unsupervised linear alignment to subj0 single trials: reaches the floor only (map over α in RUNS.md).
- Few-shot alignment on top of a good multi-subject trunk adds ≈0. The zero-shot trunk already beats a subject's own 30-min linear decoder.
- Contrastive subject-invariance at β=0.5: no gain.

## Local-vs-leaderboard gap
- 2026-10-08: b0 LB 0.203 (local 0.211); final_e1_ens3 LB 0.225 (local ensemble 0.293). Gap ≈ −0.07; the holdout predictions show top-1 collapse (61% one word), so a holdout distribution shift or a pipeline bug is suspected.
