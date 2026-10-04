# RUNS.md — every run, local held-out-subject score, LB score if submitted

Local metric = BAcc@10 on block B (src/split.py), mean over eval subjects (3, 6, 9, 12). The prior-ranking floor on block B is 0.263, not 0.20, because only ~38/50 words occur in block B. Prediction references point to PREDICTIONS.md.

| date | tag | config | seed | subj0 block B | eval-subj block B (zero-shot) | few-shot s25/s50/s100 | composite | LB | notes |
|---|---|---|---|---|---|---|---|---|---|
| 10-04 | b0_smoke | WordCNN 1% data, 1 ep | 0 | 0.278 | 0.252 | = zero-shot (no FT) | — | — | smoke only; prior floor 0.263 |
| 10-04 | b0 | WordCNN d256, 15 ep OneCycle lr1e-3, bs256, LS0.1, per-window norm, 50 Hz; subj0 train mask only | 0 | **0.498** | 0.211 (below 0.263 floor) | = zero-shot (no FT) | 0.211 | not yet (Kaggle not joined) | ep6 selected on es (subj0 S1 ses-10) 0.500, tau 0.75 |
| 10-04 | p1b_ridge | ridge sensor map X_s→X_0 on token pairs, frozen b0; α map 10/1/0.1/0.01/0.001 | — | — | s0 meanA 0.23 (best) | 0.265/0.274/0.266 (α=0.01) | 0.262 (best) | — | CPU; ≈ floor; NOT DETECTED with this instrument |
| 10-04 | p1a | per-subject 306×306 A (CE + KD to subj0 paired window, λ=1e-3), frozen b0 | 0 | — | meanA 0.266 (identity 0.211) | 0.274 / 0.297 / 0.307 | **0.289** | — | first above-floor cross-subject signal |
