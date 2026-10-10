# RUNS.md — every run, local held-out-subject score, LB score if submitted

Local metric = BAcc@10 on block B (src/split.py), mean over eval subjects (3, 6, 9, 12). The prior-ranking floor on block B is 0.263, not 0.20, because only ~38/50 words occur in block B. Prediction references point to PREDICTIONS.md.

| date | tag | config | seed | subj0 block B | eval-subj block B (zero-shot) | few-shot s25/s50/s100 | composite | LB | notes |
|---|---|---|---|---|---|---|---|---|---|
| 10-04 | b0_smoke | WordCNN 1% data, 1 ep | 0 | 0.278 | 0.252 | = zero-shot (no FT) | — | — | smoke only; prior floor 0.263 |
| 10-04 | b0 | WordCNN d256, 15 ep OneCycle lr1e-3, bs256, LS0.1, per-window norm, 50 Hz; subj0 train mask only | 0 | **0.498** | 0.211 (below 0.263 floor) | = zero-shot (no FT) | 0.211 | not yet (Kaggle not joined) | ep6 selected on es (subj0 S1 ses-10) 0.500, tau 0.75 |
| 10-04 | p1b_ridge | ridge sensor map X_s→X_0 on token pairs, frozen b0; α map 10/1/0.1/0.01/0.001 | — | — | s0 meanA 0.23 (best) | 0.265/0.274/0.266 (α=0.01) | 0.262 (best) | — | CPU; ≈ floor; NOT DETECTED with this instrument |
| 10-04 | p1a | per-subject 306×306 A (CE + KD to subj0 paired window, λ=1e-3), frozen b0 | 0 | — | meanA 0.266 (identity 0.211) | 0.274 / 0.297 / 0.307 | **0.289** | — | first above-floor cross-subject signal |
| 10-04 | b1 | multi-subject trunk + per-subj A, broad_frac 0.3, aug, 12 ep, select subj0 es | 0 | 0.487 | 0.220 (ep1 peak 0.329, unselected) | 0.225 / 0.234 / 0.261 | 0.237 | — | selection misaligned; smoke (1 ep) composite 0.316 |
| 10-04 | b1_smoke | as b1, 1 epoch | 0 | 0.378 | 0.305 | 0.313 / 0.320 / 0.322 | 0.316 | — | best composite so far; seen before B1 pre-registration |
| 10-04 | b2a | b1 + select VAL(1,5,10) zero-shot, broad_frac 0.5, 12 ep | 0 | 0.376 | **0.311** | 0.292 / 0.325 / 0.313 | **0.310** | — | VAL picks ep0; few-shot ≈ zero-shot |
| 10-04 | b2b | b2a with A frozen at identity during training (subject-agnostic trunk) | 0 | 0.312 | 0.282 | 0.298 / 0.311 / 0.306 | 0.301 | — | few-shot A helps here (+0.02–0.03) |
| 10-04 | c1 | within-subject logistic (C=1e-3, 10 Hz), own fs data only | — | — | — | 0.258 / 0.282 / 0.288 | — | — | positive control |
| 10-04 | b4 | b2a + InfoNCE β=0.5 on 79.5k all-word token pairs, 6 ep | 0 | 0.310 | 0.295 | 0.287 / 0.305 / 0.306 | 0.299 | — | contrastive not detected at β=0.5 |
| 10-04 | b6 | b2a + init from b0 + KD to subj0 teacher, lr 3e-4, 4 ep | 0 | — | 0.291 | — | 0.303 | — | ran overnight 10-04; not yet reviewed in detail |
| 10-04 | b5a | b2a, d128 p0.5 wd0.3, 4 ep | 0 | — | 0.315 | — | 0.315 | — | best local composite so far (within seed noise of b2a) |
| 10-04 | b5b | b2a, lr 3e-4, 4 ep | 0 | — | 0.294 | — | 0.296 | — | |
| 10-04 | b5c | b2a, broad_frac 0.8, 4 ep | 0 | — | 0.227 | — | 0.260 | — | |
| 10-08 | LB b0 | submissions/b0_broad.csv | — | — | local 0.211 | — | — | **0.203** | first submission |
| 10-08 | LB final_e1_ens3 | 3× final 1-ep trunk, own A 1–32, mean A 33–39 | 0–2 | — | local ens 0.293 | — | — | **0.225** | top-1 collapse: 61% of rows share one word; local-vs-LB gap ≈ −0.07, under investigation |
| 10-08 | LB final_e1_ens3_tau1.0 | as final_e1_ens3, τ=1.0 at inference (collapse removed: top-1 share 0.61 → 0.15) | 0–2 | — | — | — | — | **0.2266** | collapse hypothesis REFUTED (+0.001) |
| 10-08 | lm1_sim | GPT-2 coordinate-ascent sentence decoding, simulation | 0 | 0.450→0.448 | 0.230→0.258 | — | — | — | oracle 0.92/0.91 |
| 10-08 | lm2_sim | exact F–B trigram (Gutenberg), simulation | — | 0.450→0.497 (uni 0.508) | 0.230→0.301 (uni 0.259) | — | — | — | context gain X +0.041, S −0.011 |
| 10-08 | b7_s0/s1 (+b5a_s1 ctrl) | b5a + wav2vec2 speech InfoNCE β=1 | 0,1 | 0.383/0.351 | 0.296/0.293 | — | 0.304/0.313 | — | paired diff −0.001 vs b5a; not detected |
| 10-08 | LB CAL1 | final_e1 ens3, sampler prior, τ=1 | 0–2 | — | — | — | — | **0.2155** | sampler prior over-corrects (−0.011 vs stored τ=1) |
| 10-08 | lm2_sim_v2 | LM2 with random sentence sample (fix) | — | 0.445→0.472 (uni 0.466) | 0.231→0.276 (uni 0.240) | — | — | — | context gain X +0.036, S +0.007 |
| 10-08 | v2ref_s0 | split v2, B5a recipe, 4 ep | 0 | — | eval_zs 0.275 | seen 0.286 | **deploy 0.284** (null 0.234, lift +0.050) | — | V2-REF; pred deploy 0.30 / lift +0.05: lift HIT, level −0.015 |
| 10-08 | v2ref_s1 | same, seed 1 | 1 | — | eval_zs 0.266 | seen 0.293 | **deploy 0.288** (null 0.234, lift +0.054) | — | |
| 10-10 | v3a | continuous FIR cache, CtxEncoder, 1 s window, no ctx, 4 ep × 1500 steps | 0 | — | eval_zs 0.267 | seen 0.274 | deploy 0.272 (null 0.232, lift +0.041); 1s 0.284 | — | parity with V2-REF |
| 10-10 | v3b | as v3a, 3 s masked sentence-epoch window | 0 | — | eval_zs 0.292 | seen 0.292 | **deploy 0.292** (null 0.220, lift +0.073) | — | +0.03 lift from context window |
