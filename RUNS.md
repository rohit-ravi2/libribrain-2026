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
| 10-10 | v3c | v3b + sentence transformer (2 layers, D256) across words of a segment | 0 | — | eval_zs 0.319 | seen 0.330 | **deploy 0.328** (null 0.217, lift +0.111) | — | pred 0.33 HIT |
| 10-10 | v3c0 | v3c with timing input OFF | 0 | — | eval_zs 0.276 | seen 0.297 | deploy 0.293 (null 0.220, lift +0.073) | — | = v3b; transformer not detected |
| 10-10 | v3c, v3d | (timing ON) 0.328 / 0.506 | 0 | — | — | — | TAINTED | — | ablation: timing zeroed → 0.26 / 0.30; superseded by v3c0/v3d0 |
| 10-10 | v3d0 | v3c0 + T5-large L12 targets, duplicate-aware SigLIP, all words, 4 ep | 0 | — | eval_zs 0.339 | seen 0.371 | **deploy 0.365** (null 0.221, lift +0.144) | — | clean; pred 0.33 beaten |
| 10-10 | v3d1_s0 | v3d0 config, 8 ep × 1500 steps, seed 0 | 0 | — | eval_zs 0.456 | seen 0.464 | **deploy 0.463** (null 0.218, lift +0.244) | — | clean; ≥ 0.40 target |
| 10-10 | v3d1_s1 | as v3d1_s0, seed 1 | 1 | — | eval_zs 0.439 | seen 0.459 | **deploy 0.455** (null 0.220, lift +0.235) | — | clean |
| 10-10 | LB FINAL-1 | final_v3d1 ×2 seeds (all data, 8 ep), ce+cos τ0, no TTA | 0,1 | — | — | — | local 0.46 | **0.4108** | +0.18 over previous best |
| 10-10 | ens_v3d1 | 2-seed softmax ensemble of v3d1 | 0,1 | — | — | — | deploy 0.475 (null 0.221) | — | +0.01 over singles |
| 10-10 | v3d1_s0_adapt | seen fine-tune + CORAL/entropy TTA grid | 0 | — | — | — | deploy 0.464 vs base 0.463 | — | adaptation NOT DETECTED |
| 10-10 | v3e1 | v3d1 config, 12 ep | 0 | — | — | — | deploy 0.454 (null 0.215) | — | longer training: not detected (−0.009) |
| 10-10 | v3e2 | v3d1 config, d192 D384 | 0 | — | — | — | deploy 0.427 (null 0.216) | — | capacity: worse (−0.036) |
| 10-10 | v3e3 | v3d1 config, t5 weight 2.0 | 0 | — | eval_zs 0.458 | seen 0.470 | deploy 0.468 (null 0.221) | — | +0.005: within noise |
| 10-10 | v3e4 | v3d1 config, window [−1.5,+2.5] s | 0 | — | eval_zs 0.431 | seen 0.444 | deploy 0.441 (null 0.219) | — | wider window: worse (−0.022) |
| 10-10 | LB FINAL-2 | 5-model ensemble (v3d1 ×4 seeds + t5w2), ce+cos τ0, no TTA | 0-3 | — | — | — | local ens ≈0.48 | **0.4284** | −0.001 vs MEG-XL 0.4296 |
| 10-10 | LB FINAL-3 | 8-model ensemble (v3d1 ×7 seeds + t5w2) | 0-6 | — | — | — | — | 0.42352 | pred 0.43–0.445 |
| 10-12 | note | FINAL-3 0.4235 < FINAL-2 0.4284: 8 models not better than 5 (pred 0.43–0.445 MISS); public rank 29/44 | | | | | | | |
| 10-12 | LB LM-rerank | FINAL-2 probs + within-sentence trigram forward–backward (β=1) | — | — | — | — | — | **0.4344** | +0.006 over FINAL-2 |
| 10-12 | mv1 | v3d1 recipe + wav2vec2 SigLIP head (w=1) | 0 | — | eval_zs 0.468 | seen 0.469 | deploy 0.468 (null 0.220) | — | +0.005 vs v3d1_s0: not detected |
| 10-12 | fs100 | v3d1 recipe on 100 Hz cache (0–50 Hz) | 0 | — | eval_zs 0.422 | seen 0.409 | deploy 0.411 (null 0.217) | — | −0.052: worse |
| 10-12 | DEEP LB DEEP-0 | FINAL-2 five models on subj0 holdout, own A, ce+cos τ0 | 0-3 | — | — | — | — | **0.5662 (Deep)** | first Deep submission |
| 10-12 | LB LM-rerank-2 | FINAL-2 probs + vocab-corpus trigram forward–backward (β=1) | — | — | — | — | — | **0.4403** | new best Broad |
| 10-12 | DEEP LB rerank | DEEP-0 + vocab trigram β=1 | — | — | — | — | — | 0.5194 (Deep) | worse (−0.047); Deep stays rerank-free |
| 10-12 | deep_a | v3d1 recipe, --deep (select on subj0) | 0 | subj0 0.624 | Broad-protocol deploy 0.463 | | | — | |
| 10-12 | deep_b | bf 0.1, ctx_layers 4, --deep | 0 | **subj0 0.711** | **Broad-protocol deploy 0.594** (null 0.21) | | | — | depth/subj0-fraction confounded; ablations + DEEP-1b |
| 10-12 | deep_c | bf 0.0 (subj0 only), ctx 4 | 0 | subj0 0.705 | Broad-protocol see PREDICTIONS | | | — | |
| 10-12 | deep_d | bf 0.5, ctx 4 | 0 | subj0 0.637 | Broad-protocol: see broad_protocol.py | | | — | depth alone (with 50% Broad) is not the lever |
| 10-12 | deep_e | bf 0.1, ctx 6 | 0 | subj0 0.704 | Broad-protocol 0.606 | | | — | depth 6 ≈ 4 |
| 10-12 | deep_f | bf 0.2, ctx 4 | 0 | subj0 0.654 | Broad-protocol 0.528 | | | — | monotone in broad_frac: 0.452/.528/.594/.632 for .5/.2/.1/0 |
| 10-12 | LB FINAL-6 | subj0-only ×3 (bf 0, ctx 4): Broad raw / Broad+rerank / Deep | 0-2 | — | — | — | — | 0.54412 0.65549 0.62462  | |
| 10-13 | deep_g | bf 0, ctx 6 | 0 | subj0 0.720 | Broad-protocol 0.609 | | | — | ≈ deep_c within noise |
| 10-13 | deep_h | bf 0, ctx 4, 12 ep | 0 | subj0 0.721 | Broad-protocol 0.606 | | | — | more training: +subj0, −Broad (both ~noise) |
| 10-12 | LB FINAL-6b | subj0-only ×5 seeds: Broad / Deep | 0-4 | — | — | — | — | 0.64855 0.65351  | |
| 10-13 | deep_i | bf 0, ctx 4, 24 ep | 0 | subj0 0.708 | Broad-protocol 0.633 | | | — | = deep_c; longer training not a lever |
| 10-13 | LB BAL-2 | FINAL-6 files + per-subject Sinkhorn class balance | — | — | — | — | — | **Broad 0.6879 / Deep 0.6579** | balance re-rank: +0.032 / +0.004 |
| 10-13 | LB BAL-2b | FINAL-6 Broad + Sinkhorn T=2 | — | — | — | — | — | **Broad 0.6913** | |
| 10-13 | LB BAL-2b Deep | FINAL-6b Deep + Sinkhorn T=3 | — | — | — | — | — | **Deep 0.6639** | |
| 10-13 | proxy1 | deep_c recipe, TIMIT ses-10..14 excluded | 0 | subj0 0.701; PROXY (read sentences) 0.705 | Broad-protocol 0.598 | | | — | read sentences not harder |
| 10-13 | vr1 | deep_c recipe + --vocab_runs 0.3 | 0 | subj0 0.729 | Broad-protocol 0.599; vocab-run subset 0.535 | | | — | worse for Broad; not used |
| 10-14 | ext20 | deep_c + 20 LPP subjects (ext_frac 0.2) | 0 | subj0 0.731 | Broad-protocol 0.481 | | | — | external subjects HURT (−0.15) |
| 10-13 | LB FINAL-6c | subj0-only ×10 seeds + balance: Broad / Deep | 0-9 | — | — | — | — | 0.69213 0.66390  | |
| 10-14 | LB DRIFT-1 Deep | FINAL-6c ×10, subj0 identA / BN-recal / both, balT3 | 0-9 | — | — | — | — | Deep 0.69228 / **0.70767** / 0.70650 | BN recalibration on own holdout rows: +0.044 |
| 10-14 | LB BN-1 Broad | FINAL-6c ×10 + per-subject BN recal, balT2 | 0-9 | — | — | — | — | Broad 0.68076 | −0.011 vs 0.69213 |
| 10-14 | LB BN-α Deep | FINAL-6c ×10, AdaBN dropout-off α 1.0 / 0.5, balT3 | 0-9 | — | — | — | — | Deep 0.70767 / 0.67906 | full recal best for subj0 |
| 10-14 | LB DIV-1 Deep | FINAL-6c ×10 + deep_g/deep_h/vr1/ext20/deep_i, AdaBN, balT3 | — | — | — | — | — | Deep 0.69917 | −0.008 (noise) |
| 10-14 | LB BN-α Broad | FINAL-6 ens3 + partial AdaBN α 0.5 (dropout off), balT2 | 0-2 | — | — | — | — | Broad 0.68908 | −0.002 vs ens3_balT2 |
| 10-14 | LB SLP-1 Broad | FINAL-6c ens10 + self-learned within-subject HMM prior κ 50 / κ 200, balT2 | 0-9 | — | — | — | — | Broad **0.70366** / 0.68910 | κ 50 new best (+0.0115) |
| 10-14 | LB SLP-2 | Broad κ 20 / Deep κ 50 (on AdaBN) | 0-9 | — | — | — | — | Broad **0.70958** / Deep **0.71061** | prior strength trend real |
