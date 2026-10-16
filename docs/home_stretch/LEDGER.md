# LEDGER — every strategy tried, LibriBrain / PNPL 2026 (compiled 2026-10-15 ~9 PM ET)

Sources: PREDICTIONS.md (all pre-registrations and outcomes), RUNS.md, REPORT.md, RUN_STATE.md, docs/XSESSION_TESTS_2026-10-15.md, docs/HANDOFF_EXTERNAL_085.md, docs/COMPLIANCE.md, git log (166 commits), logs/ and runs/ JSON, and the two session transcripts (9eb3b740 main; 690819b7 cross-session from 2026-10-15 13:32 UTC on). All numbers are top-10 balanced accuracy (chance 0.20).

**Goal (Rohit):** both public leaderboards above 0.80, by any means within the rules. **Now:** Broad 0.7096, Deep 0.7161. Public top: Deep 0.813, then 0.765. The Broad top is ~0.735 and is flagged by the organisers for cross-track leakage.

**Instruments:**
- **LB** = public leaderboard. Broad uses ~11k public rows, so a paired SE is ~0.004. Deep uses ~288 rows, so its SE is ~±0.03, and Deep LB changes under ~0.02 can't be read.
- **local** = split-v2, Q3/Q4 of the shared Sherlock1 ses-11/12 stimulus, + Sinkhorn unless stated. Broad = 12–13 subjects with identity A. Deep = subj0, same recording day as training.
- **PX/LSO** = leave-recording-day-out subj0 (day-novel).
- **SIM** = simulated holdout: LLM vocab-only sentences filled with real held-out posterior rows.
- **oracle** = labels used, so it measures headroom only.

Verdicts:
- **PROMOTED**: in a final stack.
- **NOT DETECTED**: |Δ| inside noise, or below the promotion bar.
- **HARMFUL**.
- **UNTESTED-IN-FINAL**: built or measured, never on the LB.
- **TAINTED**: excluded by policy.

---

## 1. Strategy table

### 1a. Representation, target space, architecture (training-time)
| # | Strategy | What it was | Track | Instrument | Measured Δ | Verdict |
|---|---|---|---|---|---|---|
| R1 | B0 subj0 CNN applied unaligned | 1 s WordCNN on subj0, applied raw to other brains | Broad | local / LB | subj0 0.498; other brains 0.211 (floor 0.263); LB 0.2032 | baseline; sensor transfer is nil |
| R2 | Hard 50-way classifier only | CE-only multi-subject trunk (Oct 4–8) | Broad | LB | 0.2266 | superseded |
| R3 | Continuous FIR cache, 50 Hz | Kaiser/poly anti-alias instead of a boxcar | both | local | v3a parity 0.272 vs V2-REF 0.284 (1 s) | PROMOTED (infrastructure) |
| R4 | 3 s masked sentence-epoch window [−1,+2] s | context window clipped to the word's own sentence epoch | both | local | deploy 0.272→0.292; lift +0.041→+0.073 (+0.03) | PROMOTED |
| R5 | Brain-side sentence transformer, no T5 | 2-layer transformer over a segment's words | both | local | v3c0 0.293 vs 0.292 | NOT DETECTED alone |
| R6 | Relative word-timing input | inter-word intervals into the transformer | both | local | v3c 0.328, v3d 0.506; timing zeroed: 0.26/0.30; a timing model 0.30→0.51 | TAINTED (house policy; removed, `--rel 0`) |
| R7 | **T5-large L12 contextual targets** | duplicate-aware SigLIP to the T5 encoder block-12 embedding of every word (1.0M words) + CE | both | local / LB | deploy 0.293→0.365 (4 ep) →**0.463** (8 ep); lift +0.073→+0.244. LB 0.2266→**0.4108** (2 seeds) | **PROMOTED: the biggest lever** |
| R8 | 8 epochs vs 4 (OneCycle) | longer horizon | both | local | 0.365→0.463 | PROMOTED |
| R9 | 12 epochs (v3e1, 0.46 regime) | longer training | Broad | local | −0.009 | NOT DETECTED |
| R10 | Capacity d192/D384 with 50% Broad batches (v3e2) | wider | Broad | local | −0.036 | HARMFUL (in that regime) |
| R11 | T5 loss weight 2 (v3e3) | | both | local | +0.005 | NOT DETECTED (used as 1 ensemble member in FINAL-2) |
| R12 | Wider window [−1.5,+2.5] (v3e4) | | both | local | −0.022 | HARMFUL |
| R13 | 100 Hz cache, 0–50 Hz content (FS100) | | both | local | −0.052 | HARMFUL |
| R14 | wav2vec2 L8 second SigLIP head (MV1) | acoustic target in the same trunk | both | local | +0.005 | NOT DETECTED |
| R15 | wav2vec2 InfoNCE, old classifier (B7) | speech targets, in-vocab only | Broad | local | composite −0.001 paired | NOT DETECTED |
| R16 | **subj0-only training (broad_frac 0)** | drop all Broad-subject batches; identity A for every new brain | both | local / LB | Broad deploy monotone 0.452 (bf .5) → 0.528 (.2) → 0.594 (.1) → **0.632** (0). LB Broad 0.4403→**0.6555**, Deep 0.5662→0.6246 | **PROMOTED: 2nd biggest lever** |
| R17 | 4-layer vs 2-layer sentence transformer (with T5) | depth | both | local | deep_b subj0 0.711 (confounded with bf); deep_d (bf .5, ctx4) Broad 0.452, so depth alone isn't the lever. Context gain with T5 = +0.08–0.10 (ablation: random neighbours 0.27–0.30; single words 0.25–0.28) | PROMOTED (ctx 4) |
| R18 | ctx 6 (deep_e, deep_g) | deeper | both | local | subj0 0.704/0.720; Broad 0.606/0.609 (−0.02) | NOT DETECTED |
| R19 | 12 ep / 24 ep subj0-only (deep_h, deep_i) | longer | both | local | subj0 0.721 / 0.708; Broad 0.606 / 0.633 | NOT DETECTED |
| R20 | **Width d192/D384, subj0-only (DW)** | | Deep | local / PX | subj0 0.742/0.706 vs 0.705/0.703 (+0.020, both seeds); **day-novel PX 0.746 vs 0.705 (+0.041)**; Broad −0.025 | **PROMOTED, Deep only** (FINAL-7) |
| R21 | T5-3B block 12/24 targets (TX) | bigger text model | both | local | subj0 0.699/0.745 (+0.018, seeds disagree); tx_s2 run; Broad −0.016 both seeds; on top of width (DWTX) 0.731 vs 0.742 | NOT PROMOTED |
| R22 | Session-pure batches (SPB) | one recording run per batch | both | local / PX | same-day subj0 −0.026, Broad −0.031; day-novel AdaBN 0.738 vs 0.705 (+0.033, 1 seed); conflicts with width (width+SPB 0.728/0.665) | DROPPED |
| R23 | Vocab-run oversampling (VR-1) | 30% of segments from natural runs of ≥4 vocab words | both | local | subj0 +0.024; Broad −0.033; vocab-run subset 0.570→0.535 | HARMFUL for Broad |
| R24 | **Time-stretch augmentation 0.8–1.25×** (Brain2Qwerty v2), P 0.5 | | both | local (same-day) | subj0 0.737/0.739 vs 0.705/0.703 (**+0.034, both seeds**). Broad 0.602/0.622 vs 0.632/0.622 (−0.015; the single-seed +0.047 didn't replicate) | **PROMOTED, Deep only** (FINAL-9 retraining now) |
| R25 | Width + stretch (dwstr_s0) | stacking check | Deep | local | running; autopilot picks width+stretch if subj0 ≥ 0.745, else stretch only | PENDING |
| R26 | Instance norm trained in (INORM-T) | MEGConformer recipe | both | local / PX | Broad 0.610 (−0.032); Deep same-day 0.739 (−0.003); day-novel 0.698 vs 0.678 (+0.020, 1 seed) | NOT PROMOTED |
| R27 | Separate wav2vec-only model + late fusion (ENS-T) | | both | local | Broad +0.004 (w 0.2); Deep negative | NOT PROMOTED |
| R28 | Augmentation in all finals | 10% gain noise, 10% channel dropout, near-identity spatial mix; 15% isolated 1 s windows; 30% shortened context | both | (baseline recipe) | never ablated in the final regime | IN STACK (unablated) |
| R29 | Read-sentence proxy (PROXY-1) | TIMIT ses-10..14 held out | Deep | local | 0.705 vs prose 0.701 | instrument: read speech ≈ prose |

### 1b. Multi-subject / external data
| # | Strategy | Track | Instrument | Δ | Verdict |
|---|---|---|---|---|---|
| D1 | P1a per-subject A + KD to subj0 paired window, frozen B0 | Broad | local | composite 0.289 (identity 0.211); s100 0.307 | superseded |
| D2 | P1b unsupervised ridge sensor map to subj0 | Broad | local | best 0.262 (floor 0.263) | NOT DETECTED |
| D3 | B1 multi-subject trunk, selected on subj0 | Broad | local | 0.237 (ep1 peak 0.329) | selection misaligned |
| D4 | B2a trunk, VAL selection | Broad | local | 0.310; B2b subject-agnostic 0.301 | superseded |
| D5 | B4 cross-subject InfoNCE β 0.5 | Broad | local | 0.299 | NOT DETECTED |
| D6 | B5 d128/p0.5/wd0.3; lr 3e-4; bf 0.8 | Broad | local | 0.315 / 0.296 / 0.260 | noise / harmful |
| D7 | B6 trunk init from B0 + KD | Broad | local | 0.303 | NOT DETECTED |
| D8 | C1 within-subject linear (own 30 min) | Broad | local | 0.288 | positive control; the zero-shot trunk beats it |
| D9 | Broad subjects in training at 0.1–0.5 of batches | both | local | Broad 0.594→0.452; subj0 0.711→0.637 | HARMFUL |
| D10 | +20 Le Petit Prince listeners (ext20, French, SigLIP-only) | both | local | subj0 +0.026; **Broad 0.481 (−0.151)** | HARMFUL |
| D11 | MEG-XL frozen features (MEGXL-P) | both | local probe | subj0 probe ours 0.512 / MEG-XL 0.219 / concat 0.451; Broad 0.432 / 0.227 / 0.410 | HARMFUL (MEG-XL needs ~150 s context) |
| D12 | Recipe-diversity ensemble (deep_g/h/i, vr1, ext20 + FINAL-6c) | Deep | LB | 0.69917 vs 0.70767 (−0.008) | NOT DETECTED |

### 1c. Per-subject adaptation
| # | Strategy | Track | Instrument | Δ | Verdict |
|---|---|---|---|---|---|
| A1 | Fine-tune A + head (0.46 regime) | Broad | local | ±0.01 | NOT DETECTED (regime-specific) |
| A2 | CORAL α {0,.5,1}, entropy-min TTA 30 steps | Broad | local | ±0.01 | NOT DETECTED |
| A3 | **AdaBN: re-estimate encoder BN statistics on subj0's 960 holdout rows** | Deep | LB | 0.66390→**0.70767 (+0.044)**; dropout-off identical | **PROMOTED (Deep)** |
| A4 | Identity A for subj0 (drop A0) | Deep | LB | +0.028; with AdaBN 0.70650 (no gain over AdaBN) | not used |
| A5 | AdaBN α 0.5 (partial) | Deep | LB | 0.67906 (−0.029) | HARMFUL |
| A6 | AdaBN per Broad subject, α 1 (BN-1) | Broad | LB | 0.68076 (−0.011); local +0.008 | HARMFUL |
| A7 | AdaBN α 0.5 Broad | Broad | local / LB | local **+0.028** (6/7) → LB 0.68908 (**−0.002**) | NOT DETECTED (proxy unfaithful) |
| A8 | TENT-style BN affine training on balanced pseudo-labels (bnD+aff100) | Broad | local | 0.684→0.615 (subj1 Q3) | HARMFUL |
| A9 | INF-2 pseudo-label self-training of A | both | local | 0.695→0.573 (50 steps) →0.453 (150) | HARMFUL (confirmation bias) |
| A10 | SLP-ST: self-train A toward SLP+Sinkhorn targets | both | files built (`st_deep_bn*.csv`) | no LB outcome recorded | UNTESTED-IN-FINAL |
| A11 | **FITA: labelled per-subject 306×306 A, frozen trunk, CE (subjects 1–32)** | Broad | local | ORACLE O6 +0.041. FITA-2 map: deep_c +0.030..+0.044, lso1 +0.051..+0.068, **18/18 cells positive**; monotone in data fraction, flat in steps (100/300/600). Local Broad 0.642→0.687 | **PROMOTED (high)**: held as FITA-1/2/3, never on the LB (expected LB +0.02..+0.035) |
| A12 | FIR-1 spatio-temporal taps k 3/5 | Broad | local | full data +0.004; 25% −0.008 | NOT PROMOTED |
| A13 | ANAT-1 Laplacian smoothness on A−I | Broad | local | 0.671/0.676 vs 0.676/0.687 | HARMFUL |
| A14 | ANAT-1 low-rank-16 A update | Broad | local | +0.010 at 25% data, −0.007 at 100% | PROMOTED for subjects with <30 min (FITA-2/3) |
| A15 | **Fixed temporal-lobe up-weighting (TEMP-1), zero-shot** | Broad | local | k 0.5 +0.016 · 1 +0.028 · 2 +0.029 · **3 +0.033 (23/24)** · 5 +0.028 · 8 +0.022 · 15 −0.001 · 30 −0.061; Deep ≤ +0.004 | **PROMOTED for the 7 no-data subjects** (k 0.5 in FITA-2, k 3 in FITA-3) |
| A16 | Temporal k3 as the FITA base | Broad | local | 24 blocks: base 0.6425, temporal 0.6747, fit 25% 0.6846, fit 100% 0.6918 (vs plain fit 0.676/0.687): **+0.003..+0.008** | in FITA-3 |
| A17 | Per-subject spectral equalisation to subj0 (SPEQ-1) | Broad | local | type −0.016, chan −0.023, half −0.012 | HARMFUL |
| A18 | Instance norm at inference (INORM-2) | both | local / PX | Deep same-day +0.005..+0.017; day-novel +0.003/−0.003; Broad −0.007..−0.015 | NOT PROMOTED |
| A19 | Hard lead-field projection top-k (MOON-1) | both | local | Broad k30 0.648 / k50 0.629 / k70 0.627 vs 0.644; Deep 0.716/0.750/0.741 vs 0.743 | NOT DETECTED |

### 1d. Test-time structure, priors and readouts
| # | Strategy | Track | Instrument | Δ | Verdict |
|---|---|---|---|---|---|
| T1 | τ=1 (de-collapse top-1) | Broad | LB | +0.001 | NOT DETECTED |
| T2 | Sampler prior correction (CAL1) | Broad | LB | −0.011 | HARMFUL |
| T3 | Logit − 0.5·log prior (τ 0.5) | both | local fold choice | in recipe | IN STACK |
| T4 | GPT-2 coordinate ascent (LM1) | both | SIM | X +0.028, S −0.002 | NOT DETECTED |
| T5 | Exact F-B external trigram (LM2 / v2) | both | SIM | context gain X +0.036, S +0.007 | weak-evidence only |
| T6 | Trigram rerank at LB 0.43 (Gutenberg / LLM vocab corpus) | Broad | LB | +0.006 / +0.006 (0.4344 / 0.4403) | promoted then RETIRED |
| T7 | Same rerank at strong evidence | both | LB | Broad 0.6555→0.5441 (**−0.11**); Deep 0.5662→0.5194 (−0.047) | HARMFUL |
| T8 | **Per-subject Sinkhorn balance to uniform over all 50 words** | both | local / LB | local +0.030..+0.045; T map 0.5 0.655 / 1 0.674 / 2 0.687 / 3 0.688 / 4 0.689. **LB Broad +0.032 (T1) → +0.036 (T2), 0.6913; Deep +0.004 (T1) → +0.010 (T3), 0.6639** | **PROMOTED** |
| T9 | One-step marginal correction | both | local | 0.000 | NOT DETECTED |
| T10 | Unsupervised prototype pooling (cos space κ5/20; logit; encoder) | both | local | 0.609/0.472; 0.676 (= Sinkhorn); 0.659/0.547 | HARMFUL / none |
| T11 | Prior-corrected ranking p/π (true or Saerens-EM priors) | both | local | 0.377 / 0.547 vs Sinkhorn uniform | HARMFUL; uniform is the right target |
| T12 | **SLP: self-learned within-subject bigram HMM (EM on the subject's own 156 sentences)** | both | LB | Broad κ curve 5 0.6876 · 10 0.7048 · **20 0.7096** · 50 0.7037 · 200 0.6891 · ∞ 0.6921 → **+0.018**. Deep (AdaBN) 10 0.7071 · **20 0.7161** · 50 0.7106 · ∞ 0.7077 → **+0.008** | **PROMOTED** |
| T13 | SLP + learned end-of-sentence distribution | Broad | LB | −0.013 | HARMFUL |
| T14 | SLP + balance inside EM | Broad | LB | −0.008 | HARMFUL |
| T15 | SLP with a text-bigram Dirichlet base | Broad | LB | −0.047 | HARMFUL |
| T16 | SLP trigram backoff | Deep | LB | −0.009 (Broad files built, not submitted) | NOT DETECTED |
| T17 | SLP emission sharpness β 0.7 / 1.4 | both | files built | no LB outcome | UNTESTED-IN-FINAL |
| T18 | SLP on local prose vocab-runs (O5) | both | local | learned −0.009/−0.016; oracle bigram +0.029/+0.024 | local is blind to SLP |
| T19 | SIM-1 calibration of SLP | both | SIM | +0.016 (LB +0.018) / +0.007 (LB +0.008) | instrument CALIBRATED |
| T20 | GPT-2 n-best rescoring on SIM | both | SIM | Broad −0.004; Deep +0.009 (1 subject) | NOT PROMOTED |
| T21 | Class-based grammar (Brown-style categories as the SLP base) | both | — | proposed (+0.005..+0.015 expected); needs a leave-one-book-out testbed | UNTESTED |
| T22 | Rule-based sentence priors (CA rule, class-HMM K4/K8, multiway beam K20) | both | local vocab-runs | CA 0.585/0.704; HMM 0.592/0.609, 0.674/0.696; beam ≈ exact (0.641/0.746) vs SLP 0.638/0.727 | HARMFUL / equal |
| T23 | CTXM contextual target matching | both | local | oracle context Broad 0.622→0.700, Deep 0.773→0.800. Realistic (decoded context): subj0 0.773→0.736, subj1 0.614→0.604, subj2 0.610→0.558 | realistic version HARMFUL; oracle seam OPEN |
| T24 | CLOSED-1 sentence-as-unit retrieval | both | oracle | K10 0.976/0.994 · K100 0.889/0.970 · K1000 0.768/0.887 · K1e4 0.678/0.796; grammar space ~1e8 | CLOSED (only route is reconstructing the organisers' sentences = forbidden) |
| T25 | First-word temperature | both | local | Broad +0.0045, Deep 0 | NOT DETECTED |
| T26 | Condition-dependent / latent-state Sinkhorn T (PACKARD T2, STATE-1) | both | local | +0.002 / +0.0015 (= a global T retune) | NOT DETECTED |
| T27 | High-confidence anchoring inside Sinkhorn (oracle-perfect) | both | local | other rows Broad 0.621→0.578, Deep 0.723→0.693 | HARMFUL even as an oracle |
| T28 | ABS-1 linear TRF analysis-by-synthesis | both | local | stage-1 oracle 0.282; fused −0.001 Broad / −0.004 Deep | NOT DETECTED |
| T29 | Readout ce+cos τ0 vs ce τ0.5 | both | local folds | ce τ0.5 from FINAL-6 on | IN STACK |

### 1e. Ensembling
| # | Strategy | Track | Instrument | Δ | Verdict |
|---|---|---|---|---|---|
| E1 | 3-seed ensemble, 1-epoch trunks | Broad | local | +0.015 | — |
| E2 | 2 seeds (v3d1) | both | local | +0.01 (0.475) | PROMOTED |
| E3 | 5 models (4 seeds + t5w2) | Broad | LB | 0.4108→0.4284 (+0.018) | PROMOTED |
| E4 | 8 models | Broad | LB | 0.4235 (−0.005) | NOT DETECTED |
| E5 | 3→5 seeds (FINAL-6b) | both | LB | Broad −0.007 (noise); **Deep +0.029** (0.6246→0.6535) | Deep PROMOTED |
| E6 | 5→10 seeds + balance | both | LB | Broad +0.0008; Deep 0.000 | NOT DETECTED |
| E7 | Snapshot ep5–7 / weight soup / 2-seed (LIT-1) | both | local | snapshot 0.642/0.735 (none); soup **0.507**/0.731; 2-seed +0.004/−0.005 | soup HARMFUL |

### 1f. Neurophysiology, hardware and "moonshot" probes (all oracle or label-free)
| # | Probe | Δ (Broad / Deep) | Verdict |
|---|---|---|---|
| N1 | Sensor-type reweighting (mags ×0.5 / 0 / grads 0) | −0.016/−0.034/−0.169 ; −0.001/−0.021/−0.094 | HARMFUL |
| N2 | Latency shift ±60 ms, oracle / confidence-picked | +0.008/−0.006 ; +0.013/−0.021 | NOT DETECTED |
| N3 | Remove own top-k spatial PCs (k 1–8) | −0.010..−0.059 ; −0.004..−0.014 | HARMFUL |
| N4 | Pooling ceiling, labels allowed (n 2/4/8/16) | +0.074/+0.141/+0.193/+0.217 ; +0.071/+0.130/+0.171/+0.182 | headroom real, label-free unreachable |
| N5 | PURITY-1 raw-MEG kNN same-word purity (8 spaces) | 0.065–0.070 vs chance 0.067; model space 2× chance (its own signal) | closed |
| N6 | DTW-1 vs Euclidean purity | chance in both | closed |
| N7 | RIEM-1 covariance tangent space | 0.214 vs evoked 0.267 | closed |
| N8 | REP-1 within-sentence same-word pairing | raw AUC 0.514/0.487; combined lowers AUC 0.726→0.704 / 0.794→0.729 | closed |
| N9 | Jitter TTA / channel-dropout TTA (INF-1, 3 blocks, stopped early) | jit+sk +0.010/−0.011/+0.001; chd +0.007/−0.011/−0.022 | NOT DETECTED (partial) |
| N10 | PACKARD-1 brain-state pockets (alpha/theta/beta/artifact) | alpha q5−q1 −0.004 (p .8) / −0.039 (p .04), inconsistent | NOT DETECTED |
| N11 | Packard condition search, 18,032 raw conditions | best 2.15 vs null95 1.75; top-8 reverse on Q4 | NOT DETECTED |
| N12 | Sentence-initial pocket (word-controlled) | Broad +0.18/+0.32; real, from post-silence response; masking context hurts mid-sentence words (0.719→0.606) | real but not exploitable |
| N13 | Training cartography (day, run time, task, position) | flat (TIMIT confidence = memorisation; TheMoth anti-pocket ~9%) | NOT DETECTED |
| N14 | Forecast-subtraction (predictive residual) | 0.286 vs 0.287 | NOT DETECTED |
| N15 | MOON-1: reservoir/ESN, theta–gamma PAC, TDA persistence, SIREN neural field, HDC, hyperbolic | alone 0.20–0.24 (chance); HDC 0.33/0.32, hyperbolic 0.36/0.39 (re-encodings of e); fused always LOWER (0.69–0.70 / 0.58–0.60 vs 0.720/0.623) | HARMFUL / exhausted |
| N16 | Kruse / applied-EMF / biophoton channels | no physical channel into a fixed recording | closed by argument |

---

## 2. Current final stacks and files

**Submitted bests (auto-selected if nothing else is chosen):**
- Broad `submissions/slp2_broad_k20.csv` **0.70958**: FINAL-6c 10 seeds (subj0-only, 3 s, ctx 4, T5-large SigLIP + CE, 8 ep) → identity A → ce τ0.5 → SLP κ20 → Sinkhorn T2.
- Deep `submissions/slp3_deep_bn_k20.csv` **0.71611**: the same 10 models → subj0 A0 → AdaBN on its own 960 rows → SLP κ20 → Sinkhorn T3.

**Held, never submitted** (nothing has been submitted since Oct 14; 10 slots per track available):
| file | stack | basis | expected LB |
|---|---|---|---|
| `fita1_broad_ens5_k20.csv` | FINAL-6c s0–4 + labelled A (300 steps) for 1–32 + SLP + Sinkhorn T2 | A11 | 0.73–0.745 |
| `fita2_broad_ens5_k20.csv` | FITA-1 + low-rank-16 below 30 min + temporal k 0.5 for 33–39 | A11, A14, A15 | FITA-1 +0.003..+0.008 |
| `fita3_broad_ens5_k20.csv` (building in FINAL-9) | FITA-2 with temporal k 3 | A15, A16 | FITA-2 + ~0.003 |
| `final7_deep_bn_k20.csv` | width d192/D384 × 5 seeds + AdaBN + SLP + Sinkhorn T3 | R20 | 0.72–0.75 |
| `final7_deep_k20.csv` | the same without AdaBN | | AdaBN was +0.044 on the plain model, so the bn file is the default |
| `final9{dws,str}_deep_bn_k20.csv` (training now) | stretch (± width) × 5 + AdaBN + SLP + Sinkhorn | R24, R25 | 0.73–0.77 (stretch measured on same-day data only) |
| Other built, unsubmitted | `slpt_broad_k3_{10,50}`, `slpb_*_b{0.7,1.4}`, `st_deep_bn*` | T16, T17, A10 | ±0.01 |

**Autopilot** (`scratchpad/lb/queue_final9.sh` → `finalize.sh`, cross-session 690819b7):
1. dwstr_s0 screen.
2. FITA-3.
3. 5 Deep seeds: width+stretch if dwstr subj0 ≥ 0.745, else stretch only.
4. bn and no-bn files.
5. Submits FITA-3 Broad, the FINAL-9 Deep bn file and final7_deep_bn_k20, then notifies.

---

## 3. Patterns

### What the winners share
1. **Change the target or representation space, not the classifier.** T5 contextual targets were the single biggest step: LB 0.23→0.43, local lift ×3. The sentence transformer only pays inside that space (+0.08–0.10 with T5, 0 without).
2. **One deep, homogeneous reference brain.** subj0-only (80 h) beats every multi-subject mix: Broad 0.452→0.632 local, LB 0.44→0.66. Every added subject or corpus hurts zero-shot transfer (D9, D10, VR-1). Changes that help subj0 (width, ctx 6, longer training, stretch, ext data) consistently cost Broad 0.015–0.03. **Recipes must be track-specific.**
3. **Exploit the subject's own test-set structure, learned from that subject's rows, with uniform/identity as the anchor.**
   - Sinkhorn to uniform: +0.036 / +0.010.
   - SLP with Dirichlet toward uniform: +0.018 / +0.008, peak at κ≈data (κ 20).
   - Simplest forms win. The extras (end distribution, in-EM balance, trigram, text base) all lose.
4. **Same-brain, new-day calibration via normalisation statistics.** AdaBN +0.044 on Deep. Width's gain was larger on a new day (+0.041) than on the same day (+0.020).
5. **Cheap labelled per-subject linear calibration on a frozen trunk.** FITA: +0.03..+0.07 local, 18/18 configurations. It is robust because the trunk stays frozen and the map is linear. Low-rank helps when data is small.
6. **A fixed anatomical prior where no labels exist.** Temporal-lobe up-weighting: +0.033 zero-shot at k 3, 23/24 blocks.
7. **A physiologically motivated augmentation for the deep subject.** Time-stretch: +0.034 on both seeds (Deep only).
8. **Variance reduction only while the evidence is strong and seeds are few.** 2→5 seeds gave +0.018 / +0.029. 5→10 gave 0.000.

In mechanism, every winner either **raises per-word evidence from the reference model** (targets, depth with targets, width, stretch) or **removes a train→test mismatch with a few parameters estimated from abundant within-subject data**: class prior (Sinkhorn), sequence prior (SLP), BN scale (AdaBN), spatial map (FITA), anatomical weighting.

### What the failures share
1. **Importing statistics from another distribution:**
   - text n-grams or a text Dirichlet base: −0.05..−0.11 at strong evidence;
   - true or estimated class priors;
   - other subjects or languages: −0.15;
   - forcing new brains toward subj0's spectrum: SPEQ −0.016..−0.023;
   - forcing new brains toward their own BN statistics: −0.011.
2. **Features from single-trial raw MEG beyond the decoder.** All 15+ families sit at chance (purity, DTW, covariance, PAC, TDA, reservoir, neural field, MEG-XL, lead-field). Fused, they lower the score. Bottleneck = per-word SNR; ≈2× SNR would be needed for +0.15.
3. **Self-referential adaptation.** Pseudo-labels collapse (0.695→0.45), TENT −0.07, and certainty can't propagate even from oracle anchors.
4. **More capacity or training for the zero-shot track.** It specialises the trunk to subj0.
5. **Inference-time physiology knobs** (latency, sensor weights, PCs, jitter, state gating): ≤ +0.013 even as oracles.
6. **Gains seen only on an unfaithful proxy or a single seed.** BN-α +0.028 local → −0.002 LB. Stretch Broad +0.047 → −0.015 on the second seed. T5-3B Deep seeds disagree.

---

## 4. Instrument issues (local ↔ LB)

| issue | numbers | consequence |
|---|---|---|
| Early split floor | local floor 0.263 vs holdout 0.20; LB ≈ 0.20 + (local − 0.263) | superseded by split v2 |
| Split-v2 Broad → LB | ≈ +0.02 (local 0.632 → LB 0.6555; ~0.69 → 0.7096 with SLP) | Broad local is slightly pessimistic, and it is blind to SLP |
| Split-v2 Deep → LB | −0.05..−0.08 (0.705 → 0.625; 0.78 → 0.716). Cause: subj0's local test day was in training (only runs of 2024-10-04) | same-day Deep screens overstate. Use the LSO/PX day-novel instrument |
| PX for AdaBN | PX −0.010 vs LB +0.044 | judge AdaBN by the LB only |
| PX for Sinkhorn (Deep) | +0.0105 vs LB +0.010 | faithful |
| Local BN/TTA on Broad | BN-α local +0.028 → LB −0.002; BN-1 local +0.008 → LB −0.011 | local blocks lack holdout-style masking (short epochs, 10% isolated 1 s) |
| PROXY-2 balanced/short-sentence proxy | Sinkhorn +0.20 vs LB +0.036 | unfaithful; small blocks inflate Sinkhorn |
| Local Sinkhorn over present classes | 0.694 vs all-50 uniform 0.650 | use all-50 locally |
| SLP on local prose | −0.009/−0.016 vs LB +0.018/+0.008 | only SIM-1 (+0.016/+0.007) is calibrated for sentence priors |
| Short vs long sentences | local ≤ 8 s sentences 0.75 / 0.81 vs all rows 0.69 / 0.71 | holdout-like rows are easier |
| Seed noise | ±0.01–0.02 locally; promotion needs both seeds same-sign | single-seed wins replicated about half the time |
| Deep public split | ~288 rows, SE ~±0.03; FINAL-6c 10 vs 5 seeds tied to 5 decimals | Deep LB moves < 0.02 are unreadable; private 70% decides |
| Broad public split | ~11k rows, paired SE ~0.004 | Broad LB is a fine instrument (κ curve resolved) |
| LM sims v1 | alphabetical-sampling bug | v2 numbers only |
| Timing taint | v3c/v3d 0.328/0.506 were timing | `--rel 0` everywhere |

**Faithful proxies:**
- Broad LB, for everything.
- SIM-1, for sentence priors.
- PX day-novel, for Deep model and recipe changes and for Sinkhorn (not for AdaBN).
- Split-v2 local, for Broad model changes and FITA (direction; the magnitude transfers about 1:1).

**Unfaithful proxies:**
- Local BN/TTA on Broad.
- Same-day subj0 for anything drift-related.
- Balanced or short-block proxies, for Sinkhorn magnitude.
