# PREDICTIONS.md — pre-registered, written before any model run

Each entry is written and committed BEFORE the run that tests it. Outcomes are appended below the entry, never edited into it.
Metric: BAcc@10 over the 50 competition words (chance 0.20). The reference is the official Broad MEG-XL score of 0.4296.

---

## P1 — subj0-anchored alignment (the structural bet)
- **Stated:** 2026-09-21 ~17:04Z in session cfe74e11 (chat only). **Committed:** 2026-10-04 (this file), before any run.
- **Claim:** a supervised map from each Broad subject's sensors into subj0's sensor/feature space, fit on the stimulus-matched Sherlock1 ses-11 data, followed by a decoder trained on subj0's ~80 h, **beats 0.43 BAcc@10 on the ~40 min group (subj1–12)**.
- **Local estimand:** the few-shot proxy on the stimulus-disjoint split (fit the map on block A, evaluate on block B; block B is excluded from subj0's training too).
- **Holdout estimand:** the private-LB Broad score. It is not separable by group, so the group claim is tested locally only.
- **Per-group expectations (local proxy):** 100% (~40 min) 0.45; 50% 0.40; 25% 0.35; zero-shot (no map, subject-agnostic) 0.27.
- **What would falsify it:** a 100%-group local BAcc@10 ≤ 0.43, with the B0 baseline below as the positive control that the pipeline decodes at all.

## B0 — cheap subj0 decoder, applied as-is (control / first submission)
- **Committed:** 2026-10-04, before the run.
- **Setup:** a compact CNN trained on subj0 data excluding Sherlock1 ses-11/12. Per-subject channel standardisation uses each subject's own *training* statistics, or per-window statistics for zero-shot subjects. No TTA, no alignment.
- **Expectations:** subj0 within-subject on held-out Sherlock1 ses-12, about 0.45–0.55; Broad subjects with no alignment, about 0.24–0.30 (head position and anatomy differ, even though the MEGIN sensor array is shared); zero-shot the same as unaligned.
- **Why it matters:** if B0 already gets ≥0.35 cross-subject, sensor-space transfer is easier than assumed and P1's margin shrinks. If B0 is near chance, the alignment carries the whole load.

## P1a — operationalisation of P1 (committed 2026-10-04 ~14:45Z, before the run; B0 epoch-0 numbers had been seen: subj0-B 0.319, unaligned eval-B 0.211, prior floor 0.263)
- **Model:** the B0 subj0 decoder, frozen, with eval-mode BN. For each Broad subject s, a 306×306 map A_s (initialised at identity) is applied to the normalised window and fed to the frozen decoder.
- **Loss:** CE(label) + 1.0 · KL(teacher ‖ student) + 1e-3 · ‖A_s − I‖²_F. The teacher is the frozen decoder's softmax on **subj0's own window for the same stimulus token** (pairing is exact on `timechapter`, 100% matched).
- **Primary config, fixed a priori:** kd=1.0, lam=1e-3, 30 epochs, Adam lr 1e-3. The other configs in the sweep are reported as a map, not banked.
- **Zero-shot (no TTA):** A = the mean of A_s over the 28 training-pool Broad subjects.
- **Expectations (local block B, eval subjects 3/6/9/12; floor 0.263):**
  - s100 (≈30 min of fit data): 0.30
  - s50: 0.29
  - s25: 0.28
  - zero-shot mean-A: 0.25
- **Honest prior:** on these local numbers P1 (>0.43) looks unlikely with a 0.32-ceiling decoder. The subj0 decoder itself must improve first, because the alignment can only transfer what the decoder decodes. That is a decoder-capacity question, not evidence against the alignment idea.

### Outcome B0 (2026-10-04, runs/b0.json; epoch 6 selected on the early-stop set, tau=0.75)
- subj0 block B **0.498** (predicted 0.45–0.55: HIT). Unaligned eval subjects block B **0.211**, per subject 0.169 / 0.223 / 0.254 / 0.199 (predicted 0.24–0.30: MISS, lower). The prior-ranking floor is 0.263.
- Reading: sensor-space transfer without alignment is nil, even after SSS/head-position transform. Every cross-subject point has to come from alignment or multi-subject training.

## P1b — closed-form ridge sensor alignment (committed 2026-10-04 ~16:00Z, before the run; second operationalisation of P1)
- **Map:** A_s = argmin ‖A·X_s − X_0‖² + α‖A − I‖² over all time samples of paired windows (same stimulus token). Inputs are normalised per window. The output is re-normalised and fed to the frozen B0 decoder. No labels are used to fit A.
- **α:** primary α = 1 × mean eigenvalue of X_s X_sᵀ, fixed a priori; α ∈ {0.1, 10} reported as a map.
- **Zero-shot:** mean of A_s over the 28 training-pool subjects.
- **Expectations (block B, floor 0.263):** s100 0.27, s50 0.27, s25 0.26, s0 0.24. Low because the single-trial target X_0 is mostly noise; the shared stimulus-locked part is small.

### Outcome P1b (2026-10-04, runs/p1b_ridge.json + runs/p1b_ridge_low.json)
| α (rel.) | s0 meanA | s25 | s50 | s100 | composite |
|---|---|---|---|---|---|
| 10 | 0.191 | 0.207 | 0.201 | 0.201 | 0.201 |
| 1 (primary) | 0.220 | 0.216 | 0.220 | 0.217 | 0.218 |
| 0.1 | 0.232 | 0.258 | 0.282 | 0.256 | 0.259 |
| 0.01 | 0.233 | 0.265 | 0.274 | 0.266 | 0.262 |
| 0.001 | 0.232 | 0.261 | 0.252 | 0.258 | 0.252 |
- Primary α=1: MISS (predicted s100 0.27, got 0.217). The best point on the map (α=0.01) reaches the 0.263 prior floor and no further. SE ≈ ±0.01–0.02 (4 subjects × ~630 windows).
- **Instrument label: NOT DETECTED with instrument "unsupervised linear sensor map onto subj0's single-trial window + frozen B0".** The question stays OPEN.
  - Untested: label-supervised and distillation fits (P1a, queued); a multi-subject trunk (B1, queued); feature-space rather than sensor-space alignment; per-subject latency.
  - What the map shows: weaker regularisation helps up to a plateau. The subject-to-subj0 relationship is not near-identity in sensor space even after SSS/headpos.

## C1 — within-subject linear positive control (committed 2026-10-04 ~17:20Z, before the run)
- **Model:** multinomial logistic regression (L2, C=1e-3 fixed a priori, balanced class weights). Features: per-window-normalised windows pooled to 10 Hz (306×10). Trained on each eval subject's own fs_q data only; scored on that subject's block B, with tau=0 and balanced weights instead of logit adjustment.
- **Expectations:** s100 0.28, s50 0.27, s25 0.265 (floor 0.263). A pure-noise result is also possible at these data sizes.

### Outcome P1a (2026-10-04 13:30 EDT, runs/p1a.json)
- s100 **0.307** (pred 0.30, HIT); s50 0.297 (pred 0.29, HIT); s25 0.274 (pred 0.28, ≈); zero-shot mean-A 0.266 (pred 0.25, slightly above); identity 0.211. Composite 0.289. Floor 0.263.
- The score rises monotonically with fit data (s25 < s50 < s100), and the zero-shot mean-A lifts 0.211 → 0.266. That is the first cross-subject signal above the floor on this split.
- Against P1's >0.43: on the local metric, P1 is not reached. But the local metric is a sentence-only proxy with a 0.263 floor, so it is not on the holdout scale. P1 itself is tested only on the LB/private LB, with an LB-vs-local calibration from B0.

## B1 — multi-subject trunk (committed 2026-10-04 ~17:35Z, before reading the full run; the 1-epoch smoke HAD been seen: zero-shot 0.305, composite 0.316)
- **Setup:** scripts/train_b1.py defaults. The trunk trains on subj0 plus 28 pool Broad subjects (30% of each batch from Broad). Per-subject A (identity init, λ=1e-3); augmentation on; 12 epochs. Zero-shot uses the mean Broad A; few-shot fits A with the trunk frozen.
- **Expectations, full run:**
  - zero-shot 0.32
  - s25 0.33, s50 0.34, s100 0.35
  - composite 0.335
  - subj0-B 0.48 (slightly below B0, since capacity is shared)
- **Why it matters:** if zero-shot ≈ few-shot, per-subject alignment adds little over a subject-robust trunk. The zero-shot 18% is then handled without TTA.

## B2 — selection on held-out selection subjects (committed 2026-10-04 ~18:20Z, before the runs)
- **Motivation:** in B1, zero-shot peaked at epoch 1 (0.329) and decayed to ~0.22 while subj0 improved. Selecting on subj0 is misaligned with the Broad objective. Fix: VAL_SUBJECTS (1, 5, 10) are excluded from training, and epoch + τ are selected on their block-B zero-shot score. Eval subjects are untouched.
- **B2a:** per-subject A, broad_frac 0.5. Expect zero-shot 0.32, s25 0.32, s50 0.33, s100 0.34, composite 0.33.
- **B2b:** A frozen at identity during training (subject-agnostic trunk), broad_frac 0.5; few-shot still fits A afterwards. Expect zero-shot 0.31, composite 0.33.
- Both arms are reported as a map. The final choice is made on the VAL composite, never on the eval subjects.

### Outcome B1 (2026-10-04 ~14:20 EDT, runs/b1.json)
- Selected epoch 11 by subj0 es: zero-shot 0.220, s25 0.225, s50 0.234, s100 0.261, composite **0.237** (pred 0.335: MISS). subj0-B 0.487 (pred 0.48: HIT).
- Per-epoch zero-shot on eval (logged, not used for selection): 0.308, 0.329, 0.300, 0.282, 0.259, 0.269, 0.255, 0.241, 0.224, … The multi-subject trunk generalises early and then specialises.
- Instrument lesson: selecting on subj0 is the wrong instrument for a Broad objective. B2 tests VAL-subject selection. This is not evidence against multi-subject training, since the 1-epoch smoke reached 0.316.

## B4 — cross-subject contrastive invariance (charter "subject-invariance pressure") (committed 2026-10-04 ~18:55Z, before the run)
- **Setup:** B2a (VAL selection, per-subject A, broad_frac 0.5) plus β=0.5 · symmetric InfoNCE (temp 0.1, batch 128). The positive pair is the trunk embedding of a pool subject's window and subj0's window for the **same stimulus token**, over all words (vocab and non-vocab), using only pre75 tokens. 6 epochs.
- **Expectation:** VAL-selected zero-shot 0.34, few-shot s100 0.35, composite 0.345, with the zero-shot decay over epochs flatter than in B2a.
- **Prediction if wrong:** a decay as steep as B2a would mean the invariance term doesn't prevent the trunk from memorising the pool subjects.

### Outcome B2a (2026-10-04 ~15:10 EDT, runs/b2a.json)
- VAL selected epoch 0 (VAL 0.319). Eval: zero-shot 0.311, s25 0.292, s50 0.325, s100 0.313, composite **0.310**. Predicted 0.32 / 0.32 / 0.33 / 0.34 / 0.33: about 0.02 under across the board.
- VAL and eval zero-shot track each other per epoch (0.319/0.311, 0.295/0.294, 0.294/0.277, 0.278/0.259), so VAL is a valid selection proxy.
- Few-shot ≈ zero-shot. Fitting A on 10–30 min of a subject adds nothing measurable on top of this trunk (s25 is below zero-shot). The per-subject linear alignment arm is UNINFORMATIVE at this trunk quality, not refuted.

## E1 — 3-seed ensemble of 1-epoch trunks (committed 2026-10-04 ~19:20Z, before the runs)
- **Setup:** train_b1.py --select val --broad_frac 0.5 --epochs 1, seeds 0/1/2. Zero-shot (mean A) softmax average.
- **Expectations:** single seeds 0.30–0.31 eval zero-shot; ensemble 0.32.
- This recipe, with `--final` (all data, own A for subjects 1–32, mean A for 33–39), becomes submission #2.

### Outcome B2b (2026-10-04 ~16:30 EDT, runs/b2b.json)
- VAL selected epoch 0 (VAL 0.302). Eval: zero-shot 0.282 (pred 0.31: MISS), s25 0.298, s50 0.311, s100 0.306, composite **0.301** (pred 0.33: MISS).
- Compared with B2a (0.310): per-subject input layers during trunk training help the zero-shot trunk (+0.03). On the subject-agnostic trunk, post-hoc few-shot A fitting adds +0.02–0.03, unlike on B2a.

### Outcome C1 (2026-10-04, runs/c1_within_linear.json)
- Within-subject linear, own data only: s25 0.258, s50 0.282, s100 0.288 (pred 0.265 / 0.27 / 0.28: HIT). Per subject s100: 3 → 0.290, 6 → 0.265, 9 → 0.273, 12 → see json.
- Positive control passes: a subject's own 30 min carries word signal above the 0.263 floor. The zero-shot multi-subject trunk (B2a 0.311) already beats a subject's own 30-min linear decoder.

## B5 — capacity / regularisation / LR map against post-epoch-0 overfitting (committed 2026-10-04 ~21:15Z, before the runs)
- **Mode: GENERATIVE (a map).** Base is B2a (VAL selection, per-subject A, broad_frac 0.5); 4 epochs each.
  - B5a: d=128, dropout 0.5, wd 0.3 (smaller and more regularised)
  - B5b: lr 3e-4
  - B5c: broad_frac 0.8
- **Expectation:** at least one arm puts the VAL peak after epoch 0 and beats B2a's VAL 0.319 by ≥ 0.01. Most likely B5a (capacity is the suspected cause).
- The selected arm feeds the next final retrain only if it beats E1's ensemble on VAL.

### Outcome B4 (2026-10-04 ~17:30 EDT, runs/b4.json)
- VAL selected epoch 0 (VAL 0.304; per epoch 0.304, 0.301, 0.302, 0.290, 0.283, …). Eval: zero-shot 0.295, s25 0.287, s50 0.305, s100 0.306, composite **0.299** (pred 0.345: MISS). VAL decay was flatter for 3 epochs, as predicted, but the absolute level did not rise.
- **Label: NOT DETECTED with instrument "β=0.5 InfoNCE on trunk features of all-word token pairs, 6 ep".** OPEN; untested: β map, contrastive-only pretraining followed by CE, a projection from intermediate layers, longer windows.

### Outcome E1 (2026-10-04 ~18:00 EDT, runs/e1_ens.json)
- Eval zero-shot singles 0.277 / 0.292 / 0.264 (pred 0.30–0.31: MISS), ensemble **0.293** (pred 0.32: MISS). VAL singles 0.333 / 0.302 / 0.298, ensemble 0.309.
- Seed spread ±0.015 on eval and ±0.02 on VAL, so differences <0.03 between B2a/B2b/B4/E1 are not resolvable. Ensembling: +0.015.

## B6 — multi-subject trunk distilled from the subj0 teacher (committed 2026-10-04 ~22:05Z, before the run)
- **Setup:** B2a (VAL selection, per-subject A, broad_frac 0.5), with three changes:
  - the trunk is initialised from B0;
  - lr 3e-4;
  - Broad samples add kd=1.0 · KL(teacher ‖ student), where the teacher is a frozen B0 applied to **subj0's window for the same token** (in-vocab pairs, as in P1a).
  - 4 epochs.
- **Expectation:** VAL peak no longer at epoch 0. Eval zero-shot 0.33, s100 0.35, composite 0.34. subj0-B stays ≥ 0.45, since starting from B0 keeps subj0 power.

## LM1 — noisy-channel sentence decoding, simulation (committed 2026-10-08, before the run)
- **Why:** every holdout sentence word is a scored row (word_mask counts 5–9 per 2–3 s sentence), so sentences appear built from the 50-word vocab, and a sentence-level LM sees no unknown neighbours.
- **Test sentences:** natural vocab-only word runs of length 5–9 from LibriBrain transcripts (deduplicated). Not LM-generated. Split into dev/test folds by hash.
- **Brain evidence:** each word's score vector is sampled from REAL out-of-sample decoder outputs for a window of that word, ℓ(c) = log-softmax − log prior (τ=1).
  - **X:** b5a zero-shot (mean A) on eval subjects 3/6/9/12, all windows; those subjects are fully excluded from b5a training.
  - **S:** B0 on subj0 windows it never trained on (ses-10 early-stop set + block B).
- **Decoder:** coordinate ascent ("backfill"), 2 sweeps. Each slot's candidate score = β·ℓ_i(c) + log P_GPT-2(sentence with w_i=c) − δ·log prior(c). β and δ are chosen on dev; results reported on test.
- **Oracle:** the same, but the context is the true neighbouring words (a perfect-backfill upper bound).
- **Metric:** BAcc@10 over the 50 classes, rows pooled over sentences.
- **Expectations, gain over no-LM on test:**
  - X (weak evidence): +0.03 to +0.10
  - S (subj0-level evidence): +0.15 to +0.25
  - Oracle: ≥ +0.3 for both
- **Known biases:** GPT-2 may have seen Sherlock text (memorisation is optimistic). Natural fragments are less grammatical than constructed sentences (pessimistic).
