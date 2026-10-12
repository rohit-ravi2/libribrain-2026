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

### Outcome LM1 (2026-10-08, runs/lm1_sim.json; 300 dev + 300 test sentences per arm)
| arm | no-LM (test) | GPT-2 coordinate ascent (test) | gain | oracle (true context) |
|---|---|---|---|---|
| X: cross-subject b5a | 0.230 | 0.258 | **+0.028** (pred +0.03–0.10: low edge) | 0.923 |
| S: subj0 B0 | 0.450 | 0.448 | **−0.002** (pred +0.15–0.25: MISS) | 0.905 |
- The dev grid chose β=4 for S, the largest brain weight tried, so the LM prior was best nearly off.
- **Label: NOT DETECTED with instrument "greedy coordinate ascent conditioned on the argmax context".** At top-1 ≈ 10% the conditioning context is mostly wrong. The oracle shows the ceiling, provided context uncertainty is integrated.
- **Next instrument (LM2):** exact forward–backward marginals under a trigram LM over the 50-word vocab, trained on external text (not the test fragments), plus β map extended beyond 4.
- **Caveat:** natural text fragments, not holdout-style constructed sentences.

## B7 — staged sound→word: speech-embedding contrastive targets (committed 2026-10-08, before the run)
- **Targets:** facebook/wav2vec2-base, layer 8, contextual frames (chapter audio in 30 s chunks), mean-pooled over [onset, onset+0.4 s] on the stimulus clock. Session N ↔ chapter N, verified by duration. No audio for Sherlock8/TIMIT/TheMoth, so those words get CE only.
- **Model:** B5a config (d128, dropout 0.5, wd 0.3, VAL selection, per-subject A, broad_frac 0.5, 4 ep) + β=1.0 · symmetric InfoNCE between proj(trunk features) and a learned projection of the word's speech embedding (temp 0.1).
- **Expectations vs B5a (composite 0.315, zero-shot 0.315):** zero-shot +0.02, composite 0.335, subj0-B +0.03.
- **Comparison:** paired seeds 0 and 1 (b5a seed 1 is run as the matched control); the effect is the mean paired difference.
- **Why:** the shared audio supervises token-level acoustics identically across subjects, which should be a subject-invariant target.

## LM2 — exact forward–backward with an external trigram LM (committed 2026-10-08, before the run)
- **LM:** interpolated trigram over the 50 vocab words (0.6 tri / 0.3 bi / 0.1 uni, add-0.1). Counts come from vocab-only word runs in 15 Project Gutenberg novels (no Doyle), so there is no overlap with the test fragments' source.
- **Inference:** exact forward–backward over (w_{i-1}, w_i) states. Emission exp(β·ℓ_i). Ranking = log marginal − δ·log prior; β, δ chosen on dev.
- Same sentences, folds, brain-evidence pools and sampling seed as LM1.
- **Expectations, test gain over no-LM:**
  - X: +0.04
  - S: +0.08 (the context uncertainty LM1 could not handle is now integrated)

### Outcome LM2 (2026-10-08, runs/lm2_sim.json + runs/lm2_sim_unigram.json; forward–backward verified against brute force, max diff 6e-15)
| arm | no-LM | unigram control (prior only) | trigram exact F–B | context gain (tri − uni) |
|---|---|---|---|---|
| X cross-subject | 0.230 | 0.259 (+0.029) | 0.301 (**+0.070**, pred +0.04) | **+0.041** |
| S subj0 | 0.450 | 0.508 (+0.057) | 0.497 (+0.046, pred +0.08) | **−0.011** |
- Exact inference fixes LM1's failure mode on X: +0.070 vs +0.028.
- The unigram control is essential. For subj0, all of the gain is frequency-prior re-injection (dev picked δ=0), which matches the natural-frequency test fragments. It will not transfer if holdout sentences cover the 50 words more evenly. The context part is about +0.04 on weak evidence and ≈0 on subj0-level evidence, with SE ≈ ±0.01–0.015.
- **Status:** sentence context is a real but modest lever (≈+0.04 at our cross-subject level). That is not the "massive boost" the 0.9 oracle suggested. The oracle stays unreachable because neighbours are never known.
- **OPEN, untested:**
  - LM trained on holdout-style constructed sentences (unknown grammar);
  - joint decoding with the subj0-to-Broad mapping;
  - a GPT-2 marginal via sampling;
  - the prior question, which is decided by the holdout's word distribution and can be tested cheaply with one LB submission at δ=0 vs δ=1.

### Outcome B7 (2026-10-08, runs/b7_s{0,1}.json vs b5a / b5a_s1)
| seed | composite B7 / B5a | zero-shot B7 / B5a | subj0-B B7 / B5a | VAL B7 / B5a |
|---|---|---|---|---|
| 0 | 0.304 / 0.315 | 0.296 / 0.315 | 0.383 / 0.365 | 0.324 / 0.324 |
| 1 | 0.313 / 0.304 | 0.293 / 0.288 | 0.351 / 0.387 | 0.311 / 0.318 |
- Mean paired difference: composite −0.001, zero-shot −0.007, subj0 −0.009 (pred +0.02 / +0.02 / +0.03: MISS).
- The targets are valid (a linear probe on them decodes word identity at 0.881).
- **Label: NOT DETECTED with instrument:**
  - β=1 InfoNCE on in-vocab windows only;
  - 55% target coverage;
  - the selected epoch is 0–1, so the auxiliary loss gets about one epoch to act.
- **OPEN; strongest next test:** speech targets on ALL words (2.5× data), the setting where contrastive speech targets are known to matter, and speech pretraining before CE rather than joint training.

## CAL1 — LB calibration of the prior correction (committed 2026-10-08 ~16:45Z, before the submission)
- **Context:** the audit (AUDIT_AND_RECOVERY_2026-10-08.md §3) found the stored log_prior is row-weighted (8.9% Broad) while training draws 50% Broad. The local block lacks 12/50 words, so it structurally favours frequency-shaped rankings and cannot adjudicate. The LB holdout covers all 50 words.
- **Submission:** final_e1 ens3, τ=1 with the sampler prior (0.5·p_subj0 + 0.5·p_Broad), vs the existing τ=1 stored-prior submission (LB 0.2266).
- **Prediction:** +0.005 to +0.02 (≈0.23–0.245). A drop would mean the holdout's word distribution is closer to training frequencies than to uniform.
- This is a declared one-parameter calibration measurement on the only all-50-class test available, not leaderboard-driven model selection.

## Correction 2026-10-08 — LM1/LM2 sentence sampling
- scripts/sim_lm.py:91 and scripts/sim_lm2.py:93 took the first 300 fragments of an alphabetically sorted list in each fold, not a random sample (audit §4). The LM1/LM2 numbers above come from that biased sample.
- LM2 (trigram + unigram control) is re-run with a seeded random sample; LM1 is superseded, not re-run. Independent per-word resampling of neural evidence remains a limitation of both.

## V2-REF — reference run on split v2 (committed 2026-10-08, before the run)
- **Setup:** scripts/train_v2.py, B5a recipe (d128, p0.5, wd0.3, broad_frac 0.5, aug, per-subject A, 4 ep), sampler prior, seed 0. Folds A (select Q3 → test Q4) and B (select Q4 → test Q3) come from one training run.
- **Purpose:** first reference under the corrected evaluation, and the first validation of the deployed path (seen subjects with their own jointly trained layer).
- **Expectations (fold-averaged):**
  - deployment score 0.30; model-specific null 0.25; lift +0.05
  - seen subjects ≥ zero-shot by +0.01 to +0.03
  - folds may differ by up to 0.03 because block class support differs (Q3 has 47 words, Q4 38)

### Outcome CAL1 (2026-10-08, LB)
- Sampler prior, τ=1: LB **0.2155**, vs stored prior τ=1 at 0.2266 → **−0.011** (pred +0.005 to +0.02: MISS).
- **Reading:**
  - Removing the 50%-Broad sampler prior over-corrects on the holdout.
  - Either the holdout's word distribution is closer to training frequencies than to uniform, or the model's outputs are not calibrated likelihoods (label smoothing, early stopping). The two cannot be separated with one submission.
- **Action:** keep stored-prior τ=1 for submissions. No further LB tuning of τ (that would be LB-driven selection).

### Outcome LM2-v2 (2026-10-08, random sentence sample, runs/lm2_sim_v2.json + _unigram_v2.json)
| arm | no-LM | unigram | trigram | context gain (tri − uni) | v1 (alphabetical) context gain |
|---|---|---|---|---|---|
| X cross-subject | 0.231 | 0.240 (+0.009) | 0.276 (+0.045) | **+0.036** | +0.041 |
| S subj0 | 0.445 | 0.466 (+0.021) | 0.472 (+0.028) | **+0.007** | −0.011 |
- The X context gain survives the sampling fix (+0.036). For S it stays ≈0.
- Still synthetic: neural evidence is resampled per word, so it is not a leaderboard forecast (audit §4).

## T5-1 — text-target decoder, d'Ascoli-style adaptation (committed 2026-10-08, before any run)
- **Recipe (reference: neural-processing-lab/dascoli-word-decoding defaults):**
  - targets: t5-large, raw encoder block-12 output at each word's first subword, sentence context;
  - SigLIP with duplicate-aware positives (target cosine > 0.999);
  - all words as training data (~880k windows);
  - readout = cosine to vocab prototypes.
- **Departures, so this is an adaptation, not a reproduction:**
  - 1 s windows, not 3 s;
  - no cross-word Transformer (that is the sentence-context gray zone);
  - WordCNN with subject input layers instead of SimpleConvTimeAgg;
  - per-window normalisation, not RobustScaler;
  - boxcar 50 Hz, not a 0.1–40 Hz filter.
- **Run:** scripts/train_t5.py defaults (d256, p0.3, lr 5e-4, 6 ep, broad_frac 0.5), seed 0, split v2.
- **Expectations (split v2, fold-averaged, vs V2-REF on the same split):**
  - deployment score V2-REF + 0.03; lift over its own null ≈ 1.6× V2-REF's lift
  - seen subjects ≥ +0.03 over V2-REF
  - zero-shot +0.02
- **LB translation (stated now, tested later):** a T5 submission scores ~0.25–0.27 on LB without sentence context, vs the current 0.227.

## V3 ladder — build arc toward 0.40 local (committed 2026-10-10 ~12:10Z, before any v3 run)
- **Target metric:** split-v2 fold-averaged deployment score (`deploy` = (32·seen + 7·zero-shot)/39, Fold A select Q3→test Q4, Fold B reverse), with each test word given the context the holdout gives it: a [-1.0, +2.0] s window around onset at 50 Hz, masked outside [sentence start, sentence end + 1.0 s]. Isolated holdout words get only [0, 1] s, so a `deploy_mix` = 0.9·context + 0.1·one-second is also reported. Reference: V2-REF 0.284 / 0.288 (null 0.234) with 1 s windows.
- **Rules basis (Rohit, 2026-10-10):** within-subject TTA, within-subject sentence grouping, and the full sentence epoch are allowed. Cross-subject correspondence of holdout rows stays forbidden and is never used.
- **Data changes:** continuous 50 Hz cache with a proper anti-alias FIR (scipy resample_poly), replacing the 5-tap boxcar (audit §5); all words, with sentence boundaries, for every run.
- **Ladder (one trainer, flags; each rung is a separate run on split v2, seed 0):**
  | rung | change | expected deploy |
  |---|---|---|
  | v3a | 1 s window, no context, new cache (parity check) | 0.285 |
  | v3b | 3 s masked window | 0.31 |
  | v3c | + sentence transformer across words (brain-side context) | 0.33 |
  | v3d | + T5-large L12 targets, SigLIP (duplicate-aware), all words | 0.35 |
  | v3e | + per-subject fine-tuning (seen) and TTA (unseen) | 0.37 |
  | v3f | + sentence LM decoding | 0.40 |
- **Known bias, stated now:** local sentences are natural prose (mean 17 words, 5.4 s); holdout sentences are 5–9 vocab-only words, 2–3 s. Local context windows are therefore on average slightly longer than holdout ones; `deploy_mix` is an estimate, not a forecast.
- **Falsifier per rung:** a rung that does not beat the previous one by ≥ 0.01 (above the ~0.01 seed noise) is recorded as NOT DETECTED and the next rung is still run; the ladder is a map.

### Outcome V2-REF (runs finished 2026-10-08 16:45–16:50 EDT after the session OOM; runs/v2ref_s{0,1}.json)
- deploy 0.284 / 0.288 (pred 0.30: ≈, −0.015), null 0.234 (pred 0.25), lift +0.050 / +0.054 (pred +0.05: HIT). seen 0.286 / 0.293 vs zero-shot 0.275 / 0.266 (pred seen ≥ zs by +0.01–0.03: HIT, +0.01–0.03).

### Outcome v3a, v3b (2026-10-10, runs/v3a.json, runs/v3b.json)
| rung | deploy | null | lift | 1 s-only deploy | seen | eval zs | pred |
|---|---|---|---|---|---|---|---|
| v3a (1 s, no ctx, FIR cache) | 0.272 | 0.232 | +0.041 | 0.284 | 0.274 | 0.267 | 0.285: parity on the 1 s score (V2-REF 0.284/0.288) HIT |
| v3b (3 s masked window) | **0.292** | 0.220 | **+0.073** | 0.271 | 0.292 | 0.292 | 0.31: level MISS (−0.02); lift +0.03 over v3a |
- The level under-shoots because the null fell (a less frequency-shaped top-10). Lift is the honest comparison: the 3 s sentence-epoch window adds +0.03 lift. Both runs peak at epochs 0–1 and decay (sel_Q3 0.313 → 0.282 for v3b), so capacity/regularisation is still binding.

### Outcome v3c (2026-10-10, runs/v3c.json)
- deploy **0.328** (pred 0.33: HIT), null 0.217, lift **+0.111** (v3b +0.073). seen 0.330, eval zero-shot 0.319, 1 s-only deploy 0.335 (context across words works even with 1 s windows), deploy_mix 0.329.
- Per-epoch select scores still peak at epoch 1–3 with drift (Q3 0.315 → 0.342 → 0.325 → 0.334); folds disagree on epoch, so a seed/epoch ensemble is a cheap later gain.

### Outcome v3d (2026-10-10, runs/v3d.json) — TAINTED, not banked
- deploy 0.506 (pred 0.35), null 0.216, lift +0.290; fold A (cos readout) 0.542, fold B (ce+cos) 0.470.
- **Ablation (scripts/ablate_v3.py, selected epoch-0 checkpoint):** relative word-onset timing zeroed at inference → Q3 0.49→0.31, Q4 0.54→0.29 (cos readout); single-word segments (no context) → 0.26 / 0.25. **v3c likewise: 0.34→0.26 (Q3), 0.33→0.23 (Q4).**
- **Reading:** the sentence transformer learned word identity from inter-word intervals (word durations), which are stimulus metadata available in the holdout's `word_onsets_s`, not brain signal. Within-subject use is not forbidden by the rules text, but it is not MEG decoding. **Policy from now: timing input OFF (`--rel 0`, the new default); v3c/v3d numbers above are superseded.** Timing-on variants may be reported separately, flagged, if Rohit decides they are admissible.
- Caveat on the ablation: zeroing an input the model trained with is out-of-distribution, so it bounds the timing contribution from above only loosely; the retrained timing-free runs below are the clean measurement.

## V3 ladder, clean (timing-free) — committed 2026-10-10 ~14:25Z, before the runs
| rung | change | expected deploy |
|---|---|---|
| v3c0 | v3b + sentence transformer, no timing input | 0.30 |
| v3d0 | v3c0 + T5 SigLIP (all words) | 0.33 |
Reference v3b (no transformer, no timing) 0.292. If v3c0 ≤ v3b + 0.01, the brain-side context term is NOT DETECTED at this capacity.

### Outcome v3c0 (2026-10-10, runs/v3c0.json)
- deploy 0.293 (pred 0.30), null 0.220, lift +0.073 — equal to v3b (0.292 / +0.073). **Brain-side sentence context: NOT DETECTED** with instrument "2-layer transformer over ≤12-word segments, no timing input, 4 × 1500 steps". The whole v3c gain (0.328) was inter-word timing.

### Outcome v3d0 (2026-10-10, runs/v3d0.json)
- deploy **0.365** (pred 0.33: beat by +0.035), null 0.221, lift **+0.144** (v3c0 +0.073). seen 0.371, eval zero-shot 0.339, 1 s-only 0.356, deploy_mix 0.364. Readouts chosen: ce+cos / ce. Select scores still rising at epoch 3 (Q3 0.366→0.399, Q4 0.353→0.397), so training is under-run.

## V3-D1 — longer training + seeds (committed 2026-10-10 ~15:05Z, before the runs)
- v3d1_s0 / v3d1_s1: v3d0 with 8 epochs × 1500 steps (OneCycle over the longer horizon), seeds 0 and 1. Expect deploy 0.385 each (±0.01 seed), and a 2-seed softmax ensemble +0.01 → 0.395.
- Then v3e adaptation (adapt_v3.py) on the best single run: expect +0.01 to +0.03.

### Ablations on clean v3d0 (epoch-3 ckpt, 2026-10-10)
| condition | Q3 (ce+cos) seen / eval_zs | Q4 (ce+cos) seen / eval_zs |
|---|---|---|
| full segments | 0.398 / 0.383 | 0.379 / 0.341 |
| neighbours replaced by random same-subject windows (structure kept) | 0.307 / 0.292 | 0.317 / 0.266 |
| single words (no context) | 0.288 / 0.266 | 0.293 / 0.249 |
- The context gain (+0.08–0.10) comes from neighbouring MEG, not from segment structure (structure alone adds ≤ +0.02). Clean.

### Outcome v3d1_s0 (2026-10-10, runs/v3d1_s0.json; timing input OFF, verified in the checkpoint args)
- deploy **0.463** (pred 0.385: beat by +0.08), null 0.218, lift **+0.244**. seen 0.464, eval zero-shot 0.456, 1 s-only 0.454, deploy_mix 0.462. Longer OneCycle horizon (8 epochs) lifted the whole curve: epoch-1 select was already 0.43/0.46 vs v3d0's 0.34/0.37.
- **Target 0.40 local: reached on a single seed**, pending the ablations below and seed 1.
- Ablations on the fold-A checkpoint (ep7), ce+cos readout, seen / eval_zs: full Q3 0.447 / 0.415, Q4 0.493 / 0.486; single words Q3 0.252 / 0.241, Q4 0.280 / 0.222; neighbours randomised Q3 0.267 / 0.268, Q4 0.296 / 0.289. Context gain is neighbouring MEG; structure adds ≤ 0.02. **Clean.**

## FINAL-1 — first v3 leaderboard test (committed 2026-10-10 ~10:05 EDT, before the final training)
- **Model:** v3d1 config (3 s masked sentence-epoch windows, sentence transformer without timing, T5 SigLIP + CE, 8 ep × 1500 steps), trained on ALL subjects 0–32 and ALL stimulus (`--final 1`), seeds 0 and 1, last epoch; readout ce+cos, tau 0 (both local folds chose it). Subjects 1–32 use their own trained input layer, 33–39 the mean Broad layer. No TTA in this first file (TTA goes in FINAL-2 once adapt_v3 reports).
- **Holdout inference:** each subject's sentence epochs resampled with the same FIR; each sentence = one context segment; isolated words = 1 s only. Within-subject only.
- **LB prediction:** 0.35–0.42 (local deploy 0.46 is a prose proxy with 38–47/50 word coverage; holdout sentences are shorter and vocab-only, and 10% of rows are isolated words scoring ≈ the 1 s number). Anything ≥ 0.33 beats the d'Ascoli reference; ≥ 0.43 beats MEG-XL. A score < 0.30 would mean the local proxy is not transferring and the next step is diagnosing sentence-vs-isolated rows on the Deep track (subj0 file as calibration, if Rohit okays Deep).

### Outcome v3d1_s1 (2026-10-10, runs/v3d1_s1.json)
- deploy **0.455** (pred 0.385: beat), null 0.220, lift +0.235; seen 0.459, eval zero-shot 0.439, 1 s-only 0.427. Seed spread vs s0: 0.008. **Target 0.40 local reached on both seeds.**

### Outcome FINAL-1 (2026-10-10 10:20 EDT, submissions/v3d1_final_ens2.csv)
- **Public LB 0.4108** (pred 0.35–0.42: HIT, upper half). Previous best 0.2266. Local deploy 0.46 → LB 0.41: the proxy transfers with a ~0.05 discount, consistent with the stated biases (shorter vocab-only sentences, 10% isolated words).
- Against references: d'Ascoli supervised 0.331 beaten; MEG-XL 0.4296 not yet (−0.019). No TTA, no sentence LM, no adaptation in this file.

### Outcome ensemble + v3e adaptation (2026-10-10, runs/ens_v3d1.json, runs/v3d1_s0_adapt.json)
- 2-seed softmax ensemble: deploy **0.475** (singles 0.463 / 0.455), null 0.221 → +0.01 (pred +0.01: HIT).
- Adaptation on v3d1_s0: seen fine-tune (A + private head, 0/100/300 steps) and zero-shot TTA (CORAL α ∈ {0, .5, 1} × entropy steps ∈ {0, 30}) all within ±0.01 of base; deploy 0.464 vs base 0.463. **Label: NOT DETECTED** with these instruments (pred +0.01–0.03: MISS). The jointly trained per-subject input layers already carry the subject-specific information; unlabelled TTA has no measurable room on this model.

## V3-E map — model levers after the LB test (committed 2026-10-10 ~10:45 EDT, before the runs)
All vs v3d1_s0 (seed 0, deploy 0.463); seed noise ≈ ±0.01, so a lever counts as detected at ≥ +0.015.
| arm | change | expected deploy |
|---|---|---|
| v3e1 | 12 epochs (OneCycle over the longer horizon) | 0.475 |
| v3e2 | encoder d 192, embedding D 384 | 0.475 |
| v3e3 | T5 loss weight 2.0 | 0.468 |
| v3e4 | window [−1.5, +2.5] s (LB_WIN env; 200 samples) | 0.473 |
Detected levers are combined into FINAL-2 (3 seeds, all data). FINAL-2 LB prediction will be stated when that config is chosen.

### Outcome v3e1 (2026-10-10): deploy 0.454 (pred 0.475: MISS; −0.009 vs v3d1_s0). Longer training NOT DETECTED.
### Outcome v3e2 (2026-10-10): deploy 0.427 (pred 0.475: MISS; −0.036 vs v3d1_s0). More capacity hurts cross-subject generalisation.
### Outcome v3e3 (2026-10-10): deploy 0.468 (pred 0.468: HIT; +0.005 vs v3d1_s0, within noise). Not a detected lever.

## FINAL-2 — ensemble (committed 2026-10-10 ~13:40 EDT, before the runs)
- Map verdict: no single lever in V3-E is detected (12 ep −0.009, capacity −0.036, T5 weight 2 +0.005, wider window tracking below). The v3d1 configuration stands.
- **FINAL-2 file:** softmax average of five all-data models: final_v3d1 seeds 0, 1 (existing), seeds 2, 3 (new), and one T5-weight-2 model (seed 0; equal local quality, different loss balance for diversity). Readout ce+cos, tau 0, no TTA.
- **LB prediction:** 0.42–0.44 (FINAL-1 0.4108; local ensembling gives +0.01 per 2 seeds). ≥ 0.43 beats the MEG-XL reference.
### Outcome v3e4 (2026-10-10): deploy 0.441 (pred 0.473: MISS; −0.022). Wider context window hurts; [−1, +2] s stands.

### Outcome FINAL-2 (2026-10-10 13:05 EDT, submissions/v3_final2_ens5.csv)
- **Public LB 0.4284** (pred 0.42–0.44: HIT). +0.018 over FINAL-1 (0.4108). MEG-XL reference 0.4296: −0.0012. d'Ascoli 0.331: beaten by +0.10.

## FINAL-3 — 8-model ensemble (committed 2026-10-10 ~13:10 EDT, before the runs)
- Add final_v3d1 seeds 4, 5, 6 to the FINAL-2 five; same readout. **LB prediction 0.43–0.445** (diminishing returns: 2→5 models gave +0.018). This is the last autonomous step of the build arc; further gains need a new idea, not more seeds.

## Three cheap tests (committed 2026-10-12 ~09:45 EDT, before the runs)
1. **FS100** — v3d1 recipe on a 100 Hz cache (0–50 Hz content instead of 0–25 Hz), seed 0. Expect deploy within ±0.01 of v3d1_s0 (0.463); detected only if ≥ +0.015.
2. **MV1** — v3d1 recipe + a second SigLIP head to wav2vec2-base layer-8 acoustic targets (all words with audio, ~75% of rows), weight 1.0, seed 0. Expect +0.01; detected if ≥ +0.015. Earlier B7 null was on the old classifier where targets were not yet the lever.
3. **LM-rerank** — post-process of the FINAL-2 holdout probabilities: within each holdout sentence, exact forward–backward marginals under the external trigram LM over the 50 words (Gutenberg vocab-only runs), emission p^β with β = 1, δ = 0 (the LM2-v2 dev choice), isolated words untouched. One LB submission. Expect +0.00 to +0.02 over 0.4284; a drop would mean the prose-trained prior misfits the constructed holdout sentences.

### Outcome LM-rerank (2026-10-12, submissions/v3_final2_lmrerank_b1.csv)
- **LB 0.4344** vs 0.4284 for the same models without rerank: +0.006 (pred +0.00 to +0.02: HIT). Changed top-1 on 31% of sentence rows. First file above the MEG-XL reference (0.4296). β/δ not tuned on LB.

### Outcome MV1 (2026-10-12, runs/mv1.json)
- deploy 0.468 (pred +0.01 → 0.473; detected only if ≥ +0.015): +0.005 vs v3d1_s0. **NOT DETECTED.** Early epochs led by +0.03 on select scores but the fold-selected epochs converge with the T5-only model. Acoustic targets add no measurable information over contextual text targets at this evidence level (62% of rows have audio). Not used for a final retrain.

### Outcome FS100 (2026-10-12, runs/fs100.json)
- deploy 0.411 (pred within ±0.01 of 0.463): **−0.052. Worse.** Keeping 25–50 Hz content adds noise and input size without decodable information; the 50 Hz (0–25 Hz) representation stands. The "high-gamma" hypothesis from the brainstorm is refuted at the only band this data can test.

### Summary of the three tests
| test | result | kept? |
|---|---|---|
| LM rerank (LB) | 0.4284 → **0.4344** (+0.006) | yes, best file |
| MV1 acoustic targets (local) | +0.005 | no (noise) |
| FS100 0–50 Hz (local) | −0.052 | no |

## Upside arc (committed 2026-10-12 ~14:30 EDT, before any run) — Rohit: do not close with days left
Refuted-by-our-own-runs and not re-run: hard 50-way classifier (Oct 4–8, 0.22), fixed spatial filters (learned per-subject A ⊇ them; CORAL ±0.01), ridge alignment to subj0 (P1a/P1b at floor), softmax sharpening (monotone, no effect on a ranking metric).
1. **MEGXL-F** — MEG-XL (pnpl/MEG-XL) frozen features on the 8 GB card, short context (holdout sentences are 3–4 s). Step 1 is a memory/feasibility test only; if features extract, train our T5-target head + sentence transformer on them (split v2). Expect: 50% chance it does not fit or is unusable at short context; if it works, deploy ≥ 0.50 local.
2. **LM-prior-2** — n-gram prior fitted on LLM-generated vocab-only 5–9-word sentences (constrained to the 50 words) instead of Victorian prose; β chosen on a local simulation built from the same sentence family. One LB submission. Expect +0.01 to +0.03 over 0.4344.
3. **SELF-TRAIN** — per-subject pseudo-label fine-tuning of the full per-subject path on the subject's own confidently decoded holdout sentences; validated first on EVAL subjects' test block in split v2. Expect 0 to +0.02; a local loss means confirmation bias and it is dropped.

## BET A — Deep track (committed 2026-10-12 ~16:40 EDT, before any Deep submission or run). Bet B dropped by Rohit; Bet C runs as a background agent (docs/extmeg/).
- **DEEP-0 (calibration):** the existing FINAL-2 five models predict subj0's holdout (own trained input layer), ce+cos τ0, no rerank. LB prediction **0.55–0.65** (subj0 has 80 h; our local 1 s subj0 score was 0.50 before the v3 target change; d'Ascoli reference 0.732).
- **DEEP-1:** subj0-weighted v3 training (low broad_frac, deeper sentence transformer), selected on subj0's own Q3/Q4 blocks (fold A/B), seeds, + matched-corpus LM prior. Expect local subj0 deploy ≥ 0.70 and LB 0.68–0.75.

### Outcome DEEP-0 (2026-10-12, submissions/deep_v3_final2_ens5.csv)
- **Deep LB 0.5662** (pred 0.55–0.65: HIT). Broad-trained ensemble, subj0's own input layer, no Deep tuning, no rerank. Reference d'Ascoli 0.732; public top 0.765.

## DEEP-1 map (committed 2026-10-12 ~17:05 EDT, before the runs). Selection and test = subj0's own Q3/Q4 blocks (≈1.3k vocab words each, so ±0.02 noise).
| run | config | expected subj0 local |
|---|---|---|
| deep_a | v3d1 recipe (broad_frac 0.5, 2-layer ctx), selected on subj0 | 0.60 |
| deep_b | broad_frac 0.1, 4-layer ctx | 0.64 |
| deep_c | subj0 only (broad_frac 0), 4-layer ctx | 0.62 |
- If deep_b ≥ deep_a + 0.02 the subj0-weighting and depth are kept for the final Deep retrain (3 seeds, all data) → DEEP-2 submission.
