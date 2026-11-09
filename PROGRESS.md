# PROGRESS.md — leakage paper session (append-only, dated; times US Eastern)

## 2026-11-02 — session start (brief: BRIEF_LEAKAGE_PAPER_2026-11-02.md)

### Job 1 — private leaderboard numbers: DONE (~07:40 EST)
- Pulled public+private for all 35 Broad / 26 Deep submissions → `results/lb_private_20261102/` (README there).
- Final private: Broad 0.67846, rank 17/53 (public rank 8). Deep 0.69704 ranked, 26/91 (public 15). Memory "8th place" is now superseded.
- Public and private splits disagree on the self-learned sentence prior: Broad κ-curve public interior optimum at κ20 (+0.017), private strictly monotone, every prior hurts (κ20 −0.036, κ5 −0.081 vs κ∞). Deep κ20 −0.029 private. Class balancing / offset fix / trigram agree on both splits.
- Per-subject input layer (FITA-3): public +0.001, private +0.030 (confounded with prior on both files). Deep wider encoder public +0.008 → private −0.017. Deep Sinkhorn +0.010 → +0.051.
- Diagnostic: the prior changes isolated-word rows' top-10 more than sentence rows (3.1 vs 1.4 words/row at κ20) via the joint Sinkhorn offsets. Split mechanism OPEN; needs the organisers' row-level Usage column (Rohit's call to ask).
- Paper: abstract (five mechanisms), §1, §4.3, §4.4, §4.6, new §4.7, §5, §8 updated; Figure 2 `docs/figures/public_vs_private.png`.
- Implication for Job 2: the "transfer failures" of §4.4 are now partly reversed on private; seeded CIs must be local-instrument CIs for each arm (the LB is single-shot).

### Job 2a — timing-ablation seeds: PENDING (launched ~07:55 EST)
- `scripts/queue_ci_timing.sh`: arms v3b / v3c0 / v3c_rel / v3d0 / v3d_rel × seeds 0,1,2 → `runs/ci_v3*_s*.json`, `logs/ci/`. Smoke: 9.8 GB RSS with LB_MMAP=1, ~8 min/run, ~2 h total.

### Job 4 — reference verification: DONE (~07:55 EST)
- `docs/writeups/references_verified.md`: 13 entries, each fetched (arXiv API / Crossref / publisher). Fixes: Cuturi title; PNPL = Mantegna et al. 2026a (arXiv:2609.03231); MEG-XL = Jayalath & Parker Jones 2026 (arXiv:2602.02494); Packard page range dropped (not on publisher page). Added LibriBrain, LibriBrain100, Roelofs 2019, Blum & Hardt 2015, Dwork 2015 (for §4.7).
- Job 2a note: machine load ~42 (other sessions), training step rate fell ~20x after warm-up; CPU/data bound. Re-projecting after first run.

### Job 2b — §4.4 transfer seeds: PENDING (queued ~08:00 EST, starts when 2a ends)
- `scripts/queue_ci_transfer.sh`: deep_c s1,s2 + stretch s2 (same-day; s0/s1 exist) · lso1/lso_dw s1,s2 + PX eval · fita_map on deep_c s1/s2. Outputs runs/ci_*.json, logs/ci/.
- Caveat already visible: the day-novel PX block behind "width +0.041" has only 141 rows (logs/proxy_calib_px_lso_dw.log); SE of a single score ~0.03.
- 08:20 EST: first ci_v3b_s0 attempt stalled ~30 min in epoch-0 eval at load ~42 (torch/OpenMP default 16 threads, oversubscribed by other sessions). Killed (log kept as logs/ci/ci_v3b_s0.stalled_omp16.log); both queues relaunched sequentially with OMP_NUM_THREADS=4.
- Rohit's binding rule received 08:15 EST: no Co-Authored-By trailers, no AI-vendor/tool names anywhere. Checked bd9c50b, a3c8c13, fe3607b: clean, nothing to amend.
- 08:28 EST: measured 741 s/epoch for v3b (eval 44 s). Projection: job 2a ~12.5 h (ends ~21:00 EST), job 2b ~11 h more. Overrun notify threshold: 2a not done by 2026-11-03 09:00 EST.

### Job 5 — build: pipeline in place (~08:35 EST), final build waits on job 2/3 numbers
- docs/writeups/paper/build.py (+ header.tex, repro_appendix.md) → leakage_paper.pdf via pandoc 3.1.3 + xelatex; 9 pages, Figures 1–2 embedded. Fig 2 labels crowd; fix at final build.

### Job 3 — holdout-format timing + timing×MEG: IN PROGRESS
- 08:40 EST: STOPPED my cross-subject consensus-label analysis (it infers held-out labels; PNPL paper discourages this). No outputs kept; script parked in private/ (gitignored) as HELD_FOR_ROHIT. Rohit notified.
- Label-free substitute running: `docs/writeups/analyses/timing_holdout_format.py` (subj0 runs of ≥5 vocab words framed as holdout epochs; Q3/Q4 test; 3 seeds) and `timing_meg_fusion.py runs/deep_c_ep6.pt` (CPU; MEG × timing^w, w cross-fit Q3↔Q4).
- Checked: no orphan words (sentenceidx<0) in subj0 cache, so the 0.632 baseline's sentence grouping is sound.

### Job 6 — venue packet: DONE (~08:45 EST)
- `docs/writeups/paper/SUBMISSION_CHECKLIST.md`. Flags for Rohit: courtesy note to PNPL; consensus analysis; "8th place" wording on public pages; bioRxiv generative-AI disclosure vs the no-vendor rule; bioRxiv must precede arXiv; arXiv endorsement needs a personal endorser (no prior papers).
- 08:43 EST: ci_v3b_s0 deploy 0.2924 (original v3b 0.292: reproduces under current code). 23 min/run at current load → ladder ~6 h.
- Job 3 instrument problem: in holdout format the local Q3/Q4 blocks give only 74+27 scorable rows (sent59) and 6+21 (runs≥5 vocab); deep_c fusion result there (MEG 0.680, timing 0.642, fused 0.685) is UNINFORMATIVE at that n. timing_holdout_format.py removed (superseded).
  Re-run with real n: timing_meg_fusion.py on the leave-day-out checkpoint lso1 (both shared sessions MEG-novel → ~4x rows, day-novel), and timing_only_cv.py (session-grouped 5-fold CV over all subj0 data, sent59 + prose formats, 3 seeds). Running on CPU.
- 08:50 EST: timing_only_cv sent59 = 0.5724 ± 0.0022 (3 seeds, session-grouped 5-fold); prose format and lso1 fusion still running (fusion log currently stale from the masked-session bug run).
- 09:00 EST: timing_only_cv done → §4.1 updated (prose 0.593, holdout-epoch 0.572; recommendation 2 floor 0.63→0.57). lso1 fusion running.
- 09:10 EST: lso1 fusion (sent59) n=107+52 rows: MEG 0.774, fused 0.787±0.010, timing 0.656 — UNINFORMATIVE at that n. Added 'epochall' format (all sentences ≥2 words, epoch framing) for ~15x rows; rerunning lso1 and deep_c fusion.

### Job 3 — DONE (09:35 EST)
- epochall fusion (≈1300 rows/block, 3 seeds, cross-fit w): day-novel lso1 MEG 0.693 → fused 0.705±0.011; same-day deep_c 0.757 → 0.770±0.012. Timing alone 0.59–0.63. Gain ≈+0.01, positive in 10/12 cells, ≈1 seed SD: timing is largely redundant with a strong MEG decoder. §4.1 updated. sent59/runs5 formats too small (≤160 rows), reported in JSON only.

## 2026-11-02 11:25 EST — Job 2a done (timing ladder, 3 seeds)
15 runs (5 arms × seeds 0-2) plus inference-time ablations on the six timing checkpoints. `docs/writeups/analyses/seeded_cis.py` → `seeded_cis.json`.
- no context 0.309±0.015; 2-layer no timing 0.300±0.009; 2-layer + timing 0.325±0.022; T5 no timing 0.374±0.014; T5 + timing 0.493±0.014.
- Holds: transformer without timing adds nothing (−0.009, CI −0.032..0.014). Timing at T5 rung +0.119 (CI 0.073..0.166, ≥0.106 every seed).
- Weakened: timing gain at the 2-layer rung +0.025, CI −0.047..0.097; the single-run 0.29→0.33 was the favourable seed. §4.1 and abstract rewritten to say so.
- Zeroing timing at inference (T5 ckpts, seen subjects, cos readout): 0.500→0.308, 0.520→0.290; single-word 0.260/0.267.
- Fixed a wrong clause: the ladder was scored mostly on seen subjects (32+7), not held-out ones.
Commit cb248d9. Job 2b (transfer seeds) now training on the GPU.

## 2026-11-02 12:55 EST — Job 2b partial: time-stretch seeds
deep_c seeds 0/1/2: 0.705, 0.703, 0.735; stretch: 0.737, 0.739, 0.719. Paired gain +0.017 ± 0.029 (CI −0.054..0.089); seed 2 reversed sign.
The "+0.034 in both of two seeds" claim is weakened: the local gain itself is not established. §4.4 bullet and abstract rewritten (f1b8e00).
Width (PX) and FITA arms still training.

## 2026-11-02 15:30 EST — Job 2 done (all seeded CIs)
Transfer queue finished 15:09 EST. `seeded_cis.py` → `seeded_cis.json` now covers all arms.
- Wider encoder, day-novel PX (base+sk, 141 rows): seeds 0/1/2 paired +0.041, −0.019, +0.035 → +0.019 ± 0.033 (CI −0.064..0.102). Weakened: never established locally.
- Per-subject input layer: 36/36 configurations positive over 4 checkpoints, +0.030..+0.068. Holds; private split +0.030 agrees.
- Time-stretch: +0.017 ± 0.029 (earlier entry).
- New reading in §4.4/§4.7 and abstract: the one local gain robust across seeds is the one the private split confirmed; the two fragile ones (stretch, width) were never established.
Commits abc8e79, 8aed9a5. Job 5: Figure 2 labels fixed (17c56d7), appendix placeholders filled, draft note updated, PDF rebuilt (9 pp). Independent number-trace audit running.

## 2026-11-02 16:10 EST — Job 5: number-trace audit and fixes
Independent audit of every number in the draft against repo files: ~170 verified. Fixed (commit 15b6eaf):
- Correction to the 15:30 and 11:25 entries: the single-run 0.33 (v3c) equals seed 0, the middle of 0.30–0.35, not the favourable seed.
- Fusion row counts (1,300 day-novel / 650 same-day); MEG comparison range now from the fusion files (0.69 / 0.76; margins 0.49–0.56).
- "Prose trigram" → vocabulary-only trigram; Deep base 0.57 not 0.66; "8 October / twenty teams" → by 13 Oct (PREDICTIONS) and 22 teams at close (public CSV).
- Unsourced "worth about 0.2" replaced by measured 0.119 local / ~0.01 fusion. Rec. 2 floor labelled as 5–9-word epoch framing (0.59 prose).
- New script prior_topk_churn.py reproduces 3.14/1.42 (κ20) and 4.77/2.28 (κ5) row churn.
- Appendix: simulator row → sim_sweep.py (de10a30), experiment logs beyond PREDICTIONS.md named; "164 ideas" → "about 160".

## 2026-11-02 16:30 EST — All six jobs done
Appendix table reflowed (bf63048), checklist updated. PDF: docs/writeups/paper/leakage_paper.pdf, 10 pages, builds clean.
Open for Rohit (DECIDE items in SUBMISSION_CHECKLIST.md): PNPL courtesy note, consensus analysis (held), "8th place" wording, bioRxiv AI-disclosure vs no-vendor rule, venue order, pushing the scripts. Nothing pushed, posted or submitted.

## 2026-11-03 07:00 EST — Adversarial pass (referee + novelty), noise control started
Rohit asked: contributive for bioRxiv? Two independent agents: hostile referee and literature check.
- Referee verdict: do not post as is. Verified: §4.2 "signature" argument self-refuting (our clean Broad 0.722 vs top 0.729); §4.7 SE 0.004 was an estimate, not measured; "private confirmed the robust gain" pair confounded 5 ways; omitted that local prose predicted the prior's private sign (LEDGER T18) — Rec. 4 inverted.
- Literature: Jayalath & Parker Jones, arXiv 2609.40359 (30 Sep 2026, organisers) already show the timing shortcut on this data via overlapping 3 s windows (synthetic signal ≈ MEG). Our "timing-free" arms use overlapping 3 s windows clipped to the sentence span, so they may still carry timing. Also missing: Elvers et al. 2026 (PNPL 2025 retrospective; day-held-out design), Zhang et al. 2605.24524, Jo et al. 2024, Roelofs already flags non-iid splits.
- split_null.py (28519ab): Broad prior files gap −0.024..−0.102 vs prior-free −0.009..−0.022 (complete separation, dose-response in κ); Deep no separation.
- Started scripts/queue_noise_control.sh: LB_NOISE=1 (shared noise, overlap preserved) / =2 (independent) on d0, b, d_rel recipes. src/cont.py gains the LB_NOISE switch (default 0, unchanged behaviour).

## 2026-11-03 08:15 EST — Reorientation to a flagship method; cheap screen queued
Rohit: headline has prior art (2609.40359) → need a new method that beats SimpleB2T. Plan: docs/writeups/FLAGSHIP_PLAN_2026-11-03.md.
- Installing/running the external SimpleB2T code was refused by the permission check (external code): awaiting Rohit's explicit approval. Their session indexing cannot be mapped to our files without their code.
- noise1_d0_s0 = 0.272 (chance 0.20; real MEG 0.374): our "timing-free" T5 recipe leaks ~0.07 via window overlap / clipped valid masks (the mask is an input channel).
- Cheap screen (scripts/queue_clean_screen.sh, behind the noise queue): clean isolated decoder (LB_WIN=0,3, LB_NOCLIP=1, ctx 0), leave-one-day-out, scored by clean_screen.py for base / Sinkhorn / AdaBN / both, balanced top-1 and top-10; noise controls of the same recipe.

## 2026-11-03 08:45 EST — Kaggle late submissions authorised; timing-clean ladder queued
Rohit authorised Kaggle late submissions to substantiate a methodological gain. Holdout check: every sentence epoch
has at least 1.018 s after its last word onset and isolated words are 1 s epochs, so a fixed [onset, onset+1 s)
window fits every row with no boundary clipping (the geometry channel that the noise controls showed leaking: shared
noise scored 0.256 single-window and 0.272 with context, chance 0.20). Queued `scripts/queue_kaggle_clean.sh`:
3 seeds per track of the clean isolated decoder + a shared-noise twin per track; CSVs for base, +AdaBN,
+AdaBN+Sinkhorn (T=2 fixed a priori), +Sinkhorn, noise twin. Deltas will be read against split_null.json.
Caveat for the claim: Kaggle reports top-10 balanced accuracy, not the top-1 of Jayalath & Parker Jones, so the
Kaggle ladder substantiates the method deltas and the noise-twin floor; the top-1 comparison stays local.

## 2026-11-03 11:45 EST — noise controls complete; clean screen first result
Noise controls (top-10 balanced, chance 0.20; real-MEG seed means in brackets): shared noise d0 0.272 (0.374),
single-window b 0.256 (0.309), INDEPENDENT noise d0 0.288, shared noise d_rel 0.460 (0.493). The leak in the
"timing-free" arms is window geometry (valid mask clipped at sentence bounds), not shared samples; with relative
onsets fed, noise alone reaches 0.460, so nearly all of d_rel's score is timing.
Clean screen (fixed 3 s window, no clipping, ctx 0, leave-day-out sub-0 Sherlock1 ses-11/12, ~1,300 rows each,
balanced top-1, chance 0.02): base 0.0725, +Sinkhorn 0.0792, AdaBN 0.0606, AdaBN+Sinkhorn 0.0837 (top-10 0.458,
0.472, 0.425, 0.447). Top-1 SE about 0.007 per session, so Sinkhorn (+0.007) and AdaBN+Sinkhorn (+0.011) are about
one to one and a half SE: suggestive, not established. AdaBN alone lowered both sessions; it was estimated with dropout
active, so it is being rescored with dropout off (standard AdaBN; tag clean_iso_s0_dropoff). The Kaggle ladder already
uses dropout off. Training CE of the synthetic-timing twin (kc_deep_syn3_s0, 2.83) is BELOW the real-MEG model
(kc_deep_s0, 3.24): onsets inside a 1 s window are highly informative when timing is easy to read.

## 2026-11-03 12:35 EST — correction: read noise controls against the permutation null, not 0.20
In the deploy metric the label-permutation null is 0.207 to 0.227 (only ~47 of 50 words occur per block), so the
lift over the null is the honest measure. Lifts: shared noise d0 0.061, independent noise d0 0.066, single-window b
0.029, d_rel 0.252; clean 3 s fixed window on shared noise 0.028 (selection over epochs/readouts may explain part).
Real MEG: b 0.073, d0 0.144, d_rel 0.290, clean_iso_s0 0.293. The clean fixed 3 s window scores as high as d_rel on
real MEG, which points at in-window onset responses of following words (the duration channel of Jayalath & Parker
Jones). Shared noise cannot test that; queued `scripts/queue_clean_syn.sh` (synthetic onset-locked twin of the 3 s
screen). Standard AdaBN (dropout off) still lowers top-1 on the held-out day (0.0602 vs 0.0725 base).

## 2026-11-03 12:50 EST — first late submission: synthetic-timing twin scores 0.580 private on Deep
Late-submission limit is 99/day. kc_deep_syn3_fake (model trained on noise + onset-locked kernel; holdout MEG replaced by
noise + kernel at the holdout's own word onsets; isolated fixed [onset, onset+1 s) window, no context, no clipping):
public 0.5915, private 0.5805 (our final Deep entry: 0.7275 / 0.6804; chance 0.20). With no brain signal at all, a
1 s isolated window earns 0.58 from the onsets of following words inside it. The kernel (2 noise s.d.) makes timing
easier to read than real MEG does, so this is an upper bound on the timing channel, but it shows the "isolated"
framing on the Kaggle benchmark is dominated by word duration. Every clean-ladder claim must be read against it.

## 2026-11-03 18:00 EST — twins on the 3 s screen and on Broad
Clean 3 s screen, held-out day, balanced top-1 (chance 0.02) / top-10: shared-noise twin 0.023 / 0.221 (at chance:
the fixed geometry is clean); synthetic-timing twin 0.159 / 0.659; real MEG 0.073 / 0.458. Timing alone, made easy
to read, beats real MEG in the same isolated 3 s window; compare Jayalath & Parker Jones isolated MEG 9.5% and
synthetic-without-overlap 5.8%. The isolated condition is not timing-free while the window outlasts the word.
Kaggle Broad twin (Broad recipe, broad_frac 0.5, per-subject A): public 0.3066, private 0.3741, far below Deep's
0.58. Running the Deep twin model on the Broad fake holdout (identity A) to separate recipe from track.

## 2026-11-03 19:35 EST — deconvolved caches built; artifact fix
cont_dc (real MEG) and cont_syn_dc (synthetic-timing twin) written, 226 runs each, in /mnt/ssd4tb/data/libribrain_cache.
The generic onset response removes 0.0-0.5% of real MEG variance and 6-14% of the synthetic stream (its kernel).
Two source runs (sub-0 ses-3 Sherlock1, ses-9 Sherlock3) hold a few hundred inf samples and >30,000 artifacts; the first
build turned them into ~3 M non-finite samples. src/deconv.py now zeroes non-finite samples and fits the kernel on data
clipped at 20x the median |amplitude|; the two runs were rebuilt (0 non-finite). On a clean run the new kernel changes
the residual by at most 0.08 (data s.d. 26.7, fp16 rounding), so the other 224 runs were kept.

## 2026-11-03 19:50 EST — wasted late submission (degenerate), fixed
kc_deepmodel_broad_syn3_fake (Deep synthetic twin on the Broad fake holdout, --identA 1) predicted 2 distinct top-1
words (87% one word): public 0.158, private 0.201, uninformative. The Deep model never trained input layers for
subjects 1-32, so identity/mean A do not match its learned sensor map. Added --A_from k (every subject uses subject
k's layer); rerunning with --A_from 0. Rule from here: check the distinct-top-1 diagnostic in the prediction log
before every submission.

## 2026-11-09 11:40 EST — stage 0, light CPU: timing floor without a network
Held-out day check: Sherlock1 ses-11/12 run-2 (book two, chapter four) shares no 5-gram with any training run, so the
clean screen is already held-out-day AND held-out-text (Zhang et al.'s stimulus-identity condition).
`A/timing_floor.py` (event files only; boosted classifier on the onsets inside [onset, onset+W)), balanced over 50
words, mean of ses-11/12: W=0.5 s top-1 0.109 / top-10 0.554; 1 s 0.113 / 0.568; 2 s 0.112 / 0.606; 3 s 0.136 / 0.588.
Packard (1990) cross-fitted conditional-set information from the next-onset cell: 0.53-0.66 bits held out.
Reading: exact onset timing inside a fixed window predicts the word better than real MEG does in the same window
(clean_iso 3 s: 0.073 / 0.458) and better than Jayalath & Parker Jones' isolated MEG (9.5% top-1). The synthetic
twin CNN (0.159) sits near this floor, as it should. Whether real MEG makes this timing readable is the next test
(`A/onset_readability.py`, running).

## 2026-11-09 12:30 EST — stage 0a: real MEG barely carries the timing; the Nov 3 twin overstated it ~15x
`A/onset_readability.py`: generic onset TRF fitted on 6 subject-0 runs, projected to one virtual channel, boosted
classifier trained on 28 runs, tested on the held-out day/text (balanced over 50 words, chance 0.02 / 0.20).
| stream | TRF peak (resid s.d.) | held-out var explained | 1 s top-1 / top-10 | 3 s top-1 / top-10 |
| real MEG | 0.135 | 0.10% | 0.024 / 0.281 | 0.030 / 0.283 |
| twin, amp 2 (Nov 3 kernel) | 1.97 | 11.4% | 0.099 / 0.594 | 0.122 / 0.543 |
| twin, amp 0.5 | 0.49 | 0.79% | 0.091 / 0.500 | 0.073 / 0.467 |
| twin, amp matched (0.135) | 0.136 | 0.05% | 0.048 / 0.316 | 0.029 / 0.303 |
Exact onsets (timing_floor.json): 1 s 0.113 / 0.568, 3 s 0.136 / 0.588.
Reading: the Nov 3 synthetic kernel was ~15x the real onset response, so its scores (0.159 top-1 on the screen, 0.580
private on Kaggle Deep) are upper bounds far above a realistic floor. At matched amplitude, timing alone gives about
0.03-0.05 top-1 and 0.30 top-10 through this one-channel readout (white noise, so still optimistic; a 306-channel CNN
may read more). The clean 3 s real-MEG decoder (0.073 / 0.458) sits above that, so most of its score is probably
not timing; confirming needs the matched-amplitude CNN twin (GPU, ~25 min). C1 as a headline weakens: fixed-window
isolated decoding is close to timing-clean at realistic amplitude. Single seed, ~1,300 rows per session (top-1 SE ~0.006).

## 2026-11-09 12:25 EST — noise-equation pilot, decoding rung (linear decoder, CPU, peak 3.0 GB under a 4 GiB cap)
`A/noise_eq_pilot.py`, seed 0: softmax regression on 64 spatial PCs x 10 bins of [onset, onset+1 s), subject 0, test =
held-out day and text (ses-11/12 run-2). Balanced top-1 / top-10 (chance 0.02 / 0.20):
| train runs (rows) | uniform | whiten (Sigma^-1/2) | state precision | state permuted | state->test CE corr |
| 15 (24k)  | 0.031 / 0.313 | 0.033 / 0.309 | 0.026 / 0.297 | 0.027 / 0.296 | 0.068 |
| 38 (69k)  | 0.047 / 0.352 | 0.049 / 0.354 | 0.043 / 0.354 | 0.045 / 0.345 | 0.058 |
| 76 (139k) | 0.049 / 0.333 | 0.038 / 0.349 | 0.044 / 0.346 | 0.046 / 0.348 | 0.043 |
| 152 (300k)| 0.045 / 0.326 | 0.039 / 0.347 | 0.039 / 0.327 | 0.035 / 0.340 | 0.038 |
Grade: null in this operationalization, and underpowered. The linear decoder saturates by ~70k rows and sits at the
amplitude-matched timing floor (0.03-0.05 / 0.30), so it cannot show a precision gain or fit an exponent alpha (the
curve is flat after 25%). Whitening: top-10 +0.02, top-1 -0.01 at full data, within noise (top-1 SE ~0.006). State
weights vs their permutation: no difference. Prediction rung: pre-onset state predicts the decoder's per-word loss on
the unseen day weakly (r = 0.04-0.07, ~2-3 SE at n = 2,672), driven by log pre-onset power (positive coefficient,
stable across budgets). OPEN, not closed: the test needs a decoder well above the floor (the CNN, 0.073 / 0.458) with
the precision weights inside its loss, plus the matched twin; that needs a GPU slot. Feature cache 0.4 GB in
/mnt/ssd4tb/data/libribrain_cache/noise_eq.
