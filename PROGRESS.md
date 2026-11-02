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
