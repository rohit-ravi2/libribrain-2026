# TUNING — every tunable knob in the final pipelines (2026-10-15, ~21:40 ET)

**Scope and constraints.** CPU only, no submissions, no edits to existing repo files. All new code, banks and candidate files are in `docs/home_stretch/tuning/`.

**Metric.** Top-10 balanced accuracy, chance 0.20.

**Goal.** Both tracks above 0.80 on the leaderboard. Current bests: Broad 0.7096, Deep 0.7161.

## Bottom line

Even with every knob below set right, the expected total is about **Broad +0.003 to +0.010** and **Deep +0.005 to +0.020**. Tuning does not reach 0.80 on either track.

Tonight's tests found three real problems:

1. **The isolated-word rows are decoded at chance in both tracks.** These are 92 rows per subject, 9.6% of rows. For Deep, a dedicated 1 s decoder that already exists (`runs/b0.pt`) recovers them. In a holdout-shaped simulation this is worth **+0.008 to +0.012 on Deep**. The files are built, on CPU.
2. **FITA-3 does not do what the ledger says (A16).** The temporal-lobe base is applied only to the 7 no-data subjects. The 32 fitted subjects are still fitted from the identity matrix.
3. **The low-rank cut-off is mis-set.** "+0.010 at 25%" referred to 25% of the local fit pool, which is about 6 min of data. It did not refer to the 25%-supervision group, which has 12 min. Because of this mix-up, the 24-min group (subjects 13–22) is on low-rank, where local measured −0.007.

Fixing 2 and 3 together is worth about **Broad +0.002 to +0.005**. It needs one GPU prediction run, and the script is ready.

Eight other knobs were checked tonight and are dead or harmful. Don't spend slots on them; they are listed under "Dead knobs" below.

---

## 1. Ranked knobs with plausible ≥ +0.005 gain (or a mis-setting)

| rank | knob | current | recommended | evidence | expected LB | instrument / cost |
|---|---|---|---|---|---|---|
| 1 | **Isolated-word rows, Deep** (source == word, 92/960) | v3 3 s-context model fed a lone 1 s window, then SLP passes it through and joint Sinkhorn T3 | Replace those rows' raw posteriors with **B0 + AdaBN** (`runs/b0.pt`, 1 s subj0 WordCNN, BN statistics from subj0's own 960 holdout windows), w = 1, then SLP κ20 + Sinkhorn T3 as usual | See §2. Local subj0 Q4 (held out for both models), lone 1 s windows: v3 **0.231** (null 0.190), B0 **0.449**. Holdout-shaped simulation (868 context + 92 isolated rows, joint T3, 30 draws): **+0.0104 ± 0.003** (natural frequencies), **+0.0083 ± 0.002** (balanced); isolated rows 0.14 → 0.40. On the real holdout, v3 puts "good" as top-1 on 25.5× its uniform share of isolated rows, and its top-1 agrees with B0 on 0/92 rows. B0 agrees with v3 on sentence rows above chance (B0 top-1 in v3 top-10: 0.36 vs 0.20), so B0 keeps signal on the holdout day | **Deep +0.005 to +0.012** (discounted for day drift) | Deep LB can't read it (only ~28 public isolated rows). CPU, 1 min. Files: `tuning/final7_deep_bn_isoB0bn_k20.csv` (B0+AdaBN), `…_isoB0w1.0_k20.csv` (B0 raw), `…_isoB0w0.5_k20.csv` (mixed) |
| 2 | **FITA fit base** for fitted subjects 1–32 | identity (`predict_holdout_v3.py` initialises the fit at I and ridges it to I; `--temporal_base` only reaches subjects without labels) | Temporal k3 as init and ridge anchor: `tuning/predict_holdout_fita4.py --fit_base temporal` | FITA × temporal maps (deep_c, 12 subjects × 2 blocks, + Sinkhorn): plain fit 0.6757 (f0.25) / 0.6865 (f1.0); temporal-k3 base 0.6834 / 0.6894 → +0.008 / +0.003. Same direction at k0.5: +0.003 / +0.001 | Broad +0.001 to +0.003 | Broad LB (paired SE about 0.003). GPU, about 1 h (same cost as FITA-3) |
| 3 | **Low-rank threshold** `--lowrank_below_min` | 30 min, so the 24-min group 13–22 and the 12-min group 23–32 both get rank-16 | **18 min**: 13–22 full A, 23–32 low-rank | Local fit fraction f = share of each subject's Q1–Q2 pool, which is about 23.5 labelled min (subjects 1–12 have about 47 min). So f0.25 ≈ 6 min and f1.0 ≈ 24 min. Low-rank vs full: **+0.010 at ~6 min, −0.007 at ~24 min**. 24 min is exactly subjects 13–22. No measurement at ~12 min (interpolated ≈ 0), so 23–32 stays low-rank | Broad +0.001 to +0.002 (10 of 39 subjects × ~+0.007) | Ships with rank 2 in the same FITA-4 run. Optional check: `tuning/fita_lr_temporal.py` (GPU, about 10–15 min) maps low-rank+temporal at f 0.25/0.5/1.0 |
| 4 | **Sinkhorn T, Broad**, on the FITA file | 2 (LB only ever compared T1 against T2) | Try **T3** as one LB slot on the final FITA file | Local, three instruments, all saying higher T is slightly better: BAL map T2 0.687 / T3 0.688 / T4 0.689; tonight's bank (5 FINAL-6c seeds, 24 blocks) T3 +0.0012 ± 0.0017, T4 +0.0027 ± 0.0019; SIM after SLP T3 +0.0006 ± 0.0016. FITA posteriors are sharper than the files T2 was tuned on (raw entropy 3.21 vs 3.29) | Broad +0.000 to +0.004 | Broad LB is the right instrument: a 1-D knob with about 11k public rows, so little overfitting risk. CPU: `hmm_prior.py --T 3` on the raw FITA file |
| 5 | **Deep cross-recipe ensemble** (FINAL-7 width × 5 + FINAL-9 stretch × 5) | single recipe | Average the two raw `_bn` files (`tuning/avg_csv.py`), then SLP κ20 T3 (+ the isolated-row fix) | Same-recipe 5 → 10 seeds gave 0.000. Width (+0.041 day-novel) and stretch (+0.034 same-day) work through different mechanisms. The D12 diversity ensemble (−0.008) used weaker members | Deep +0.000 to +0.010 | No faithful instrument tonight (Deep LB ±0.03; the day-novel test would need lso-stretch checkpoints). CPU, 1 min. A hedge, not a measured gain |

**Stacked, honest:**
- **Broad**, ranks 2–4: about +0.003 to +0.008 over FITA-3. From the FITA-3 expectation (0.73–0.75), that lands at about 0.735–0.755.
- **Deep**, ranks 1 and 5: about +0.005 to +0.02 over the FINAL-9/FINAL-7 file, so about 0.73–0.78.
- Neither track has a path to 0.80 from tuning.

---

## 2. The isolated-word finding (new tonight)

**Diagnosis on the real holdout files, label-free.** On isolated rows, every final file collapses onto a few words, while sentence rows do not:
- **Broad `fita2_broad_ens5.csv`:** "so" is top-1 on 38.7× its uniform share and "good" on 11.3×. Even after SLP + joint Sinkhorn, "so" is still 27.8×, which is the top-1 on 56% of isolated rows.
- **Deep `final7_deep_bn.csv`:** "good" 25.5× and "so" 23.9×.
- **Sentence rows** stay within about 5×.
- The isolated rows are also *sharper*: max-p 0.21–0.23 vs 0.18–0.19 on sentence rows.

**Why.** The 3 s context recipe gets its strength from neighbouring words' overlapping windows. Its 1 s window inside a segment still scores 0.69–0.73 (`ctx1s`). Fed a lone 1 s window as a 1-word segment, which is how `holdout2_word` rows are fed, it drops to chance: subj0 0.229 / 0.231 (Q3 / Q4), Broad subjects 0.14–0.23.
- **Grouping doesn't help.** Grouping lone words into random 7-word pseudo-segments of the same subject also fails: 0.21 / 0.28.
- **Mirrors the holdout collapse.** Locally the lone-window posteriors collapse the same way: the top class gets 11× its uniform mass, against 10× on the holdout.
- **Training doesn't cover it.** The 15% isolated windows used in training sit inside multi-word segments, so the model never learns a lone window.

**The fix that exists today is Deep only.**
- **B0 recovers the rows.** B0 is the October 4 subj0 1 s WordCNN; Q4 is its held-out block. It decodes lone 1 s windows at 0.449.
- **B0 on the real holdout:** it runs on the Deep holdout on CPU, through its own `predict_holdout.py` path. Output: `tuning/b0_deep_raw.csv`, and with AdaBN `tuning/b0bn_deep_raw.csv`.
- **Why AdaBN.** It is the default because day drift was the main Deep discount (v3 AdaBN gave +0.044 on the LB). On sentence rows, B0+AdaBN agrees with v3 slightly more than raw B0 (top-10 overlap 0.318 vs 0.290). AdaBN changes B0's top-1 on 87% of rows, so which variant to pick is a judgement call. Both are built.

**Broad.** B0 does not transfer to other brains (0.211, chance).
- **Per-source Sinkhorn on Broad's isolated block does nothing.** I simulated balancing the isolated block on its own (`tuning/iso_sim.py`, 12 subjects × 10 draws). Isolated rows improved +0.03 to +0.09, but pooled 960-row accuracy came out −0.0004 (natural) and −0.013 (balanced). Not recommended.
- **The real Broad fix needs GPU.** It would be a lone-1 s model with a per-subject labelled input layer (FITA-style), or retraining with some 1-word segments. Upper bound about +0.008 Broad. Not feasible before the deadline alongside the autopilot.

**Training-time knob that was never tuned.** `--p_iso 0.15` builds isolated windows *inside* segments. A recipe that also trains 1-word segments would fix the isolated rows in both tracks. Note this for any post-deadline work.

**Compliance.**
- Within-subject only.
- Uses the row's own source file (holdout2_word vs sentence), which the pipeline already uses to choose the window, and a fixed 1 s canonical window.
- No timing features, no cross-subject information, no labels.

---

## 3. Full inventory

**Status key:**
- **MAPPED**: swept on a named instrument.
- **PARTIAL**: two points, or the wrong instrument.
- **NEVER**: not tuned.
- **DEAD**: verified to have no effect or to hurt.

### Post-processing (all CPU)

| knob | setting (Broad / Deep) | status | instrument & result | verdict |
|---|---|---|---|---|
| Readout τ (logit − τ·log prior) | 0.5 / 0.5 | **DEAD (tonight)** | Bank, 5 seeds: with Sinkhorn after, τ 0 / 0.5 / 1 give identical scores (logit-mean: exactly; prob-mean: ±0.001). τ is a column scaling, and Sinkhorn removes it | Leave as is |
| ce vs ce+cos readout | ce | MAPPED (local folds, T29) | ce τ0.5 chosen | Leave as is |
| Ensemble combination | softmax-probability mean | **NEVER → tested tonight** | Bank, 24 blocks: logit-mean +0.0017 ± 0.0014 (T2), +0.0038 ± 0.0023 (T3, against prob T2) | < +0.005; would need GPU re-prediction (only averaged CSVs exist). Skip |
| Seeds | 5 (FITA) / 5 (FINAL-7/9) | MAPPED (LB) | 2→5 +0.018/+0.029; 5→10 0.000. Bank: 3 vs 5 seeds +0.002 ± 0.0035 | Leave as is |
| Which epoch(s) | ep7 only | MAPPED (local, LIT-1) | Snapshot ep5–7: nothing; weight soup harmful | Leave as is |
| Sinkhorn target | uniform over all 50, per subject over all rows | MAPPED (LB, PRIOR-1) | True or estimated priors harmful | Leave as is |
| Sinkhorn T | 2 / 3 | PARTIAL (LB: T1 vs T2 Broad, T1 vs T3 Deep) | Local saturates upward (above); never re-tuned after SLP or FITA | **Rank 4**: one Broad LB slot at T3. Deep: leave (unreadable) |
| Sinkhorn iterations | 10 | **DEAD (tonight)** | Every final file's column sums are exactly on target (max deviation 0.0000) | Converged |
| Top-10 inclusion equalisation (balancing per-class top-10 counts instead of mass) | none | **NEVER → HARMFUL (tonight)** | λ 0.25 / 0.5 / 1: −0.020 / −0.092 / −0.221 | Dead |
| Per-source Sinkhorn (isolated block separate) | joint | **NEVER → tested tonight** | Broad sim ±0 / −0.013 | Don't use |
| SLP κ | 20 / 20 | MAPPED (LB, both tracks) | Interior peak at 20 on both tracks | Leave as is |
| SLP EM iterations | 10 | **NEVER → MAPPED (tonight, SIM)** | The Deep holdout changes a lot (10 vs 30 iterations: 8.5% of top-10 slots differ), so it is a real knob. SIM, 12 subjects × 2 draws (κ20): it 3 +0.0004, 5 +0.0008, 20 −0.0021, **40 −0.0081**; κ10: it 3–5 +0.003, it 20–40 −0.010..−0.012. Early stopping at 10 is acting as a hidden regulariser | Keep 10. Never raise it |
| SLP emission β | 1.0 | Files built, untested | **SIM tonight: β 0.7 −0.0057, β 1.4 −0.0051** | Don't spend LB slots on `slpb_*` |
| SLP trigram | off | PARTIAL (Deep LB −0.009) | `slpt_broad_*` not submitted | Low priority; SIM not run |
| SLP end-dist / in-EM balance / text base | off | MAPPED (LB) | −0.013 / −0.008 / −0.047 | Leave off |
| SLP marginal (`marg` includes the collapsed isolated rows) | all rows | NEVER | Isolated collapse inflates "so"/"good" in `marg` about 2×, which slightly down-weights those words' sentence emissions | Option `--marg_sent 1` in `tuning/hmm_prior_split.py`; untested, expected ±0.002 |
| Self-training toward SLP targets (SLP-ST) | off | Files built (`st_deep_bn*`), untested | Pseudo-label self-training collapsed every time it was tried (A8, A9) | Don't submit |

### Per-subject adaptation (Broad)

| knob | setting | status | result | verdict |
|---|---|---|---|---|
| FITA steps | 300 | MAPPED (local, 2 checkpoints) | Flat over 100 / 300 / 600 | Leave as is |
| FITA learning rate / label smoothing / ridge | 3e-4 / 0.1 / 1e-3 | NEVER | — | Flat in steps, so lr is probably flat too. Skip |
| **FITA base for fitted subjects** | identity | **MIS-SET** (A16 measured the temporal base, which never reached production) | +0.003..+0.008 local | **Rank 2** |
| **Low-rank threshold** | 30 min | **MIS-SET** (fraction ↔ minutes mislabelled) | Full beats low-rank at 24 min by 0.007 | **Rank 3**: 18 min |
| Low-rank rank | 16 | NEVER | — | Skip |
| Temporal k, no-data subjects 33–39 | 3 (FITA-3) | MAPPED (local, 24 blocks: 1–5 flat 0.028–0.033) | k 3 | Leave as is |
| Temporal k as the fit base | (3 in FITA-4) | MAPPED k 0.5 vs 3 | k3 ≥ k0.5 at both fractions | Use k 3 |
| AdaBN, Broad | off | MAPPED (LB α1 −0.011, α0.5 −0.002; never on FITA) | — | Leave off |
| Ensemble members in FITA | 5 (s0–4) | 10 available | 5→10 = 0 | Leave as is |

### Deep

| knob | setting | status | result | verdict |
|---|---|---|---|---|
| AdaBN | α 1, dropout on | MAPPED (LB on FINAL-6c: +0.044; α0.5 −0.029; dropout irrelevant) | Untested on width or stretch models | Keep the bn files as default |
| A0 vs identity A (with AdaBN) | A0 | MAPPED (LB tie) | — | Leave as is |
| Width × stretch | stretch only (the autopilot's rule: dwstr_s0 subj0 0.7385 < 0.745) | MAPPED (one seed) | Width and stretch don't stack (dwstr 0.7385 vs width 0.742, stretch 0.737) | Ensemble the two recipes instead (rank 5) |
| **Isolated-word decoder** | v3 lone-window (≈ chance) | **MIS-SET** | §2 | **Rank 1** |
| Sinkhorn T / κ | 3 / 20 | MAPPED on FINAL-6c | Deep LB can't resolve changes under ±0.02 | Leave as is |

### Training-time (GPU; none feasible before 8 AM alongside the autopilot)

| knob | setting | status | note |
|---|---|---|---|
| Stretch probability / range | 0.5 / 0.8–1.25 | NEVER | Single setting, measured same-day only |
| `p_iso` (isolated windows in segments) | 0.15 | NEVER | Should include **1-word segments** (§2). This is the most valuable post-deadline training change |
| `p_short`, `maxw` (12; holdout sentences are 5–9 words) | 0.3 / 12 | NEVER | Low prior |
| Dropout / lr / wd / epochs | 0.3 / 1e-3 / 0.1 / 8 | PARTIAL (epochs mapped: 12 and 24 no gain) | — |
| Augmentation pack (gain noise, channel dropout, spatial mix) | on | never ablated in the final regime | — |

---

## 4. Recommended actions, in order

1. **Now (CPU, done):** Deep candidate `docs/home_stretch/tuning/final7_deep_bn_isoB0bn_k20.csv`, which is FINAL-7 plus the B0+AdaBN isolated-row fix.
   - When FINAL-9 finishes, build the same for it:
     ```
     cd ~/projects/libribrain-2026; T=docs/home_stretch/tuning
     ~/venvs/libribrain/bin/python $T/iso_replace.py submissions/final9str_deep_bn.csv $T/b0bn_deep_raw.csv $T/final9str_deep_bn_isoB0bn.csv 1.0
     ~/venvs/libribrain/bin/python scripts/hmm_prior.py --track deep --T 3 --kappa 20 --inp $T/final9str_deep_bn_isoB0bn.csv --out $T/final9str_deep_bn_isoB0bn_k20.csv
     ```
   - Cross-recipe hedge (rank 5):
     ```
     avg_csv.py $T/f79_bn.csv submissions/final7_deep_bn.csv submissions/final9str_deep_bn.csv
     ```
     Then run `iso_replace` and `hmm_prior` on it the same way.
   - **Final Deep pick:** the isolated-row-fixed version of whichever Deep recipe Rohit trusts most. My order: the width+stretch average > width (FINAL-7) > stretch (FINAL-9). Width is the only one measured day-novel.
2. **After the autopilot frees the GPU (about 1 h of GPU):** FITA-4 Broad.
   ```
   cd ~/projects/libribrain-2026; PY=~/venvs/libribrain/bin/python
   CK=$(ls runs/final6_s{0..4}_ep7.pt | paste -sd,)
   scripts/gpu_run.sh fita4 $PY docs/home_stretch/tuning/predict_holdout_fita4.py --track broad --ckpt $CK --readout ce --tau 0.5 --fitA_steps 300 --lowrank_below_min 18 --temporal_base 1 --temporal_k 3 --fit_base temporal --out submissions/fita4_broad_ens5.csv
   $PY scripts/hmm_prior.py --inp submissions/fita4_broad_ens5.csv --out submissions/fita4_broad_ens5_k20.csv --kappa 20 --T 2
   $PY scripts/hmm_prior.py --inp submissions/fita4_broad_ens5.csv --out submissions/fita4_broad_ens5_k20_T3.csv --kappa 20 --T 3
   ```
   Note: `submissions/` belongs to the orchestrator; this agent wrote nothing there.
3. **Broad LB slots (about 11k rows, readable to about 0.004):**
   - FITA-3, already queued.
   - FITA-4 at T2.
   - FITA-4 at T3.
   - Pick the best of the three for the final selection.
   - Spend nothing on β, trigram, SLP-ST or top-k files.
4. **Final selection:** Kaggle scores the private 70% of the same subjects. On Broad, pick the LB-best of these 1-D variants; with about 11k rows the winner's curse is about 0.002. On Deep, pick by the evidence above, not by the public LB (±0.03).

## 5. Files (all new, under `docs/home_stretch/tuning/`)

**Banks**
- `bank.py`, `bank_final6.npz`: CPU posterior bank, FINAL-6c s0–4, Broad 1–12, Q3/Q4, out-of-sample.
- `iso_bank.py`, `iso_bank.npz`: lone 1 s windows for the same rows.

**Sweeps**
- `pp_sweep.py`, `pp_sweep.json`, `paired.py`: τ, T, ensemble combination, top-k equalisation.
- `sim_sweep.py`, `sim_sweep.json`: SIM-1 on the bank (EM iterations × κ, β, T after SLP).

**Isolated-word tests**
- `iso_check.py`, `iso_check2.py`: holdout diagnostics.
- `iso_subj0.py`: lone-window decoding.
- `iso_b0.py`: B0 vs v3.
- `iso_sim.py`: Broad per-source Sinkhorn.
- `iso_deploy_sim.py`: Deep deployed simulation.

**Isolated-word tools**
- `b0_adabn.py`, `iso_replace.py`, `avg_csv.py`.

**Deep candidate files**
- `b0_deep_raw.csv`, `b0bn_deep_raw.csv`: B0 posteriors on the Deep holdout.
- `final7_deep_bn_isoB0{bn,w1.0,w0.5}{,_k20}.csv`: the three FINAL-7 variants with the isolated-row fix (each raw, plus SLP κ20 + Sinkhorn T3).

**Ready scripts**
- `predict_holdout_fita4.py`: FITA-4 (adds `--fit_base temporal`).
- `fita_lr_temporal.py`: optional GPU map.
- `hmm_prior_split.py`: per-source Sinkhorn / `--marg_sent` options; its defaults reproduce `hmm_prior.py` exactly (regression-tested).
