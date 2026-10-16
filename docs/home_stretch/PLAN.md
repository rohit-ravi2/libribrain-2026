# PLAN: home stretch, LibriBrain / PNPL 2026 (written 2026-10-15, 21:45 ET)

**Sources:** LEDGER.md, PATTERN_EXTENSIONS.md, TUNING.md, OTHER_PRIORS.md and DATA.md, all in this directory. I checked the claims below against the disk at 21:43 ET.

**Goal:** both leaderboards above 0.80, by any means within the rules.
**Now:** Broad 0.70958, Deep 0.71611.
**Kaggle closes:** 08:00 ET Oct 16. New work runs until 06:00 ET.

## State verified on disk (21:43 ET)

- **FITA-3 Broad is built.** `submissions/fita3_broad_ens5.csv` and `_k20.csv` were written at 21:41. The builder has started making the T3/κ variants in `files/`.
- **The autopilot chose stretch-only.** dwstr_s0 scored 0.739, below the 0.745 cut-off, so the recipe is `final9str`.
  - Seed 0 is at about 10 steps/s, which is about 20 min per seed.
  - 5 seeds plus prediction means the GPU frees at about **23:30 ET**, with ±20 min of slack.
  - The autopilot then submits three files: `fita3_broad_ens5_k20.csv`, `final9str_deep_bn_k20.csv` and `final7_deep_bn_k20.csv`.
- **TUNING's FITA-3 claim is confirmed in code.**
  - `scripts/predict_holdout_v3.py:182` applies `TEMPW` only to subjects without labels.
  - Fitted subjects start at I and are ridged back to I.
  - The `--fit_base temporal` fix exists only in `tuning/predict_holdout_fita4.py`, which has never been run.
- **CPU throttle.** The stretch augmentation is CPU-side; a contended stretch run earlier took 83 min per seed. Any CPU work tonight runs at `nice -n 19` with `OMP_NUM_THREADS<=2`.

## Conflicts resolved by de-duplication

- **Stratified (separate) balancing of the 92 isolated rows.** OTHER_PRIORS ranks it #1. It was tested twice tonight and came out negative both times:
  - ISO-1: Broad −0.011, 0 of 4 subjects improved.
  - TUNING iso_sim: −0.0004 with natural frequencies, −0.013 with balanced ones.

  The collapse on isolated rows acts as a coverage reservoir, and joint Sinkhorn already exploits it. **Dropped. Do not spend a slot on it.**
- **Evidence-raising on isolated rows (B0 swap, Deep) is a different mechanism.** It raises per-word evidence; it does not rebalance. It stays in the plan (A3).
- **Population input map Ā:**
  - OTHER_PRIORS #2 expected +0.002 to +0.006.
  - MFA-1 measured it equal to temporal k3, and the two don't stack (+0.003 on 4 of 8 blocks, so ≤ +0.001 on the leaderboard).

  **Dropped.**
- **Grammar-class SLP base (T21).** Estimates ranged from +0.005..+0.015 (ledger) to +0.000..+0.008 with downside (OTHER_PRIORS). It is still untested, and SIM-1 is its gate. Low priority.

## Ranked actions

Expected gains are calibrated leaderboard deltas on the final (private) score. **Deep public changes under ~0.02 cannot be read (~288 rows).**

| # | Action | Track | Expected LB gain | Confidence | Cost | Depends on | Verify by submission? |
|---|---|---|---|---|---|---|---|
| A1 | **FITA-3 Broad** (`submissions/fita3_broad_ens5_k20.csv`): labelled per-subject A, low-rank below 30 min, temporal k3 for subjects 33–39, SLP κ20, Sinkhorn T2 | Broad | **+0.020 .. +0.035** vs 0.7096 | High (local +0.045, 18/18 positive; never on the LB) | 0: built; the autopilot submits it at ~23:30 | autopilot | Yes. The Broad LB is faithful (SE ~0.004) |
| A2 | **FITA-4 Broad**: `--fit_base temporal` (k3) for the fitted subjects + `--lowrank_below_min 18` (subjects 13–22 get a full A) | Broad | **+0.002 .. +0.005** over FITA-3 | Medium (A16 +0.003..+0.008 local; low-rank −0.007 at 24 min; the script copy has never run) | ~60 min GPU, plus a 2-min smoke test on one subject first | GPU free (~23:30) | Yes: T2 and T3 variants, 2 Broad slots |
| A3 | **Deep isolated-row fix**: replace the 92 isolated rows' raw posteriors with B0+AdaBN (`tuning/b0bn_deep_raw.csv`), then SLP κ20 + Sinkhorn T3. Apply to whichever Deep file is final | Deep | **+0.005 .. +0.012** | Medium (holdout-shaped sim +0.0104 ± 0.003 / +0.0083 ± 0.002; B0 0.449 vs v3 0.231 on lone windows; day drift is the discount; B0 vs B0+AdaBN is a judgement call because AdaBN changes 87% of B0's top-1) | 1 min CPU per file. `tuning/final7_deep_bn_isoB0bn_k20.csv` is already built | FINAL-9 raw `submissions/final9str_deep_bn.csv` for the mix version | Submit only so the file can be selected; the LB cannot read it (~28 public isolated rows) |
| A4 | **Deep XR-1 cross-recipe mix**: width ×5 (FINAL-7) + stretch ×5 (FINAL-9), AdaBN raw, 50/50, **then A3**, then SLP κ20 T3 | Deep | **+0.000 .. +0.010** over the better single file; mainly cuts the downside of stretch's untested day-novel behaviour | Low-medium (same-day w2+s2 0.762 vs c2+s2 0.753 and c2+w2 0.741; cross-pairs +0.003..+0.005) | 1 min CPU. The builder auto-makes `files/xr1_deep_bn_w5s5_k20.csv`; the iso-fixed version needs `iso_replace.py` on `files/xr1_deep_bn_w5s5.csv` before `hmm_prior` | FINAL-9 done | Submit so it can be selected; unreadable on the LB |
| A5 | **LSO-STR day-novel check**: stretch recipe, the 2024-10-04 day excluded, 1 seed, scored on PX (compare lso1 0.705, lso_dw 0.746) | Deep (selection) | 0 directly; protects against a **−0.02..−0.04** wrong final pick | Decision value high; single seed (±0.015) | ~30 min GPU (more if CPU-contended) | GPU after A2, or before it if A2's smoke test fails | No: local PX instrument |
| A6 | **Broad Sinkhorn T3** on the best FITA file (`files/fita3_broad_ens5_k20_T3.csv`, then the FITA-4 T3) | Broad | **+0.000 .. +0.004** | Medium-low (local +0.0012..+0.0027, 18/24 blocks; FITA posteriors are sharper) | 0: auto-built | A1 (and A2) | Yes, 1 slot per base file |
| A7 | **Broad SLP κ15 / κ30** on the FITA base (`files/fita3_broad_ens5_k{15,30}.csv`) | Broad | **±0.003** | Low | 0: auto-built | A1 | Yes, 2 slots. Submit only after A1/A2/A6, and only if slots are spare |
| A8 | **FITA with the T5-SigLIP auxiliary loss** (fit A on all words of the labelled sessions, not just the vocabulary) | Broad | **+0.000 .. +0.008** | Low (never run; FITA is monotone in data) | ~45 min coding (CPU) + ~25 min GPU + local check | A2 and A5 finished; GPU idle by ~01:30 | Yes, 1 slot, only if the local split-v2 check is positive on 2 or more subjects |
| A9 | **LPP (Le Petit Prince) 6th Deep member**: 1 seed `train_v3_ext.py` with the FINAL-9 recipe, `--ext_frac 0.2`, added to the A4 mix in a separate file | Deep | **0 ± 0.01** (P(real gain ≥ 0.01) ≈ 25–30%) | Low (ext20 +0.026: 1 seed, same day; the D12 diversity ensemble was −0.008 on the LB) | ~90 min GPU | GPU idle by ~03:00 | Unreadable. Do not select it over A4 without a PX check |
| A10 | **Class-grammar SLP base (T21)**, gated on SIM-1 | both | **+0.000 .. +0.008**, real downside | Low | ~60–90 min CPU (venv-local pip for a treebank / Brown classes) | none; CPU only | Only if SIM-1 beats κ20 by ≥ 0.005; then 1 Broad slot |

### Do not submit (measured dead or negative)

| Candidate | Why |
|---|---|
| Separate/stratified isolated-row Sinkhorn (`files/iso1_*`) | Measured negative twice tonight (see above) |
| `slpb_*` (emission sharpness) | −0.005 each on SIM |
| `st_deep_bn*` (self-training) | Pseudo-labelling collapsed every time it was tried |
| Top-k inclusion balancing | −0.02 to −0.22 |
| Logit averaging | +0.002 |
| Readout τ | No effect once Sinkhorn runs after it |
| SLP iterations > 10 | −0.008 at 40 |
| FITA on the width or stretch trunks for Broad | −0.023 to −0.026 |
| Ā maps | ≤ +0.001 |
| Per-class calibration | Null |
| Input winsorising | Null |
| Unbalanced OT | Null |
| Any external data for Broad | −0.15 |
| Partial or full Broad AdaBN | −0.002 / −0.011 on the LB |

## GPU timeline (one queue; every job through `scripts/gpu_run.sh`)

| ET | Job |
|---|---|
| ~23:30 | Autopilot done. It submits FITA-3 Broad, FINAL-9 Deep and FINAL-7 Deep. |
| 23:30–23:35 | **A2 smoke test** (FITA-4 on one subject, to check that the script runs and A is initialised at TEMPW). |
| 23:35–00:35 | **A2 full FITA-4 Broad.** Build its T2 and T3 files with `hmm_prior.py`. |
| 00:35–01:10 | **A5 LSO-STR.** If A2 fails its smoke test, run A5 first. |
| 01:10–01:40 | **A8**, only if the code is ready and checked on CPU. Otherwise idle. |
| 01:40–03:15 | **A9**, optional and lowest value. Skip it if anything before it has run late. |
| 06:00 | Stop all new work. |
| 06:00–07:30 | Final selections. |

## CPU tasks (niced, ≤ 2 threads)

- A3 and A4 files, as soon as `submissions/final9str_deep_bn.csv` appears.
- A8 code.
- A10 on SIM-1.

## Submission budget (10 per track for the Oct 16 UTC day, which began at 20:00 ET)

| Track | Autopilot uses | Use before 02:00 ET (keep ≥ 5 in reserve) | Reserve after 02:00 |
|---|---|---|---|
| Broad | 1 (FITA-3 T2) | ≤ 4: FITA-3 T3 (A6), FITA-4 T2 (A2), FITA-4 T3 (A2/A6), and one more only if A2 is late (FITA-3 κ15) | ≥ 5 (κ30, A8, A10, fallbacks) |
| Deep | 2 (FINAL-9 bn, FINAL-7 bn) | ≤ 3: XR-1 + iso-fix (A4+A3), FINAL-7 + iso-fix (A3) | ≥ 5 (FINAL-9 + iso-fix if A5 favours stretch; the A9 mix) |

### Selection rules

**Broad: the public LB is faithful.**
- Select the best Broad file by the public LB.
- Count a variant as better only if it wins by ≥ 0.004.
- If FITA-3 scores below 0.7096 + 0.010, distrust FITA and fall back to `fita2_broad_ens5_k20.csv`. Its T3 variant is in `files/`.

**Deep: choose on evidence, not on the public LB.** The default pair is:
1. the XR-1 mix with the iso fix;
2. FINAL-7 with the iso fix.

Change the pair according to A5:
- If stretch scores at least as well as width on PX, use the mix and FINAL-9 with the iso fix.
- If stretch scores at or below lso1 (0.705), use FINAL-7 with the iso fix and FINAL-7 without it.

**Unverified:** I did not check how many final selections Kaggle allows per track (usually 2); check the competition page. With no selection, Kaggle picks by public score, which on Deep is a coin flip.

## Honest bottom line

**Broad**
- Expected final: **0.735–0.755**, central about 0.742.
- What gets it there: FITA-3 (+0.02..+0.035), then FITA-4 and T3 (+0.002..+0.009).
- **P(Broad ≥ 0.80) ≈ 1%.** It needs +0.09, and the largest remaining measured lever is +0.035. Nothing tonight raises per-word SNR on new brains. Averaging same-word trials (+0.14) needs labels, and every label-free route to it measured at chance.

**Deep**
- Expected final, private: **0.725–0.775**, central about 0.745.
- What gets it there:
  - FINAL-7 or FINAL-9 over FINAL-6c: +0.01..+0.04;
  - the A3 iso fix: +0.005..+0.012;
  - the A4 mix: +0..+0.01.
- **P(Deep private ≥ 0.80) ≈ 8–12%.**
  - Stretch has never been tested on a new recording day.
  - Width and stretch did not stack inside one model.
  - The Deep local-to-LB discount is −0.05..−0.08.
  - The 0.80 cases need a true score near 0.77 plus private-split luck.
- The public Deep number could show ≥ 0.80 by noise (SE ±0.03) with about 5% probability. That would not be a real gain.

**Both tracks ≥ 0.80: ≈ 0.1–0.2%.**

**What could change this:** only a new evidence-raising trunk on the scale of the T5 target step (+0.18 on the LB). Candidates are a T5 layer sweep, brain-aligned LM layers, and training on 1-word segments, which fixes isolated rows on both tracks. None of them can be trained, replicated on 2 seeds and ensembled in the GPU hours left. They are the first post-deadline work.
