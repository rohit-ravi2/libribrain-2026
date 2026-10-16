# PATTERN EXTENSIONS: closest untested neighbours of the measured wins (2026-10-16, ~01:45 UTC)

Scope: for each pattern in LEDGER.md §3 that produced gains, the nearest untested extension that could move the **leaderboard** before the deadline (2026-10-16 12:00 UTC). Four of the candidates were tested tonight on the CPU while the FINAL-9 autopilot held the GPU. All numbers are top-10 balanced accuracy (chance 0.20).

**Target:** both tracks above 0.80. **Honest grade:** no extension found tonight has a measured path to 0.80 on Broad. Deep can reach 0.80 on the private board only with luck. The details are in §4.

Test scripts and raw JSON are in `docs/home_stretch/tests/`. Candidate files (built, **not submitted**) are in `docs/home_stretch/files/`. The builder `docs/home_stretch/build_after_autopilot.sh` is running in the background, niced. It waits for FINAL-9's raw outputs, then writes the remaining candidate files and logs to `files/build.log`.

**Operational note.** At 4 threads × 3 processes, my CPU tests slowed the autopilot's FITA-3 step (GPU utilisation fell to 17–25%, ~4 min per subject against ~0.6 min for FITA-1). I killed them, and FITA-3 recovered to ~0.5 min per subject. Every later test ran at `nice 19` with 2–3 threads. Any further CPU work tonight should be niced the same way.

---

## 1. Tested tonight (CPU, local split-v2 unless stated)

| id | pattern extended | mechanism | result | verdict |
|---|---|---|---|---|
| **XR-1** | seeds/variance (8) × Deep evidence (width R20, stretch R24) | Width and stretch don't stack inside one model: dwstr_s0 scored 0.739 against width 0.742 and stretch 0.737, which is why the autopilot fell back to stretch-only. So test whether they stack as **ensemble members**. Setup: subj0 Q3/Q4 same-day, A0, ce τ0.5, all-50 Sinkhorn. | 4-member mix, by Sinkhorn T:<br>• w2+s2: **0.762** (T3), 0.762 (T2), 0.750 (T1), 0.765 (T3, geometric mean)<br>• c2+w2: 0.741 / 0.740 / 0.748 / 0.743<br>• c2+s2: 0.753 / 0.752 / 0.736 / 0.756<br>• all six: 0.756 / 0.749 / 0.751 / 0.757<br>Equal-size pairs: the mean of the 4 cross w+s pairs beats the mean of the same-recipe w and s pairs by +0.003 (T3), +0.005 (T2) and 0.000 (T1). | **PROMOTE as the Deep final pick (hedge).** The best 4-member mix at every T. Adding the d128 family to w+s lowers the score at T2/T3 and is neutral at T1. The gain is small and same-day only. |
| **FT-1** | FITA (5) × Deep recipe (R20/R24) | Does the labelled per-subject A cancel the Broad penalty of the width and stretch trunks? If it did, those trunks could serve Broad as well. Setup: 100 steps, f=1.0, + Sinkhorn T2. | deep_c 0.638 → FITA **0.693** (subjects 3 and 6, 4 blocks).<br>stretch_s0 0.589 → FITA 0.670 (4 blocks): **−0.023** vs deep_c+FITA.<br>dw_s0 subject 3: FITA 0.660 vs deep_c 0.686 (**−0.026**). | **NOT PROMOTED.** FITA closes about half of the trunk-specialisation gap but not all of it. Broad keeps the FINAL-6c trunk. The seed-1 arms were killed for the autopilot, so this is seed 0 only, but it agrees in sign with every earlier Broad measurement. |
| **ISO-1** | Sinkhorn/structure (3) | **Holdout diagnosis (new):** on the 92 isolated-word rows per subject, the model collapses onto a few sentence-initial classes ("so", "and", "good", "but"). In `slp2_broad_k20.csv`, "so" is top-1 on 2,923 of 3,587 Broad isolated rows. Per subject, only ~13 (Broad) or ~22 (Deep) distinct words ever enter an isolated row's top-10. Proposed fix: a separate Sinkhorn on the isolated block. | Local holdout-shaped mixes: 868 contextual rows + 92 isolated 1 s rows, class-balanced, 30 resamples.<br>• Isolated rows only: subj0 0.208 → 0.315; Broad 0.188 → 0.213.<br>• **All rows: subj0 −0.001; Broad 0.564 → 0.553 (−0.011; 4/4 subjects ≤ 0).** | **HARMFUL. Do not submit.** The collapse acts as a coverage reservoir. Joint Sinkhorn sends the top-10 slots of low-evidence rows to the classes that are otherwise under-recalled, and balanced recall rewards that. Files `files/iso1_*` are kept only as a record. |
| **TKB-1** | Sinkhorn (3) | Metric-aligned balancing: choose the 50 per-class biases so that every class enters ~10n/50 top-10 lists (soft inclusion), instead of equalising probability mass. Same parameter count and the same uniform anchor as Sinkhorn. | Broad, natural blocks: Sinkhorn T2 0.634 and T4 0.639 vs top-k balancing at its best ε 0.631.<br>Broad, balanced blocks: 0.640/0.642 vs 0.640.<br>Sinkhorn → top-k balancing chained: 0.631/0.640. | **NOT DETECTED.** Sinkhorn already sits at this optimum. |
| **MFA-1** | FITA (5) × anatomical prior (6) | For the 7 zero-shot subjects: the population-mean labelled map Ā, from 6 other Broad subjects at 100 steps, used alone and composed with temporal k3. Leave-subjects-out: fit on 13–18, test on 3/6/9/12 Q3/Q4. | Score + Sinkhorn: identity 0.634, temporal k3 0.674, **Ā 0.672**, (I+Ā)/2 0.671, **Ā·temporal 0.677** (+0.003 vs temporal; 4/8 blocks). Raw: Ā·temporal 0.687 vs temporal 0.671. | **NOT PROMOTED** (≤ +0.001 LB, since only 18% of rows are zero-shot). Scientifically useful: a data-driven population map recovers the same +0.04 as the hand-set temporal-lobe prior (convergent validity), and the two don't stack. |

Instrument notes:
- XR-1 is same-day subj0. That instrument is unfaithful for drift, but it is fair for comparing ensemble members of equal size and recipe-independent noise.
- Seed noise is ±0.015 per single model (dw_s1 0.714 vs dw_s0 0.746). Read only the 4-member comparisons.
- FT-1, TKB-1 and MFA-1 use the split-v2 Broad instrument, which transfers to the LB at about 1:1 in direction (LEDGER §4).

---

## 2. Ranked list: what can still move the leaderboard

Expected gains are against the current submitted bests: Broad 0.70958, Deep 0.71611. The "faithful instrument" column names the instrument that can actually confirm each item.

| rank | candidate | pattern neighbour | mechanism | faithful instrument | expected LB gain | time to deploy | status |
|---|---|---|---|---|---|---|---|
| 1 | **FITA-3 Broad** (`fita3_broad_ens5_k20.csv`) | 5 + 6 | Labelled per-subject A, low-rank below 30 min, temporal k3 for zero-shot | Broad LB (SE ~0.004) | **+0.02 .. +0.035** (local +0.045, 18/18) | 0: the autopilot submits it | in flight |
| 2 | **XR-1 Deep mix**: width×5 (FINAL-7) + stretch×5 (FINAL-9), AdaBN raw, 50/50, then SLP κ20 and Sinkhorn T3 | 8 × R20/R24 | Two evidence-raising recipes that conflict inside one model but stack as ensemble members. It also hedges stretch's unverified day-novel behaviour with width's verified +0.041 day-novel gain. | PX day-novel for the members; Deep LB unreadable (±0.03) | **+0.000 .. +0.010** over the better single file; lower downside than stretch-only | ~1 min CPU after FINAL-9 (the builder makes `files/xr1_deep_bn_w5s5_k20.csv`) | auto-built |
| 3 | **LSO-STR**: stretch recipe trained with the 2024-10-04 day excluded, 1 seed, scored on PX | 7 × instrument rule | Stretch's +0.034 was measured same-day only. Deep same-day runs 0.05–0.08 above the LB because of day drift, and width's gain doubled day-novel. This is the one test that tells whether FINAL-9 or FINAL-7 should carry Deep on the private board. | PX day-novel (compare lso1 0.705, lso_dw 0.746) | 0 directly. Protects against a −0.02..−0.04 wrong pick. | ~30 min GPU after the autopilot (~04:00 UTC):<br>`P_STRETCH=0.5 train_stretch.py --tag lso_str --exclude "sub-0_ses-1[12]_task-Sherlock1"` plus the deep_c-recipe flags, then `PX=... eval` | proposed |
| 4 | **Broad Sinkhorn T3 on FITA-3+SLP** (`files/fita3_broad_ens5_k20_T3.csv`) | 3 | FITA sharpens the posteriors, and the local T map still rose at T3–T4 (+0.002..+0.005 local tonight) | Broad LB | +0.000 .. +0.004 | 1 slot (auto-built) | auto-built |
| 5 | **Broad SLP κ 15 / 30 on FITA-3** (`files/fita3_broad_ens5_k{15,30}.csv`) | 3 | The κ optimum ≈ data strength. Sharper emissions may move it. | Broad LB | ±0.003 | 2 slots (auto-built) | auto-built |
| 6 | **FITA + T5-SigLIP auxiliary loss** (fit A with CE on vocab words + SigLIP to T5 targets on *all* words of the subject's labelled sessions) | 5 × 1 | The FITA gain is monotone in labelled data: 25% → 100% gave +0.009. The T5 targets supervise ~2× more tokens and train the A map in the space that produced the biggest win. | split-v2 Broad (FITA map protocol), then Broad LB | +0.000 .. +0.008 | ~1 h (new predict script copying `predict_holdout_v3.fit_A_labelled`, ~25 min GPU) | proposed, only if the GPU is idle after rank 3 |
| 7 | Ā·temporal for zero-shot (MFA-1, with all 32 subjects at 300 steps) | 5 × 6 | see §1 | Broad LB | ≤ +0.001 | ~30 min GPU | not worth a slot |
| — | ISO-1, FT-1, TKB-1 | | | | negative or zero (§1) | | **do not submit** |

**Expected finals if ranks 1–5 land as expected:**
- **Broad:** 0.73–0.75. FITA-3 is the only large step left.
- **Deep:** 0.72–0.77 on the public board, and the private board is equally wide.

---

## 3. Slot plan and final selection (submissions are Rohit's call)

**Broad** (readable instrument; ~7 slots remain after the autopilot):
1. After the autopilot's FITA-3 lands, submit `files/fita3_broad_ens5_k20_T3.csv`, then `files/fita3_broad_ens5_k15.csv` and `k30`.
2. Keep whichever is best by ≥ 0.004. Otherwise keep `fita3_broad_ens5_k20.csv`.
3. If FITA-3 comes in below FITA-2, submit `fita2_broad_ens5_k20.csv`. Its T3 variant is already built at `files/fita2_broad_ens5_k20_T3.csv`.

**Deep** (public board unreadable below ~0.02; choose on evidence, not on public rank):
1. Submit `files/xr1_deep_bn_w5s5_k20.csv` so that it is scored and selectable.
2. **Final selection:** XR-1 mix + `final7_deep_bn_k20.csv` (width, the only day-novel-verified recipe).
3. Switch to FINAL-9 only if LSO-STR shows stretch ≥ width on PX. Switch to pure FINAL-7 if LSO-STR ≤ lso1.

**Selection rule.** The number of final selections per track on Kaggle is UNVERIFIED here (usually 2); check the competition page. If nothing is selected, Kaggle auto-picks by public score, and on Deep that is close to a coin flip.

---

## 4. Honest grade

**Broad.**
- No extension tested tonight adds more than ~0.005 beyond FITA-3.
- Reaching 0.80 needs about +0.09 over the current best. The largest effect ever measured on this track besides T5 and subj0-only is FITA, worth +0.02..+0.035.
- The measured route to that size is per-word SNR. Averaging same-word trials is worth +0.14, but it needs labels; every label-free proxy sits at chance (LEDGER 1f).
- **P(Broad ≥ 0.80) ≈ 1–2%.**

**Deep.**
- Public SE is ±0.03, and the 70% private split decides.
- Width plus stretch plus the XR-1 hedge plausibly gives a true score of 0.73–0.77.
- **P(Deep private ≥ 0.80) ≈ 10–15%.** That is slightly below the ledger's 15–20%, because stretch is still unverified on a new recording day and stretch and width did not stack inside one model.

**What would change this.** Only a new evidence-raising trunk on the scale of the T5 step, which gave +0.18 on the LB. T5-3B (TX) already failed. A new target family cannot be trained, screened and seeded inside the ~8 GPU-hours left, under the house rule that a recipe must win on both seeds before it is promoted. Not credible tonight.

**Lessons recorded:**
- The isolated-row collapse is beneficial under balanced recall (a coverage reservoir), not a bug.
- A data-driven population input map rediscovers the temporal-lobe prior.
- FITA does not rescue Deep-specialised trunks for Broad.
