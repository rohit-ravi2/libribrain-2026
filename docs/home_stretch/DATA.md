# DATA.md: can any data source raise the score? (home stretch, 2026-10-15 ~21:00 ET)

Scope: every data source that could be added before the Kaggle close (2026-10-16 12:00 UTC = 08:00 ET). I checked availability live on 2026-10-15 (HF API, curl on the URLs) and checked usage against the local disk and the code (`scripts/build_cont.py`, `scripts/train_v3.py --final 1`, `src/split2.py`, `scripts/extmeg/`). Nothing was downloaded.

## Bottom line

- **No data source can "for sure" raise either score, and none gets either track near 0.80.**
- **Broad:** every data addition we measured lowered zero-shot transfer:
  - own Broad subjects in training: −0.04 to −0.18;
  - +20 Le Petit Prince listeners: −0.151;
  - vocab-run oversampling: −0.033.

  The no-label subjects (33–39) have no data anywhere except their own 960 holdout rows, and we already use those (Sinkhorn, SLP, temporal prior).
- **Deep:** all released subj0 data is already in training. `--final 1` trains on every `*/derivatives/serialised*` file: Sherlock1–9, TIMIT, MOCHA-TIMIT, TheMoth, ~80 h.
- **Single most promising source:** the Le Petit Prince listen corpus (OpenNeuro ds007523, 58 subjects, Neuromag/MEGIN 306), as a T5-SigLIP-only side stream, **Deep only**. Why it leads:
  - It is already preprocessed on disk (`/mnt/ssd4tb/data/libribrain_cache/ext_lpp_listen/`, 58 subjects).
  - Its trainer (`scripts/train_v3_ext.py`) exists.
  - It is the only external data with a measured positive subj0 effect: ext20 gave subj0 0.731 vs 0.705, **+0.026**.

  Its weaknesses: that was 1 seed, measured same-day, never stacked with width or stretch, and never on the leaderboard. The recipe-diversity ensemble that included it scored −0.008 on the Deep leaderboard. Expected Deep leaderboard gain: **0.000 to +0.02**, which can't be read on ~288 public rows. P(real gain ≥ 0.01 on private) ≈ 25–30%. It needs ~1.5 h of GPU per seed, and the GPU is booked by FINAL-9 until near the deadline. **Verdict: not worth displacing FINAL-9 tonight.** At most, it is one extra ensemble member if a GPU slot opens after FINAL-9 finishes.
- **The highest-confidence lever tonight is not new data.** It is the subject's own labelled rows through FITA (+0.03 to +0.07 locally, 18/18 configurations), already built as FITA-1/2/3 and still unsubmitted. Submitting FITA-3 for Broad is the surest raise available.

## Ranked candidates

| Rank | Source | Availability (verified) | Licence / rules | Helmet / format | Evidence here | Time to integrate before 08:00 ET | Expected LB gain | Verdict |
|---|---|---|---|---|---|---|---|---|
| 1 | **Le Petit Prince MEG, listen task** (OpenNeuro ds007523, 58 subj, French) | On disk, cached (14 GB, 58 subjects); pipeline built (`scripts/extmeg/`) | CC0 / public external data: allowed | Neuromag TRIUX 306: **same helmet family** | Deep: ext20 subj0 **+0.026** (1 seed, same-day). Broad: **−0.151**. Diversity ensemble incl. ext20: Deep LB −0.008 | ~1.5 h GPU per seed. 5 seeds = 7+ h. **No GPU slot**: FINAL-9 holds it | Deep 0.000..+0.02 (unreadable on public); Broad negative | **Deep only, only if the GPU frees up.** Never Broad. Not a "for sure" |
| 2 | **Phoneme annotations in the released events** (`kind=phoneme`, ~9.6k per session; never used) | On disk | Released data: allowed | n/a | Never run. The nearest analogue, acoustic targets (wav2vec2 head / InfoNCE / late fusion), was null (+0.005, −0.001, +0.004) | Phoneme cache + aux head + retrain: ≥ 3 h code + GPU. No slot | 0..+0.01 | Post-deadline |
| 3 | **Other text-target layers** (T5-large blocks 16/18/24, multi-layer concat, or a decoder LLM's hidden states) | HF `google-t5/t5-large` (200); weights local | Public pretrained: allowed | n/a | Target space was the biggest lever (+0.18 LB), but the second-generation target (T5-3B) was null: Deep seeds disagree, Broad −0.016. L12 was inherited from MEG-XL/d'Ascoli and never swept | 1.0M-word target recompute (GPU) + 5-seed retrain: > 10 h | −0.01..+0.02 | Post-deadline; the best long-run bet in this table |
| 4 | **LibriBrain-Competition-2025 holdout** (`pnpl/LibriBrain-Competition-2025`) | HF, 1.06 GB. Labels uploaded 2026-07-29 | CC-BY-NC-4.0: allowed | Same sub-0, same pipeline (`ses-2025`), h5 + pt | It is a **new subj0 recording day**, but labels are speech/silence (run-1) and averaged phonemes (run-2) only. No word labels, no audio, no transcript, so no word supervision is possible. As unlabelled drift data, AdaBN on the actual 2026 holdout day already does that job (+0.044) | 1–2 h to wire as unlabelled BN/SSL data, plus a retrain | ~0 | No |
| 5 | **MegNIST** (`pnpl/MegNIST`, single participant, inner-speech digits 0–9) | HF, 63.4 GB (> 5 GB cap; not pulled) | CC-BY-NC-4.0 | MEGIN, with a `derivatives/neo` folder (TRIUX **neo** calibration). Whether the participant is LibriBrain's sub-0 is not stated | Different task (inner speech, visual digits); digits aren't the target vocabulary | Days | ~0 | No. (Side note: the `neo` folder suggests pnpl now records on a TRIUX neo. If the 2026 holdout came from a newer system, that is one more drift source, which AdaBN already targets.) |
| 6 | **Armeni et al. 2022** (3 subjects × ~10 h, English Sherlock Holmes stories; Donders DSC_3011085.05_995) | URL 200; Donders account needed; ≫ 5 GB | Donders DUA (click-through): allowed as external public data | **CTF 275 axial gradiometers**: different helmet; needs its own input layer | Shape matches our winner (deep single-subject data, same author, English), but every added-subject experiment hurt Broad, and cross-helmet sharing is untested | Days (download + CTF preprocessing + new stream) | Broad negative by precedent; Deep ≤ LPP | No (post-deadline curiosity) |
| 7 | **MOUS** (Schoffelen 2019, 204 subj, Dutch; Donders DSC_3011020.09_236) | URL 200; ≫ 5 GB | Donders DUA | CTF 275 | Shallow subjects hurt (−0.15 with 20 LPP) | Days | Negative | No |
| 8 | **MEG-MASC** (Gwilliams 2023, 27 subj, English; OSF ag3kj) | URL 200; ~100+ GB | CC-BY | **KIT 208**: different helmet | Shallow subjects hurt; different sensor geometry | Days | Negative | No |
| 9 | **Cam-CAN** (~650 subj) | Portal 200; **application required** (not same-night) | DUA | MEGIN 306 (same helmet) | No narrative speech (rest / sensorimotor / passive tones). Shallow subjects hurt | Not obtainable tonight | Negative | No |
| 10 | **SMN4Lang** (OpenNeuro ds004078, 12 subj, Mandarin narratives) | URL 200; ≫ 5 GB | CC0 | Reported as Elekta/MEGIN 306 (not verified from the files) | Another language. LPP (French) on the same helmet: Broad −0.151 | Days | Negative | No |
| 11 | **Audio / speech-model targets** (wav2vec2 done; Whisper / HuBERT untried) | Audio local (`LibriBrain100_audio`, no audio map for Sherlock8/TIMIT/TheMoth) | Allowed | n/a | All three acoustic-target variants null (R14, R15, R27) | Hours of recompute + retrain | ~0 | No |
| 12 | **MEG-XL weights** (`pnpl/MEG-XL`, med / med-v2) | HF | MIT | Same data | Frozen features: alone 0.219, and concat lowered our probe 0.512 → 0.451. Needs 150 s context; fine-tuning needs ≥ 80 GB VRAM | n/a | Negative | No |
| 13 | **More LibriBrain100** (new subj0 sessions; second halves of subj13–32) | LibriBrain2 last commit 2026-09-11; LibriBrain 2026-09-11. Nothing new. Subj13–32 second halves **embargoed until after the competition** (README) | — | — | — | — | — | Nothing to add; everything released is used |
| 14 | **Data for the 7 zero-shot subjects (33–39)** | None exists outside their own holdout. No MRI or headshape released | — | — | Their own 960 rows are already used. Fixed anatomy (temporal-lobe weighting k 3) gave +0.033 zero-shot | — | — | Nothing to acquire. The anatomical prior is the "other scientific prior" here, and it's already in FITA-3 |
| 15 | **Local subj0 not in final training** | `--final 1` trains on all stimulus. The split-v2 Q3/Q4 exclusion applies only to local runs | — | — | — | — | 0 | Already used |

## Other scientific priors that are data, not MEG

| Prior | Data | Status / evidence | Verdict |
|---|---|---|---|
| Sequence prior learned from the subject's own sentences (SLP) | Each subject's own 156 holdout sentences | +0.018 Broad / +0.008 Deep on the LB, in the stack | Done |
| Text n-gram / LLM sentence priors (Gutenberg, GPT-2, LLM-vocab corpora) | Public text | −0.05..−0.11 at our evidence level | Harmful; don't add more text |
| Word-class (Brown-cluster) grammar as the SLP base | Public text for the class map only; transitions learned on own rows | Untested (expected +0.005..+0.015; 1 class map + EM; CPU-feasible) | The only text-derived resource with a plausible small gain tonight. It is a prior method, not a data acquisition, and it needs a leave-one-book-out check first (SIM-1 is the calibrated instrument) |
| Anatomy / lead-field priors | `data/metadata/sensor_xyz.json`, MNE fsaverage | Temporal weighting +0.033 zero-shot; lead-field top-k null | Done |
| Psycholinguistic norms (frequency, concreteness, AoA) as class priors | Public (SUBTLEX, Brysbaert) | Any non-uniform class prior lost to uniform Sinkhorn (T11: 0.377/0.547) | No |

## Rules check (all candidates)

- External public data and pretrained models: allowed (arXiv 2609.03231, verified in RULES.md §5).
- Nothing above uses `word_onsets_s` absolute timing, `shuffle_seed`, cross-subject row mapping, hand labels or sentence-list reconstruction.
- The 2025 holdout labels are public organiser releases for a different recording day and stimulus. Using them as training data is allowed. They carry no information about the 2026 holdout rows.

## Recommendation (data agent)

1. Do not change the FINAL-9 autopilot to make room for any new data. No candidate has replicated, day-novel, leaderboard-faithful evidence that beats stretch (+0.034, both seeds) or width (+0.041 day-novel).
2. If the GPU frees up before ~05:00 ET: train **one** `train_v3_ext.py` seed with the FINAL-9 Deep recipe plus `--ext_frac 0.2 --ext_subjects 58` (or 20, the measured dose). Add it as a 6th Deep ensemble member in a *separate* submission file, never replacing the FINAL-9 file. Expected effect ≈ 0 ± 0.01, and the public leaderboard can't resolve it.
3. The surest score raise tonight is the already-built **FITA-3 Broad** file (subject-own labelled data; expected LB +0.02..+0.035 over 0.7096). That puts Broad at about 0.73–0.745, still well short of 0.80.
4. Post-deadline, the most valuable data experiment is a T5-large layer sweep / multi-layer target. Target space is the only lever that ever moved the leaderboard by > 0.05.
