# Bet C — external MEG subjects (extmeg) — PLAN

Written 2026-10-12 ~15:40 EDT by the Bet C background agent. Scope: add many external MEG subjects to v3 training
(d'Ascoli et al. 2024, arXiv 2412.17829: one encoder over several helmets via a sensor-position front end; accuracy
rises roughly log-linearly with the number of training participants, no plateau seen at 723). Nothing under `src/` or
`scripts/` (pre-existing) is modified; all new code is in `scripts/extmeg/`, notes in `docs/extmeg/`.

Status log (append-only, newest last):
- 15:23 download of ds007523 started (`scripts/extmeg/download_lpp_listen.sh`, log `logs/extmeg_download_lpp_listen.log`).
- 15:35 one LPP run run through the full SSS→50 Hz chain: 65 s/run, amplitudes match the LibriBrain h5 (mag rms 3.1e-13 vs 2.7e-13 T).

## 1. Candidate corpora (all facts verified today from the repository APIs or the papers; "my estimate" marked)

| Corpus | Access | Licence / gate | Size | Format | Sensors | Subj | Hours | Lang | Task | Word onsets |
|---|---|---|---|---|---|---|---|---|---|---|
| **Le Petit Prince MEG listen** (Bel, Bonnaire, Pallier, King 2026; d'Ascoli's own data) | OpenNeuro **ds007523** v1.0.1, anonymous S3 | **CC0, no gate** | 478 GB total, **445 GiB without anat** (2408 files) | BIDS, raw **MaxShield fif** (SSS NOT applied; fine-cal + crosstalk shipped per subject), 1000 Hz, hp 0.1 / lp 330 | **Elekta Neuromag TRIUX 306** — sensor xyz identical to LibriBrain to 1e-5 m (checked on sub-03 vs `metadata/sensor_xyz.json`), same channel names | **58** (sub-01..58, 1 session, 9 runs of ~10.8 min) | **94** (d'Ascoli Table 1) | French | audiobook listening | **yes**: `events.tsv` one row per word (onset, duration, stimulus); `sourcedata/task-listen_run-XX_extra_info.tsv` adds `sequence_id` (sentence), `is_last_word`, `pos`, `content_word`; `sourcedata/stimuli/txt_raw/` has the punctuated text |
| Le Petit Prince MEG **read** (RSVP) | OpenNeuro ds007524 v1.1.0 | CC0 | 321 GB | same as above | TRIUX 306 | 50 | 59 | French | reading | yes (same family) |
| **MEG-MASC** (Gwilliams et al. 2023) | **OSF** osf.io/ag3kj + 3 child components (h2tzn, u5327, dr4wy "Part 2–4"); osfclient / OSF API, public | **CC0, no gate**; OSF speed typically 5–20 MB/s | not stated; my estimate 100–150 GB (27 subj × ~2 h × 208 ch × 1 kHz) | BIDS via MNE-BIDS, KIT raw (`.con` + `.mrk`/`.elp`/`.hsp`, to verify) | **KIT 208 axial gradiometers** (NYU Abu Dhabi), 1 kHz | 27 (22 with two sessions) | 57 (d'Ascoli Table 1) | **English** | 4 MASC short stories + word lists | **yes**: events.tsv rows for words and phonemes (`trial_type` dict kind/word/story/sequence), sentence vs word-list flag |
| **Appleseed Audiobook MEG** (Brodbeck et al. 2022, eLife) | OpenNeuro **ds007870** v1.3.2, anonymous S3 | CC0, no gate | **41.6 GiB** (317 files) | BIDS **fif** (converted KIT), 1 kHz, continuous ~58 min/subject | **KIT 157** axial (UMD; sidecar says "Elekta", 157 MEG + 3 ref) | 12 (+ empty room) | ~11.5 (my estimate: 12 × 3464 s recording) | **English** | continuous narrative ("Appleseed"), 11 segments | **indirect**: `events.tsv` has only segment onsets/offsets; word onsets from `stimuli/<seg>.TextGrid` (Praat, word + phone tiers) shifted by the segment onset |
| MEG-SCANS (Daeglau et al. 2025) | OpenNeuro ds006468 v1.1.2, S3 | CC0 | 204 GiB w/o anat (raw + MaxFiltered copies) | BIDS fif | **Neuromag TRIUX 306** | 24 | ~16 audiobook (+ OLSA sentences, chirps) | German | 2 audiobooks ~40 min/subj | yes: TextGrids per audiobook |
| SMN4Lang (Wang et al. 2022) | OpenNeuro ds004078 (NOTE: this is NOT MEG-MASC) | CC0 | 678 GB (MEG + fMRI + MRI) | BIDS, MEG + fMRI | MEG (Neuromag 306 at Beijing; to verify) | 12 | 6 h listening each (72 total) | Mandarin | 60 story runs | yes (time-aligned characters/words) |
| **Armeni et al. 2022** | Donders `DSC_3011085.05_995` | **DUA RU-DI-HD-1.0: account + click-through — GATED, for Rohit** | 247 GB (collection metadata) | BIDS, **CTF 275** `.ds`, 1200 Hz | CTF 275 axial | 3 | 30 (10 h each) | English | Sherlock Holmes audiobook (same material as LibriBrain!) | yes (events + alignments) |
| **MOUS / Schoffelen 2019** | Donders `DSC_3011020.09_236` | **DUA — GATED, for Rohit** | **1.4 TB** total (MEG + MRI + fMRI) | BIDS, CTF 275 `.ds` | CTF 275 | 204 (≈96 listen / 99 read usable) | 81 listen + 106 read | Dutch | sentences & word lists, listening or reading | yes (word-level) |
| BABA MEG (ds005346) | OpenNeuro, CC0 | — | 42 GB | fif | ? | 30 | ~1 h each | Mandarin | sitcom video | yes | not speech-only; skip |

Not MEG (for completeness, all used by d'Ascoli): Broderick (EEG, English listen), Nieuwland (EEG read), Accou (EEG Dutch).

## 2. Chosen first corpus: **Le Petit Prince MEG listen (ds007523)** — downloading now

Why this and not the English MEG-MASC first:
1. **Same helmet.** Sensor xyz and channel names are identical to LibriBrain (verified numerically). The first scaling
   test therefore needs **no spatial-attention front end at all**: LPP subjects enter `train_v3.py` exactly like 58
   extra broad subjects, each with its own 306×306 `A`. That isolates the question we care about (do more
   *subjects* help cross-subject generalisation?) from the front-end engineering, which can then be judged separately.
2. **Largest ungated listening set** (58 subj, 94 h) — the log-linear scaling claim is about *participants*.
3. **Downloadable now at ~70 MB/s** (anonymous S3; 14 GB in the first 3 min). OSF for MEG-MASC is slower and
   split over 4 components.
4. Word onsets + sentence ids + POS are shipped; SSS calibration files are shipped.
Cost: French. The 50-word CE head gets no labels (all `label = -1`); LPP words train only the T5 SigLIP branch.
t5-large was pretrained with en↔fr translation and its SentencePiece vocabulary covers French, so block-12 embeddings
are meaningful; the SigLIP positives are per-token, so French and English never collide as "same word" anyway.
MEG-MASC is the second corpus (English, needs the front end; download command below), Appleseed the third (20 min).

Download (running): `scripts/extmeg/download_lpp_listen.sh` =
`rclone copy ":s3,provider=AWS,region=us-east-1:openneuro.org/ds007523/" /mnt/ssd4tb/data/extmeg/lpp_listen --exclude "**/anat/**" ...`
445 GiB at the observed ~70 MB/s → **ETA ≈ 1.8–2 h from 15:23 EDT (≈ 17:15–17:30)**. Resumable: re-run the script.

Second/third corpus commands (not started; one download at a time):
- MEG-MASC: `pip install osfclient` (venv) then `osf -p ag3kj clone /mnt/ssd4tb/data/extmeg/megmasc/part1` and the same for `h2tzn`, `u5327`, `dr4wy`.
- Appleseed: `rclone copy ":s3,provider=AWS,region=us-east-1:openneuro.org/ds007870/" /mnt/ssd4tb/data/extmeg/appleseed --transfers 8`.

Gated (Rohit must register and accept the Donders DUA, then `webdav.data.donders.ru.nl`): **Armeni** (English Sherlock
Holmes, 3 subj × 10 h, CTF; the closest stimulus match to LibriBrain) and **MOUS** (Dutch, 204 subj, 1.4 TB).

## 3. Integration design

### (a) Preprocessing → `/mnt/ssd4tb/data/libribrain_cache/ext_lpp_listen/`
Script: `scripts/extmeg/preprocess_lpp_listen.py` (Phase 2; see its docstring). Per run `sub-XX_ses-01_task-listen_run-0R`:
1. `read_raw_fif(allow_maxshield=True)`, `fix_mag_coil_types`, `find_bad_channels_maxwell` (39 s) → bads,
   `maxwell_filter(calibration, cross_talk, coord_frame="head")` (7.5 s) — reproduces LibriBrain's `bads+sss` steps.
2. pick MEG, high-pass 0.1 Hz, `resample(250)` (12.8 s) → the same 250 Hz stage LibriBrain's `bp+ds` h5 is at.
3. reorder channels to LibriBrain order (MEGxxx1 mag, xxx2, xxx3 per triplet; by name against `metadata/channels.tsv`),
   × `src.common.SCALE`, `resample_poly(1, 5)` → **(306, T50) float16**, identical to `scripts/build_cont.py`.
4. events: `events.tsv` words aligned (difflib on word strings) to `sourcedata/.._extra_info.tsv` to pull
   `sequence_id` → `sentenceidx`; `sent_start_s/sent_end_s` as in build_cont; `label = -1` everywhere;
   `timechapter = onset` (no split-v2 quartering for ext subjects); extras `pos`, `content_word`, `is_last_word`.
   Saved as `<base>_ev.npz` with the build_cont keys so `text_targets` logic reads it unchanged.
5. once per corpus: `sensors.npz` — `xyz (306,3)`, `ori (306,3)` (= `loc[9:12]`), `kind` (0 mag / 1 planar grad),
   `names`, in LibriBrain order, device coordinates — the table the front end consumes.
Measured: **65 s per run on 4 threads** → 522 runs ≈ 9.4 CPU-hours → **≈ 2.5 h with 4 workers** (RAM ≈ 4 GB/worker).
Output size: 306 × 31.5k × 2 B ≈ 19 MB/run → **10 GB** for the corpus (fits RAM beside the LibriBrain cache; `LB_MMAP=1` otherwise).

### (b) Spatial-attention front end (for helmets ≠ Neuromag-306) — `scripts/extmeg/spatial_attention.py`
Défossez 2023 / d'Ascoli 2024 style, extended for mixed sensor types:
- per channel c: feature φ_c = [Fourier(xyz_c) ⊕ orientation_c ⊕ type-embedding_c] → MLP → key k_c ∈ R^h.
  Fourier: random Fourier features of the *normalised* xyz (centre, scale to the unit ball per helmet; 64 frequencies
  → 128 dims). Orientation and type are needed because the 3 Neuromag sensors of a triplet share xyz.
- 270 learned queries q_j; weights `A_jc = softmax_c(q_j·k_c / √h)` with masked (absent/bad) channels at −∞;
  `virtual = A @ x` → (B, 270, T). Position jitter (σ = 1 cm) at train time as in Défossez.
- then a **per-subject 270×270 linear** (init I) replaces the current per-subject 306×306 `A`.
- Drop-in: in `train_v3.py` the single line `x = einsum(Am, x)` becomes `x = einsum(Am270, front(x, pos_b, mask_b))`,
  and `CtxEncoder(n_ch=271)` (270 + valid-mask channel). Everything downstream (temporal blocks, SentCtx, heads,
  losses, eval) is untouched. `normalize()` needs a type-aware twin (median per sensor *kind*, not `IS_MAG`).
- For LibriBrain and LPP the channel set is constant, so `pos_b` is a fixed table; for KIT/CTF it is that corpus'
  `sensors.npz`. The front end is only *needed* for corpus #2 (MEG-MASC); step (d) runs first without it.

### (c) T5 targets for the new corpus
`scripts/extmeg/text_targets_ext.py` = `scripts/text_targets.py` with the cache dir parameterised
(`ext_lpp_listen/*_ev.npz` → `ext_lpp_listen/t5/<base>.npy`). Same t5-large block-12 raw output at each word's first
subword, sentence = `sequence_id` group. All 58 subjects hear the same 9 texts, so the memo makes the whole corpus
≈ one subject's worth of T5 forward passes (≈15k words): minutes on GPU, ~30 min on CPU.

### (d) Local test that would show the gain
Split v2 (`src/split2.py`): EVAL subjects (3, 6, 9, 12) never trained, VAL (1, 5, 10) for selection, SEEN
(2, 4, 7, 8, 11); deploy = (32·seen + 7·zero-shot)/39. Baseline = v3d1 recipe, seed 0: **deploy 0.463**.
Treatment = same recipe + an `ExtWords` loader (`scripts/extmeg/ext_words.py`, subject ids 1000+) feeding LPP words as a
third sampling stream (`--ext_frac`, T5 SigLIP loss only, own `A` per LPP subject). Score the same EVAL/VAL blocks.
Detection rule as in PREDICTIONS.md: **≥ +0.015 deploy**; the zero-shot component (`val_zs`) is the one d'Ascoli's
claim predicts most. Secondary: a 3-point subject-count curve (0 / 20 / 58 LPP subjects) to see the log-linear slope.
Prediction to commit before the run: +0.00 to +0.02 (the LPP words are off-language and only reach the T5 branch;
the encoder gains sensor-level generality, the CE head gains nothing).

### (e) Honest time estimates
| step | wall-clock | who |
|---|---|---|
| download ds007523 (running) | ~2 h, done ≈ 17:30 EDT | machine |
| preprocess 522 runs (4 workers) | ~2.5 h (overnight) | machine, `nohup` command in §4 |
| T5 targets (memoised) | 10–30 min | machine |
| `ext_words.py` loader + `train_v3_ext.py` (copy of train_v3 with the third stream) | 2–3 h agent time | agent |
| one training run v3d1-ext (data in RAM: +10 GB) | ~ same as v3d1 (check `runs/*.json` for v3d1_s0 duration) | GPU (shared with Bet A) |
| **first local result** | **≈ 1 day after the download**, i.e. Oct 13 evening if preprocessing runs tonight | |
| spatial front end + MEG-MASC (English) | +1 day download/preproc (KIT .con via MNE), +0.5 day integration | after freeze; post-competition unless Bet A frees the GPU |

## 4. Phase 2 status and launch commands
See the end of this file (appended by the agent after the preprocessing test).

## 4. Phase 2 status (2026-10-12 ~16:00 EDT)

Done and tested:
- `scripts/extmeg/download_lpp_listen.sh` — running (pid in `pgrep -f "rclone copy"`); 27% at 15:51, 62–87 MB/s, **ETA ≈ 17:20 EDT**. Re-run the script to resume if it dies.
- `scripts/extmeg/preprocess_lpp_listen.py` — run on sub-03 (9 runs). Output validated: `(306, 31550)` fp16 per run,
  rms in SCALE units matches the LibriBrain `cont/` cache, all 1614–1860 words per run aligned to `extra_info`
  (`sequence_id` → sentenceidx; ~10–12 words/sentence), `sensors.npz` written (xyz/ori/kind/names, LibriBrain order).
  Bads found per run: 2–5 (MEG0213 flat and MEG1933 bad in every sub-03 run).
- `scripts/extmeg/text_targets_ext.py lpp_listen cpu run-01` — T5 targets for one run (see `ext_lpp_listen/t5/`).
- `scripts/extmeg/spatial_attention.py` — smoke test passes (masked channels get zero weight, rows sum to 1, 72k params).

Timing caveat (measured): alone, one run takes **65 s** on 4 threads (find_bads 39 s, SSS 7.5 s, filter+resample 13 s).
With Bet A training on the CPU (240%) plus 4 workers, `find_bad_channels_maxwell` slowed to **270–480 s per run**
(BLAS oversubscription), i.e. ~7 min/run → 522 runs ≈ 15 h at 4 workers. Launch the full pass when the GPU job's
data loader is idle, with 1–2 BLAS threads per worker, or use `--skip_bads 1` (SSS alone: 10× cheaper; LibriBrain's
"bads" step is then approximated by SSS's own regularisation — acceptable for a first result, flag it in the run note).

### Overnight launch (after the download finishes; check `grep ETA logs/extmeg_download_lpp_listen.log | tail -1`)
```bash
cd ~/projects/libribrain-2026 && OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 nohup \
  ~/venvs/libribrain/bin/python scripts/extmeg/preprocess_lpp_listen.py --nproc 6 \
  > logs/extmeg_preprocess_lpp_listen.log 2>&1 &
# then T5 targets (GPU if free, else cpu):
nohup ~/venvs/libribrain/bin/python scripts/extmeg/text_targets_ext.py lpp_listen cuda > logs/extmeg_t5_lpp_listen.log 2>&1 &
```
Expected: 6 workers × ~1 run/65–130 s → **3–5 h** uncontended; the T5 pass is minutes (memoised: all subjects hear the same text).
Disk: `ext_lpp_listen/` ≈ 10 GB MEG + 522 × (N×1024×4 B ≈ 7 MB) ≈ 3.5 GB T5.

### Not done (next session, in order)
1. `scripts/extmeg/ext_words.py`: a `Words`-compatible loader over `ext_lpp_listen/*_ev.npz` (+ `t5/`), subject ids 1000+sub,
   no split-v2 quartering (all LPP words trainable), same `windows()`/`normalize()` code path (306 channels, LibriBrain order).
2. `scripts/extmeg/train_v3_ext.py`: copy of `scripts/train_v3.py` with a third sampling stream (`--ext_frac`), per-LPP-subject `A`,
   LPP rows in the T5 SigLIP loss only (`label = -1` → excluded from CE automatically). Commit the prediction in PREDICTIONS.md
   first (+0.00 to +0.02 deploy; detected ≥ +0.015), run seed 0 vs the v3d1_s0 baseline (0.463), then the 0/20/58-subject curve.
3. Only if (2) detects a gain: MEG-MASC download (OSF, English) + `preprocess_megmasc.py` (KIT `.con` via `mne.io.read_raw_kit`,
   208 axial grads, `sensors.npz` from `info["chs"][i]["loc"]`), and wire `spatial_attention.py` in place of the 306×306 `A`.

- 16:05 sub-03 fully processed (9/9 runs, 100% words aligned, 159 sentences/run, T5 (1614,1024) for run-01, no zero rows). The last
  run, alone on the CPU, took 71 s — the 480–640 s figures above are pure contention. Download 37% at 16:01, ETA ≈ 16:55 EDT.
