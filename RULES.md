# RULES.md — PNPL Competition 2026, verified 2026-09-21

Sources, all saved under `ref/`:
- arXiv 2609.03231 (abstract)
- libribrain.com `/editions/2026/rules/` and `/participate/`
- Kaggle pages via API: `ref/kaggle/{broad,deep}_pages.md` (description, rules, evaluation, data-description, foundational-rules)
- Kaggle announcement #741919 (2026-09-18)
- pnpl 0.2.0 source: `pnpl/competition/holdout.py` and `pnpl/datasets/libribrain100/selectors.py`
- Holdout `.npz` files, inspected directly

Status tags: **VERIFIED** (read from a primary source or from the data), **UNVERIFIED** (not found yet), **INTERPRETATION** (my reading, not a stated rule).

## 1. Tracks

| Track | Kaggle slug | Evaluates | Teams (2026-09-21) | Public LB top |
|---|---|---|---|---|
| Deep | `pnpl-competition-2026-deep` | subj0 only | 53 | 0.7475 |
| **Broad** | `pnpl-competition-2026-broad` | subj1–39, one submission surface | **22** | 0.7292 (contaminated, see §4) |

**Broad supervision groups** (VERIFIED from the Kaggle description and rules page, and from the pnpl holdout.py comment):

| Group | n subjects | Labelled data per subject | Condition |
|---|---|---|---|
| 100% | 12 | ~40 min | few-shot |
| 50% | 10 | ~20 min | few-shot |
| 25% | 10 | ~10 min | few-shot |
| **0%** | **7 (subj33–39)** | **none; subject ID only** | **zero-shot, never seen in training** |

**Brainstorm check** ("cross-subject track including unseen subjects"): **HOLDS, with a refinement.** Broad is a *mixture* of 32 few-shot subjects and 7 zero-shot subjects, scored together in one BAcc@10 over 37,439 rows. Zero-shot rows are 7/39 ≈ 18% of Broad rows. Charter step 2 (stop if the tracks differ) is therefore not triggered.

Group map (VERIFIED, arXiv 2609.03231 HTML): **subj1–12 = 100%, subj13–22 = 50%, subj23–32 = 25%, subj33–39 = zero-shot** (the paper says 33–40; the holdout contains 33–39).

## 2. Data

- LibriBrain100 = `pnpl/LibriBrain` (sub-0, Sherlock1–7) ∪ `pnpl/LibriBrain2` (sub-0 Sherlock8/9, TIMIT, MOCHATIMIT, Podcasts; sub-1..32 Sherlock1 ses-11 and ses-12 only). VERIFIED from the pnpl constants.
- Broad subjects 1–32 heard **the same Sherlock1 ses-11/ses-12 audio that sub-0 heard**, so the cross-subject data is stimulus-matched. In pnpl's standard splits, ses-11 = validation and ses-12 = test. sub-1..32 have **no `train` partition**; pnpl says to use `validation` as fine-tuning data.
- 306-channel MEGIN (102 magnetometers + 204 gradiometers), 250 Hz.
- Word window [onset, onset+1 s] = (306, 250). The holdout loader enforces this window.
- Licence: **CC-BY-NC-4.0** (VERIFIED from the HF card).
- Size: the full HF repos are 552 + 238 GB, mostly raw FIF and wav. **Only the serialised h5 + tsv are needed: 60.0 + 54.0 GB.** Of that, sub-0 is ≈96 GB and each broad subject is ≈0.44–0.9 GB. Holdout: 5.7 GB.
- Local paths: `data/` → `/mnt/ssd4tb/data/LibriBrain100`; `holdout_data/` → `/mnt/ssd4tb/data/LibriBrain-Competition-2026`. The ssd had 2.4 TB free at start.
- Download: `scripts/download.py` (resumable), log in `logs/download.log`. Holdout DONE 2026-09-21. Order: broad subjects, then sub-0.

## 3. Holdout and submission (VERIFIED)

- Final rankings use **independent holdout data recorded with stimuli entirely different from anything released**.
- Every one of the 40 subjects has the *identical* holdout structure: 156 sentences (868 word windows) + 92 isolated words (subj04 has 91). Broad = 37,439 rows; Deep = 960 rows. **All subjects heard the same holdout stimuli.**
- The holdout MEG is raw (≈1e-12 T scale). Standardise per subject with that subject's own channel statistics.
- Format: CSV with `index` + 50 competition-vocab columns (+ optionally 50 `moses_*` columns), 51 or 101 columns. Values are non-negative, finite scores; only the ranking matters. Build it with `pnpl.competition.write_submission`, using `holdout.indices` for row order.
- Metric: BAcc@10 = mean over 50 words of per-word Recall@10. Chance ≈ 0.20. Tie-break: BAcc@1. Moses-50 is secondary and not ranked.
- Submission limit: **10 per day** (VERIFIED via `kaggle competitions submission-limits`).
- Public/private: standard Kaggle split. Winners are chosen on the **private** LB plus code review. **Split percentage and row assignment: UNVERIFIED** (not stated on any page I could reach). Public and private rows are presumably drawn from the same 39 subjects, so the subjects overlap and only the rows differ (INTERPRETATION).
- Deadline: libribrain.com says 15 Oct 2026 AoE; the Kaggle API says 2026-10-16 12:00 UTC (the same instant). Charter: freeze Oct 13, submit Oct 14.
- Prizes: top 3 per track, which must beat the baselines; one prize per team. The Kaggle listing says "Kudos", so any money goes through the organizers (the brainstorm's figure of $5k is UNVERIFIED here). Top teams must provide runnable checkpoints and possibly training code.

## 4. Leakage rules (VERIFIED, announcement 2026-09-18)

- `word_onsets_s` absolute timing, plus `shuffle_seed`, let predictions be mapped between subjects on the shared holdout stimulus. The **Broad public LB is contaminated**: scores may be inflated. Final ranking = code-reviewed submissions only.
- **Forbidden:** mapping predictions across tracks using timing or shuffle metadata. Code that does this makes the team ineligible in *both* tracks. "Broad-track predictions must be generated for the broad-track subjects themselves."
- **Our policy (INTERPRETATION, conservative):** we never use `word_onsets_s` absolute timing, `shuffle_seed`, or any cross-subject correspondence of holdout rows, including *within* Broad. Each holdout window is predicted from that subject's MEG alone.
- **Gray zone, unresolved:** unlabelled **test-time adaptation** using a subject's *own* holdout MEG statistics (e.g. fitting a per-subject sensor alignment or normalisation on that subject's unlabelled holdout windows, with no row correspondence to other subjects). This is not forbidden by any text I found. It is the natural route for the 0% group, whose only data is the holdout. Per-subject channel standardisation from the holdout is *already required* by the official pipeline, which is a precedent for using the holdout's unlabelled statistics. Plan: build it as a switchable variant and always keep a non-TTA fallback submission. Confirming with the organizers (libribrain@robots.ox.ac.uk or Discord) is Rohit's call.

## 5. External data, pretrained models, compute

- **VERIFIED (arXiv 2609.03231):** "Participants may use any training data for either track. This includes (i) external, publicly available datasets, (ii) self-collected or synthetically-generated data, or (iii) pre-trained models available from public repositories on Hugging Face or GitHub." The paper has no rule on test-time adaptation.
- The paper also states that the holdout is split into **two disjoint partitions**, public and private.

### Official baselines (VERIFIED, arXiv 2609.03231)
| Track | Model | BAcc@10 | BAcc@1 |
|---|---|---|---|
| Deep | d'Ascoli et al. 2025 supervised word decoder, all ~80 h sub-0 | 0.7323 | 0.2560 |
| **Broad** | **MEG-XL** (pretrained ~300 h / 800 subj; fine-tuned; per subj1–32, half of ses-11 fine-tune, half validation) | **0.4296** | 0.0620 |
| Broad | d'Ascoli | 0.3313 | 0.0514 |

The paper gives no per-group (100/50/25/0%) breakdown. MEG-XL: github.com/neural-processing-lab/MEG-XL (MIT, cloned to `ext/`); weights `pnpl/MEG-XL` (283 MB). Its libribrain100 config uses 150 s segments at 50 Hz, a 0.1–40 Hz band-pass, a −0.5 s window offset, a T5-large layer-12 contrastive word target, and subject FiLM. The README requires **≥80 GB VRAM** for fine-tuning (activation memory from the 150 s context). The config has a gradient-checkpointing flag.

- Compute: no limit stated. Teams: one team per person; the maximum team size is not stated.

## 6. Consequences for the local validation design (INTERPRETATION)

The final test is **new subjects-or-few-shot subjects × new stimuli**. Because all broad subjects and sub-0 heard Sherlock1 ses-11/12, a split that holds out subjects but reuses ses-11/12 stimuli in training leaks stimulus identity: the same word tokens with the same acoustics and context. The local split must be disjoint on **both axes**:

- **Zero-shot proxy:** hold out K whole subjects from subj1–32. Evaluate on a stimulus block that *no* training subject (sub-0 included) was trained on.
- **Few-shot proxy:** for the same held-out subjects, fine-tune on a stimulus block A and evaluate on a disjoint block B. B is also excluded from everyone's training.
- The score is reported by group (0/25/50/100%) and combined with the holdout's row weights (18% zero-shot, 82% few-shot).
