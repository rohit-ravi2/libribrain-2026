# Compliance note (for organiser code review) — 2026-10-14

Sources checked on 2026-10-14 (live, via Kaggle CLI; snapshot in `ref/kaggle_20261014/`): Broad and Deep pages (Description, Data, Rules, Evaluation, Foundational Rules; byte-identical to the 2026-09-21 snapshot), every discussion topic (Broad: the 2026-09-18 leakage announcement; Deep: final-selection Q&A, submission-system bug), the pnpl holdout loader docs, and the competition paper (arXiv 2609.03231).

**Explicit prohibitions found:** (1) mapping predictions between tracks or subjects using `word_onsets_s` absolute timing or `shuffle_seed` ("or a similar form of cross-track leakage"); (2) hand labelling / human prediction of test data (Kaggle Foundational Rules 4b); (3) paper: "We discourage any attempt to infer held-out labels outside of the official evaluation procedure." **Not addressed anywhere:** test-time adaptation, transductive inference, use of unlabelled holdout statistics. Training data: unrestricted ("any training data", public pretrained models allowed).

**What our final files do**
- Training: subj0's released LibriBrain100 recordings only (`--broad_frac 0`); T5-large (public) for text targets. No holdout data in training. Checkpoints `runs/final6_s{0..9}_ep7.pt`, all `rel=0` (no word-timing input), `final=1`.
- Inference (`scripts/predict_holdout_v3.py`): each subject's holdout predicted from that subject's own MEG. `word_onsets_s` is used only as the official loader uses it, to cut each word's window inside its own sentence epoch; relative timing is not a model input. `shuffle_seed` is never read.
- Within-subject test-time steps, each computed over one subject's own holdout rows, never across subjects or tracks: per-subject class balancing (`balance_submission.py`, Sinkhorn); for Deep, re-estimating the encoder's BatchNorm statistics on subj0's own holdout epochs (`--bnrecal 1`, AdaBN); a sentence prior learned by EM from the subject's own decoded sentences (`hmm_prior.py --kappa 20`).
- Scalars chosen with public-leaderboard submissions (official procedure): Sinkhorn T, κ, AdaBN on/off.
- No public LB entry ever used cross-track or cross-subject mapping, so there is no score discrepancy to declare.
