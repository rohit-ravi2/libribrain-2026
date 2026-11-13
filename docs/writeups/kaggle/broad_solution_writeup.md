# 17th of 53 on Broad: a single-subject encoder, within-subject balancing, and a sentence prior that did not survive the private split

## Context

This is my writeup for the Broad track of PNPL LibriBrain 2026. The task is to name which of 50 words a listener heard from
306 channels of MEG, scored as top-10 balanced accuracy, so chance is 0.20. Broad has 39 listeners, most with ten to forty
minutes of labelled data and seven with none. My selected submission scored 0.72230 on the public leaderboard, eighth at the
close, and 0.67846 on the private leaderboard, which placed 17th of 53. The private winner scored 0.758. A file I submitted
but did not select scored 0.68037 privately, and I come back to that below, because part of the gap between my public and
private positions comes from one test-time step that I trusted too far.

The code, the experiment log with predictions written before most runs, and every submission with its public and private
score are in the repository at https://github.com/rohit-ravi2/libribrain-2026. A longer account is on the project page,
https://www.rohitravi.com/projects/libribrain-2026/.

## Overview

The final pipeline has three parts. An encoder of about five million parameters is trained on subject 0, the Deep
listener, and nobody else, with a contextual text embedding as its main target. It is applied to the Broad listeners
through a per-subject 306 × 306 input layer, fitted on labelled data where a listener has it. At test time, each listener's
posteriors are averaged over seeds, rescored by a bigram hidden Markov model learned from that listener's own holdout
sentences, and balanced across classes with a Sinkhorn step. Every test-time computation reads one subject's rows and
nothing else.

## Data and preprocessing

Continuous MEG is resampled from 250 to 50 Hz with a Kaiser-windowed FIR filter. Each word becomes a 3 s window running from
one second before onset to two seconds after, zeroed outside the word's own sentence epoch and accompanied by a validity-mask
channel so that the network knows which samples are real. Isolated single-word epochs use the window [0, 1) s. Windows are
normalised over their valid samples only.

I excluded relative word timing as an input. A model given inter-word intervals rose from 0.30 to 0.51 locally, which says
more about the stimulus metadata than about the brain, so the feature was removed and the transformer described below has its
timing inputs zeroed. That did not make the recipe timing-free. After the competition I found that the sentence-clipped
windows and the validity-mask channel still expose word duration: trained and scored on shared noise in place of MEG, the
same recipe reaches 0.272 top-10 against 0.374 on real MEG, so roughly 0.06 to 0.07 of a score in this family comes from
window geometry rather than the brain.

## Model and training target

The word encoder is convolutional: a 1 × 1 projection to 128 channels, three residual temporal blocks, two strided
convolutions and a linear map to a 256-dimensional embedding. A four-layer pre-norm transformer then attends over the words
of a sentence. Two heads sit on top. One is a 50-way classifier. The other is a 1024-dimensional projection trained with a
SigLIP loss against the block-12 encoder states of T5-large for every word in the training audiobooks, about a million words,
whether or not the word is in the competition vocabulary.

The target was one of the two largest steps in the project, though it arrived bundled. Moving to the contextual T5 target
together with sentence-clipped windows and the sentence transformer took the public Broad score from 0.225 to 0.428
(+0.203); training on subject 0 alone, described next, added +0.227. Transformer depth,
window length and training time each moved it by a point or two afterwards. Larger targets (T5-3B, −0.016), acoustic
wav2vec2 targets (+0.005), 100 Hz input (−0.052) and wider windows (−0.022) did not help. Logits are corrected by half the
log training prior before averaging.

## Cross-subject transfer

The second large step went against my expectations. A model trained on subject 0 alone and applied to the Broad listeners
with identity input layers scored 0.655 on the public board, against 0.428 for the best multi-subject recipe at the time.
Mixing the Broad listeners' own labelled data into training lowered transfer steadily, from 0.632 locally with none of it to
0.452 when it made up half of each batch, and adding twenty external listeners from the Le Petit Prince MEG corpus cost 0.15.
My reading is that ten to forty minutes per head contributes more inter-subject variance in sensor geometry and noise than
information about words, whereas about eighty hours of one brain yields a mapping that carries across people.

The Broad listeners' labelled minutes were more useful for alignment than for training. In the final pipeline each labelled
subject gets its own input layer fitted on its labelled data, with a low-rank form for subjects under thirty minutes. The
seven unlabelled subjects receive the average of six donor subjects' input layers applied to their own MEG. On the public
board these two steps added +0.001 each, but the per-subject input layer looked better privately (+0.0013 public, +0.0304
private). That file pair is not a clean comparison, since the two files also differ in seed count (five against ten) and in
how the subjects without labels were handled. The direction is supported locally, where the per-subject layer was positive
in all 36 configurations tested over four checkpoints, by +0.030 to +0.068.

## Test-time steps

All three steps operate on one subject at a time.

Sinkhorn class balancing. The model inherits the skew of ordinary prose, in which frequent function words receive too much
mass. Each subject's posteriors are rescaled so that every class receives equal total mass, at temperature T = 2, later 1.75,
with the offsets fitted on sentence rows only.

Self-learned sentence prior. Every subject heard 156 holdout sentences built from the 50 words. A bigram HMM is fitted by EM
to one subject's holdout sentences, with emissions given by the posterior divided by that subject's mean posterior and
transitions smoothed toward uniform with Dirichlet strength κ = 20; its forward-backward marginals replace the posteriors.
No text corpus is involved.

BatchNorm re-estimation (AdaBN). Statistics are recomputed on the subject's holdout rows. On Broad this was mixed, and the
selected file does not use it.

## What worked and what did not

The table gives the measured effect of each intervention on the local instrument, the public board and the private board.
Changes on the boards are differences between two submitted files that differ only in that step.

| Intervention | Local | Public | Private |
|---|---|---|---|
| Sinkhorn balancing, T = 2 | +0.030 | +0.0358 | +0.0366 |
| Offsets on sentence rows only, γ = 0.85 | +0.0079 | +0.0085 | +0.0087 |
| Per-subject input layer from labelled data | +0.041 | +0.0013 | +0.0304 |
| AdaBN, α = 0.5 | +0.028 | −0.0023 | +0.0072 |
| AdaBN, full | +0.008 | −0.0114 | −0.0041 |
| Self-learned sentence prior, κ = 20 | +0.016 | +0.0174 | −0.0363 |
| Trigram rerank from a vocabulary text corpus | | −0.1114 | −0.0898 |

Sinkhorn balancing is the cleanest result in the project. The public and private changes agree to within a thousandth, the
local estimate was +0.030, and the refinement that fits the offsets on sentence rows held up in the same way.

The sentence prior is the opposite case. Measured against the ten-seed balanced file without it, its public gain was
+0.0174, and a holdout simulator built from real held-out posteriors arranged into vocabulary-only sentences reproduced it
(+0.016). That agreement was weaker evidence than it looked, because I had calibrated the simulator against public
leaderboard effects. A local test of the prior on held-out prose had shown it losing, and I set that aside on the grounds
that prose was a poor model of the constructed test sentences. On the private split the prior cost 0.036. The damage scales with the prior's strength: across the κ values I submitted, the private scores were
0.665 at κ = 200, 0.646 at 50, 0.634 at 20, 0.616 at 10 and 0.590 at 5, against 0.671 for the same ensemble with no prior.
Publicly κ = 20 looked best. The public-private gaps also suggest the split is not exchangeable with respect to this step.
Among Broad files scoring at least 0.60 publicly, the seven without the prior had gaps between −0.009 and −0.022 and the
eighteen with it had gaps between −0.024 and −0.102, with no overlap. The prior acts unevenly across row types: at κ = 20,
counting words that enter or leave a row's top ten, it changes isolated-word rows by 3.14 words per row against 1.42 for
sentence rows, because the balancing that follows it is fitted over both. A private split that weights isolated words,
subjects or sentences differently from the public one would see the prior differently. I cannot check which rows are
private, so this is a reading of the evidence rather than a confirmed cause. An external text trigram was
worse still, since a text prior overrides good brain evidence as readily as bad.

AdaBN generalised only in part. The full version hurt on both boards, while a half-strength version (α = 0.5) lost a little
publicly and gained 0.007 privately.

Many other ideas stayed within a hundredth of baseline: CORAL, entropy minimisation, soft pseudo-labelling on the holdout,
fine-tuning the input layer and head, deeper transformers, longer training, and a long list of alternative feature families
including reservoir computing, Riemannian and DTW features, and MEG-XL embeddings. One probe did replicate. Sentence-initial
words decode 0.12 to 0.32 better than the same words mid-sentence for unseen subjects, consistent with a stronger response
after silence, though it depends on position and cannot be used at test time.

## Public versus private, and final selection

My selected file (public 0.72230, private 0.67846) carries the prior. The best file I submitted privately was an AdaBN
α = 0.5 ensemble with Sinkhorn balancing at T = 2 and no prior, which scored 0.68908 publicly and 0.68037 privately; that
would have placed about 16th. The margin is small on Broad, one place, while on Deep the same choice cost eight. A step
whose support came from the public split, or from a tool calibrated to it, amounted to a single measurement however many
submissions seemed to confirm it, and one of the two final slots should have gone to the strongest file without such
steps.

## Compute

Everything ran on one RTX 4060 Ti with 8 GB of memory over 26 days. One training seed took about 40 minutes, and the
selected Broad file averages five seeds.

## Rules and compliance

Every subject heard the same 156 holdout sentences, so absolute onset times and the shuffle seed would allow rows to be
matched across subjects and across the two tracks. The organisers disqualify teams that do this. Nothing in this entry uses
that correspondence or any other cross-subject or cross-track information about the holdout; each test-time computation reads
a single subject's rows. As noted above, I also excluded relative word timing as a model input, although the clipped windows
and validity mask still carry some duration information. The audit is in
`docs/COMPLIANCE.md` in the repository.

## Links

- Code, experiment log and submissions: https://github.com/rohit-ravi2/libribrain-2026
- Project page: https://www.rohitravi.com/projects/libribrain-2026/
