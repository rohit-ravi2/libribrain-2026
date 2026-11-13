# 17th of 53 on Broad: a single-subject encoder, within-subject balancing, and a sentence prior that did not survive the private split

## Context

This is my writeup for the Broad track of PNPL LibriBrain 2026. The task is to name which of 50 words a listener heard from
306 channels of MEG, scored as top-10 balanced accuracy, so chance is 0.20. Broad has 39 listeners, most with ten to forty
minutes of labelled data and seven with none. My selected submission scored 0.72230 on the public leaderboard, eighth at the
close, and 0.67846 on the private leaderboard, which placed 17th of 53. The private winner scored 0.758. A file I submitted
but did not select scored 0.68037 privately, and I come back to that below, because the gap between my public and private
positions is mostly explained by one test-time step that I trusted too far.

The code, the experiment log with predictions written before each run, and every submission with its public and private
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
timing inputs zeroed.

## Model and training target

The word encoder is convolutional: a 1 × 1 projection to 128 channels, three residual temporal blocks, two strided
convolutions and a linear map to a 256-dimensional embedding. A four-layer pre-norm transformer then attends over the words
of a sentence. Two heads sit on top. One is a 50-way classifier. The other is a 1024-dimensional projection trained with a
SigLIP loss against the block-12 encoder states of T5-large for every word in the training audiobooks, about a million words,
whether or not the word is in the competition vocabulary.

The target was the largest single change in the project. Moving from a plain classifier to the contextual T5 target, with
sentence-clipped windows and the sentence transformer, took the public Broad score from 0.225 to 0.428. Transformer depth,
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
board these two steps added +0.001 each, but the per-subject input layer turned out to matter more privately than publicly
(+0.0013 public, +0.0304 private, measured between two otherwise matched files).

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

Sinkhorn balancing is the cleanest result in the project. The local estimate, the public change and the private change agree
to within a thousandth or so, and the refinement that fits the offsets on sentence rows held up in the same way.

The sentence prior is the opposite case. Its public gain of +0.017 was predicted almost exactly by a holdout simulator built
from real held-out posteriors arranged into vocabulary-only sentences (+0.016), which is why I trusted it. On the private
split it cost 0.036. The damage scales with the prior's strength: across the κ values I submitted, the private scores were
0.665 at κ = 200, 0.646 at 50, 0.634 at 20, 0.616 at 10 and 0.590 at 5, against 0.671 for the same ensemble with no prior.
Publicly κ = 20 looked best. I chose κ with public feedback, and the private numbers suggest the self-learned transitions
fitted something specific to the public rows; I do not yet have a clean explanation beyond that. An external text trigram was
worse still, since a text prior overrides good brain evidence as readily as bad.

AdaBN generalised only in part. The full version hurt on both boards, while a half-strength version (α = 0.5) lost a little
publicly and gained 0.007 privately.

Many other ideas stayed within a hundredth of baseline: CORAL, entropy minimisation, soft pseudo-labelling on the holdout,
fine-tuning the input layer and head, deeper transformers, longer training, and a long list of alternative feature families
including reservoir computing, Riemannian and DTW features, and MEG-XL embeddings. One probe did replicate. Sentence-initial
words decode 0.12 to 0.32 better than the same words mid-sentence for unseen subjects, consistent with a stronger response
after silence, though it depends on position and cannot be used at test time.

## Public versus private, and final selection

Every Broad file built on the sentence prior with κ of 50 or below lost between 0.04 and 0.10 going from public to private
(the weak κ = 200 version lost 0.024), while the late files without it lost about 0.01 to 0.02. My selected file (public 0.72230, private 0.67846) carries the prior. The best file I submitted
privately was an AdaBN α = 0.5 ensemble with Sinkhorn balancing at T = 2 and no prior, which scored 0.68904 publicly and
0.68037 privately; that would have placed about 16th. The margin is small on Broad, but it is the same lesson as on Deep,
where it cost eight places. A step whose public gain came through a parameter tuned against the public board deserved a
hedge in final selection, and the safer second slot would have been the strongest file without it.

## Compute

Everything ran on one RTX 4060 Ti with 8 GB of memory over 26 days. One training seed took about 40 minutes, and the
selected Broad file averages five seeds.

## Rules and compliance

Every subject heard the same 156 holdout sentences, so absolute onset times and the shuffle seed would allow rows to be
matched across subjects and across the two tracks. The organisers disqualify teams that do this. Nothing in this entry uses
that correspondence or any other cross-subject or cross-track information about the holdout; each test-time computation reads
a single subject's rows. As noted above, I also excluded relative word timing as a model input. The audit is in
`docs/COMPLIANCE.md` in the repository.

## Links

- Code, experiment log and submissions: https://github.com/rohit-ravi2/libribrain-2026
- Project page: https://www.rohitravi.com/projects/libribrain-2026/
