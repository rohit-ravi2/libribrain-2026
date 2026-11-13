# 26th of 91 on Deep: a contextual-embedding target, recording-day BatchNorm, Sinkhorn balancing, and a selection I would make differently

## Context

This is my writeup for the Deep track of PNPL LibriBrain 2026. The task is to name which of 50 words a listener heard from
306 channels of MEG, scored as top-10 balanced accuracy with chance at 0.20. Deep has a single listener, subject 0, with
about 80 hours of recordings. My best public score was 0.72928, 15th on the public board at the close. The selected file
that counted on the private leaderboard scored 0.72727 public and 0.69704 private, which placed 26th of 91. The private winner scored 0.763. Two files I submitted but did not select
scored 0.71779 privately, which would have placed about 18th. Most of this writeup is about the method, but the gap between
those numbers is the most useful thing I can report, and it has its own section.

Code, the experiment log with predictions written before most runs, and every submission with public and private scores are at
https://github.com/rohit-ravi2/libribrain-2026. The project page is https://www.rohitravi.com/projects/libribrain-2026/.

## Overview

An encoder of about five million parameters is trained on subject 0 with a contextual text embedding as its main target and
a four-layer sentence transformer on top. At test time I re-estimate BatchNorm statistics on the holdout recording day,
average seeds, rescore with a bigram HMM learned from the holdout sentences themselves, and apply Sinkhorn class balancing at
temperature 3. Isolated single-word rows get a separate 1 s decoder.

## Data and preprocessing

Continuous MEG is resampled from 250 to 50 Hz with a Kaiser-windowed FIR filter. Each word is a 3 s window from one second
before onset to two seconds after, zeroed outside its own sentence epoch, with a validity-mask channel; windows are normalised
over valid samples only. Isolated single-word epochs use [0, 1) s.

I excluded relative word timing as an input. Given inter-word intervals, a model rose from 0.30 to 0.51 locally, and I judged
that to be a property of the stimulus rather than of the brain. The sentence transformer runs with its timing inputs zeroed.
That did not make the recipe timing-free. After the competition I found that the sentence-clipped windows and the
validity-mask channel still expose word duration: trained and scored on shared noise in place of MEG, the same recipe reaches
0.272 top-10 against 0.374 on real MEG, so roughly 0.06 to 0.07 of a score in this family comes from window geometry.

## Model and training target

The word encoder is a 1 × 1 projection to 128 channels, three residual temporal blocks, two strided convolutions and a
linear map to a 256-dimensional embedding; for Deep's final models I widened it (d 192, D 384). A four-layer pre-norm
transformer attends across the words of each sentence. One head is a 50-way classifier. The other is a 1024-dimensional
projection trained with SigLIP against block-12 encoder states of T5-large, computed for every word in the training
audiobooks, roughly a million words, in or out of the vocabulary. Logits are corrected by half the log training prior.

The contextual target was one of the two largest steps. On Broad, together with sentence-clipped windows and the sentence
transformer, it took the public score from 0.225 to 0.428, and the same recipe gave 0.566 on Deep with subject 0's own input layer. Training on subject 0 only with the four-layer sentence
transformer and five seeds then reached 0.654 (+0.088). Bigger target models, acoustic targets, 100 Hz input, wider windows, deeper
transformers and longer training did not help.

## Test-time steps

BatchNorm re-estimation (AdaBN). The BIDS acquisition dates support the view that the Deep holdout was recorded on a day absent from
training, and recomputing BatchNorm statistics on the holdout rows was the largest test-time step on the public board: +0.044.

Self-learned sentence prior. A bigram HMM is fitted by EM to the 156 holdout sentences, with emissions given by the posterior
divided by the mean posterior and transitions smoothed toward uniform with Dirichlet strength κ = 20. Its forward-backward
marginals replace the posteriors. No text corpus is used.

Sinkhorn class balancing. Posteriors are rescaled so that each class receives equal total mass, at T = 3, with offsets
fitted on sentence rows and γ = 0.85 in the last version.

Isolated words. The isolated single-word rows are decoded by a separate 1 s model of subject 0 (+0.003 public).

## What worked and what did not

Effects are differences between pairs of submitted files. For the first, second, fifth and sixth rows the pair differs only
in the named step; the wider-encoder and time-stretch rows compare five-seed ensembles against a ten-seed one, so they mix
the step with ensemble size. The local column is the instrument I used to screen each step: a model with a whole recording
day held out for the first three rows, a held-out prose split for time-stretch and the holdout simulator for the prior. The
local values for width and time-stretch are single seeds.

| Intervention | Local | Public | Private |
|---|---|---|---|
| Sinkhorn balancing, T = 3 | +0.0105 | +0.0104 | +0.0512 |
| AdaBN, full | −0.010 | +0.0438 | +0.0148 |
| Wider encoder | +0.041 | +0.0082 | −0.0171 |
| Time-stretch augmentation | +0.034 | −0.0041 | −0.0255 |
| Self-learned sentence prior, κ = 20 | +0.007 | +0.0084 | −0.0291 |
| Trigram rerank from a vocabulary text corpus | | −0.0468 | −0.0215 |

Sinkhorn balancing generalised, and privately it was worth five times its public gain. AdaBN was positive on both boards but
far smaller privately (+0.015 against +0.044), and my day-held-out local estimate had it slightly negative, so all three
instruments disagree on its size. The sentence prior gained 0.008 publicly and lost 0.029 privately. The damage again tracks
the prior's strength: on the same AdaBN ensemble with balancing, the private scores were 0.71779 with no prior, 0.71553 at
κ = 50, 0.68874 at κ = 20 and 0.64295 at κ = 10. The wider encoder and the time-stretch augmentation both looked strong
on one seed and both lost privately. Re-run over three seeds after the competition, their local gains were +0.019 ± 0.033
and +0.017 ± 0.029, so neither had been established. A five-and-five mix of the two was nonetheless my best selected file. Many other things stayed
within a hundredth of baseline, including CORAL, entropy minimisation, pseudo-labelling on the holdout and a long list of
alternative feature families.

## Public versus private, and final selection

The Deep board moved in the opposite direction to Broad for many of the files without the prior. Among files scoring at
least 0.60 publicly, the prior-free ones had public-to-private gaps from −0.030 to +0.039, most of them positive, while those
carrying the sentence prior at κ = 20 or below lost 0.027 to 0.064. The two groups overlap, so the Deep evidence is weaker
than on Broad. My selected file with the best private score combined the wide and time-stretched
encoders with AdaBN, the isolated-word decoder, the κ = 20 prior and Sinkhorn at T = 3: 0.72727 public, 0.69704 private. The
file with my best public score (0.72928) dropped to 0.68537.

The two best private files were the simplest late ones: ten seeds of the subject-0 recipe, full AdaBN and Sinkhorn at T = 3,
with no prior, no width change and no augmentation, at 0.70767 public and 0.71779 private. Each step I stacked on top of
that file was chosen on public gains of 0.003 to 0.008, or on none for the time-stretch half of the mix, and together they
cost 0.021 privately and eight places. The local support was thin as well: the width gain rested on one day-held-out seed,
time-stretch on one seed of a prose split, and the prior on a simulator that I had calibrated against public effects. In hindsight, gains of that size on a single public split were within the noise of the split, and one
of the two final slots should have gone to the plain AdaBN and balancing file.

## A calibration point for this benchmark

After the competition I measured how much of the score timing alone can supply. On a held-out day of subject 0, an
event-only model that knows word onsets exactly and sees no MEG reaches 0.588 top-10 with fixed 3 s windows. A synthetic
timing twin, trained and scored on noise plus a fixed onset kernel at twice the noise standard deviation, with no word
identity, an isolated 1 s window and no sentence context, scored 0.5915 public and 0.5805 private as a late submission on
Deep. Scores on this track should be read against that level as well as against chance. The comparison with my entry is
not like for like: the twin's kernel is far louder than the brain's onset response, and my model uses 3 s windows with
sentence context and, as noted above, still sees some duration through its clipped windows. I would treat 0.58 as a
reference point rather than a floor to subtract.

## Compute

One RTX 4060 Ti with 8 GB of memory, over 26 days. One training seed took about 40 minutes.

## Rules and compliance

Every subject heard the same 156 holdout sentences, so absolute onset times and the shuffle seed would allow rows to be
matched across subjects and across the Broad and Deep tracks. The organisers disqualify teams that do this. This entry uses no
cross-subject or cross-track information about the holdout; every test-time computation reads one subject's rows. I also
excluded relative word timing as a model input, although the clipped windows and validity mask still carry some duration
information. The audit is in `docs/COMPLIANCE.md` in the repository.

## Links

- Code, experiment log and submissions: https://github.com/rohit-ravi2/libribrain-2026
- Project page: https://www.rohitravi.com/projects/libribrain-2026/
