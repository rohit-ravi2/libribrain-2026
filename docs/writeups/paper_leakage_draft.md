# When the stimulus decodes itself: shortcuts and evaluation failures in a non-invasive word-decoding benchmark

Rohit Ravi (New York University)

*Draft, 2026-10-18. Numbers trace to `PREDICTIONS.md`, `results/lb_history/` and `runs/*.json` in the accompanying
repository. Items marked [VERIFY] need a check before submission.*

## Abstract

Benchmarks for decoding language from non-invasive brain recordings are usually judged by a single accuracy figure,
and the figure is easy to inflate without decoding anything from the brain. We report what happened while building an
entry for the LibriBrain 2026 competition, in which a 50-word vocabulary is decoded from 306-channel MEG and scored by
top-10 balanced accuracy (chance 0.20). Four distinct mechanisms produced accuracy that did not come from neural
evidence, or produced validation estimates that did not hold on the leaderboard. First, a sentence-level transformer
given the relative onset times of the words in a sentence reached 0.51 locally; with timing withheld the same
architecture scored 0.29, no better than a model without sentence context at all. Second, because every test subject
heard the same 156 sentences, rows could be aligned across subjects through stimulus metadata, and the public Broad
leaderboard filled with scores close to the single-subject Deep ceiling. Third, a validation split that shared a
recording day with training hid a 0.044 effect of recording-day drift that only the leaderboard revealed. Fourth,
several method changes with consistent local gains (+0.03 to +0.07 in every one of 18 configurations, or +0.034 in
both of two seeds) transferred as +0.001 or as a loss. We describe how each was detected, a simulator of the test set
calibrated against leaderboard effects that recovered a usable instrument, and an oracle analysis showing that the
remaining limit is single-trial evidence per word. We close with design recommendations for future decoding
benchmarks.

## 1 Introduction

A decoder that classifies heard words from magnetoencephalography (MEG) has two possible sources of information: the
brain's response to the word, and whatever else in the test input correlates with the word's identity. Only the first
is of scientific interest. The second can be substantial, because naturalistic listening experiments carry rich
stimulus structure (word durations, sentence lengths, word order, the identity of the recording session) and modern
models are good at finding it. The general problem is familiar as shortcut learning (Geirhos et al., 2020) and as
leakage in machine-learning-based science (Kapoor & Narayanan, 2023). It has particular force in neural decoding,
where single-trial signal-to-noise is low, so a weak non-neural cue can account for a large share of measured
accuracy.

This paper is a case study. Over 26 days we built an entry for both tracks of the PNPL LibriBrain 2026 competition
(PNPL, 2026) on a single 8 GB GPU, pre-registering each experiment with a predicted effect before running it. The
final entry scored 0.722 on the Broad track (39 listeners with at most 40 minutes of labelled data each) and 0.729 on
the Deep track (one listener with about 80 hours), against published references of 0.430 (MEG-XL) and 0.732 (d'Ascoli
et al., 2024) respectively. Those results are reported elsewhere. Here we document the ways the evaluation, ours and
the benchmark's, could have misled us, because each of them would generalise to other decoding benchmarks.

## 2 Task and data

Each test row is one heard word, to be assigned a probability over a fixed 50-word vocabulary consisting almost
entirely of function words. The score is top-10 balanced accuracy: for each word, the fraction of its occurrences in
which it appears among a row's ten highest-ranked words, averaged over the 50 words.

Training data are the LibriBrain recordings: Elekta/MEGIN 306-channel MEG at 250 Hz, already processed with signal
space separation and a head-position transform, with word-level event annotations. The held-out set has the same
structure for every subject: 156 sentence epochs (each from the first word's onset to one second after the last word,
5 to 9 words, every word drawn from the vocabulary) and 92 isolated one-second word epochs, 960 rows in total. Rows
are shuffled within subject. The public leaderboard scores about 30% of rows; the private ranking uses the remainder
of the same files.

Two properties of this design matter below. Every held-out sentence is built only from the 50 vocabulary words, so the
sentences are constructed rather than natural. And every subject heard the same 156 sentences.

## 3 The decoder, briefly

Our model is a convolutional word encoder over a 3 s window around each word onset (zeroed outside the word's own
sentence epoch), followed by a four-layer transformer across the words of a sentence. It is trained with
cross-entropy over the vocabulary and a SigLIP loss towards contextual embeddings of every heard word from block 12 of
the T5-large encoder, following d'Ascoli et al. (2024). It was trained on the Deep subject's data alone; adding the
Broad subjects' short recordings to training reduced transfer to new subjects monotonically. At test time each
subject's predictions are adjusted with statistics of that subject's own unlabelled rows: Sinkhorn balancing of class
mass (Cuturi, 2013), a bigram hidden Markov model learned by expectation-maximisation from the subject's own
sentences, and, for the Deep subject only, re-estimated batch-normalisation statistics (Li et al., 2016). No
computation ever combines rows from different subjects.

## 4 Four ways accuracy was not what it seemed

### 4.1 The stimulus timing shortcut

The first version of the sentence-level transformer received, alongside each word's MEG embedding, the word's onset
time relative to the first word of its sentence. These offsets are part of the test-set metadata, which is required
to cut each word's window from its sentence epoch. With this input, local validation accuracy rose from 0.29 (no
sentence context) to 0.33 with a two-layer transformer and to 0.51 once the T5 targets were added.

We tested the source of the gain by zeroing the timing input at inference on the trained checkpoint. Accuracy fell
from 0.49 to 0.31 and from 0.54 to 0.29 on the two validation blocks, close to the 0.26 and 0.25 obtained with no
sentence context at all. Because zeroing an input the model was trained with is out of distribution, we then retrained
both rungs without timing. The two-layer transformer without timing scored 0.293, identical to the model without a
transformer (0.292): its whole apparent contribution had been timing. With T5 targets, the timing-free model scored
0.365, and with longer training 0.463; ablations of that model (single-word segments, or neighbouring words replaced by
random windows from the same subject) confirmed that its context gain came from neighbouring MEG rather than from
segment structure.

The mechanism is simple. Inter-word intervals encode word durations, and durations identify short function words
well. To measure how well, we trained a gradient-boosted classifier on timing metadata alone, with no MEG input. Its
features were each word's onset relative to its sentence, the gap to the next word's onset (or to the sentence end),
its position, the number of words and the sentence duration, all of which a test row exposes. Trained on the Deep
subject's training sentences and scored on the same held-out blocks as the MEG decoder, it reached 0.632 top-10
balanced accuracy. The duration proxy alone gave 0.468 and position alone 0.361 (chance 0.20). For comparison, the
timing-free MEG decoder scored about 0.70 to 0.78 on the Deep subject's blocks and 0.64 to 0.69 on new subjects. A
metadata model with no access to the brain thus comes within a few points of a brain decoder on this benchmark's
format. [Exploratory, run after the competition; `docs/writeups/analyses/timing_only_baseline.py`.] The model learned to read word identity from the audio file's timing rather than from the brain. Nothing in the
competition rules forbade within-subject use of relative timing, and on the leaderboard it would very likely have been
worth about 0.2. We excluded it. A decoder that improves when given the stimulus's own temporal structure is decoding
the stimulus.

### 4.2 Cross-subject correspondence of the test set

All test subjects heard the same 156 sentences, and the absolute onset times and a shuffle seed in the released files
make it possible to align a row for one subject with the same word position for another subject, or with the
single-subject Deep track. A strong Deep model can then be copied onto every Broad subject. The organisers identified
this channel during the competition, prohibited it, and announced that prize-eligible code would be reviewed.

The signature on the public leaderboard was visible before the notice was widely understood. On 8 October the top
Broad score (0.729, for subjects with at most 40 minutes of data) nearly matched the top Deep score (0.765, for one
subject with 80 hours), and twenty teams sat between 0.60 and 0.73 while the clean published reference was 0.43.
Few-shot cross-subject decoding matching deep single-subject decoding is not a result anyone has reported. We read the
pattern as leakage and treated the public Broad ranking as uninformative about method quality.

The same correspondence makes a class of tempting test-time methods illegitimate even when they never copy a
prediction: an average class histogram across subjects, a grammar estimated from all subjects' decoded sentences, or
prototypes clustered across subjects all summarise the shared sentence inventory. We restricted every test-time
computation to one subject's own rows.

### 4.3 Validation that shared a recording day with training

Our local validation for the Deep subject held out quarters of one stimulus (two Sherlock Holmes runs) while training
on the other quarters. Local accuracy for the Deep subject ran about 0.05 to 0.08 above its leaderboard score, and
methods that should have helped on a new recording day showed nothing locally.

The BIDS acquisition dates explained it. The Deep subject's 262 runs span 47 recording days, and the two held-out runs
were the only runs recorded on 4 October 2024; their first halves were in every local training set. Every local Deep
number had therefore been a same-day number, while the test set came from a session the training data did not cover.
Re-estimating batch-normalisation statistics on the Deep subject's own test rows improved the Deep leaderboard score by
0.044, an effect our same-day validation could not see. A leave-recording-day-out model, trained with that day removed
entirely, gave a day-novel instrument. It reproduced the leaderboard effect of class balancing on the Deep subject
(+0.0105 against +0.010), but not the effect of batch-normalisation re-estimation (−0.010 against +0.044), so even the
corrected instrument was faithful for some interventions and not others.

The same re-estimation lowered the Broad score by 0.011, and a partial version that improved Broad locally by 0.028
lowered it by 0.002 on the leaderboard. A new recording day of a known brain benefits from re-standardisation; a new
brain is better served by the reference subject's statistics.

### 4.4 Local gains that did not transfer

Three changes passed strict local tests and failed on the leaderboard.

- **Per-subject input layers fitted on labelled data (Broad).** Fitting a 306 × 306 input map for each Broad subject on
  that subject's labelled recordings, with the network frozen, improved local accuracy by 0.03 to 0.07 in all 18
  configurations tested (two checkpoints, three data fractions, three step counts). On the leaderboard it added 0.001.
  A variant initialised from a temporal-lobe sensor weighting lost 0.003.
- **Time-stretch augmentation (Deep).** Training with windows stretched by 0.8 to 1.25 around the onset improved the
  Deep subject's local accuracy by 0.034 in both of two seeds. On the leaderboard the same pipeline with it scored 0.712,
  against 0.716 without it.
- **Partial batch-normalisation re-estimation (Broad):** +0.028 locally, −0.002 on the leaderboard (Section 4.3).

Figure 1 (`docs/figures/local_vs_lb.png`, data in `docs/writeups/analyses/local_vs_lb.csv`) plots every intervention we
measured both locally and on the leaderboard. Estimates from the calibrated simulator (Section 5) lie near the
diagonal, apart from one smoothing setting; estimates from same-session prose validation fall below it for every
per-subject or training-time change; the day-novel instrument is right for class balancing and wrong in sign for
batch-normalisation re-estimation.

The common factor is that all three local estimates were made on held-out quarters of the same recording sessions
used for fitting, in long prose sentences with mostly fully valid windows. The leaderboard rows came from a different
session, in short constructed sentences with heavy masking and isolated one-second words. Each change could learn
something specific to the validation session, and our split could not tell.

### 4.5 A pocket of predictability, and the word confound

Following Packard (1990), we searched for conditions under which the decoder's errors were concentrated, scoring
conjunctions of binned covariates (brain-state band power, artefact level, word position, sentence length) by the
divergence of correctness inside the condition from the base rate, with a sample-size penalty set from a
shuffled-label null. The best condition (first word of a sentence, in one length bin) had a top-10 hit rate of 0.929
on the block where it was found and 0.716 on the other block, where the base rate was 0.708. That pattern looks like a
fluke of the kind Packard warns the search will find.

It was not. Sentence-initial words are a different word mix from mid-sentence words, and harder on average. With word
identity controlled (each row's hit compared with the mean hit of the same word outside the condition), the effect
replicated: +0.28 and +0.12 on the two blocks, and +0.18 to +0.32 for sentence-initial words in new subjects. Removing
the preceding context from mid-sentence windows lowered their accuracy (0.72 to 0.61), which rules out interference
from earlier words' responses and leaves stronger responses after silence as the likely mechanism. The general point
for decoding studies is that any comparison of accuracy across conditions has to control for which words occur in
each condition before it can be read as a statement about neural signal.

### 4.6 Text priors flip sign with evidence strength

A trigram language model over vocabulary words, applied by exact forward-backward marginals within each sentence,
improved the Broad leaderboard score by 0.006 to 0.012 when the decoder scored about 0.43. Applied to a decoder scoring
0.66, the same prior lowered it by 0.11, and on the Deep track by 0.047. A prior learned without labels from each
subject's own 156 decoded sentences, by contrast, helped at the higher level (+0.018 Broad, +0.008 Deep). Text priors
substitute for evidence rather than complementing it: when brain evidence is strong, a mismatched prior overrides it.
Results reported for language-model rescoring on weak decoders need not hold for strong ones.

## 5 A calibrated simulator of the test set

Because local prose was a poor proxy for the constructed test sentences, we built a simulator: language-model-generated
5- to 9-word sentences using only the vocabulary, with each word replaced by a real held-out posterior row of that word
from the same subject. The substitution is valid to the extent that per-word evidence is conditionally independent
given the word, which our same-word similarity tests support (Section 6). The simulator reproduced the leaderboard
effect of the self-learned sentence prior (+0.016 against +0.018 on Broad, +0.007 against +0.008 on Deep), and we used
it to screen 164 post-processing ideas in the final two days. It still misranked one setting badly: it preferred a
weaker smoothing strength for the sentence prior that cost 0.022 on the leaderboard. Calibration against known
leaderboard effects is necessary, and it is not sufficient.

## 6 The information ceiling

If one averages a subject's posteriors over several occurrences of the same word, using labels to group them, top-10
balanced accuracy rises by 0.07 with two occurrences, 0.14 with four and 0.22 with sixteen. The information exists in
aggregate. Without labels, however, same-word trials are not findable: k-nearest-neighbour label purity among a
subject's rows was at chance in each of eight raw-MEG representations (post-onset windows of 0.4 and 0.8 s, 10 or 40
spatial components, with and without per-feature standardisation), and dynamic-time-warping distance did no better
than Euclidean. Even two occurrences of the same word in one sentence, heard seconds apart in the same brain state, were no more similar in raw MEG than different words (AUC 0.51). Everything the pair carries is already in the
decoder's posteriors. The binding constraint is single-trial evidence per word, and no test-time method we tried added
to it.

## 7 Recommendations for decoding benchmarks

1. **Do not share test stimuli across subjects or tracks when metadata allows alignment.** If shared stimuli are
   needed for scientific reasons, release no absolute timing or ordering information that links rows.
2. **Report a timing-only baseline.** A model given only the stimulus metadata a decoder can see (onsets, durations,
   epoch lengths) sets the floor that any brain-based claim must clear. On this benchmark's format that floor is about
   0.63, not 0.20.
3. **Hold out recording days, not just stimuli.** Publish acquisition dates and require validation splits disjoint
   in day as well as in subject and stimulus.
4. **Score held-out data in the test set's own format.** If test sentences are constructed and short, validation
   should be too, including the isolated-word rows.
5. **Control for word identity before comparing conditions.** Accuracy differences between positions, states or
   subjects are confounded by which words occur.
6. **Report the oracle pooling ceiling.** It separates representation limits from evidence limits.

## 8 Limitations

All numbers come from one competition, one model family and one research group. Many effects rest on one or two seeds;
the Deep public split has about 288 rows, so differences under about 0.03 there are not interpretable. The private
leaderboard scores were not yet released at the time of writing. [UPDATE when released.]

## References

- Cuturi, M. (2013). Sinkhorn distances: lightspeed computation of optimal transport. *NeurIPS*.
- d'Ascoli, S., et al. (2024). Decoding individual words from non-invasive brain recordings across 723 participants.
  arXiv:2412.17829.
- Geirhos, R., et al. (2020). Shortcut learning in deep neural networks. *Nature Machine Intelligence* 2, 665–673.
- Kapoor, S., & Narayanan, A. (2023). Leakage and the reproducibility crisis in machine-learning-based science.
  *Patterns* 4(9).
- Li, Y., Wang, N., Shi, J., Liu, J., & Hou, X. (2016). Revisiting batch normalization for practical domain adaptation.
  arXiv:1603.04779.
- Packard, N. H. (1990). A genetic learning algorithm for the analysis of complex data. *Complex Systems* 4, 543–572.
- PNPL (2026). The 2026 PNPL Competition. arXiv:2609.03231. [VERIFY exact title and authors]
- MEG-XL reference. [VERIFY citation]

## To finish before posting

- Seeded confidence intervals for the timing ablation and the transfer failures (at least three seeds each).
- Private-leaderboard numbers for Section 4.4.
- Timing-only baseline: done (0.632 locally). Still to do: the same model on holdout-format constructed sentences, and timing + MEG combined.
- Figure 1 (local vs leaderboard): done.
- Decide venue: bioRxiv (no endorsement needed) now; arXiv cs.LG / q-bio.NC once an endorser is available.
