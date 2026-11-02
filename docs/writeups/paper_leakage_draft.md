# When the stimulus decodes itself: shortcuts and evaluation failures in a non-invasive word-decoding benchmark

Rohit Ravi (New York University)

*Draft, 2026-10-18. Numbers trace to `PREDICTIONS.md`, `results/lb_history/` and `runs/*.json` in the accompanying
repository. Items marked [VERIFY] need a check before submission.*

## Abstract

Benchmarks for decoding language from non-invasive brain recordings are usually judged by a single accuracy figure,
and the figure is easy to inflate without decoding anything from the brain. We report what happened while building an
entry for the LibriBrain 2026 competition, in which a 50-word vocabulary is decoded from 306-channel MEG and scored by
top-10 balanced accuracy (chance 0.20). Five distinct mechanisms produced accuracy that did not come from neural
evidence, or produced validation estimates that did not hold on the leaderboard. First, a sentence-level transformer
given the relative onset times of the words in a sentence scored 0.493 ± 0.014 locally over three seeds; retrained
with timing withheld it scored 0.374 ± 0.014, and a smaller transformer without timing was no better than a model with
no sentence context at all. Second, because every test subject
heard the same 156 sentences, rows could be aligned across subjects through stimulus metadata, and the public Broad
leaderboard filled with scores close to the single-subject Deep ceiling. Third, a validation split that shared a
recording day with training hid a 0.044 effect of recording-day drift that only the leaderboard revealed. Fourth,
method changes with local gains transferred to the public leaderboard as +0.001 or as a loss: a per-subject input
layer gained 0.03 to 0.07 locally in every one of 36 configurations on four checkpoints, and a time-stretch
augmentation gained 0.034 in two seeds, a gain that a third seed reduced to 0.017, indistinguishable from zero. The
private split later confirmed the robust gain and not the fragile one. Fifth, the public leaderboard itself
disagreed with the private split released after the close: a self-learned sentence prior that gained 0.017 on the
public Broad split lost 0.036 on the private one, ranking six prior strengths in reverse order, while class balancing
agreed to within 0.001 on both. We describe how each was detected, a simulator of the test set calibrated against
public leaderboard effects that inherited the public split's bias, and an oracle analysis showing that the remaining
limit is single-trial evidence per word. We close with design recommendations for future decoding
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
(Mantegna et al., 2026a), built on the LibriBrain recordings (Özdogan et al., 2025; Mantegna et al., 2026b), on a single 8 GB GPU, pre-registering each experiment with a predicted effect before running it. The
final entry scored 0.722 on the public Broad leaderboard (39 listeners with at most 40 minutes of labelled data each)
and 0.729 on the public Deep leaderboard (one listener with about 80 hours); on the private split, released after the
close, the same files scored 0.678 and 0.685 (the best-scoring of the selected Deep files, 0.697, is the one Kaggle
ranked). These compare with published references of 0.430 (MEG-XL; Jayalath & Parker Jones, 2026) and 0.732 (d'Ascoli
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

## 4 Ways accuracy was not what it seemed

### 4.1 The stimulus timing shortcut

The first version of the sentence-level transformer received, alongside each word's MEG embedding, the word's onset
time relative to the first word of its sentence. These offsets are part of the test-set metadata, which is required
to cut each word's window from its sentence epoch. In the first single runs, local validation accuracy rose from
0.29 (no sentence context) to 0.33 with a two-layer transformer and to 0.51 once the T5 targets were added.

We repeated every rung of this ladder, with and without the timing input, over three training seeds (mean ± standard
deviation; intervals are 95% t intervals over seeds, and contrasts are paired by seed). The model without sentence
context scored 0.309 ± 0.015. The two-layer transformer scored 0.300 ± 0.009 without timing and 0.325 ± 0.022 with it.
Without timing the transformer added nothing over the context-free model (paired difference −0.009, interval −0.032
to 0.014). At this rung the timing gain itself (+0.025, interval −0.047 to 0.097) is not distinguishable from zero
with three seeds, and the single run's 0.33 was the favourable end of the range. With T5 targets the picture is
unambiguous. The timing-free model scored 0.374 ± 0.014 and the model with timing 0.493 ± 0.014. Timing added 0.119
(interval 0.073 to 0.166, at least 0.106 on every seed), more than the 0.065 (interval 0.025 to 0.105) that the T5
targets and neighbouring MEG added over the context-free model.

We tested the source of that gain by zeroing the timing input at inference on each of the three trained T5
checkpoints. On the subjects seen in training, accuracy fell from 0.500 ± 0.033 to 0.308 ± 0.010 on one validation
block and from 0.520 ± 0.018 to 0.290 ± 0.007 on the other. Breaking each sentence into single-word segments gave
0.260 and 0.267. Zeroing an input the model was trained with is out of distribution, which is why the retrained
timing-free rungs above are the primary evidence. A timing-free T5 model trained for longer reached 0.463 in a single
run. Ablations of that model (single-word segments, or neighbouring words replaced by random windows from the same
subject) confirmed that its context gain came from neighbouring MEG rather than from segment structure.

The mechanism is simple. Inter-word intervals encode word durations, and durations identify short function words
well. To measure how well, we trained a gradient-boosted classifier on timing metadata alone, with no MEG input. Its
features were each word's onset relative to its sentence, the gap to the next word's onset (or to the sentence end),
its position, the number of words and the sentence duration, all of which a test row exposes. Trained on the Deep
subject's training sentences and scored on the same held-out blocks as the MEG decoder, it reached 0.632 top-10
balanced accuracy. The duration proxy alone gave 0.468 and position alone 0.361 (chance 0.20). Those blocks are one
split. Over five folds grouped by recording session, covering all of the Deep subject's labelled vocabulary words, the
same model scored 0.593 (three seeds, standard deviation below 0.001), the duration proxy alone 0.469 and position
alone 0.380. The holdout exposes less than prose does: its epochs end one second after the last onset, so the last
word's gap and the sentence duration carry no information about where the sentence really ended. Restricting the data
to sentences of 5 to 9 words and framing each as a holdout epoch, the timing model still reached 0.572 ± 0.002 (gap
alone 0.441, position alone 0.433; `docs/writeups/analyses/timing_only_cv.py`). For comparison, the
timing-free MEG decoder scored about 0.70 to 0.78 on the Deep subject's blocks and 0.64 to 0.69 on new subjects. A
metadata model with no access to the brain thus recovers most of the margin above chance that a brain decoder reaches
on this benchmark's format: timing alone clears chance by 0.37, against 0.50 to 0.58 for the MEG decoder on the
Deep subject's blocks and 0.44 to 0.49 on new subjects. Combining the two adds little, though. We
fused the timing model's posteriors with those of a timing-free Deep decoder (log-linear weight chosen on one block and
scored on the other, all sentences framed as holdout epochs, about 1,300 scored words per block). On a checkpoint that
never saw either evaluation session, MEG alone scored 0.693 and the fusion 0.705 ± 0.011 over three timing-model seeds;
on a same-day checkpoint, 0.757 and 0.770 ± 0.012. The gain of about 0.01 was positive in 10 of 12 cross-fitted
comparisons and is about one seed standard deviation. A strong decoder already carries most of what the timing carries,
plausibly because the evoked response itself reflects word duration. The shortcut is therefore largest where brain
evidence is weakest, as for the cross-subject transformer above, which scored 0.374 without timing and gained 0.119
from it; a timing gain measured on a weak decoder says little about a strong one. (All timing analyses were run after the competition;
`docs/writeups/analyses/timing_only_baseline.py`, `timing_only_cv.py` and `timing_meg_fusion.py`.) The transformer learned to read word identity from the audio file's timing rather than from the brain. Nothing in the
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
corrected instrument was faithful for some interventions and not others. On the private split the re-estimation gain was
0.015 rather than 0.044, still positive and still outside what the same-day split could see.

The same re-estimation lowered the Broad score by 0.011, and a partial version that improved Broad locally by 0.028
lowered it by 0.002 on the leaderboard. A new recording day of a known brain benefits from re-standardisation; a new
brain is better served by the reference subject's statistics.

### 4.4 Local gains that did not transfer

Three changes passed strict local tests and failed on the public leaderboard. Seeds added after the competition
(`docs/writeups/analyses/seeded_cis.py`) separate them.

- **Per-subject input layers fitted on labelled data (Broad).** Fitting a 306 × 306 input map for each Broad subject on
  that subject's labelled recordings, with the network frozen, improved local accuracy by 0.03 to 0.07 in all 18
  configurations tested (two checkpoints, three data fractions, three step counts). On the public leaderboard it added
  0.001. A variant initialised from a temporal-lobe sensor weighting lost 0.003. On the private split the same pair of
  files differs by +0.030, close to the local estimate (Section 4.7). Repeated on two further checkpoints trained with
  new seeds, all 18 new configurations gained again (0.036 to 0.065); across the four checkpoints and 36
  configurations the gain ranges from 0.030 to 0.068 with no exception.
- **Time-stretch augmentation (Deep).** Training with windows stretched by 0.8 to 1.25 around the onset improved the
  Deep subject's local accuracy by 0.032 and 0.036 in two seeds, the estimate we acted on (and the one Figure 1
  plots). On the public leaderboard the same pipeline with it scored 0.712, against 0.716 without it; on the private
  split, 0.663 against 0.689 (−0.025). A third seed of both arms, trained after the competition, reversed the local
  sign (−0.016): over three seeds the paired gain is 0.017 ± 0.029 (95% interval −0.054 to 0.089), and the baseline
  alone varied from 0.703 to 0.735 across seeds. Part of this failure was therefore local. Two seeds were too few to
  establish a gain of this size, and the leaderboard loss is consistent with an effect near zero.
- **Partial batch-normalisation re-estimation (Broad):** +0.028 locally, −0.002 on the public leaderboard, +0.007 on the
  private split (Section 4.3).

Figure 1 plots every intervention we measured both locally and on the public leaderboard. Estimates from the
calibrated simulator (Section 5) lie near the diagonal, apart from one smoothing setting; estimates from same-session
prose validation fall below it for every per-subject or training-time change; the day-novel instrument is right for
class balancing and wrong in sign for batch-normalisation re-estimation.

![Local estimate against public-leaderboard effect for every intervention measured both ways, by local instrument. Data: `docs/writeups/analyses/local_vs_lb.csv`; script: `docs/writeups/analyses/fig_local_vs_lb.py`.](../figures/local_vs_lb.png){#fig:local width=75%}

All three local estimates were made on held-out quarters of the same recording sessions used for fitting, in long
prose sentences with mostly fully valid windows. The leaderboard rows came from a different session, in short
constructed sentences with heavy masking and isolated one-second words. Each change could learn something specific to
the validation session, and our split could not tell. That reading has to be qualified twice. The private split
restores most of the per-subject fit's local gain and leaves the time-stretch loss in place (Section 4.7). And the
seeds show that the time-stretch gain, like the wider encoder's day-novel gain (Section 4.7), was never established
locally: each rested on one or two seeds and shrank toward zero with a third. The one local gain that was robust
across seeds and configurations is also the one the private split confirmed. Against the public split alone, a
same-session local gain was uninformative about transfer. Against the private split, with enough seeds to separate
a gain from training noise, it was right in the one case where it was solid.

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
improved the public Broad score by 0.006 to 0.012 when the decoder scored about 0.43. Applied to a decoder scoring
0.66, the same prior lowered it by 0.11, and on the Deep track by 0.047. On the private split those losses were 0.090 and 0.022. A prior learned without labels from each subject's own 156
decoded sentences, by contrast, appeared to help at the higher level on the public split (+0.017 Broad, +0.008 Deep),
and it is the clearest casualty of Section 4.7: on the private split it lost 0.036 and 0.029. Text priors substitute
for evidence rather than complementing it: when brain evidence is strong, a mismatched prior overrides it.
Results reported for language-model rescoring on weak decoders need not hold for strong ones.

### 4.7 The public split disagreed with the private split

The private scores, released after the close, cover the roughly 70% of rows the public leaderboard did not. Every
submission was scored on both, so each leaderboard comparison in this paper can be read twice (Figure 2;
`results/lb_private_20261102/`, built by `docs/writeups/analyses/private_lb_table.py`). Several readings agree.
Sinkhorn class balancing gained 0.036 on the public Broad split and 0.037 on the private one, the prose trigram prior
lost 0.111 and 0.090, and restricting the Sinkhorn offsets to sentence rows gained 0.0085 and 0.0087. The family we
tuned hardest on the public split does not agree. The self-learned sentence prior traced an interior optimum on the
public Broad split over its smoothing strength κ (0.688, 0.705, 0.710, 0.704, 0.689, 0.692 for κ = 5, 10, 20, 50, 200,
∞), and the private split ranks the same six files in strictly reverse order of prior strength (0.590, 0.616, 0.634,
0.646, 0.665, 0.671). On Broad the public split holds about 11,000 rows and the private split about 26,000, so the
paired standard error of a difference is near 0.004 and 0.003; a sign reversal of 0.05 is not sampling noise under a
row-random split. The two splits are measuring different populations of rows, in a way the prior is sensitive to and
class balancing is not.

We can say where the prior acts, though not yet which rows are private. The hidden Markov model rewrites only sentence
rows, but the Sinkhorn balancing that follows it is fitted jointly over each subject's sentence and isolated-word rows.
Measured as the number of words entering or leaving a row's top ten relative to the prior-free file, the prior at κ =
20 changes isolated-word rows more than sentence rows (3.1 against 1.4 words per row) and at κ = 5 more again (4.8
against 2.3). A split that weights the two row types differently would therefore see the prior differently. The
direction of the private result implies that the prior's indirect effect on isolated-word rows, or its effect on some
subset of subjects or sentences, is harmful and the public split under-sampled it. Resolving this needs the row-level
public/private assignment, which only the organisers hold.

The private split also reverses the reading of two interventions in Section 4.4. The labelled per-subject input layer,
which we reported as a transfer failure on the public split (+0.001), gained 0.030 on the private one, close to its
local estimate of 0.041; this pair is confounded with the sentence prior, which both files carry, so the honest reading
is that the per-subject fit helped once the prior's damage is held fixed. The Deep wider encoder, promoted on a
day-novel local gain of 0.041 and a public gain of 0.008, lost 0.017 on the private split. Its local gain was one seed
on a 141-row block; two further seeds gave −0.019 and +0.035, a paired mean of 0.019 ± 0.033 (95% interval −0.064 to
0.102), so the local instrument never separated it from zero. Deep class balancing, +0.010
on the public split, gained 0.051 on the private one, about five times as much. Deep comparisons rest on about 290
public and 670 private rows and differences below about 0.03 there remain uninterpretable on either split.

![The same interventions on the two leaderboard splits. (a) Local estimate against the private-split effect. (b) Public-split against private-split effect for the same pairs of submission files. Data: `results/lb_private_20261102/interventions_public_private.csv`; script: `docs/writeups/analyses/fig_public_vs_private.py`.](../figures/public_vs_private.png){#fig:pubpriv width=100%}

For benchmark design the consequence is direct. A public leaderboard computed on a fixed subset of rows is itself a
validation instrument, and it can be unfaithful in the same way as a same-session split when the subset is not
exchangeable with the rest. Participants who tune against it, as we did over 61 submissions, inherit its bias. Roelofs et al. (2019) found little adaptive overfitting across 120 Kaggle
competitions, and the reusable-holdout and Ladder mechanisms (Dwork et al., 2015; Blum & Hardt, 2015) address the
adaptive kind. What we observed is different in kind: a split whose two parts are not exchangeable, which no amount of
restraint in querying the public score would correct. Our own
final selections were made on public scores; the best private Broad file we submitted (0.680, partial
batch-normalisation re-estimation without the sentence prior) was not among them, and neither was the best private Deep
file (0.718, full re-estimation and class balancing, again without the prior), against 0.697 for the one ranked.

## 5 A calibrated simulator of the test set

Because local prose was a poor proxy for the constructed test sentences, we built a simulator: language-model-generated
5- to 9-word sentences using only the vocabulary, with each word replaced by a real held-out posterior row of that word
from the same subject. The substitution is valid to the extent that per-word evidence is conditionally independent
given the word, which our same-word similarity tests support (Section 6). The simulator reproduced the public
leaderboard effect of the self-learned sentence prior (+0.016 against +0.017 on Broad, +0.007 against +0.008 on Deep),
and we used it to screen 164 post-processing ideas in the final two days. It still misranked one setting badly: it
preferred a weaker smoothing strength for the sentence prior that cost 0.022 on the public split. Worse, the private
split shows that the effect it was calibrated to was itself a property of the public rows: the prior lost 0.036 (Broad)
and 0.029 (Deep) on the private split (Section 4.7). The simulator was faithful to the public split, which was
unfaithful to the test set. Calibration against known leaderboard effects is necessary, and it is not sufficient, and
it is only as good as the leaderboard split it is calibrated to.

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
   epoch lengths) sets the floor that any brain-based claim must clear. On this benchmark's epoch format that floor is
   about 0.57 for a single listener's prose, not 0.20.
3. **Hold out recording days, not just stimuli.** Publish acquisition dates and require validation splits disjoint
   in day as well as in subject and stimulus.
4. **Score held-out data in the test set's own format.** If test sentences are constructed and short, validation
   should be too, including the isolated-word rows.
5. **Control for word identity before comparing conditions.** Accuracy differences between positions, states or
   subjects are confounded by which words occur.
6. **Report the oracle pooling ceiling.** It separates representation limits from evidence limits.

## 8 Limitations

All numbers come from one competition, one model family and one research group. Many effects rest on one or two seeds;
the Deep public split has about 288 rows, so differences under about 0.03 there are not interpretable. The row-level
public/private assignment is not public, so Section 4.7 establishes that the two splits disagree but not why.

## References

- Blum, A., & Hardt, M. (2015). The Ladder: A reliable leaderboard for machine learning competitions. *ICML*.
  arXiv:1502.04585.
- Cuturi, M. (2013). Sinkhorn distances: lightspeed computation of optimal transportation distances. *NeurIPS* 26.
  arXiv:1306.0895.
- d'Ascoli, S., Bel, C., Rapin, J., Banville, H., Benchetrit, Y., Pallier, C., & King, J.-R. (2024). Decoding individual
  words from non-invasive brain recordings across 723 participants. arXiv:2412.17829.
- Dwork, C., Feldman, V., Hardt, M., Pitassi, T., Reingold, O., & Roth, A. (2015). The reusable holdout: Preserving
  validity in adaptive data analysis. *Science* 349(6248), 636–638.
- Geirhos, R., Jacobsen, J.-H., Michaelis, C., Zemel, R., Brendel, W., Bethge, M., & Wichmann, F. A. (2020). Shortcut
  learning in deep neural networks. *Nature Machine Intelligence* 2(11), 665–673.
- Jayalath, D., & Parker Jones, O. (2026). MEG-XL: Data-efficient brain-to-text via long-context pre-training. *ICML*.
  arXiv:2602.02494.
- Kapoor, S., & Narayanan, A. (2023). Leakage and the reproducibility crisis in machine-learning-based science.
  *Patterns* 4(9), 100804.
- Li, Y., Wang, N., Shi, J., Liu, J., & Hou, X. (2016). Revisiting batch normalization for practical domain adaptation.
  arXiv:1603.04779.
- Mantegna, F., Elvers, G., Jayalath, D., Landau, G., et al. (2026a). The 2026 PNPL Competition: Word classification and
  efficient cross-subject generalisation in LibriBrain100. arXiv:2609.03231.
- Mantegna, F., Jayalath, D., Elvers, G., et al. (2026b). LibriBrain100: One hundred hours of broad and deep MEG data for
  neural speech decoding at scale. arXiv:2608.25204.
- Özdogan, M., Landau, G., Elvers, G., Jayalath, D., Somaiya, P., Mantegna, F., Woolrich, M., & Parker Jones, O. (2025).
  LibriBrain: Over 50 hours of within-subject MEG to improve speech decoding methods at scale. arXiv:2506.02098.
- Packard, N. H. (1990). A genetic learning algorithm for the analysis of complex data. *Complex Systems* 4(5).
- Roelofs, R., Shankar, V., Recht, B., Fridovich-Keil, S., Hardt, M., Miller, J., & Schmidt, L. (2019). A meta-analysis
  of overfitting in machine learning. *NeurIPS* 32, 9175–9185.

Every entry was checked by direct fetch; URLs in `docs/writeups/references_verified.md`.

## To finish before posting

- Seeded confidence intervals: done (`seeded_cis.py`), Sections 4.1, 4.4, 4.7.
- Private-leaderboard numbers for Section 4.4.
- Timing-only baseline: done, including holdout epoch format (0.572) and timing + MEG fusion (+0.01).
- Figure 1 (local vs leaderboard): done.
- Decide venue: bioRxiv (no endorsement needed) now; arXiv cs.LG / q-bio.NC once an endorser is available.
