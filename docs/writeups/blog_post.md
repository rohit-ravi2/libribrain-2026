---
title: "Fifty words from a magnetic field"
startDate: 2026-10-16
description: "Twenty-six days decoding heard words from MEG in the LibriBrain 2026 competition: what moved the score, what did not, and why the remaining distance looks like a property of the measurement rather than of the model."
tags: ["MEG", "Neural Decoding", "Neuroscience", "Machine Learning", "Kaggle", "Research"]
---

<!-- DRAFT for review. Two places marked [ROHIT] want a sentence in your own words. -->

When someone hears a word, the neurons that respond to it produce currents, and the currents produce a magnetic field outside
the head. The field is tiny, on the order of a hundred femtotesla, a few billionths of the Earth's field, and it can be measured
with an array of superconducting sensors in a shielded room. The LibriBrain 2026 competition, run by the PNPL group at Oxford,
asked a narrow version of an old question about that signal: given the field recorded while a person listened to a word, can
you tell which word it was?

The question was narrowed on purpose. There were fifty words, mostly function words such as *the*, *was*, *not* and *they*,
and a prediction counted as correct if the true word appeared among a model's ten best guesses. Chance under that metric is
0.20. Two tracks shared the setup. In the Deep track a single listener had been recorded for about eighty hours, mostly
Sherlock Holmes audiobooks. The Broad track had thirty-nine other people, most with between ten and forty minutes of labelled
listening and seven with none at all.

[ROHIT: a few sentences on why this problem in particular, for example the connection to philosophy of mind and to what it
would mean to read content from a brain, in your own voice.]

I finished eighth, with 0.722 on Broad and 0.729 on Deep. What follows is an account of how the score got there, which ideas
carried it, and why I think the distance that remained is set more by the physics of the measurement than by anything a
model could still learn.

## A slow start, and an audit

The first two weeks went badly. The approach I began with was the natural one for a multi-subject problem: a shared
convolutional trunk trained on everyone, with a small learned input layer per subject to absorb differences in head shape and
sensor position. On my local evaluation it reached about 0.31 for held-out subjects. On the public leaderboard it scored 0.225,
barely above chance, and several variations moved it by less than a point.

At that stage I stopped running new experiments and audited the existing ones. The audit, which is still in the repository,
compared the pipeline line by line against the published reference implementations and found two kinds of problem. Some were
plain defects. Others were departures from recipes that had already been shown to work, most importantly in what the model
was trained to predict. It did not show that any single fix would help, and I was careful not to write it as if it did. It
did make the case for rebuilding from the reference design rather than continuing to patch.

## The target matters more than the network

The rebuild followed the recipe of d'Ascoli and colleagues. Instead of training the network only to pick one of fifty
labels, I trained it to predict, for every heard word, the contextual embedding that a large text model assigns to that word
in its sentence. The text model was T5-large, read out at the twelfth of its twenty-four encoder blocks, and the brain
embeddings were pulled toward the text embeddings with a contrastive (SigLIP) loss. Every word in the audiobooks was used,
about a million of them, not just the fifty that would be scored.

On the local evaluation this single change raised held-out accuracy from 0.29 to 0.46. On the leaderboard it took Broad from
0.225 to 0.428, just short of the strongest published reference, a model pretrained on roughly three hundred hours from eight
hundred people. The architectural changes I tried around the same time, longer windows and a transformer over the words of a
sentence, mattered little on their own, and only paid off once the text targets gave them something worth attending to.

The transformer also produced the project's most instructive mistake. Its first version received the time between words as
an input. Local accuracy jumped to 0.51, and when I zeroed the timing at inference it fell to 0.30. The model had learned to
identify words from their spacing in the audio, which is a property of the stimulus file and says nothing about the brain.
Nothing in the rules forbade the feature. I removed it anyway, and every later result was obtained without it.

## Less data, better transfer

The second large improvement came from deleting training data. While testing how much each Broad subject's few minutes of
labelled data helped, I trained a series of models in which those subjects made up a decreasing share of each batch. Transfer
to new people improved at every step: 0.452 when they were half the batch, 0.528 at a fifth, 0.594 at a tenth, and 0.632 when
the model saw nothing but the Deep subject's eighty hours. Adding twenty listeners from an external MEG dataset made it worse
by 0.15.

This went against my expectations, and I still hold the explanation loosely. The most plausible account is that ten to forty
minutes from a new head adds more variation in sensor geometry and noise than it adds information about words, so a model
fitted to one brain in great depth learns a cleaner mapping, which then transfers more gracefully. On the leaderboard the
subject-0-only model scored 0.655 on Broad, the largest single jump of the competition.

## Structure without labels

The holdout had properties that a subject's own unlabelled recordings could reveal, and two of them were worth a good deal.

The first was class balance. Training data is ordinary prose, in which *the* is vastly more common than *people*, and the model
inherited that skew: one word took up to sixteen percent of its top predictions. The test words were far more evenly spread.
Rescaling each subject's predictions so that every word received equal total probability, a Sinkhorn iteration run over that
subject's rows alone, added 0.036 on Broad.

The second was grammar. Every holdout sentence was built only from the fifty words, and a given subject heard 156 of them. That
is enough to learn something about which words follow which. I fitted a small hidden Markov model by expectation-maximisation to
each subject's own predicted sequences, with no labels and no text corpus, and replaced each prediction with the model's
posterior. That added 0.018 on Broad and 0.008 on Deep. A grammar learned from real English text did the opposite and cost up
to 0.11, because once the decoder is good a text prior overrides correct brain evidence as readily as incorrect evidence.

The Deep track had a problem of its own. Its local scores ran five to eight points above its leaderboard scores, and the gap
turned out to be the recording day. The holdout appears to have been recorded on a day that the training data did not cover. Simply
re-estimating the network's normalisation statistics from the holdout recordings, a standard technique called AdaBN, recovered
0.044. The same step hurt the Broad subjects slightly. A new day of the same brain needs re-standardising; a new brain is
better off measured against the reference brain's scale.

## Where it stopped

The last two days were spent on smaller things. Fitting each Broad subject's input layer on their own labelled data, a
dedicated decoder for the single-word trials, careful tuning of the balancing step, and an ensemble of other subjects' input
layers for the seven people with no labels together added a little over a point on each track.

Before ending I wanted to know how much room remained, so I measured it directly. If one cheats and averages a subject's
predictions over several hearings of the same word, accuracy rises by 0.07 with two hearings, 0.14 with four and 0.22 with
sixteen. The information exists. Finding those repeated hearings without labels is another matter: the similarity between two
trials of the same word was at chance in every representation of the raw signal I tried, eight in all. In the final
forty-eight hours I tested 164 post-processing ideas against a simulator of the holdout that I had calibrated on the
leaderboard. None moved the score by more than a few thousandths.

What limits a decoder like this, then, is the amount of evidence a single hearing of a single word leaves in the sensors. A
model can be made to use that evidence well, and the ladder above is a record of doing so. It cannot be made to use evidence
that is not there, and the sensible next steps are about the measurement rather than the model: more repetitions per word,
sensors closer to the scalp (optically pumped magnetometers allow that), or source models that pool the signal more
efficiently.

## On the leaderboard

During the competition the organisers posted a notice about leakage. All forty subjects heard the
same sentences, and timing information in the holdout files made it possible to match a row for one subject to the same
sentence position for another, or between the two tracks. Doing so is grounds for disqualification, and the notice concerned entries near the top of the
board. I do not know how the final review will treat them, and I am wary of reading too much into a ranking that
may still change.

What I can say is that every number above was obtained from each person's own brain data, and that the project is documented
well enough for anyone to check that. Each idea was written down with a predicted effect before it was run, and the outcome
was recorded afterward, misses included. The code, that log and the full history are public at
[github.com/rohit-ravi2/libribrain-2026](https://github.com/rohit-ravi2/libribrain-2026).

[ROHIT: a closing sentence or two on what you take from it, or what comes next.]
