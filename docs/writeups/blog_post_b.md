---
title: "Arguing with the stopping points"
startDate: 2026-10-21
description: "How a month of decoding heard words from brain recordings actually unfolded: an audit that changed the target, a single-subject model that beat every multi-subject one, a hidden pocket of predictability, and two lines I held to the end."
tags: ["MEG", "Neural Decoding", "Research Process", "Kaggle"]
draft: false
order: 2
---

Part of the [LibriBrain](/posts/libribrain/) series. Technical companion: [The margin above 0.63](/posts/libribrain/the-margin-above-063/).

LibriBrain began as one dataset among many in a broader investigation of public neuroscience data. In late September
I gave it a dedicated session, and on October 8 I entered both tracks of the PNPL competition: decode which of fifty
words a listener heard from 306 channels of magnetoencephalography. My first question was whether a route to the top
of the leaderboard existed at all. This post is about how we found one, and how far it went.

## Changing the question

My opening idea came from earlier work on handwriting decoding, where guessing a word from several uncertain letters
had proved far easier than guessing each letter alone. The same principle should hold for speech. Decode each word
roughly, then let the sentence correct its weakest members. That intuition would return in a more disciplined form
later in the month.

The first model, a classifier choosing among fifty labels, stalled near chance. An independent audit of the pipeline
made the cause clear, and I redirected the work around a different question taken from published research. Instead of
asking which of fifty words the brain heard, ask where that word sits in a language model's space of meaning, in the
context of its sentence. I set one condition for the rebuild: keep going until the local score passed 0.4. Two days
later the leaderboard score had doubled, from 0.22 to 0.43. The architecture had barely changed. The target had.

## One brain teaches the rest

I worked in two registers throughout the month. Ideas from AI assistants, the literature and my own notes went into
an open-ended brainstorming session. Anything that survived scrutiny went to an execution session with a cheap test
attached. Most borrowed ideas did not survive their first measurement. The value of the process was its speed: an idea
could be proposed, tested and closed within an hour.

The decisive result came from a run aimed at the other track. A model trained on the one listener with eighty hours
of recordings, and no one else, transferred to the thirty-nine other listeners better than any model trained on all of
them. Their short recordings added more variation than information. The Broad score rose from 0.43 to 0.66 in a single
evening.

That result raised a deeper question. If different people's brains respond to the same heard word so similarly that
one person's model reads everyone else, where does individuality live? My working answer is production. Perception is
shaped by a shared sensory pathway, while speaking is a program each person builds over years of development. I think
that asymmetry reflects how tightly language is coupled to the timing of plasticity in the circuits behind cognition,
and it is now a research question I intend to pursue.

## Rules first

Near the top of the leaderboard, I paused to verify one thing before any further work: that nothing in our pipeline
used the shortcut the organisers had prohibited. Every listener heard the same sentences, and the test files leaked
enough timing to align rows across people. We had used none of it.

I held a second line of my own. Early on, a model given the spacing between words jumped from 0.30 to 0.51 locally. It
was reading word durations from the audio, not the brain. The rules allowed it; I removed it. After the competition I
measured how far that shortcut alone reaches. A model with no brain data at all scores 0.63 on our test blocks, which
means any honest decoder on this benchmark should be judged by its margin above 0.63, not above chance.

## Pockets of predictability

The most original stretch borrowed from finance. Norman Packard, before co-founding a trading firm, designed an
algorithm that searches for the conditions under which a chaotic system briefly becomes predictable. Applying it to
brain recordings seemed like a reach, but conceptually close enough to test. The search found one condition with a
hit rate of 0.92. The first analysis called it a fluke. I asked why we were ruling it out so early, and a second
look, controlling for which words occur where, showed the effect was real: words at the start of a sentence decode
far better than the same words mid-sentence, most likely because the brain responds more strongly after silence.

## The home stretch

The final day and a half was a sustained effort to stack every gain that had shown signal. A self-learned grammar built
from each listener's own decoded sentences added nearly two points; that was the opening intuition about sentence
correction, in its disciplined form. Correcting for recording-day drift added four points on the Deep track. A
parallel session generated and screened ideas in volume while the main session handled the finals. The more ambitious
ideas, ten moonshots and a turn toward formal linguistics, did not move the score. At four in the afternoon I called
the lock-in and we consolidated what had worked.

The entry finished at 0.722 on Broad, eighth on the public leaderboard, and 0.729 on Deep, against published references
of 0.430 and 0.732.

## What I take from it

Two practices carried this work. Keeping wide-open exploration separate from disciplined execution let ideas move
quickly without contaminating the results. Refusing to accept a stopping point until it had been measured was right
more often than it was wrong; several limits we had called hard turned out to be soft. And the lines I held under
deadline pressure, I would hold again.

The code, the pre-registered experiment log and the full development history are at
[github.com/rohit-ravi2/libribrain-2026](https://github.com/rohit-ravi2/libribrain-2026), and the technical summary is
on the [project page](/projects/libribrain-2026/).
