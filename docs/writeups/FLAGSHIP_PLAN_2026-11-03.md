# Flagship plan: a timing-clean decoder that beats SimpleB2T on its own protocol

Written 2026-11-03 after the adversarial pass. The leakage paper's timing headline has prior art from the organisers
(Jayalath & Parker Jones, arXiv 2609.40359, 30 Sep 2026). New orientation: a methodological advance measured on their
protocol, with the synthetic-signal control proving it is timing-clean.

## The target (their protocol, subject 0 of LibriBrain100)

- Split: validation = session 11; test = sessions 0, 7, 13, 14, 15, 16, 18, 20, 21, 22, 29, 30; train = the rest.
- Word level: balanced top-1 over the 50 most frequent words. Isolated MEG 9.5%; leaky joint 22.3%; synthetic joint 22.0%.
- Sentence level: 200 clinical sentences (Core / Expanded), 92-word vocabulary, up to 5 test observations per word.
  SimpleB2T Full WER 71.3 (k=1), 41.0 (k=5); Core 65.6 / 36.6. Qwen3-8B-Base prior, beam 50, lambda 0.5.
- Code: github.com/neural-processing-lab/SimpleB2T (commit 51ef953, cloned to `external/`, not yet installed).
  All 388 pinned data files are already on disk under /mnt/ssd4tb/data/LibriBrain100.

## Candidate advances, with predictions (to be graded at the strength the numbers support)

1. Recording-day test-time adaptation. Their 12 test sessions are 12 recording days; SimpleB2T adapts nothing.
   Our own measurement: a 0.044 day-drift effect (Deep, public split), and AdaBN gains on day-novel data.
   Method: per-session re-estimation of normalisation statistics plus an unsupervised per-session alignment of the
   input (Euclidean alignment of sensor covariance, or the per-subject input layer fitted label-free).
   Prediction: +1 to +3 points balanced top-1 at the word level; WER gain largest at k=1.
2. Class-marginal calibration of emissions (Sinkhorn / optimal-transport label-shift correction) before beam search.
   Our measured gains: +0.036 to +0.05 top-10 balanced accuracy on both leaderboard splits.
   Prediction: +1 to +2 points word top-1; small WER gain once the LM prior is on (the LM already shapes marginals).
3. Timing-clean sentence context. Encode each window independently (as SimpleB2T), then let a small transformer read
   the neighbours' independent embeddings with ordinal positions only, no onsets, no overlap. Must pass both noise
   controls (shared and independent noise score at the position-prior level, not above it).
   Prediction: uncertain; 0 to +2 points word top-1. If the noise control fails, it is dropped.
4. Reliability-weighted aggregation of the k observations (per-observation confidence from the posterior entropy and
   session drift, replacing the single agreement weight alpha). Prediction: WER at k=2..5 down 1 to 3 points.

The headline claim, if it holds: on the organisers' own benchmark, adapting to the recording day and calibrating class
marginals improves a timing-clean decoder by X points of word accuracy and Y points of WER at k=1 and k=5, with the
synthetic-signal control at chance.

## Steps

0. Noise control on our own decoder (running: `scripts/queue_noise_control.sh`). Tells us whether our existing
   "timing-free" arms were timing-free.
1. Reproduce SimpleB2T's isolated encoder (9.5% top-1) and WER at k=1 and k=5 on this machine. Qwen3-8B does not fit
   in 8 GB at FP32; run it 4-bit on the GPU and report the gap to their published WER as a quantisation check.
   Needs Rohit's approval to install and run the external code.
2. Add candidates 1 and 2 to their pipeline (inference-time only: no retraining), five seeds, report per-session gains.
3. Candidates 3 and 4.
4. Rewrite the paper around the result; the evaluation-instrument material (public/private split, same-day validation,
   discarded instrument) becomes the second part.

Compute: their encoder is about 20M parameters, batch 128, up to 50 epochs on 66 h of data. On the RTX 4060 Ti, to be
measured with a 1% smoke run before the full run.
