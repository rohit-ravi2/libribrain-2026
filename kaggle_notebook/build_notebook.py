"""Builds timing-floor-libribrain.ipynb from the cell sources below (run once before `kaggle kernels push`)."""
import os
import nbformat as nbf

HERE = os.path.dirname(os.path.abspath(__file__))
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip("\n")))

md(r"""
# How much of LibriBrain's word label is in the timing?

The 2026 PNPL competition asks for the identity of a heard word, one of 50, from a window of MEG recorded around its onset. The Deep track scores a single, heavily recorded listener and the Broad track scores many listeners with less data each; both use balanced top-10 accuracy, for which chance is 0.20. Every word in running speech evokes a response in the MEG, and the gaps between word onsets are not random with respect to word identity. Short function words tend to be followed quickly by the next word, while content words and phrase-final words tend to be followed by longer gaps. A window that outlasts its word therefore carries information about the label through the timing of the onsets inside it, before any neural representation of the word enters the picture.

Jayalath and Parker Jones (arXiv 2609.40359) have already shown how strong this channel is on LibriBrain: a model given only inter-word intervals reaches 22.9% balanced top-1, against 22.3% for their joint MEG model, and they note that a 3 s window can contain the responses to subsequent words. This notebook builds on that work with a narrower measurement, an exact-onset, event-only floor for a fixed window. It uses no MEG at all. It downloads only the BIDS event files of subject 0 (the Deep subject) from the public LibriBrain release, builds a classifier whose only input is the pattern of word onsets falling inside a fixed window after the target onset, and scores it on a held-out recording day. The result is a timing floor: the accuracy that onset timing alone supports under each window length. Because the competition supplies word onsets as metadata with every holdout epoch, this floor is a legitimate baseline rather than a leak, and the quantity that speaks to brain decoding is the score of a model minus the floor.

The notebook also shows how much Sinkhorn class balancing changes balanced accuracy on these timing-only probabilities, and closes with a discussion of what an honest noise control has to contain for this dataset. The analysis is a port of `timing_floor.py` from my competition repository, and the reference numbers from that local run are printed beside the Kaggle numbers for comparison.
""")

md(r"""
## Data

LibriBrain is distributed on Hugging Face as `pnpl/LibriBrain` (Sherlock Holmes books 1 to 7) and `pnpl/LibriBrain2` (books 8 and 9, *The Moth*, TIMIT and MOCHA-TIMIT). Each run has an `*_events.tsv` file under `<task>/derivatives/events/` listing every word and phoneme with its onset in MEG time (`timemeg`). The cell below fetches the 162 event files of subject 0, roughly 640 MB in total because each file also lists phonemes, and nothing else; the MEG `.h5` files are never touched.

The split follows the one I used throughout the competition. Training uses every subject-0 run except two, and the test runs are Sherlock1 run 2 from sessions 11 and 12. Those two sessions form one held-out recording day, and the text of Sherlock1 run 2 shares no 5-gram with any training run, so neither the day nor the passage was seen in training. All 162 runs are used for training rather than only the Sherlock books because that is the training set of the reference result; the onset statistics of the other stories are similar enough that this matters little, but reproducing the reference exactly requires it.
""")

code(r"""
import os, glob, time, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from huggingface_hub import snapshot_download

T0 = time.time()
print("scikit-learn", sklearn.__version__, "| numpy", np.__version__, "| pandas", pd.__version__)

EVENTS = "/tmp/libribrain_events"      # outside /kaggle/working, so the 640 MB of tsv files are not saved as output
for repo in ("pnpl/LibriBrain", "pnpl/LibriBrain2"):
    snapshot_download(repo_id=repo, repo_type="dataset", local_dir=EVENTS, token=False, max_workers=8,
                      allow_patterns=["*/derivatives/events/sub-0_*_events.tsv"])
files = sorted(glob.glob(f"{EVENTS}/*/derivatives/events/sub-0_*_events.tsv"))
print(len(files), "event files downloaded in", round(time.time() - T0), "s")
""")

md(r"""
The competition's vocabulary is the 50-word list shipped with the `pnpl` package (`pnpl/competition/data/vocabulary.csv`), written out here so the notebook does not depend on the package. Words are matched after lower-casing and replacing the typographic apostrophe, so that *It's* in the transcript maps to *it’s* in the vocabulary. Every word event is kept for the onset timeline, whether or not it belongs to the vocabulary, since the brain responds to all of them; only vocabulary words become rows to classify.
""")

code(r"""
VOCAB = ("is,the,a,to,it,i,not,was,we,be,he,that,have,this,they,of,there,and,are,in,but,will,so,all,my,for,she,"
         "were,any,really,at,out,our,am,it’s,had,him,an,very,has,do,can,time,think,good,always,new,people,as,on").split(",")
norm = lambda w: str(w).strip().lower().replace("’", "'")
VOCAB_ID = {norm(w): i for i, w in enumerate(VOCAB)}
assert len(VOCAB) == 50 and len(VOCAB_ID) == 50

def load_events(path):
    # "None" is a word in the transcripts, so only empty cells count as missing
    e = pd.read_csv(path, sep="\t", keep_default_na=False, na_values=[""], usecols=["kind", "segment", "timemeg"])
    w = e[e.kind == "word"]
    onset = pd.to_numeric(w.timemeg, errors="coerce").to_numpy(float)
    label = np.array([VOCAB_ID.get(norm(s), -1) for s in w.segment.astype(str)], int)
    return onset, label

RUNS = {os.path.basename(f)[:-len("_events.tsv")]: load_events(f) for f in files}
TEST = ("sub-0_ses-11_task-Sherlock1_run-2", "sub-0_ses-12_task-Sherlock1_run-2")
TRAIN = [b for b in RUNS if b not in TEST]
assert all(t in RUNS for t in TEST)

n_words = sum(len(o) for o, _ in RUNS.values()); n_vocab = sum((y >= 0).sum() for _, y in RUNS.values())
print(f"{len(RUNS)} runs, {n_words:,} word events, {n_vocab:,} vocabulary tokens ({n_vocab / n_words:.1%})")
print("training runs:", len(TRAIN), "| test runs:", TEST)
""")

md(r"""
## Features and estimators

For a vocabulary word at onset *t* and a window [*t*, *t* + *W*), the features are the number of later word onsets inside the window and the offsets of the next *K* = 4 of them, with *W* standing in for an onset that does not occur. This is the timing a fixed-window decoder could see through onset-locked responses if it located every onset exactly. The resulting score is what an exact-onset classifier of this capacity extracts from the window. It is neither a lower nor an upper bound for a given MEG model, which may recover less timing than this (onsets blurred by noise) or more (a stronger learner, or cues such as word duration that the features here omit).

Two estimators read these features. The first is a gradient-boosted classifier (scikit-learn's `HistGradientBoostingClassifier`, 200 iterations, 31 leaves, balanced class weights), scored by balanced top-1 and top-10 accuracy over the 50 words, the competition's metric. The second is information in bits, in the conditional-set form of Packard (1990): the onset pattern is reduced to a cell (the next-onset offset in 20 ms bins crossed with the onset count), the word distribution in each cell is estimated on the training runs with add-½ smoothing, and the held-out information is the mean of log₂ *P*(*y* | cell) / *P*(*y*) over test rows. Because the table is fitted on training runs and evaluated on the held-out day, the estimate is cross-fitted, and over-fitted cells lower it rather than raise it.
""")

code(r"""
K, BIN, WINDOWS = 4, 0.02, (0.5, 1.0, 2.0, 3.0)

def rows(base, W):
    onset, label = RUNS[base]
    o = np.sort(onset); X = []
    for t in onset[label >= 0]:
        nxt = o[(o > t + 1e-6) & (o < t + W)] - t
        X.append([len(nxt)] + [nxt[k] if k < len(nxt) else W for k in range(K)])
    return np.array(X, float), label[label >= 0]

def balanced_topk(P, y, k):
    top = np.argsort(-P, 1)[:, :k]; hit = (top == y[:, None]).any(1)
    return float(np.mean([hit[y == c].mean() for c in np.unique(y)]))

def packard_bits(Xtr, ytr, Xte, yte, W):
    cell = lambda X: (np.minimum(X[:, 1], W) / BIN).astype(int) * 16 + np.minimum(X[:, 0], 15).astype(int)
    ctr, cte = cell(Xtr), cell(Xte)
    P0 = (np.bincount(ytr, minlength=50) + 0.5) / (len(ytr) + 25)
    tab = {c: np.bincount(ytr[ctr == c], minlength=50) + 0.5 for c in np.unique(ctr)}
    lp = [np.log2(tab[c][y] / tab[c].sum()) if c in tab else np.log2(P0[y]) for c, y in zip(cte, yte)]
    return float(np.mean(lp) - np.mean(np.log2(P0[yte])))
""")

md(r"""
## The timing floor across window lengths

For each window length the classifier is trained on all training rows and applied to both held-out sessions. Gradient boosting on this many rows stops early on an internal validation split of 10% that is drawn at random, so the fit is repeated with three seeds. Seed 0 is the configuration of the reference run and supplies the probabilities used in the rest of the notebook; the other two seeds measure how much the classifier's score moves for reasons unrelated to timing. Each fit takes well under a minute on Kaggle's CPU.
""")

code(r"""
SEEDS = (0, 1, 2)
results, probs, seed_runs = {}, {}, []
for W in WINDOWS:
    t0 = time.time()
    tr = [rows(b, W) for b in TRAIN]
    Xtr = np.concatenate([a for a, _ in tr]); ytr = np.concatenate([b for _, b in tr])
    test = {b.split("_")[1]: rows(b, W) for b in TEST}
    bits = {s: packard_bits(Xtr, ytr, Xte, yte, W) for s, (Xte, yte) in test.items()}
    for seed in SEEDS:
        clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, max_leaf_nodes=31,
                                             class_weight="balanced", random_state=seed).fit(Xtr, ytr)
        r = {"n_train": int(len(ytr)), "n_iter": int(clf.n_iter_)}
        for s, (Xte, yte) in test.items():
            P = np.zeros((len(yte), 50)); P[:, clf.classes_] = clf.predict_proba(Xte)
            if seed == 0:
                probs[(W, s)] = (P, yte)
            r[s] = {"n": int(len(yte)), "top1": balanced_topk(P, yte, 1), "top10": balanced_topk(P, yte, 10), "bits": bits[s]}
        r["mean"] = {m: float(np.mean([r[s][m] for s in test])) for m in ("top1", "top10", "bits")}
        seed_runs.append({"window (s)": W, "seed": seed, "iterations": r["n_iter"],
                          "top-1": round(r["mean"]["top1"], 4), "top-10": round(r["mean"]["top10"], 4)})
        if seed == 0:
            results[W] = r
    sr = [x for x in seed_runs if x["window (s)"] == W]
    print(f"W = {W:.1f} s  n_train {len(ytr):,}  {results[W]['mean']['bits']:.3f} bits  "
          f"seed 0: top-1 {results[W]['mean']['top1']:.4f}  top-10 {results[W]['mean']['top10']:.4f}  |  "
          f"seeds {SEEDS}: top-1 {min(x['top-1'] for x in sr):.3f}-{max(x['top-1'] for x in sr):.3f}  "
          f"top-10 {min(x['top-10'] for x in sr):.3f}-{max(x['top-10'] for x in sr):.3f}  ({time.time() - t0:.0f} s)")
""")

md(r"""
The reference values below come from the same code, seed 0, run locally on the same event files with scikit-learn 1.9.1. Kaggle's image carries an older scikit-learn, and the internal validation split, the histogram binning and the early-stopping point can all differ between versions, so the classifier's numbers are not expected to agree to the last digit. The Packard estimate involves no random fitting and should agree exactly. The second table gives the seed-to-seed spread of the classifier on this image, which is the right yardstick for judging the gap to the reference.
""")

code(r"""
REF = {  # timing_floor.json, mean of ses-11 and ses-12
    0.5: {"top1": 0.1086, "top10": 0.5535, "bits": 0.6573},
    1.0: {"top1": 0.1132, "top10": 0.5677, "bits": 0.5963},
    2.0: {"top1": 0.1121, "top10": 0.6057, "bits": 0.5605},
    3.0: {"top1": 0.1358, "top10": 0.5880, "bits": 0.5333},
}
tab = pd.DataFrame([{"window (s)": W,
                     **{f"{m} (Kaggle)": round(results[W]["mean"][m], 4) for m in ("top1", "top10", "bits")},
                     **{f"{m} (local)": REF[W][m] for m in ("top1", "top10", "bits")},
                     **{f"{m} diff": round(results[W]["mean"][m] - REF[W][m], 4) for m in ("top1", "top10", "bits")}}
                    for W in WINDOWS]).set_index("window (s)")
per_ses = pd.DataFrame([{"window (s)": W, "session": s, "n rows": results[W][s]["n"],
                         "top-1": round(results[W][s]["top1"], 4), "top-10": round(results[W][s]["top10"], 4),
                         "bits": round(results[W][s]["bits"], 4)} for W in WINDOWS for s in ("ses-11", "ses-12")])
display(tab)
display(per_ses)
display(pd.DataFrame(seed_runs).set_index(["window (s)", "seed"]))
""")

md(r"""
Against chance levels of 0.02 for top-1 and 0.20 for top-10, onset timing alone reaches about 0.09 to 0.14 top-1 and 0.55 to 0.62 top-10 on a day and a passage that the classifier never saw, across window lengths, seeds and the two library versions. Top-1 is roughly five times chance. The classifier's score moves by one to two points between seeds and between versions, and the differences between window lengths are of the same size, so the honest summary is that the floor is nearly flat in *W*: a 0.5 s window, shorter than the span most people would consider safe from neighbouring words, already carries almost all of it. The reason is that the first gap after the target word does most of the work, and that gap usually falls inside half a second.

The information estimate, between about half a bit and two thirds of a bit, declines as the window grows. This is a property of the estimator more than of the data. The Packard cell crosses the first gap with the onset count, so a longer window yields more cells with fewer training rows each, and the cross-fitted estimate pays for that sparsity on the held-out day. It is a conservative number. For scale, the label itself carries about 4.9 bits (the entropy of the vocabulary's word frequencies in these recordings), so timing alone accounts for roughly a tenth to an eighth of it.
""")

md(r"""
## Which words carry the timing?

The per-word top-10 recall at the 3 s window shows where the floor comes from. Recall is pooled over the two held-out sessions. Words with fewer than 20 occurrences on the held-out day are left out, since their recall rests on a handful of rows.
""")

code(r"""
P = np.concatenate([probs[(3.0, s)][0] for s in ("ses-11", "ses-12")])
y = np.concatenate([probs[(3.0, s)][1] for s in ("ses-11", "ses-12")])
hit = (np.argsort(-P, 1)[:, :10] == y[:, None]).any(1)
per_word = pd.DataFrame({"word": [VOCAB[c] for c in np.unique(y)],
                         "n": [int((y == c).sum()) for c in np.unique(y)],
                         "top10_recall": [hit[y == c].mean() for c in np.unique(y)]}).sort_values("top10_recall")
print(f"{len(per_word)} vocabulary words occur on the held-out day; {(per_word.n >= 20).sum()} occur at least 20 times")
per_word = per_word[per_word.n >= 20].reset_index(drop=True)
print("best:", ", ".join(f"{w} {r:.2f}" for w, r in zip(per_word.word[::-1][:8], per_word.top10_recall[::-1][:8])))
print("worst:", ", ".join(f"{w} {r:.2f}" for w, r in zip(per_word.word[:8], per_word.top10_recall[:8])))
""")

md(r"""
## Sinkhorn class balancing

Balanced accuracy rewards a classifier for spreading its top ranks evenly across the 50 words. A cheap way to do that at test time, which I used in my competition submissions, is Sinkhorn balancing (Cuturi, 2013): the probability matrix of one subject's rows is alternately rescaled so that every word receives the same total mass and every row sums to one, optionally after sharpening or flattening with a temperature *T*. It uses no labels and only one subject's own rows, which keeps it inside the competition rules. Here it is applied separately to each held-out session.

The classifier above already uses balanced class weights, so this is a demanding case for the method. A second classifier without class weights, trained on the 3 s window only, shows the more common situation in which the raw posteriors follow the skewed word frequencies of English.
""")

code(r"""
def sinkhorn(P, T=1.0, iters=10):
    Q = np.clip(P, 1e-12, None) ** (1.0 / T); Q /= Q.sum(1, keepdims=True); n = len(Q)
    for _ in range(iters):
        Q = Q / np.clip(Q.sum(0, keepdims=True), 1e-12, None) * (n / 50.0)
        Q = Q / Q.sum(1, keepdims=True)
    return Q

W = 3.0
tr = [rows(b, W) for b in TRAIN]
Xtr = np.concatenate([a for a, _ in tr]); ytr = np.concatenate([b for _, b in tr])
clf_raw = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, max_leaf_nodes=31, random_state=0).fit(Xtr, ytr)
for b in TEST:
    Xte, yte = rows(b, W); Pr = np.zeros((len(yte), 50)); Pr[:, clf_raw.classes_] = clf_raw.predict_proba(Xte)
    probs[("raw", b.split("_")[1])] = (Pr, yte)

sk_rows = []
for name, key in (("balanced weights, 0.5 s", 0.5), ("balanced weights, 3 s", 3.0), ("no class weights, 3 s", "raw")):
    for T in (None, 1.0, 2.0, 3.0):
        m = {k: [] for k in ("top1", "top10")}
        for s in ("ses-11", "ses-12"):
            P, yte = probs[(key, s)]; Q = P if T is None else sinkhorn(P, T)
            m["top1"].append(balanced_topk(Q, yte, 1)); m["top10"].append(balanced_topk(Q, yte, 10))
        sk_rows.append({"classifier": name, "Sinkhorn": "none" if T is None else f"T = {T:g}",
                        "top-1": round(np.mean(m["top1"]), 4), "top-10": round(np.mean(m["top10"]), 4)})
sk = pd.DataFrame(sk_rows)
display(sk.set_index(["classifier", "Sinkhorn"]))
""")

code(r"""
def top1_share(P):
    return np.bincount(P.argmax(1), minlength=50).max() / len(P)
for key, label in ((3.0, "balanced weights"), ("raw", "no class weights")):
    P, _ = probs[(key, "ses-11")]
    print(f"{label:17s} largest share of top-1 predictions on ses-11: {top1_share(P):.3f} raw, "
          f"{top1_share(sinkhorn(P, 1.0)):.3f} after Sinkhorn (uniform would be 0.020)")
""")

md(r"""
The two classifiers move in opposite directions. Without class weights, the posteriors follow word frequency, about a third of the top-1 predictions on a session go to a single word, and balancing lifts top-10 by roughly seven points (from about 0.47 to 0.54 in the Kaggle run) while leaving top-1 nearly where it was. With balanced class weights the prior has already been removed, and balancing lowers top-10 by roughly six points (about 0.59 to 0.53 at 3 s, 0.57 to 0.51 at 0.5 s), again with top-1 almost unchanged. Temperature makes little difference in either case. A plausible reading is that equal total mass is the wrong target for a test session on which a few vocabulary words never occur and the rest are far from uniform; once the classifier has spread its top ranks, forcing a further spread spends top-10 slots on words that are absent. I have not tested that explanation here.

The practical point for a competition entry is that a balancing step has to be credited against an equally prior-corrected baseline, on a held-out day, before it is trusted. Against an uncorrected baseline it looks like a large gain, and most of that gain is the class weighting it implicitly supplies. In my own submissions the Sinkhorn step did help on the private leaderboard, adding 0.037 on Broad (*T* = 2) and 0.051 on Deep (*T* = 3) to a model whose logits had been corrected by half the log training prior, so only partly. That was a different model on a different test set. The result above is a reason to check the step on each new model, not a reason to drop it.
""")

md(r"""
## Where the floor sits among the controls

The figure on the left plots the floor against window length. The figure on the right places the 3 s floor among measurements from my competition work on the same held-out day and split, all for a fixed 3 s window with no sentence context. These come from trained networks (on GPU, not rerun here). The real-MEG bar is a screening model, not my competition decoder: no sentence context, a fixed 3 s window, a single seed and no onset metadata. A competition-style model with sentence context scores 0.693 top-10 on the same held-out day, above the floor; it is not plotted because its input differs from the fixed-window setting of the other bars. The pure-noise control is shared Gaussian noise in place of MEG. The two timing twins add one fixed onset kernel at every word onset to that noise and train the same network on it: one with the kernel at 2 noise standard deviations, and one with the kernel at the amplitude of the real onset response (0.135 residual s.d. per channel) over a white, spatially independent background. Both twins overstate what a realistic onset response would yield. The first has a kernel stronger than the real response. The second matches the real response amplitude channel by channel but places it on white, spatially independent noise, which is far easier to see through than the correlated background of real MEG.
""")

code(r"""
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw={"width_ratios": [1, 1.25]})
BLUE, ORANGE, GREY, INK = "#2a78d6", "#eb6834", "#8a8984", "#52514e"
Ws = list(WINDOWS)
a1.plot(Ws, [results[W]["mean"]["top10"] for W in Ws], "-o", color=BLUE, lw=2, ms=7, label="top-10 (chance 0.20)")
a1.plot(Ws, [results[W]["mean"]["top1"] for W in Ws], "-s", color=ORANGE, lw=2, ms=7, label="top-1 (chance 0.02)")
for c in (0.20, 0.02):
    a1.axhline(c, color=GREY, lw=1, ls="--")
a1.set_xlabel("window after onset (s)"); a1.set_ylabel("balanced accuracy, held-out day"); a1.set_ylim(0, 0.7)
a1.set_xticks(Ws); a1.set_title("Event-only timing floor", loc="left"); a1.legend(frameon=False, loc="center right")

bars = [("pure shared noise", 0.221, GREY), ("real MEG, no-context screening model", 0.458, INK),
        ("event-only floor (this notebook)", results[3.0]["mean"]["top10"], BLUE),
        ("twin, per-channel TRF amplitude,\nwhite independent noise (optimistic)", 0.611, ORANGE), ("twin, 2 s.d. kernel (optimistic)", 0.659, ORANGE)]
yy = np.arange(len(bars))[::-1]
a2.barh(yy, [b[1] for b in bars], color=[b[2] for b in bars], height=0.6)
for yi, (name, v, _) in zip(yy, bars):
    a2.text(v + 0.01, yi, f"{v:.3f}", va="center", color=INK)
a2.set_yticks(yy); a2.set_yticklabels([b[0] for b in bars])
a2.axvline(0.20, color=GREY, lw=1, ls="--"); a2.set_xlim(0, 0.78)
a2.set_xlabel("balanced top-10, fixed 3 s window, no context, held-out day"); a2.set_title("Floor among noise controls", loc="left")
for ax in (a1, a2):
    ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="x" if ax is a2 else "y", color="#e6e5e0", lw=0.8)
    ax.set_axisbelow(True)
plt.tight_layout(); plt.savefig("timing_floor.png", dpi=130); plt.show()
""")

code(r"""
fig, ax = plt.subplots(figsize=(12, 3.6))
ax.bar(range(len(per_word)), per_word.top10_recall, color=BLUE, width=0.8)
ax.axhline(0.20, color=GREY, lw=1, ls="--")
ax.set_xticks(range(len(per_word)))
ax.set_xticklabels([f"{w} ({n})" for w, n in zip(per_word.word, per_word.n)], rotation=70, fontsize=9)
ax.set_ylabel("top-10 recall"); ax.set_ylim(0, 1.02)
ax.set_title("Per-word top-10 recall of the 3 s timing floor, held-out day, words with n >= 20 (dashed: chance)", loc="left")
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout(); plt.savefig("timing_floor_per_word.png", dpi=130); plt.show()
""")

md(r"""
## Discussion

Jayalath and Parker Jones (arXiv 2609.40359) already include a Timing condition on LibriBrain, in which a model sees only the inter-word intervals; it reaches 22.9% balanced top-1, slightly above the 22.3% of their joint MEG model. Their Section 3.1 and Appendix C.1 also point out that a 3 s window can contain responses to subsequent words. The floor in this notebook is a narrower companion to that result: exact onsets, a fixed window starting at the target onset, a held-out day and passage, and the competition's balanced top-10 alongside top-1. Zhang et al. (arXiv 2605.24524) study MEG-to-audio retrieval on the Gwilliams and MOUS datasets rather than LibriBrain. They show that signal-blind Gaussian noise falls to chance once windows have a fixed length and the split holds out stimulus identity, and they propose Group Context Bias, an inference-time additive logit bias that pools sentence-consistent evidence. The pure-noise bar above agrees with their control on this data: shared noise in a fixed 3 s window on a held-out day scores 0.221 top-10, at chance.

What a pure-noise control does not contain is an onset-locked response. Real MEG has one at every word, and a fixed window that outlasts its word still contains the responses to the next few words. The event-only floor measures what those onsets say about the label when they are known exactly, which is about 0.58 to 0.59 top-10 and 0.11 to 0.14 top-1 at 3 s on a held-out day and passage, and still 0.55 to 0.57 top-10 at 0.5 s. Once the noise control is given a realistic response at each onset, it no longer sits at chance. The twin at 2 noise s.d. reached 0.659 top-10 (0.159 top-1), and the twin at the amplitude of the real onset response over a white background reached 0.611 (0.119 top-1); both are optimistic for the reasons given above, but both lie close to the event-only floor, so a network can in principle read much of the timing from evoked responses alone. A no-context screening model on real MEG, trained in the same fixed-window configuration with a single seed, reached 0.458 top-10 (0.0725 top-1). That number describes a weak screening setup and should not be read as the level of MEG decoding on this day: a competition-style model with sentence context reaches 0.693 top-10 on the same held-out sessions, above the floor of about 0.59, although it was scored on prose sentence segments, which favour it somewhat over the holdout's epoch format. What the comparison does show is that a fixed-window score cannot be read as word decoding until it is placed against the floor for the same window and split.

In this competition the timing is not forbidden. Each holdout epoch comes with its word onsets as metadata, so a model may use them, and the floor computed here is a baseline that any entry could reproduce without looking at the brain. The quantity that measures decoding is real minus floor: the improvement of a model that sees both MEG and onsets over a model that sees only the onsets, under the same split. For an honest noise control on this data I would therefore suggest three things, each of which is cheap. First, compute the event-only floor for the window and split in use, as done here in a few minutes of CPU. Second, if a synthetic control is used, give it an onset-locked kernel at the amplitude of the real response, estimated from training recordings, and a background with the spatial and temporal correlation of real MEG, since white noise flatters the twin. Third, report the decoder's score as a difference from the floor, on a held-out day and a held-out passage, so that neither recording-day drift nor repeated text inflates it.

Two limitations apply. The floor is what an exact-onset classifier of this capacity extracts; it is neither a lower nor an upper bound for a given MEG model, since such a model may recover the onsets imperfectly, may use a stronger learner, or may read timing cues such as word duration that these features leave out. And the held-out test here is one day of one subject; the Broad track's many-subject setting has different text and different recording days, and its floor should be measured on its own split before the same reasoning is applied there.

**References.** Cuturi, M. (2013). Sinkhorn distances: lightspeed computation of optimal transportation distances. NeurIPS 26. Jayalath, D., & Parker Jones, O. (2026). arXiv:2609.40359 (Timing condition; Section 3.1 and Appendix C.1). Packard, N. H. (1990). A genetic learning algorithm for the analysis of complex data. Complex Systems 4(5). Zhang et al. (2026). arXiv:2605.24524 (Gwilliams and MOUS; Group Context Bias). Özdogan, M., et al. (2025). LibriBrain: Over 50 hours of within-subject MEG to improve speech decoding methods at scale. arXiv:2506.02098.
""")

code(r"""
out = {f"W{W}": {k: v for k, v in results[W].items()} for W in WINDOWS}
out["sinkhorn"] = sk_rows
json.dump(out, open("timing_floor_kaggle.json", "w"), indent=1)
print(f"total runtime {time.time() - T0:.0f} s")
""")

nb = nbf.v4.new_notebook(cells=cells, metadata={
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"}})
nbf.write(nb, os.path.join(HERE, "timing-floor-libribrain.ipynb"))
print("wrote", len(cells), "cells")
