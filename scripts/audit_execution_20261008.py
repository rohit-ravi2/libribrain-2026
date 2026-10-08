"""Read-only checks of the current experiment design; writes a separate audit JSON.

Run from the repository: ~/venvs/libribrain/bin/python scripts/audit_execution_20261008.py
No training, holdout-label inference, submissions, or checkpoint changes.
"""
import ast
import collections
import glob
import hashlib
import itertools
import json
import pathlib
import sys

import numpy as np
import pandas as pd
import torch
import h5py
from scipy import signal

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import split as S
from src.common import CACHE, VOCAB, VOCAB_ID, norm_word, downsample, IS_MAG
from src.data import bacc_at_k, normalize

torch.set_num_threads(2)
out = {}
labels, subjects, audio_ok, tokens = [], [], [], []
test_labels = collections.defaultdict(list)
pool_classes = {"X": set(), "S": set()}
for f in sorted(glob.glob(f"{CACHE}/runs/*.npz")):
    base = pathlib.Path(f).stem
    sub, ses = S.parse(base)
    with np.load(f) as d:
        tc, y = d["timechapter"], d["label"]
        if sub in S.EVAL_SUBJECTS:
            pool_classes["X"].update(y.tolist())
        if sub == 0:
            pm = np.ones(len(tc), bool) if base == "sub-0_ses-10_task-Sherlock1_run-1" else S.block_masks(base, tc)["B"]
            pool_classes["S"].update(y[pm].tolist())
        m = S.train_mask(base, tc, val=True)
        if base == "sub-0_ses-10_task-Sherlock1_run-1":
            m[:] = False
        if m.any():
            e = np.load(f"{CACHE}/runs_emb/{base}.npy", mmap_mode="r")
            ok = ~np.isnan(e).any(1)
            labels.extend(y[m].tolist())
            subjects.extend([sub] * int(m.sum()))
            audio_ok.extend(ok[m].tolist())
            # Same source audio and onset means exactly the same target across subjects.
            tokens.extend([(ses if ses in S.SHARED else base, round(float(t), 2)) for t in tc[m]])
        if sub in S.EVAL_SUBJECTS:
            test_labels[sub].extend(y[S.block_masks(base, tc)["B"]].tolist())
y, sub, ok = np.array(labels), np.array(subjects), np.array(audio_ok)
tokid = {t: i for i, t in enumerate(sorted(set(tokens)))}
tid = np.array([tokid[t] for t in tokens])
i0, ib = np.flatnonzero(sub == 0), np.flatnonzero(sub != 0)
p0 = np.bincount(y[i0], minlength=50) / len(i0)
pb = np.bincount(y[ib], minlength=50) / len(ib)
stored = np.bincount(y, minlength=50) / len(y)
effective = 0.5 * (p0 + pb)
out["training_distribution"] = {
    "n_subj0": len(i0), "n_broad": len(ib),
    "broad_fraction_in_stored_prior": len(ib) / len(y),
    "broad_fraction_in_sampler": 0.5,
    "audio_coverage_rows": float(ok.mean()),
    "audio_coverage_sampler": float(0.5 * (ok[i0].mean() + ok[ib].mean())),
    "broad_draws_per_row_per_epoch": len(i0) // 128 * 128 / len(ib),
    "prior_total_variation": float(abs(stored - effective).sum() / 2),
    "largest_prior_ratio_changes": sorted(
        [{"word": VOCAB[c], "effective_over_stored": float(effective[c] / stored[c])}
         for c in range(50) if stored[c] > 0],
        key=lambda r: abs(np.log(r["effective_over_stored"])), reverse=True)[:10],
}
rng = np.random.default_rng(20261008)
collision = []
for _ in range(300):
    b = np.r_[rng.choice(i0, 128, replace=False), rng.choice(ib, 128, replace=True)]
    b = b[ok[b]]
    same_token = tid[b, None] == tid[b][None, :]
    same_word = y[b, None] == y[b][None, :]
    np.fill_diagonal(same_token, False)
    np.fill_diagonal(same_word, False)
    collision.append([len(b), same_token.any(1).mean(), same_token.sum(1).mean(), same_word.sum(1).mean()])
out["speech_batch_collision"] = dict(zip(
    ["mean_audio_batch_size", "fraction_with_duplicate_audio_token", "mean_duplicate_token_negatives", "mean_same_word_negatives"],
    np.mean(collision, axis=0).tolist()))
out["local_metric"] = {}
for s, ys in test_labels.items():
    yt = torch.tensor(ys, dtype=torch.long)
    counts = np.bincount(ys, minlength=50)
    floor = bacc_at_k(torch.tensor(stored).expand(len(ys), 50), yt)
    out["local_metric"][s] = {
        "n": len(ys), "present_classes": int((counts > 0).sum()),
        "missing_words": [VOCAB[c] for c in np.flatnonzero(counts == 0)],
        "classes_with_at_most_5_examples": int(((counts > 0) & (counts <= 5)).sum()),
        "constant_prior_ranking": floor,
        "uniform_random_expected": 10 / 50,
    }

# Recreate the sentence selection exactly, without rerunning model inference.
runs = set()
for f in sorted(glob.glob("/mnt/ssd4tb/data/LibriBrain100/*/derivatives/events/sub-0_*.tsv")):
    d = pd.read_csv(f, sep="\t")
    d = d[d.kind == "word"]
    cur, cs = [], None
    for w, s in zip([norm_word(x) for x in d.segment] + [None], d.sentenceidx.tolist() + [None]):
        if w is not None and w in VOCAB_ID and (not cur or s == cs):
            cur.append(w); cs = s
        else:
            if 5 <= len(cur) <= 9:
                runs.add(" ".join(cur))
            cur = [w] if w is not None and w in VOCAB_ID else []
            cs = s
runs = sorted(runs)
fold = lambda r: int(hashlib.md5(r.encode()).hexdigest(), 16) % 2
out["sentence_selection"] = {"total_fragments": len(runs), "arms": {}}
for arm, classes in pool_classes.items():
    valid = [r for r in runs if all(VOCAB_ID[w] in classes for w in r.split())]
    out["sentence_selection"]["arms"][arm] = {}
    for f in (0, 1):
        all_s = [r for r in valid if fold(r) == f]
        selected = all_s[:300]
        words = [w for r in selected for w in r.split()]
        out["sentence_selection"]["arms"][arm][f] = {
            "available": len(all_s), "selected": len(selected),
            "first": selected[:3], "last": selected[-3:],
            "first_word_counts": dict(collections.Counter(r.split()[0] for r in selected)),
            "present_classes": len(set(words)),
        }

# Verify the actual forward-backward function against enumeration on a small vocabulary.
tree = ast.parse((ROOT / "scripts/sim_lm2.py").read_text())
fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "marginals")
v, n = 3, 4
p = rng.dirichlet(np.ones(v), size=(v + 1, v + 1))
ns = {"np": np, "V": v, "logP": np.log(p)}
exec(compile(ast.Module(body=[fn], type_ignores=[]), "marginals_audit", "exec"), ns)
ell = rng.normal(size=(n, v)); beta = 0.7
seqs = list(itertools.product(range(v), repeat=n))
lp = []
for seq in seqs:
    a, b = v, v; val = 0.0
    for i, c in enumerate(seq):
        val += np.log(p[a, b, c]) + beta * ell[i, c]; a, b = b, c
    lp.append(val)
probs = np.exp(np.array(lp) - np.logaddexp.reduce(lp))
exact = np.zeros((n, v))
for seq, pr in zip(seqs, probs):
    for i, c in enumerate(seq): exact[i, c] += pr
out["forward_backward_max_error"] = float(abs(np.exp(ns["marginals"](ell, beta)) - exact).max())

# The legacy seed-0 control predates insertion of a randomly initialized audio projection.
def rng_after_init(with_audio):
    from src.models import WordCNN
    torch.manual_seed(0)
    WordCNN(d=128, p=0.5)
    torch.nn.Linear(128 // 2 * 25, 256)
    if with_audio: torch.nn.Linear(768, 256)
    return torch.randperm(20).tolist()
out["seed0_batch_rng"] = {"legacy_control": rng_after_init(False), "speech_code": rng_after_init(True)}

# Check one raw file/cache pair and quantify the boxcar decimator's stop-band.
rawpath = glob.glob('/mnt/ssd4tb/data/LibriBrain100/Sherlock1/derivatives/serialised/sub-1_ses-11*h5')[0]
base = pathlib.Path(rawpath).name.split('_proc')[0]
with h5py.File(rawpath) as h, np.load(f'{CACHE}/runs/{base}.npz') as d:
    names = h.attrs['channel_names'].split(', ')
    ix = np.linspace(0, len(d['onset']) - 1, 12).astype(int)
    samples = [int(round(float(d['onset'][i]) * 250)) for i in ix]
    raw = np.stack([h['data'][:, s:s+250] for s in samples])
    rebuilt = downsample(raw).astype(np.float16)
    cached = d['X'][ix]
    seg = h['data'][:, 250*60:250*120]
    f, psd = signal.welch(seg, fs=250, nperseg=1000, axis=-1)
    ratio = psd[:, f > 25].sum(-1) / psd.sum(-1)
    out['preprocessing'] = {
        'file': rawpath,
        'mag_channel_order_matches': bool(np.array_equal(np.array([n.endswith('1') for n in names]), IS_MAG)),
        'sample_frequency': float(h.attrs['sample_frequency']),
        'source_lowpass_hz': float(h.attrs['lowpass_cutoff']),
        'cache_samples_bitwise_equal': bool(np.array_equal(rebuilt, cached)),
        'median_raw_power_fraction_above_25hz': float(np.median(ratio)),
        'boxcar_amplitude_at_30hz': float(abs(np.mean(np.exp(-2j*np.pi*30*np.arange(5)/250)))),
        'boxcar_amplitude_at_40hz': float(abs(np.mean(np.exp(-2j*np.pi*40*np.arange(5)/250)))),
        'note': 'Spectral fraction is descriptive for one minute of one recording, not measured score impact.'
    }

# Submission integrity without using holdout timing or cross-subject correspondences.
out['submissions'] = {}
for p in sorted((ROOT / 'submissions').glob('*.csv')):
    d = pd.read_csv(p)
    values = d.iloc[:, 1:].to_numpy()
    out['submissions'][p.name] = {
        'shape': list(d.shape), 'unique_indices': bool(d.iloc[:, 0].is_unique),
        'vocab_order_matches': list(d.columns[1:]) == VOCAB,
        'finite_nonnegative': bool(np.isfinite(values).all() and (values >= 0).all())
    }

dst = ROOT / "results/audit_execution_20261008.json"
dst.write_text(json.dumps(out, indent=2) + "\n")
print(json.dumps(out, indent=2))
