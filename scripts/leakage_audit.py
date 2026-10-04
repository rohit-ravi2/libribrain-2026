"""Leakage audit for the local split (charter Phase 1 step 3). Writes results/leakage_audit.md. Any FAIL exits non-zero."""
import glob, os, re, subprocess, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.common import CACHE, DATA, norm_word
from src import split as S

out, fails = [], 0
def rep(ok, msg):
    global fails
    fails += (not ok); out.append(f"- {'PASS' if ok else 'FAIL'}: {msg}")

runs = sorted(glob.glob(f"{CACHE}/runs/*.npz"))
n_train = n_B = 0; min_gap = np.inf; eval_in_train = 0
for f in runs:
    base = os.path.basename(f)[:-4]; sub, ses = S.parse(base)
    t = np.load(f)["timechapter"]
    tm = S.train_mask(base, t); n_train += tm.sum()
    if sub in S.EVAL_SUBJECTS: eval_in_train += tm.sum()
    if ses in S.SHARED:
        B = S.block_masks(base, t)["B"]
        if sub in S.EVAL_SUBJECTS: n_B += B.sum()
        if tm.any():  # gap between latest train window end and earliest block-B stimulus time
            min_gap = min(min_gap, 0.75 * S.T_MAX[ses] + S.GUARD - (t[tm].max() + 1.0))
rep(eval_in_train == 0, f"subject axis: {eval_in_train} training windows from eval subjects {S.EVAL_SUBJECTS}")
rep(min_gap >= 0.99, f"stimulus axis (shared Sherlock1 ses-11/12): min gap between any training window end and block B start = {min_gap:.2f} s (>= 1 s required)")
out.append(f"- INFO: training windows {n_train}; block-B eval windows {n_B} over {len(S.EVAL_SUBJECTS)} subjects")

# Text-level: do block-B word 8-grams occur anywhere in the training text (all sub-0 tasks + other broad subjects)?
def words(ev):
    e = pd.read_csv(ev, sep="\t"); w = e[e.kind == "word"]
    return [norm_word(x) for x in w.segment], w.timechapter.values if "timechapter" in w else np.full(len(w), np.nan)
B_grams, T_grams = set(), set()
seen = set()
for ev in sorted(glob.glob(f"{DATA}/*/derivatives/events/sub-*_events.tsv")):
    base = os.path.basename(ev).replace("_events.tsv", ""); sub, ses = S.parse(base)
    key = (ses, sub if sub in S.EVAL_SUBJECTS else "train")
    ws, t = words(ev)
    if ses in S.SHARED:
        m = S.block_masks(base, t)
        if sub in S.EVAL_SUBJECTS:
            idx = np.nonzero(m["B"])[0]; tgt = B_grams
        else:
            idx = np.nonzero(m["pre75"])[0]; tgt = T_grams
    else:
        if sub in S.EVAL_SUBJECTS: continue
        idx = np.arange(len(ws)); tgt = T_grams
    # n-grams over contiguous index runs only
    for a in range(len(idx) - 7):
        if idx[a + 7] - idx[a] == 7:
            tgt.add(tuple(ws[idx[a]:idx[a] + 8]))
ov = B_grams & T_grams
rep(len(ov) <= 0.01 * max(len(B_grams), 1), f"text axis: {len(ov)} of {len(B_grams)} block-B word 8-grams also occur in training text"
    + (f" (examples: {[' '.join(g) for g in list(ov)[:3]]})" if ov else ""))

# Holdout metadata policy (RULES.md §4): no model code may read timing/shuffle/cross-subject fields.
code = subprocess.run(["grep", "-rn", "-E", "word_onsets_s|shuffle_seed|onset_s", "src", "scripts", "--include=*.py"],
                      capture_output=True, text=True).stdout
code = "\n".join(l for l in code.splitlines() if "leakage_audit.py" not in l)
rep(code.strip() == "", "holdout metadata: no reference to word_onsets_s / shuffle_seed / onset_s in src/ or scripts/" + (f"\n```\n{code}\n```" if code else ""))

os.makedirs("results", exist_ok=True)
txt = "# Leakage audit\n\nSplit: `src/split.py` (EVAL_SUBJECTS=%s, GUARD=%.1f s)\n\n" % (S.EVAL_SUBJECTS, S.GUARD) + "\n".join(out) + "\n"
open("results/leakage_audit.md", "w").write(txt); print(txt)
sys.exit(1 if fails else 0)
