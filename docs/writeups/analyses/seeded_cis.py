"""Seeded confidence intervals for the timing ladder (leakage paper, Section 4.1).
Inputs: runs/ci_v3{b,c0,c_rel,d0,d_rel}_s{0,1,2}.json (scripts/queue_ci_timing.sh) and the inference-time ablations
logs/ci/ablate_ci_v3{c_rel,d_rel}_s*.log (scripts/queue_ci_transfer.sh, first loop), and the Section 4.4
transfer seeds (runs/deep_c, ci_deepc_s*, stretch_s*, ci_stretch_s2; PX and FITA arms added when present). Mean, sample sd and a two-sided
95% t interval over seeds (n = 3, t = 4.303); contrasts are paired by seed. Output: seeded_cis.json"""
import os, json, ast, re
import numpy as np
R = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../.."))
ARMS = {"b": "no context", "c0": "2-layer transformer, no timing", "c_rel": "2-layer transformer + timing",
        "d0": "+ T5 targets, no timing", "d_rel": "+ T5 targets + timing"}
SEEDS = (0, 1, 2); T95 = 4.303


def ci(v):
    v = np.asarray(v, float); m, sd = v.mean(), v.std(ddof=1)
    return {"mean": round(m, 4), "sd": round(sd, 4), "ci95": [round(m - T95 * sd / np.sqrt(len(v)), 4), round(m + T95 * sd / np.sqrt(len(v)), 4)], "seeds": [round(x, 4) for x in v]}


dep = {a: [json.load(open(f"{R}/runs/ci_v3{a}_s{s}.json"))["summary"]["deploy"] for s in SEEDS] for a in ARMS}
out = {"metric": "deploy (top-10 balanced accuracy, 32 seen + 7 new subjects, local blocks), chance 0.20",
       "arms": {a: {"desc": ARMS[a], **ci(dep[a])} for a in ARMS},
       "contrasts": {f"{x}-{y}": ci(np.subtract(dep[x], dep[y])) for x, y in (("c_rel", "c0"), ("d_rel", "d0"), ("c0", "b"), ("d0", "b"), ("c_rel", "b"))}}
abl = {}
for a in ("c_rel", "d_rel"):
    rows = {}
    for s in SEEDS:
        for ln in open(f"{R}/logs/ci/ablate_ci_v3{a}_s{s}.log"):
            m = re.match(r"(Q[34]) (full|rel_zero|single_word) (\{.*\})$", ln.strip())
            if m:
                d = ast.literal_eval(m.group(3)); ro = "cos" if "cos" in d else "ce"
                rows.setdefault((m.group(1), m.group(2)), []).append(d[ro]["seen"][0])
    abl[a] = {"readout": "cos" if a == "d_rel" else "ce", "group": "seen subjects",
              **{f"{b}_{c}": ci(v) for (b, c), v in sorted(rows.items())}}
out["ablation"] = abl
# Section 4.4 transfer arms (scripts/queue_ci_transfer.sh), paired by seed; arms with missing runs are skipped.
def dep_of(t):
    f = f"{R}/runs/{t}.json"; return json.load(open(f))["summary"]["deploy"] if os.path.exists(f) else None
tr = {}
base, trt = [dep_of(t) for t in ("deep_c", "ci_deepc_s1", "ci_deepc_s2")], [dep_of(t) for t in ("stretch_s0", "stretch_s1", "ci_stretch_s2")]
if None not in base + trt:
    tr["time_stretch_deep"] = {"metric": "Deep subject, local same-session blocks (deploy)", "base": ci(base), "treated": ci(trt), "paired_diff": ci(np.subtract(trt, base))}
out["transfer"] = tr
json.dump(out, open(f"{R}/docs/writeups/analyses/seeded_cis.json", "w"), indent=1); print(json.dumps(out, indent=1))
