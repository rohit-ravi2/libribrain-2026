"""Average the row probabilities of raw (pre-SLP, pre-Sinkhorn) submission CSVs of the same track (equal weights unless W=w1,w2,..).
Usage: avg_csv.py out.csv in1.csv in2.csv [...]; then hmm_prior.py --track deep --T 3 --kappa 20 --inp out.csv --out out_k20.csv"""
import sys, os, numpy as np, pandas as pd
out, ins = sys.argv[1], sys.argv[2:]; w = [float(x) for x in os.environ.get("W", ",".join(["1"] * len(ins))).split(",")]
dfs = [pd.read_csv(f) for f in ins]; assert all((d.iloc[:, 0].values == dfs[0].iloc[:, 0].values).all() and list(d.columns) == list(dfs[0].columns) for d in dfs)
P = sum(wi * (d.iloc[:, 1:].to_numpy(float) / d.iloc[:, 1:].to_numpy(float).sum(1, keepdims=True)) for wi, d in zip(w, dfs)) / sum(w)
o = dfs[0].copy(); o.iloc[:, 1:] = P; o.to_csv(out, index=False); print("wrote", out, P.shape)
