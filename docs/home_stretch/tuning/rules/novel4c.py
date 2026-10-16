"""Confound check: is the kappa-average gain just a lower single kappa?"""
import sys, os, numpy as np
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, HERE)
from simrule import load, score, slp
D = load(); idx = {id(b): i for i, b in enumerate(D)}
C = {(i, k): slp(b["raw"], kappa=k)[0] for i, b in enumerate(D) for k in [5, 10]}
for k in [5, 10]: print("single kappa", k, score(lambda b, k=k: C[(idx[id(b)], k)], D), flush=True)
