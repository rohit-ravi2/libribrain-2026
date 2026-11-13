import os

# Tiny synthetic problems: avoid OpenMP oversubscription in the boosted classifier.
os.environ.setdefault("OMP_NUM_THREADS", "2")
