"""libribrain-floor: evaluation tools for measuring the word-timing component of MEG word decoding."""
from .metrics import balanced_topk
from .balance import sinkhorn
from .floor import timing_features, timing_floor, packard_bits
from .twin import onset_kernel, twin_signal
from .deconv import fit_trf, subtract_trf

__version__ = "0.1.0"
__all__ = ["balanced_topk", "sinkhorn", "timing_features", "timing_floor", "packard_bits",
           "onset_kernel", "twin_signal", "fit_trf", "subtract_trf"]
