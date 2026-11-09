"""What the residual runner measures. The completion flag is not evidence of an effect."""
from __future__ import annotations

from .protocol import STATE_CHANNELS, TARGET_CHANNELS


def residual_audit_record() -> dict:
    return {
        "target_channels": list(TARGET_CHANNELS),
        "state_channels": list(STATE_CHANNELS),
        "channel_rule": "fixed file-order indices, not chosen by correlation or anatomy",
        "target_definition": (
            "Per channel, the mean of the raw onset-to-+1s samples times the sensor scale. "
            "The window mean is not removed."
        ),
        "official_decoder_removes_that_mean": True,
        "state_definition": (
            "run_fraction among the kept windows, and the log mean-square of the state "
            "channels over the same second, including the offset."
        ),
        "labels_enter_the_residual_fit": False,
        "task_metric_on_this_stream_is_interpreted": False,
        "windows_may_overlap_in_time": True,
        "heldout_logscore_meaning": (
            "Mean Gaussian log density of the 4 window-means on a held-out acquisition day, "
            "with mean, variance, and z-scoring fit on the other days. It is a scoring rule "
            "under a conditional-independence assumption the overlapping windows violate. "
            "It is not the likelihood of a word and not a decoding accuracy."
        ),
        "uncertainty_unit": "held-out acquisition day",
        "endpoint_is_decoding": False,
        "adequate_competition_decoder": False,
        "completion_flag_is_not_an_effect": True,
        "local_pilot_reading": (
            "A finished 4-day by 48-window run with M2-M0 about -0.099 and SE about 0.42 "
            "is not distinguishable from zero under the two-SE reading. The negative point "
            "estimate does not justify precision-weighted decoding."
        ),
    }
