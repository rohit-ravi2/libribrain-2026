"""How an encoding likelihood would enter a decoder. The intervention is not run."""
from __future__ import annotations


def encoding_to_decoder_note() -> dict:
    return {
        "executed": False,
        "neural_decoder_trained": False,
        "encoding_negative_log_likelihood": (
            "0.5 * (sum_d log(2*pi*sigma_d^2(s)) + (y_d - mu_d(s))^2 / sigma_d^2(s))"
        ),
        "intervention_not_run": (
            "A later structured-noise training intervention would replace an isotropic "
            "feature loss with that encoding likelihood, which weights each residual "
            "dimension by the precision 1/sigma_d^2(s). s is a conditioning feature "
            "available at inference. Labels, absolute onset, shuffle seeds, and "
            "stimulus-clock alignment are not inputs to sigma. "
            "Tenfold fewer labeled minutes is an aspirational target on a later rung, not a result."
        ),
    }
