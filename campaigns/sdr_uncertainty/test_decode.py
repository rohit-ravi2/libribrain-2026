"""Decode-probe tests. A synthetic pass is not a decoding result."""
from __future__ import annotations

import os
import subprocess
import unittest

import numpy as np

from campaigns.sdr_uncertainty.decode import (
    collect_decode_windows,
    evaluate_decoding,
    fit_rho,
    keep_nonoverlapping,
    shrunk_weights,
    summarize_demeaned,
)
from campaigns.sdr_uncertainty.outday import contrast_reading
from campaigns.sdr_uncertainty.protocol import TARGET_CHANNELS, VOCAB
from campaigns.sdr_uncertainty.stream import Hyperslab


def _signal_batch():
    rng = np.random.default_rng(0)
    n_days, n_per, n_classes = 3, 40, 20
    n = n_days * n_per
    labels = np.resize(np.arange(n_classes), n)
    y = rng.normal(scale=0.05, size=(n, n_classes))
    y[np.arange(n), labels] += 4.0
    return {
        "y": y,
        "state": rng.normal(size=(n, 2)),
        "label": labels,
        "day": np.repeat(np.arange(n_days), n_per),
        "time_index": np.tile(np.arange(n_per), n_days),
        "dates": ["2024-01-02", "2024-03-01", "2024-06-01"],
    }


class AuditReadingTests(unittest.TestCase):
    def test_reported_pilot_contrast_is_not_distinguishable(self):
        reading = contrast_reading(-0.099, 0.42)
        self.assertFalse(reading["distinguishable_from_zero"])
        self.assertTrue(reading["science_flag_is_not_evidence"])

    def test_constant_window_is_removed_and_overlaps_are_dropped(self):
        summary = summarize_demeaned(np.ones((4, 250)), TARGET_CHANNELS)
        self.assertTrue(np.allclose(summary, 0.0))
        kept = keep_nonoverlapping(np.array([0.0, 0.2, 1.0, 1.1]))
        self.assertEqual(kept.tolist(), [0, 2])


class DecodeTests(unittest.TestCase):
    def test_probe_reports_arms_without_claiming_a_gain(self):
        result = evaluate_decoding(_signal_batch())
        self.assertEqual(result["claim_status"], "decode_probe_completed")
        self.assertFalse(result["real_meg_scientific_validation"])
        self.assertFalse(result["effect_claimed"])
        self.assertFalse(result["adequate_for_competition_claim"])
        self.assertTrue(result["endpoint_is_decoding"])
        frozen = result["frozen_eval"]
        self.assertIn("uniform", frozen["arms"])
        self.assertIn("shrunk_weight", frozen["arms"])
        self.assertIn("shuffled_weight", frozen["arms"])
        self.assertIn("quadratic_features", frozen["arms"])
        self.assertIn("causal_ar", frozen["arms"])
        self.assertEqual(len(frozen["weighted_minus_uniform"]["per_day"]), 3)
        self.assertIsNotNone(frozen["weighted_minus_uniform"]["se"])
        self.assertFalse(frozen["effect_claimed"])
        fractions = result["exploratory"]["labeled_word_fractions"]
        ten = [row for row in fractions if row["fraction_of_labeled_words"] == 0.10][0]
        self.assertTrue(ten["aspirational"])
        self.assertIsNone(ten["weighted_minus_uniform_mean"])
        self.assertEqual(ten["status"], "underpowered")
        self.assertFalse(result["exploratory"]["label_permutation"]["diagnose_required"])

    def test_sparse_labels_produce_no_score(self):
        batch = _signal_batch()
        batch["label"] = np.full(len(batch["label"]), -1)
        batch["label"][::20] = 0
        result = evaluate_decoding(batch)
        self.assertEqual(result["claim_status"], "decoding_inadequate")
        self.assertIsNone(result["frozen_eval"])
        self.assertFalse(result["effect_claimed"])

    def test_rho_ignores_the_held_out_day_and_weights_stay_clipped(self):
        y = np.random.default_rng(1).normal(size=(40, 3))
        day = np.repeat([0, 1], 20)
        train = day == 0
        rho = fit_rho(y, day, train)
        shifted = y.copy()
        shifted[day == 1] += 100.0
        self.assertEqual(fit_rho(shifted, day, train), rho)
        weights = shrunk_weights(np.linspace(-1, 1, 30), np.random.default_rng(2).normal(size=(30, 4)), np.linspace(-1, 1, 10))
        self.assertGreaterEqual(float(weights.min()), 0.5)
        self.assertLessEqual(float(weights.max()), 1.5)

    def test_stream_is_nonoverlapping_and_not_a_full_read(self):
        rows = [
            {"segment": "the", "timemeg": 0.0},
            {"segment": "a", "timemeg": 0.2},
            {"segment": "of", "timemeg": 2.0},
        ]
        dataset = Hyperslab(np.zeros((306, 2000)))
        batch = collect_decode_windows(dataset, rows, VOCAB, max_windows=10, max_bytes=32 * 1024 * 1024)
        self.assertEqual(batch["n_windows"], 2)
        self.assertTrue(batch["nonoverlapping"])
        self.assertTrue(np.allclose(batch["y"], 0.0))
        self.assertLess(dataset.bytes_read, 32 * 1024 * 1024)
        with self.assertRaises(Exception):
            dataset[:]
        self.assertNotIn("timemeg", batch)


class DecodeGuardTests(unittest.TestCase):
    def test_cli_fails_closed_and_script_keeps_caps(self):
        env = os.environ.copy()
        env["SDR_FORBID_LIVE_COMPUTE"] = "1"
        env["LIBRIBRAIN_DATA"] = "/no/such/libribrain"
        proc = subprocess.run(
            [os.environ.get("PYTHON", "python3"), "-m", "campaigns.sdr_uncertainty.runner", "decode-probe"],
            cwd=os.path.join(os.path.dirname(__file__), "..", ".."),
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("refusing", proc.stderr)
        self.assertNotIn("does not exist", proc.stderr)
        path = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", "..", "scripts", "sdr_uncertainty_decode_probe.sh"))
        with open(path, encoding="utf-8") as handle:
            text = handle.read()
        for needle in ("MemoryMax=4G", "MemorySwapMax=0", "CPUQuota=100%", "TasksMax=16", "RuntimeMaxSec=600", "decode-probe"):
            self.assertIn(needle, text)
        self.assertNotIn("acquire_lock.sh", text)
        subprocess.run(["bash", "-n", path], check=True)


if __name__ == "__main__":
    unittest.main()
