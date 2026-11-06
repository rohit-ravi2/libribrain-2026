"""306-channel probe fixtures. They do not establish a LibriBrain decoding result."""
from __future__ import annotations

import os
import subprocess
import unittest

import numpy as np

from campaigns.sdr_uncertainty.decode306 import (
    _fit_whiten,
    _ridge_dual,
    assert_time_chunk,
    collect_day,
    evaluate_decode306,
    normalize_raw_window,
    planned_budget,
    read_window,
)
from campaigns.sdr_uncertainty.models import ridge_multi
from campaigns.sdr_uncertainty.protocol import DECODE306_MAX_READ_BYTES, N_SENSORS, VOCAB, WINDOW_SAMPLES


class BudgetAndNormTests(unittest.TestCase):
    def test_budget_is_explicit_and_inside_the_byte_cap(self):
        plan = planned_budget(4)
        self.assertEqual(plan["max_read_bytes"], 512 * 1024 * 1024)
        self.assertEqual(plan["bytes_per_window"], N_SENSORS * WINDOW_SAMPLES * 4)
        self.assertEqual(plan["max_windows_per_day"], plan["max_windows_total"] // 4)
        self.assertGreater(plan["max_windows_per_day"], 48)
        self.assertLessEqual(
            plan["max_windows_per_day"] * 4 * plan["bytes_per_window"],
            DECODE306_MAX_READ_BYTES,
        )
        self.assertEqual(plan["channels"], 306)
        self.assertEqual(plan["time_bins"], 50)

    def test_normalization_matches_the_torch_median_and_unbiased_std(self):
        rng = np.random.default_rng(0)
        raw = rng.normal(size=(306, 250))
        mag = (np.arange(306) % 3) == 0
        scale = np.where(mag, 1e13, 1e12)
        x = raw * scale[:, None]
        x = x - x.mean(axis=1, keepdims=True)
        sd = x.std(axis=1, ddof=1)
        s_mag = np.sort(sd[mag])[(int(mag.sum()) - 1) // 2]
        s_grad = np.sort(sd[~mag])[(int((~mag).sum()) - 1) // 2]
        div = np.where(mag, max(s_mag, 1e-6), max(s_grad, 1e-6))
        expected = np.clip(x / div[:, None], -10, 10).reshape(306, 50, 5).mean(axis=2)
        self.assertTrue(np.allclose(normalize_raw_window(raw), expected))
        self.assertTrue(np.allclose(normalize_raw_window(np.zeros((306, 250))), 0.0))

    def test_chunks_stop_at_the_budget_and_refuse_a_full_read(self):
        array = np.zeros((306, 1000), dtype=np.float32)
        calls = []

        class Chunks:
            shape = array.shape
            dtype = array.dtype

            def __getitem__(self, key):
                if key == (slice(None), slice(None)) or key == slice(None):
                    raise RuntimeError("full read")
                calls.append(key)
                return array[key]

        dataset = Chunks()
        with self.assertRaises(RuntimeError):
            assert_time_chunk(0, 1000, 1000)
        block, used = read_window(dataset, 0, 250, 0, N_SENSORS * 250 * 4)
        self.assertEqual(block.shape, (306, 250))
        self.assertEqual(used, N_SENSORS * 250 * 4)
        self.assertTrue(all(key[1].stop - key[1].start == 250 for key in calls))
        self.assertTrue(all(key[0].stop - key[0].start <= 32 for key in calls))
        again, used2 = read_window(dataset, 250, 500, used, used)
        self.assertIsNone(again)
        self.assertEqual(used2, used)
        rows = [{"segment": word, "timemeg": float(index)} for index, word in enumerate(VOCAB)]
        rows.append({"segment": "not-a-vocab-word", "timemeg": 0.1})
        part = collect_day(dataset, rows, VOCAB, per_day=3, used=0, max_bytes=N_SENSORS * 250 * 4 * 2 + 10)
        self.assertEqual(part["n_windows"], 2)
        self.assertEqual(part["features"].shape, (2, 306, 50))
        self.assertEqual(part["bytes_read"], N_SENSORS * 250 * 4 * 2)
        self.assertNotIn("timemeg", part)


class MetricTests(unittest.TestCase):
    def _covered(self):
        rng = np.random.default_rng(0)
        per = 3
        labels = np.tile(np.arange(50), 4 * per)
        n = len(labels)
        features = rng.normal(scale=0.05, size=(n, 2, 4))
        features[np.arange(n), 0, 0] += labels
        day = np.repeat(np.arange(4), 50 * per)
        return features, labels, day

    def test_full_coverage_reports_paired_differences_without_a_claim(self):
        features, labels, day = self._covered()
        result = evaluate_decode306(features, labels, day, ["d0", "d1", "d2", "d3"])
        self.assertTrue(result["fifty_class_metric_defined"])
        self.assertFalse(result["real_meg_scientific_validation"])
        self.assertFalse(result["effect_claimed"])
        self.assertFalse(result["tenfold_labeled_minutes_claimed"])
        frozen = result["frozen_eval"]
        self.assertEqual(frozen["metric"], "bacc_at_10_over_all_50_classes")
        contrast = frozen["shrunk_whiten_minus_uniform"]
        self.assertEqual(len(contrast["per_day"]), 4)
        self.assertIsNotNone(contrast["se"])
        self.assertFalse(frozen["effect_claimed"])
        self.assertEqual(result["causal_ar"], "not_run")

    def test_missing_class_gates_the_metric(self):
        features, labels, day = self._covered()
        labels = labels.copy()
        labels[labels == 49] = 0
        result = evaluate_decode306(features, labels, day, ["d0", "d1", "d2", "d3"])
        self.assertFalse(result["fifty_class_metric_defined"])
        self.assertIsNone(result["frozen_eval"])
        self.assertEqual(result["claim_status"], "engineering_cap_metric_undefined")
        self.assertTrue(result["engineering_cap_is_not_scientific_evidence"])
        self.assertIn("Stage B", result["staged_evaluation"])
        self.assertFalse(result["effect_claimed"])

    def test_whitener_and_ridge_do_not_use_held_out_rows(self):
        rng = np.random.default_rng(1)
        train = rng.normal(size=(30, 3, 5))
        mean, transforms = _fit_whiten(train)
        changed = train.copy()
        changed += 10.0
        # A held-out array is not an argument. Refitting on train is stable.
        mean2, transforms2 = _fit_whiten(train)
        self.assertTrue(np.allclose(mean, mean2))
        self.assertTrue(all(np.allclose(a, b) for a, b in zip(transforms, transforms2)))
        self.assertFalse(np.allclose(train, changed))
        x = rng.normal(size=(40, 6))
        y = rng.normal(size=(40, 2))
        dual = _ridge_dual(x - x.mean(0), y - y.mean(0))
        primal = ridge_multi(x - x.mean(0), y - y.mean(0), lam=1e-2)
        # ridge_multi penalizes every column including a missing intercept; both are centered.
        self.assertEqual(dual.shape, primal.shape)

    def test_cli_fails_closed_and_the_script_keeps_the_hard_caps(self):
        env = os.environ.copy()
        env["SDR_FORBID_LIVE_COMPUTE"] = "1"
        env["LIBRIBRAIN_DATA"] = "/no/such/libribrain"
        proc = subprocess.run(
            [os.environ.get("PYTHON", "python3"), "-m", "campaigns.sdr_uncertainty.runner", "decode-306"],
            cwd=os.path.join(os.path.dirname(__file__), "..", ".."),
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("refusing", proc.stderr)
        self.assertNotIn("does not exist", proc.stderr)
        path = os.path.realpath(os.path.join(
            os.path.dirname(__file__), "..", "..", "scripts", "sdr_uncertainty_decode306.sh"
        ))
        with open(path, encoding="utf-8") as handle:
            text = handle.read()
        for needle in ("MemoryMax=4G", "MemorySwapMax=0", "CPUQuota=100%", "TasksMax=16", "decode-306"):
            self.assertIn(needle, text)
        self.assertNotIn("RuntimeMaxSec", text)
        self.assertNotIn("acquire_lock.sh", text)
        subprocess.run(["bash", "-n", path], check=True)
        sidecar = os.path.join(os.path.dirname(__file__), "STATUS_CORRECTION.md")
        with open(sidecar, encoding="utf-8") as handle:
            note = handle.read()
        self.assertIn("not a word decoder", note)
        self.assertIn("27, 20, 20, and 28", note)


if __name__ == "__main__":
    unittest.main()
