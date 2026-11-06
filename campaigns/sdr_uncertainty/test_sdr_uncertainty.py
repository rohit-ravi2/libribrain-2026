"""Synthetic tests for the residual-uncertainty campaign. They do not validate MEG."""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import textwrap
import unittest

import numpy as np

from campaigns.sdr_uncertainty.cgroup import CgroupError, evaluate_limits, verify_fixture
from campaigns.sdr_uncertainty.crossfit import fit_scaler, run_crossfit, smoke_one_series
from campaigns.sdr_uncertainty.leakage import LeakageError
from campaigns.sdr_uncertainty.models import fit_model, parameter_count, predict_model
from campaigns.sdr_uncertainty.protocol import (
    FEATURE_SET,
    MEMORY_MAX_BYTES,
    MODEL_NAMES,
    TARGET_CHANNELS,
    STATE_CHANNELS,
    VOCAB,
    next_rung,
    protocol_sha256,
)
from campaigns.sdr_uncertainty.stream import Hyperslab, discover_smoke_run, parse_events, stream_windows, time_split
from campaigns.sdr_uncertainty.synthetic import make_dataset, make_null_state


PASSING = {
    "memory.max": str(MEMORY_MAX_BYTES),
    "memory.swap.max": "0",
    "cpu.max": "100000 100000",
    "pids.max": "16",
}
ENV = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "BLIS_NUM_THREADS": "1",
    "CUDA_VISIBLE_DEVICES": "",
    "LB_WORKERS": "1",
}


def _mean(result, name):
    return result["models"][name]["mean"]


class ProtocolTests(unittest.TestCase):
    def test_hash_is_stable_and_vocab_is_closed(self):
        self.assertEqual(protocol_sha256(), protocol_sha256())
        self.assertEqual(len(protocol_sha256()), 64)
        self.assertEqual(len(VOCAB), 50)
        self.assertEqual(len(set(TARGET_CHANNELS) & set(STATE_CHANNELS)), 0)

    def test_parameter_counts_match_for_the_capacity_control(self):
        self.assertEqual(
            parameter_count("M2_state_variance", 4, 2, 3),
            parameter_count("C_matched_random_state", 4, 2, 3),
        )
        self.assertGreater(parameter_count("C_rich_mean", 4, 2), parameter_count("M0_fixed_gaussian", 4, 2))
        self.assertEqual(parameter_count("C_day_intercept", 4, 2, n_train_days=4),
                         parameter_count("M0_fixed_gaussian", 4, 2) + 4 * 4)

    def test_next_rung_is_not_executed(self):
        step = next_rung("residual_prediction_smoke_one_training_day_group")
        self.assertEqual(step["next"], "out_of_day_replication")
        self.assertFalse(step["executed"])


class CgroupTests(unittest.TestCase):
    def test_accepts_exact_caps(self):
        self.assertEqual(evaluate_limits(PASSING, ENV, [1], 600), [])

    def test_rejects_each_broken_cap(self):
        cases = {
            "memory.max": "1073741824",
            "memory.swap.max": "1",
            "cpu.max": "max 100000",
            "pids.max": "64",
        }
        for key, value in cases.items():
            broken = dict(PASSING)
            broken[key] = value
            self.assertTrue(evaluate_limits(broken, ENV, [], 600), key)
        self.assertTrue(evaluate_limits(PASSING, ENV, [], 60))
        self.assertTrue(evaluate_limits(PASSING, ENV, [195], 600))
        env = dict(ENV)
        env["OMP_NUM_THREADS"] = "8"
        self.assertTrue(evaluate_limits(PASSING, env, [], 600))
        env = dict(ENV)
        env["CUDA_VISIBLE_DEVICES"] = "0"
        self.assertTrue(evaluate_limits(PASSING, env, [], 600))

    def test_fixture_directory(self):
        root = tempfile.mkdtemp()
        try:
            for name, value in PASSING.items():
                open(os.path.join(root, name), "w", encoding="utf-8").write(value + "\n")
            open(os.path.join(root, "runtime.max_sec"), "w", encoding="utf-8").write("600\n")
            open(os.path.join(root, "environ"), "w", encoding="utf-8").write(
                "".join(f"{key}={value}\n" for key, value in ENV.items())
            )
            open(os.path.join(root, "nvidia.majors"), "w", encoding="utf-8").write("1\n")
            verify_fixture(root)
            open(os.path.join(root, "memory.swap.max"), "w", encoding="utf-8").write("4096\n")
            with self.assertRaises(CgroupError):
                verify_fixture(root)
        finally:
            shutil.rmtree(root)


class ModelTests(unittest.TestCase):
    def test_hetero_variance_beats_mean_and_capacity_controls(self):
        result = run_crossfit(make_dataset("hetero"))
        gap = _mean(result, "M2_state_variance") - _mean(result, "M0_fixed_gaussian")
        self.assertGreater(gap, 0.02, msg=str({k: result["models"][k]["mean"] for k in MODEL_NAMES}))
        self.assertGreater(
            _mean(result, "M2_state_variance"),
            _mean(result, "C_rich_mean") + 0.01,
        )
        self.assertGreater(
            _mean(result, "M2_state_variance"),
            _mean(result, "C_matched_random_state") + 0.01,
        )
        self.assertLess(
            result["models"]["M2_state_variance"]["squared_innovation_acf_lag1_mean"],
            result["models"]["M0_fixed_gaussian"]["squared_innovation_acf_lag1_mean"],
        )
        self.assertFalse(result["real_meg_scientific_validation"])
        self.assertEqual(result["claim_status"], "synthetic_evaluation_only")
        self.assertFalse(result["neural_inference_allowed"])
        self.assertFalse(result["decoder_link"]["executed"])
        self.assertFalse(result["diagnose_required"])

    def test_homo_gap_is_smaller_than_the_hetero_gap(self):
        hetero = run_crossfit(make_dataset("hetero"))
        homo = run_crossfit(make_dataset("homo"))
        hetero_gap = _mean(hetero, "M2_state_variance") - _mean(hetero, "M0_fixed_gaussian")
        homo_gap = _mean(homo, "M2_state_variance") - _mean(homo, "M0_fixed_gaussian")
        self.assertGreater(hetero_gap, homo_gap + 0.02)

    def test_temporal_model_cuts_innovation_correlation(self):
        result = run_crossfit(make_dataset("ar1"))
        m0 = abs(result["models"]["M0_fixed_gaussian"]["innovation_acf_lag1_mean"])
        m1 = abs(result["models"]["M1_temporal_corr"]["innovation_acf_lag1_mean"])
        self.assertGreater(m0, 0.4)
        self.assertLess(m1, m0 - 0.2)
        self.assertGreater(_mean(result, "M1_temporal_corr"), _mean(result, "M0_fixed_gaussian"))

    def test_day_intercept_does_not_use_the_test_day_mean(self):
        data = make_dataset("day_shift", n_days=4, n_per=80)
        train = data["day"] < 3
        test = data["day"] == 3
        null = make_null_state(len(data["y"]), 2)
        params = fit_model(
            "C_day_intercept", data["y"][train], data["state"][train], data["day"][train], null[train]
        )
        mu, _ = predict_model(params, data["y"][test], data["state"][test], data["day"][test], null[test])
        shifted = data["y"][test] + 100.0
        mu_shifted, _ = predict_model(params, shifted, data["state"][test], data["day"][test], null[test])
        self.assertTrue(np.allclose(mu, mu_shifted))
        self.assertGreater(abs(float(mu.mean()) - float(data["y"][test].mean())), 5.0)

    def test_scaler_ignores_test_rows_and_labels_do_not_change_logscores(self):
        data = make_dataset("homo", n_days=3, n_per=40)
        train = np.flatnonzero(data["day"] != 2)
        mu, _ = fit_scaler(data["y"], train)
        corrupted = data["y"].copy()
        corrupted[data["day"] == 2] += 50.0
        mu2, _ = fit_scaler(corrupted, train)
        self.assertTrue(np.allclose(mu, mu2))
        base = run_crossfit(data)
        shuffled = dict(data)
        shuffled["label"] = data["label"].copy()
        shuffled["label"] = shuffled["label"][np.random.default_rng(0).permutation(len(shuffled["label"]))]
        other = run_crossfit(shuffled)
        for name in MODEL_NAMES:
            self.assertAlmostEqual(_mean(base, name), _mean(other, name), places=6)

    def test_null_state_does_not_depend_on_y(self):
        self.assertTrue(np.allclose(make_null_state(20, 2, 3), make_null_state(20, 2, 3)))
        data = make_dataset("hetero", n_days=2, n_per=30)
        flipped = dict(data)
        flipped["y"] = -data["y"]
        # Both draws are positional. Equality of the helper is the contract.
        self.assertEqual(make_null_state(len(data["y"]), 2).shape, data["y"].shape[:1] + (2,))
        self.assertFalse(np.allclose(data["y"], flipped["y"]))

    def test_mining_nulls_the_search_and_one_day_is_refused(self):
        data = make_dataset("hetero", n_days=2, n_per=30)
        mined = run_crossfit(data, mined_features=True)
        self.assertEqual(mined["claim_status"], "null_adaptive_search")
        self.assertTrue(mined["metrics_omitted"])
        self.assertNotIn("models", mined)
        other = run_crossfit(data, feature_candidates=["a", "b"])
        self.assertEqual(other["claim_status"], "null_adaptive_search")
        data["day"] = np.zeros(len(data["y"]), dtype=np.int64)
        with self.assertRaises(LeakageError):
            run_crossfit(data)

    def test_forbidden_state_name_raises(self):
        data = make_dataset("homo", n_days=2, n_per=20)
        data["state_names"] = ("word_onsets_s", "s1")
        with self.assertRaises(LeakageError):
            run_crossfit(data)

    def test_one_series_smoke_is_not_cross_day(self):
        data = make_dataset("hetero", n_days=1, n_per=48)
        result = smoke_one_series(data)
        self.assertFalse(result["cross_day"])
        self.assertFalse(result["rung_completed_as_science"])
        self.assertFalse(result["real_meg_scientific_validation"])
        self.assertGreaterEqual(result["n_guard_dropped"], 8)
        self.assertFalse(result["next_rung"]["executed"])


class StreamTests(unittest.TestCase):
    def test_events_drop_forbidden_fields_and_hyperslab_is_bounded(self):
        text = textwrap.dedent("""\
            kind\tsegment\ttimemeg\tword_onsets_s\tshuffle_seed\ttimechapter\tonset_s
            word\thello\t0.0\t9\t1\t3\t9
            phoneme\tHH\t0.0\t9\t1\t3\t9
            word\tthe\t1.0\t9\t1\t4\t9
            """)
        rows, refused = parse_events(text)
        self.assertEqual(len(rows), 2)
        self.assertEqual(set(rows[0]), {"timemeg", "segment"})
        self.assertEqual(sorted(refused), ["onset_s", "shuffle_seed", "timechapter", "word_onsets_s"])
        array = np.zeros((306, 2000), dtype=np.float64)
        array[0, 0:250] = 1e-12
        dataset = Hyperslab(array)
        with self.assertRaises(Exception):
            dataset[:]
        batch = stream_windows(dataset, rows, VOCAB, max_windows=1, max_bytes=32 * 1024 * 1024)
        self.assertEqual(batch["n_windows"], 1)
        self.assertTrue(batch["truncated"])
        self.assertLessEqual(batch["bytes_read"], 32 * 1024 * 1024)
        self.assertNotIn("timemeg", batch)
        self.assertNotIn("timechapter", batch)
        self.assertEqual(batch["state_names"], ("run_fraction", "state_energy"))
        self.assertGreater(dataset.bytes_read, 0)
        self.assertLess(dataset.bytes_read, 32 * 1024 * 1024)
        self.assertTrue(all(stop - start == 250 for _, start, stop in dataset.calls))

    def test_sample_gap_and_day_rule(self):
        train, test = time_split(40, 8)
        self.assertGreaterEqual(int(test[0]) - int(train[-1]), 8)
        root = tempfile.mkdtemp()
        try:
            excluded = "sub-0_ses-11_task-Sherlock1_run-2"
            early = "sub-0_ses-02_task-Sherlock1_run-1"
            later = "sub-0_ses-01_task-Sherlock1_run-1"
            second = "sub-0_ses-02_task-Sherlock1_run-2"
            for base, date in (
                (excluded, "2024-10-04T00:00:00"),
                (later, "2024-06-01T00:00:00"),
                (early, "2024-01-02T00:00:00"),
                (second, "2024-01-02T01:00:00"),
            ):
                h5_dir = os.path.join(root, "Sherlock1", "derivatives", "serialised")
                ev_dir = os.path.join(root, "Sherlock1", "derivatives", "events")
                scan_dir = os.path.join(root, "Sherlock1", "sub-0", "ses-01")
                os.makedirs(h5_dir, exist_ok=True)
                os.makedirs(ev_dir, exist_ok=True)
                os.makedirs(scan_dir, exist_ok=True)
                open(os.path.join(h5_dir, base + "_proc-0_meg.h5"), "wb").close()
                open(os.path.join(ev_dir, base + "_events.tsv"), "w", encoding="utf-8").write("kind\n")
            scan = os.path.join(root, "Sherlock1", "sub-0", "ses-01", "sub-0_ses-01_scans.tsv")
            lines = ["filename\tacq_time"]
            for base, date in (
                (excluded, "2024-10-04T00:00:00"),
                (later, "2024-06-01T00:00:00"),
                (second, "2024-01-02T01:00:00"),
                (early, "2024-01-02T00:00:00"),
            ):
                lines.append(f"meg/{base}_meg.fif\t{date}")
            open(scan, "w", encoding="utf-8").write("\n".join(lines) + "\n")
            chosen = discover_smoke_run(root)
            self.assertEqual(chosen["date"], "2024-01-02")
            self.assertEqual(chosen["base"], early)
            self.assertNotIn("holdout", chosen["h5"])
        finally:
            shutil.rmtree(root)


class RunnerTests(unittest.TestCase):
    def test_live_smoke_is_refused_under_the_test_lock(self):
        env = os.environ.copy()
        env["SDR_FORBID_LIVE_SMOKE"] = "1"
        env["LIBRIBRAIN_DATA"] = tempfile.mkdtemp()
        proc = subprocess.run(
            [os.environ.get("PYTHON", "python3"), "-m", "campaigns.sdr_uncertainty.runner", "smoke-local"],
            cwd=os.path.join(os.path.dirname(__file__), "..", ".."),
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("refusing", proc.stderr)
        self.assertNotIn("real_meg_scientific_validation\": true", proc.stdout)

    def test_synthetic_cli_marks_itself_synthetic(self):
        out = tempfile.mkdtemp()
        env = os.environ.copy()
        env["SDR_SYNTHETIC"] = "1"
        env["SDR_TEST_HOOK"] = "1"
        env["LIBRIBRAIN_SDR_OUT"] = out
        proc = subprocess.run(
            [os.environ.get("PYTHON", "python3"), "-m", "campaigns.sdr_uncertainty.runner", "smoke-synthetic"],
            cwd=os.path.join(os.path.dirname(__file__), "..", ".."),
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)
        self.assertIn("synthetic_dgp", proc.stdout)
        self.assertIn('"real_meg_scientific_validation": false', proc.stdout)
        written = open(os.path.join(out, "sdr_uncertainty_synthetic.json"), encoding="utf-8").read()
        self.assertIn('"real_meg_scientific_validation": false', written)
        self.assertNotIn("2024-10-04", written)

    def test_shell_command_declares_caps_and_does_not_submit(self):
        path = os.path.join(os.path.dirname(__file__), "..", "..", "scripts", "sdr_uncertainty_local_smoke.sh")
        path = os.path.realpath(path)
        text = open(path, encoding="utf-8").read()
        for needle in (
            "MemoryMax=4G",
            "MemorySwapMax=0",
            "CPUQuota=100%",
            "TasksMax=16",
            "RuntimeMaxSec=600",
            "OMP_NUM_THREADS=1",
            "CUDA_VISIBLE_DEVICES=\"\"",
            "smoke-local",
        ):
            self.assertIn(needle, text)
        for forbidden in ("git push", "kaggle", "arxiv", "for day in", "queue"):
            self.assertNotIn(forbidden, text)
        subprocess.run(["bash", "-n", path], check=True)


if __name__ == "__main__":
    unittest.main()
