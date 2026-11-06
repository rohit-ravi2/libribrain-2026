"""Tests for the out-of-day runner. Passing them does not validate MEG."""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import time
import unittest

import numpy as np

from campaigns.sdr_uncertainty.crossfit import smoke_one_series
from campaigns.sdr_uncertainty.models import parameter_count
from campaigns.sdr_uncertainty.outday import (
    classify_out_of_day,
    evaluate_out_of_day,
    stream_out_of_day,
)
from campaigns.sdr_uncertainty.protocol import (
    EXCLUDED_SMOKE_DATE,
    FEATURE_SET,
    MODEL_NAMES,
    OOD_MIN_DAYS,
    OOD_MIN_WINDOWS_PER_DAY,
    OOD_SELECTION_RULE,
    VOCAB,
)
from campaigns.sdr_uncertainty.stream import Hyperslab, discover_out_of_day_runs, parse_events
from campaigns.sdr_uncertainty.synthetic import make_dataset


def _part(day_mean: float, n: int, date: str, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    y = np.full((n, 4), day_mean) + 0.05 * rng.normal(size=(n, 4))
    state = rng.normal(size=(n, 2))
    return {
        "y": y,
        "state": state,
        "label": np.arange(n) % 4,
        "date": date,
        "n_windows": n,
        "bytes_read": n * 16,
    }


def _qualified_parts():
    return [
        _part(2.0, OOD_MIN_WINDOWS_PER_DAY, "2024-01-02", 1),
        _part(8.0, OOD_MIN_WINDOWS_PER_DAY, "2024-03-01", 2),
        _part(20.0, OOD_MIN_WINDOWS_PER_DAY, "2024-06-01", 3),
    ]


def _lay_runs(root, rows):
    lines = ["filename\tacq_time"]
    for base, date, with_events in rows:
        h5_dir = os.path.join(root, "Sherlock1", "derivatives", "serialised")
        ev_dir = os.path.join(root, "Sherlock1", "derivatives", "events")
        os.makedirs(h5_dir, exist_ok=True)
        os.makedirs(ev_dir, exist_ok=True)
        open(os.path.join(h5_dir, base + "_proc-0_meg.h5"), "wb").close()
        if with_events:
            with open(os.path.join(ev_dir, base + "_events.tsv"), "w", encoding="utf-8") as handle:
                handle.write("kind\ttimemeg\tsegment\n")
        lines.append(f"meg/{base}_meg.fif\t{date}T00:00:00")
    scan_dir = os.path.join(root, "Sherlock1", "sub-0", "ses-01")
    os.makedirs(scan_dir, exist_ok=True)
    with open(os.path.join(scan_dir, "sub-0_ses-01_scans.tsv"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


class DayRuleTests(unittest.TestCase):
    def test_excluded_day_and_missing_events_do_not_become_candidates(self):
        root = tempfile.mkdtemp()
        try:
            _lay_runs(root, [
                ("sub-0_ses-11_task-Sherlock1_run-2", EXCLUDED_SMOKE_DATE, True),
                ("sub-0_ses-02_task-Sherlock1_run-9", "2024-01-02", True),
                ("sub-0_ses-02_task-Sherlock1_run-1", "2024-01-02", False),
                ("sub-0_ses-03_task-Sherlock1_run-1", "2024-03-01", True),
                ("sub-0_ses-04_task-Sherlock1_run-1", "2024-05-01", True),
                ("sub-0_ses-05_task-Sherlock1_run-1", "2024-07-01", True),
                ("sub-0_ses-06_task-Sherlock1_run-1", "2024-08-01", True),
            ])
            chosen = discover_out_of_day_runs(root, max_days=3)
            dates = [row["date"] for row in chosen]
            self.assertEqual(dates, ["2024-01-02", "2024-03-01", "2024-05-01"])
            self.assertNotIn(EXCLUDED_SMOKE_DATE, dates)
            self.assertEqual(chosen[0]["base"], "sub-0_ses-02_task-Sherlock1_run-9")
            self.assertTrue(all(row["selection_rule"] == OOD_SELECTION_RULE for row in chosen))
            self.assertEqual(dates, sorted(dates))
        finally:
            shutil.rmtree(root)


class StreamAndLeakTests(unittest.TestCase):
    def test_hyperslab_stream_stops_at_the_shared_byte_budget(self):
        text = "kind\tsegment\ttimemeg\tword_onsets_s\tshuffle_seed\ttimechapter\n" + "".join(
            f"word\tthe\t{index}.0\t9\t1\t{index}\n" for index in range(4)
        )
        _rows, refused = parse_events(text)
        self.assertIn("timechapter", refused)
        dataset = Hyperslab(np.zeros((306, 4000), dtype=np.float64))
        specs = [{"date": "2024-01-02", "h5": "a.h5", "events": "a.tsv", "text": text, "dataset": dataset}]

        def opener(spec):
            parsed, seen = parse_events(spec["text"])
            return spec["text"], parsed, seen, spec["dataset"], lambda: None

        streamed = stream_out_of_day(specs, VOCAB, deadline=time.monotonic() + 30, opener=opener, max_bytes=20000, max_windows=10)
        self.assertTrue(streamed["read_budget_hit"])
        self.assertEqual(streamed["parts"], [])
        self.assertGreater(dataset.bytes_read, 0)
        self.assertLess(dataset.bytes_read, 32 * 1024 * 1024)
        self.assertTrue(all(stop - start == 250 for _, start, stop in dataset.calls))
        with self.assertRaises(Exception):
            dataset[:]

    def test_deadline_does_not_open_the_next_day(self):
        def opener(_spec):
            raise AssertionError("deadline passed and a day was still opened")

        streamed = stream_out_of_day(
            [{"date": "2024-01-02", "h5": "a", "events": "a"}],
            VOCAB,
            deadline=time.monotonic() - 1,
            opener=opener,
        )
        self.assertTrue(streamed["timed_out"])
        self.assertEqual(streamed["parts"], [])

    def test_held_out_day_does_not_enter_the_scaler_and_controls_stay(self):
        parts = _qualified_parts()
        fitted = evaluate_out_of_day(
            parts,
            regime="synthetic_dgp",
            timed_out=False,
            cgroup_live=True,
            selection_rule=OOD_SELECTION_RULE,
        )
        self.assertFalse(fitted["real_meg_scientific_validation"])
        self.assertEqual(fitted["claim_status"], "synthetic_evaluation_only")
        self.assertEqual(set(fitted["models"]), set(MODEL_NAMES))
        self.assertEqual(
            parameter_count("M2_state_variance", 4, 2, 2),
            fitted["models"]["M2_state_variance"]["n_params"],
        )
        self.assertEqual(
            fitted["models"]["M2_state_variance"]["n_params"],
            fitted["models"]["C_matched_random_state"]["n_params"],
        )
        y = np.concatenate([part["y"] for part in parts], axis=0)
        for fold in fitted["folds"]:
            held = fold["held_out_day"]
            train = y[held * OOD_MIN_WINDOWS_PER_DAY:(held + 1) * OOD_MIN_WINDOWS_PER_DAY]
            others = np.concatenate([
                part["y"] for index, part in enumerate(parts) if index != held
            ], axis=0)
            self.assertTrue(np.allclose(fold["scaler_mean_y"], others.mean(axis=0), atol=1e-8))
            self.assertGreater(abs(float(np.mean(fold["scaler_mean_y"])) - float(train.mean())), 1.0)
        report = fitted["uncertainty"]
        self.assertEqual(len(report["primary_contrast_by_day"]["per_day"]), 3)
        self.assertIsNotNone(report["primary_contrast_by_day"]["std"])
        self.assertIsNotNone(report["primary_contrast_by_day"]["se"])
        self.assertIsNone(report["best_day_index"])
        self.assertIsNone(report["selected_model"])
        self.assertFalse(report["day_selected_by_score"])
        self.assertFalse(report["confirmation_quarters_used"])
        self.assertNotIn("Q3", report)
        self.assertNotIn("Q4", report)


class ClaimTests(unittest.TestCase):
    def test_science_plumbing_timeout_and_forbidden_day_stay_distinct(self):
        powered = dict(
            regime="local_out_of_day_stream",
            n_days=OOD_MIN_DAYS,
            windows_per_day=[OOD_MIN_WINDOWS_PER_DAY] * OOD_MIN_DAYS,
            timed_out=False,
            cgroup_live=True,
            dates=["2024-01-02", "2024-03-01", "2024-06-01"],
            q3q4_used=False,
            selection_rule=OOD_SELECTION_RULE,
        )
        science = classify_out_of_day(**powered)
        self.assertFalse(science["real_meg_scientific_validation"])
        self.assertTrue(science["residual_protocol_completed"])
        self.assertFalse(science["endpoint_is_decoding"])
        self.assertEqual(science["claim_status"], "residual_proxy_completed")
        self.assertFalse(science["neural_inference_allowed"])

        synthetic = classify_out_of_day(**{**powered, "regime": "synthetic_dgp"})
        self.assertFalse(synthetic["real_meg_scientific_validation"])
        smoke = classify_out_of_day(**{**powered, "regime": "plumbing"})
        self.assertEqual(smoke["claim_status"], "plumbing")
        self.assertFalse(smoke["real_meg_scientific_validation"])

        timed = classify_out_of_day(**{**powered, "timed_out": True})
        self.assertEqual(timed["claim_status"], "timeout")
        self.assertFalse(timed["real_meg_scientific_validation"])

        short = classify_out_of_day(**{**powered, "n_days": 2, "windows_per_day": [48, 48]})
        self.assertEqual(short["claim_status"], "underpowered")
        weak = classify_out_of_day(**{**powered, "windows_per_day": [OOD_MIN_WINDOWS_PER_DAY - 1] * 3})
        self.assertEqual(weak["claim_status"], "underpowered")

        fixture = classify_out_of_day(**{**powered, "cgroup_live": False})
        self.assertEqual(fixture["claim_status"], "cgroup_not_live")
        forbidden = classify_out_of_day(**{**powered, "dates": ["2024-01-02", EXCLUDED_SMOKE_DATE, "2024-06-01"]})
        self.assertEqual(forbidden["claim_status"], "protocol_violation")
        quarters = classify_out_of_day(**{**powered, "q3q4_used": True})
        self.assertEqual(quarters["claim_status"], "protocol_violation")
        for row in (synthetic, smoke, timed, short, weak, fixture, forbidden, quarters):
            self.assertFalse(row["real_meg_scientific_validation"])

        live = evaluate_out_of_day(
            _qualified_parts(),
            regime="local_out_of_day_stream",
            timed_out=False,
            cgroup_live=True,
            selection_rule=OOD_SELECTION_RULE,
        )
        self.assertFalse(live["real_meg_scientific_validation"])
        self.assertTrue(live["residual_protocol_completed"])
        self.assertFalse(live["audit"]["endpoint_is_decoding"])
        self.assertFalse(live["next_rung"]["executed"])
        self.assertEqual(live["uncertainty"]["models"]["M0_fixed_gaussian"]["n_days"], 3)

        partial = evaluate_out_of_day(
            _qualified_parts(),
            regime="local_out_of_day_stream",
            timed_out=True,
            cgroup_live=True,
            selection_rule=OOD_SELECTION_RULE,
        )
        self.assertEqual(partial["claim_status"], "timeout")
        self.assertFalse(partial["fit_completed"])
        self.assertIsNone(partial["models"])

        smoke_batch = make_dataset("homo", n_days=1, n_per=40)
        smoke_batch["feature_set"] = FEATURE_SET
        smoke_result = smoke_one_series(smoke_batch)
        self.assertFalse(smoke_result["real_meg_scientific_validation"])
        self.assertFalse(smoke_result["cross_day"])


class RunnerGuardTests(unittest.TestCase):
    def test_out_of_day_cli_fails_closed_before_data(self):
        env = os.environ.copy()
        env["SDR_FORBID_LIVE_COMPUTE"] = "1"
        env["LIBRIBRAIN_DATA"] = "/no/such/libribrain"
        proc = subprocess.run(
            [os.environ.get("PYTHON", "python3"), "-m", "campaigns.sdr_uncertainty.runner", "out-of-day"],
            cwd=os.path.join(os.path.dirname(__file__), "..", ".."),
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("refusing", proc.stderr)
        self.assertNotIn("does not exist", proc.stderr)
        self.assertNotIn('"real_meg_scientific_validation": true', proc.stdout)

    def test_scope_script_keeps_the_caps_and_skips_the_unverified_lock(self):
        path = os.path.realpath(os.path.join(
            os.path.dirname(__file__), "..", "..", "scripts", "sdr_uncertainty_out_of_day.sh"
        ))
        with open(path, encoding="utf-8") as handle:
            text = handle.read()
        for needle in (
            "MemoryMax=4G",
            "MemorySwapMax=0",
            "CPUQuota=100%",
            "TasksMax=16",
            "RuntimeMaxSec=600",
            "OMP_NUM_THREADS=1",
            "out-of-day",
        ):
            self.assertIn(needle, text)
        for forbidden in ("acquire_lock.sh", "release_lock.sh", "MemoryMax=8G", "RuntimeMaxSec=1200", "kaggle", "arxiv"):
            self.assertNotIn(forbidden, text)
        subprocess.run(["bash", "-n", path], check=True)
        contract = os.path.join(os.path.dirname(__file__), "SCHEDULER_CONTRACT.md")
        with open(contract, encoding="utf-8") as handle:
            prose = handle.read()
        self.assertIn("acquire_lock.sh", prose)
        self.assertIn("does not call", prose)


if __name__ == "__main__":
    unittest.main()
