"""Chronological expansion past the frozen earliest-4 cache. Fixtures are not a result."""
from __future__ import annotations

import json
import os
import shutil
import tempfile
import unittest

import numpy as np

from campaigns.sdr_uncertainty.decode306 import evaluate_decode306, score_consumed_days, score_memory_ok
from campaigns.sdr_uncertainty.decode306_expand import (
    frozen_action,
    gate_from_counts,
    launch_expansion,
    select_later_runs,
)
from campaigns.sdr_uncertainty.decode306_resume import PREPROCESS_ID, SELECTION_RULE
from campaigns.sdr_uncertainty.protocol import (
    DECODE306_CACHE_VERSION,
    DECODE306_MIN_HELDOUT_DAYS,
    DECODE306_MIN_PER_CLASS,
    DECODE306_VERIFIED_FROZEN_DATES,
    DECODE306_VERIFIED_FROZEN_RUNS,
    EXCLUDED_SMOKE_DATE,
    N_SENSORS,
    WINDOW_SAMPLES,
)


def _write_json(path, payload):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)
        handle.write("\n")


def _frozen_cache(root, dates, runs, progress_index, n_batches, label=0, day=0):
    os.makedirs(root, exist_ok=True)
    _write_json(os.path.join(root, "manifest.json"), {
        "cache_version": DECODE306_CACHE_VERSION,
        "preprocess_id": PREPROCESS_ID,
        "selection_rule": SELECTION_RULE,
        "excluded_date": EXCLUDED_SMOKE_DATE,
        "dates": list(dates),
        "runs": runs,
    })
    _write_json(os.path.join(root, "progress.json"), {
        "run_index": progress_index,
        "word_index": 0,
        "last_kept_onset": None,
        "last_kept_run": None,
        "kept": n_batches,
        "skipped_overlap": 0,
        "in_vocab_considered": n_batches,
    })
    for index in range(n_batches):
        np.savez_compressed(
            os.path.join(root, f"batch_{index:04d}.npz"),
            features=np.zeros((1, 2, 2), np.float32),
            label=np.array([label], np.int16),
            day_index=np.array([day], np.int16),
            onset_s=np.array([0.0]),
            run_index=np.array([0], np.int16),
        )


def _verified_runs():
    dates = list(DECODE306_VERIFIED_FROZEN_DATES)
    return [{"date": dates[index % 4], "base": f"run-{index:02d}"} for index in range(DECODE306_VERIFIED_FROZEN_RUNS)]


class SelectionTests(unittest.TestCase):
    def test_later_dates_follow_acquisition_order_and_not_a_score(self):
        frozen = list(DECODE306_VERIFIED_FROZEN_DATES)
        discovered = [
            {"date": "2024-08-20", "base": "b", "score": 9},
            {"date": "2024-08-15", "base": "c", "score": 1},
            {"date": "2024-08-15", "base": "a", "score": 8},
            {"date": frozen[0], "base": "already", "score": 100},
        ]
        selected = select_later_runs(discovered, frozen)
        self.assertEqual([(row["date"], row["base"]) for row in selected], [
            ("2024-08-15", "a"),
            ("2024-08-15", "c"),
            ("2024-08-20", "b"),
        ])
        with self.assertRaises(RuntimeError):
            select_later_runs([{"date": EXCLUDED_SMOKE_DATE, "base": "x"}], frozen)
        with self.assertRaises(RuntimeError):
            select_later_runs([{"date": "2024-08-01", "base": "x"}], frozen)
        with self.assertRaises(RuntimeError):
            select_later_runs([{"date": "2024-07-01", "base": "x"}], frozen)

    def test_verified_cache_actions(self):
        root = tempfile.mkdtemp()
        try:
            self.assertEqual(frozen_action(root), "absent")
            cache = os.path.join(root, "frozen")
            _frozen_cache(cache, DECODE306_VERIFIED_FROZEN_DATES, _verified_runs(), 21, 7)
            self.assertEqual(frozen_action(cache), "expand")
            _write_json(os.path.join(cache, "progress.json"), {
                "run_index": 3,
                "word_index": 0,
                "last_kept_onset": None,
                "last_kept_run": None,
                "kept": 1,
                "skipped_overlap": 0,
                "in_vocab_considered": 1,
            })
            self.assertEqual(frozen_action(cache), "resume_frozen")
            _write_json(os.path.join(cache, "progress.json"), {
                "run_index": 21,
                "word_index": 0,
                "last_kept_onset": None,
                "last_kept_run": None,
                "kept": 7,
                "skipped_overlap": 0,
                "in_vocab_considered": 7,
            })
            os.remove(os.path.join(cache, "batch_0006.npz"))
            with self.assertRaises(RuntimeError):
                frozen_action(cache)
        finally:
            shutil.rmtree(root)


class GateTests(unittest.TestCase):
    def test_always_on_one_early_day_does_not_define_a_partial_metric(self):
        dates = list(DECODE306_VERIFIED_FROZEN_DATES) + ["2024-08-20"]
        counts = np.full((5, 50), DECODE306_MIN_PER_CLASS, dtype=np.int64)
        counts[:, 45] = 0
        counts[1, 45] = 5
        partial = gate_from_counts(counts, dates, [True, True, True, True, False])
        self.assertFalse(partial["label_gate_met"])
        self.assertFalse(partial["partial_class_accuracy_reported"])
        self.assertNotIn("frozen_eval", partial)
        august_7 = partial["rows"][1]
        self.assertEqual(august_7["date"], "2024-08-07")
        self.assertTrue(august_7["heldout_eligible"])
        self.assertIn("always", august_7["classes_absent_from_train"])
        self.assertFalse(august_7["fifty_class_metric_defined"])
        counts[4] = DECODE306_MIN_PER_CLASS
        full = gate_from_counts(counts, dates, [True] * 5)
        self.assertTrue(full["label_gate_met"])
        self.assertGreaterEqual(len(full["eligible_days"]), DECODE306_MIN_HELDOUT_DAYS)
        self.assertIn(1, full["eligible_days"])
        self.assertIn(4, full["eligible_days"])
        self.assertTrue(all(full["rows"][index]["train_gate"] for index in full["eligible_days"]))
        self.assertIn("2024-07-31", full["not_held_out_below_min"])

    def test_streaming_score_matches_the_in_memory_probe_and_skips_incomplete_days(self):
        rng = np.random.default_rng(0)
        per = DECODE306_MIN_PER_CLASS
        labels = np.tile(np.arange(50), 4 * per)
        features = rng.normal(scale=0.05, size=(len(labels), 2, 4))
        features[np.arange(len(labels)), 0, 0] += labels
        day = np.repeat(np.arange(4), 50 * per)
        dates = ["d0", "d1", "d2", "d3"]

        def factory():
            yield {"features": features, "label": labels, "day": day}

        streamed = score_consumed_days(factory, dates, [0, 1, 2, 3])
        stored = evaluate_decode306(features, labels, day, dates)
        self.assertTrue(streamed["fifty_class_metric_defined"])
        self.assertFalse(streamed["effect_claimed"])
        self.assertFalse(streamed["partial_class_accuracy_reported"])
        self.assertFalse(streamed["frozen_eval"]["effect_claimed"])
        for arm in streamed["frozen_eval"]["arms"]:
            np.testing.assert_allclose(
                streamed["frozen_eval"]["arms"][arm]["per_day"],
                stored["frozen_eval"]["arms"][arm]["per_day"],
                atol=1e-5,
            )
        short = labels.copy()
        short_day = day.copy()
        # Drop class 49 from day 0 only. Days 1 and 2 remain complete.
        drop = (day == 0) & (labels == 49)
        short = short[~drop]
        short_features = features[~drop]
        short_day = short_day[~drop]
        keep = short_day < 3
        short, short_features, short_day = short[keep], short_features[keep], short_day[keep]

        def short_factory():
            yield {"features": short_features, "label": short, "day": short_day}

        held = score_consumed_days(short_factory, ["d0", "d1", "d2"], [1, 2])
        self.assertTrue(held["fifty_class_metric_defined"])
        self.assertEqual(held["frozen_eval"]["heldout_dates"], ["d1", "d2"])
        self.assertEqual(len(held["frozen_eval"]["arms"]["uniform"]["per_day"]), 2)
        self.assertFalse(held["effect_claimed"])
        self.assertNotIn("d0", held["frozen_eval"]["heldout_dates"])
        blocked = score_consumed_days(short_factory, ["d0", "d1", "d2"], [1])
        self.assertFalse(blocked["fifty_class_metric_defined"])
        self.assertIsNone(blocked["frozen_eval"])
        self.assertFalse(blocked["effect_claimed"])
        text = json.dumps(blocked)
        self.assertNotIn("bacc_at_10", text)

    def test_large_gram_does_not_score_without_headroom(self):
        self.assertTrue(score_memory_ok(8, None))
        self.assertFalse(score_memory_ok(N_SENSORS * 50, None))
        self.assertFalse(score_memory_ok(N_SENSORS * 50, 2 * 1024 ** 3))
        self.assertTrue(score_memory_ok(N_SENSORS * 50, 4 * 1024 ** 3))


class LaunchTests(unittest.TestCase):
    def _paths(self):
        root = tempfile.mkdtemp()
        frozen = os.path.join(root, "decode306_cache")
        expand = os.path.join(root, "decode306_expand_cache")
        _frozen_cache(
            frozen,
            ["2024-01-02", "2024-01-03"],
            [{"date": "2024-01-02", "base": "a"}, {"date": "2024-01-03", "base": "b"}],
            2,
            1,
        )
        runs = [{
            "date": "2024-02-01",
            "base": "later-1",
            "h5": "later.h5",
            "events_sha256": "abc",
            "h5_nbytes": 4,
            "h5_shape": [306, 5000],
            "h5_dtype": "float32",
            "words": [[0.0, 0], [2.0, 1]],
            "fields_seen_not_used": ["timechapter"],
        }]
        return root, frozen, expand, runs

    def test_one_batch_resumes_and_leaves_the_frozen_cache_unchanged(self):
        root, frozen, expand, runs = self._paths()
        calls = []

        def read_raw(run_index, onset):
            calls.append((run_index, onset))
            return np.zeros((N_SENSORS, WINDOW_SAMPLES))

        try:
            with open(os.path.join(frozen, "batch_0000.npz"), "rb") as handle:
                before = handle.read()
            one = N_SENSORS * WINDOW_SAMPLES * 4
            first = launch_expansion(frozen, expand, runs, read_raw, batch_bytes=one, itemsize=4)
            self.assertEqual(first["next_action"], "rerun_this_command")
            self.assertEqual(first["claim_status"], "resume_required")
            self.assertFalse(first["effect_claimed"])
            self.assertFalse(first["fifty_class_metric_defined"])
            self.assertIsNone(first["frozen_eval"])
            self.assertFalse(first["real_meg_scientific_validation"])
            self.assertEqual(calls, [(0, 0.0)])
            with open(os.path.join(frozen, "batch_0000.npz"), "rb") as handle:
                self.assertEqual(handle.read(), before)
            self.assertTrue(first["frozen_provenance"]["cache_preserved"])
            blob = json.dumps(first)
            self.assertNotIn("bacc_at_10", blob)
            second = launch_expansion(frozen, expand, runs, read_raw, batch_bytes=one * 4, itemsize=4)
            self.assertEqual(calls, [(0, 0.0), (0, 2.0)])
            self.assertEqual(second["claim_status"], "dataset_limitation")
            self.assertEqual(second["next_action"], "stop")
            self.assertFalse(second["effect_claimed"])
            self.assertIsNone(second["frozen_eval"])
            with open(os.path.join(frozen, "batch_0000.npz"), "rb") as handle:
                self.assertEqual(handle.read(), before)
            frozen_batches = os.listdir(frozen)
            self.assertEqual([name for name in frozen_batches if name.startswith("batch_")], ["batch_0000.npz"])
            with open(os.path.join(frozen, "batch_0000.npz"), "ab") as handle:
                handle.write(b"\n")
            before_expand = sorted(name for name in os.listdir(expand) if name.startswith("batch_"))
            with self.assertRaises(RuntimeError):
                launch_expansion(frozen, expand, runs, read_raw, batch_bytes=one, itemsize=4)
            self.assertEqual(calls, [(0, 0.0), (0, 2.0)])
            after_expand = sorted(name for name in os.listdir(expand) if name.startswith("batch_"))
            self.assertEqual(before_expand, after_expand)
        finally:
            shutil.rmtree(root)

    def test_scheduler_contract_loops_the_same_command(self):
        root = os.path.join(os.path.dirname(__file__), "..", "..")
        with open(os.path.join(root, "campaigns", "sdr_uncertainty", "SCHEDULER_CONTRACT.md"), encoding="utf-8") as handle:
            contract = handle.read()
        with open(os.path.join(root, "scripts", "sdr_uncertainty_decode306.sh"), encoding="utf-8") as handle:
            script = handle.read()
        self.assertIn("rerun_this_command", contract)
        self.assertIn("scripts/sdr_uncertainty_decode306.sh", contract)
        self.assertIn("decode306_cache is not rewritten", script)
        self.assertNotIn("RuntimeMaxSec", script)
        self.assertNotIn("huggingface", script.lower())


if __name__ == "__main__":
    unittest.main()
