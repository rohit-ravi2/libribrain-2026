"""Resumable 306-channel cache. Fixtures are not a decoding result."""
from __future__ import annotations

import os
import shutil
import tempfile
import unittest

import numpy as np

from campaigns.sdr_uncertainty.decode306_resume import resume_batch


def _runs():
    return [{
        "date": "2024-01-02",
        "base": "run-1",
        "h5": "a.h5",
        "events_sha256": "abc",
        "h5_nbytes": 10,
        "h5_shape": [306, 5000],
        "h5_dtype": "float32",
        "words": [[0.0, 0], [0.2, 1], [1.0, 2]],
        "fields_seen_not_used": ["timechapter"],
    }, {
        "date": "2024-01-02",
        "base": "run-2",
        "h5": "b.h5",
        "events_sha256": "def",
        "h5_nbytes": 10,
        "h5_shape": [306, 5000],
        "h5_dtype": "float32",
        "words": [[0.0, 3]],
        "fields_seen_not_used": [],
    }]


class ResumeTests(unittest.TestCase):
    def test_nonoverlap_resume_does_not_reread_and_stops_when_exhausted(self):
        root = tempfile.mkdtemp()
        calls = []

        def read_raw(run_index, onset):
            calls.append((run_index, onset))
            return np.zeros((306, 250))

        try:
            one = 306 * 250 * 4
            first = resume_batch(root, _runs(), read_raw, batch_bytes=one, itemsize=4)
            self.assertEqual(first["claim_status"], "resume_required")
            self.assertEqual(first["next_action"], "rerun_this_command")
            self.assertEqual(calls, [(0, 0.0)])
            self.assertEqual(first["skipped_overlap"], 1)
            self.assertIsNone(first.get("frozen_eval"))
            second = resume_batch(root, _runs(), read_raw, batch_bytes=one * 10, itemsize=4)
            self.assertEqual(calls, [(0, 0.0), (0, 1.0), (1, 0.0)])
            self.assertEqual(second["claim_status"], "dataset_limitation")
            self.assertEqual(second["next_action"], "stop")
            self.assertFalse(second["effect_claimed"])
            self.assertIn("the", second["per_date"][0]["classes_below_min"])
            before = len(calls)
            third = resume_batch(root, _runs(), read_raw, batch_bytes=one * 10, itemsize=4)
            self.assertEqual(len(calls), before)
            self.assertEqual(third["claim_status"], "dataset_limitation")
            changed = _runs()
            changed[0]["events_sha256"] = "nope"
            with self.assertRaises(RuntimeError):
                resume_batch(root, changed, read_raw, batch_bytes=one, itemsize=4)
        finally:
            shutil.rmtree(root)

    def test_a_class_seen_on_only_one_day_has_train_min_zero(self):
        root = tempfile.mkdtemp()
        runs = [{
            "date": "2024-01-02",
            "base": "run-1",
            "h5": "a.h5",
            "events_sha256": "aaa",
            "h5_nbytes": 1,
            "h5_shape": [306, 5000],
            "h5_dtype": "float32",
            "words": [[0.0, 0], [1.0, 0], [2.0, 0]],
            "fields_seen_not_used": [],
        }, {
            "date": "2024-03-01",
            "base": "run-1",
            "h5": "b.h5",
            "events_sha256": "bbb",
            "h5_nbytes": 1,
            "h5_shape": [306, 5000],
            "h5_dtype": "float32",
            "words": [[0.0, 1], [1.0, 1], [2.0, 1]],
            "fields_seen_not_used": [],
        }]

        def read_raw(_index, _onset):
            return np.zeros((306, 250))

        try:
            report = resume_batch(root, runs, read_raw, batch_bytes=306 * 250 * 4 * 20, itemsize=4)
            self.assertEqual(report["claim_status"], "dataset_limitation")
            by_date = {row["date"]: row for row in report["heldout_coverage"]}
            self.assertEqual(by_date["2024-01-02"]["train_min"], 0)
            self.assertIn("is", by_date["2024-01-02"]["classes_absent_from_train"])
            self.assertNotIn("frozen_eval", report)
        finally:
            shutil.rmtree(root)


if __name__ == "__main__":
    unittest.main()
