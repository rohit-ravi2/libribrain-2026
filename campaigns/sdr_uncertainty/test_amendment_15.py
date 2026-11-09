"""Amendment 1.5. Fixtures are not a measurement and not a claim."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
import unittest

import numpy as np

from campaigns.sdr_uncertainty.amendment_15 import (
    BACC_EQUIVALENCE_MARGIN,
    CANDIDATES_15,
    EVAL_EXTENSION_SUBJECTS,
    FIT_ROW_FLOOR,
    FIT_SUBJECTS,
    INNER_CHUNK_DEADLINE_SEC,
    PARTIAL_ARMS_15,
    PARTIAL_SCHEDULE_15,
    PRIMARY_METRIC,
    REDUCED_ROW_FLOOR,
    RUNTIME_CEILING_SEC,
    append_exposure,
    assert_timer_contract,
    bacc10_present,
    begin_chunk,
    choose_support,
    contrast_resolved,
    decision_from_contrasts,
    fit_equations,
    fit_reduced_state,
    initial_partial_arms_15,
    log_variance_15,
    matched_partial_steps_15,
    measurement_complete_15,
    metric_declaration,
    next_action_15,
    plan_recordings,
    power_report,
    read_exposure,
    rows_for_one_slope,
    timers_coherent,
    validation_stop_15,
)
from campaigns.sdr_uncertainty.strong_decoder_15_runner import (
    OUT_DIRNAME,
    _save_json,
    assert_output_dir,
    fit_directory,
    main,
    run_dry,
    run_timing_pilot,
    score_frozen_directory,
    score_heldout_directory,
    score_partial_directory,
)
from campaigns.sdr_uncertainty.strong_noise import (
    CANDIDATES,
    CHECKPOINT_BYTES,
    EVAL_SUBJECTS,
    N_CLASSES,
    VAL_SUBJECTS,
    ProtocolError,
    assemble_features,
    bacc10_from_logits,
    fit_noise,
    gaussian_nll,
    log_variance,
    missing_acquisition_date,
)


def _rows(base, subject, date, split, n=4, scale=1.0, quarter="Q3"):
    rows = []
    for index in range(n):
        rows.append({
            "base": base,
            "subject": subject,
            "date": date,
            "split": split,
            "quarter": quarter,
            "run_fraction": scale * (index + 1) / n,
            "artifact_rms": scale * (0.2 + index),
            "bandpower": scale * (1.0 + 0.1 * index),
            "group_windows": n,
            "residual": 0.4 + 0.05 * index,
        })
    return rows


def _write_forward(folder, base, subject, labels, timechapter, seed=0):
    os.makedirs(folder, exist_ok=True)
    rng = np.random.default_rng(seed)
    n = len(labels)
    path = os.path.join(folder, base + ".npz")
    np.savez_compressed(
        path,
        base=np.array(base),
        subject=np.int64(subject),
        label=np.asarray(labels, dtype=np.int64),
        logits=rng.normal(size=(n, N_CLASSES)).astype(np.float32),
        embedding=rng.normal(size=(n, 3)).astype(np.float32),
        run_fraction=np.linspace(0.1, 0.9, n),
        artifact_rms=np.linspace(0.2, 1.5, n),
        bandpower=np.linspace(0.3, 1.2, n),
        timechapter=np.asarray(timechapter, dtype=np.float64),
        checkpoint_sha=np.array("abc"),
        timing=np.array("zero"),
    )
    return path


def _q3_file(folder, base, subject, n_q3, seed):
    labels = np.zeros(n_q3 + 1, dtype=np.int64)
    timechapter = np.concatenate([np.full(n_q3, 60.0), np.array([99.0])])
    return _write_forward(folder, base, subject, labels, timechapter, seed=seed)


class Amendment15Design(unittest.TestCase):
    def test_power_floor_is_above_the_680_row_fit(self):
        self.assertEqual(rows_for_one_slope(), 349)
        self.assertEqual(FIT_ROW_FLOOR, 1745)
        self.assertEqual(fit_row_floor := FIT_ROW_FLOOR, 349 * 5)
        self.assertEqual(REDUCED_ROW_FLOOR, 698)
        report = power_report()
        self.assertFalse(report["observed_1_4_fit_meets_learned_floor"])
        self.assertFalse(report["observed_1_4_fit_meets_reduced_floor"])
        self.assertEqual(report["rows_observed_in_1_4"], 680)
        self.assertLess(680 / 5, report["rows_per_parameter_required"])
        del fit_row_floor

    def test_subject_lists_do_not_overlap_and_session_tokens_are_not_days(self):
        self.assertFalse(set(FIT_SUBJECTS) & set(EVAL_SUBJECTS))
        self.assertFalse(set(FIT_SUBJECTS) & set(VAL_SUBJECTS))
        self.assertFalse(set(FIT_SUBJECTS) & set(EVAL_EXTENSION_SUBJECTS))
        dates = {"sub-0_ses-11_task-Sherlock1_run-2": "2024-10-04"}
        bases = [
            "sub-2_ses-12_task-Sherlock1_run-1",
            "sub-0_ses-11_task-Sherlock1_run-2",
            "sub-13_ses-11_task-Sherlock1_run-1",
            "sub-3_ses-11_task-Sherlock1_run-1",
            "sub-1_ses-11_task-Sherlock1_run-1",
            "sub-25_ses-11_task-Sherlock1_run-1",
            "sub-0_ses-1_task-audio_run-1",
            "sub-0_ses-11_task-Sherlock1_run-9_holdout",
        ]
        plan = plan_recordings(bases, dates)
        pool = [row["base"] for row in plan["phases"]["noise_pool"]]
        self.assertEqual(pool, [
            "sub-0_ses-11_task-Sherlock1_run-2",
            "sub-2_ses-12_task-Sherlock1_run-1",
            "sub-13_ses-11_task-Sherlock1_run-1",
        ])
        self.assertIsNone(plan["phases"]["noise_pool"][1]["date"])
        self.assertEqual(plan["phases"]["noise_pool"][0]["date"], "2024-10-04")
        self.assertEqual(plan["phases"]["eval"][0]["subject"], 3)
        self.assertEqual(plan["phases"]["eval_extension"][0]["subject"], 25)
        self.assertEqual(plan["phases"]["selection"][0]["subject"], 1)
        self.assertEqual(plan["phases"]["train"][0]["base"], "sub-0_ses-1_task-audio_run-1")
        self.assertEqual(plan["refused_holdout"][0]["base"], "sub-0_ses-11_task-Sherlock1_run-9_holdout")
        self.assertFalse(any(row["date"] == "12" for row in plan["phases"]["noise_pool"]))
        self.assertTrue(all(row["session_ordinal_used_as_day"] is False for row in plan["phases"]["noise_pool"]))
        with self.assertRaises(ProtocolError) as caught:
            plan_recordings(["sub-0_ses-11_task-Sherlock1_run-2"], {"sub-0_ses-11_task-Sherlock1_run-2": "11"})
        self.assertIn("not YYYY-MM-DD", str(caught.exception))

    def test_primary_metric_is_defined_when_classes_are_absent(self):
        rng = np.random.default_rng(2)
        labels = np.array([cls for cls in range(N_CLASSES) if cls not in (29, 46) for _ in range(3)], dtype=np.int64)
        logits = rng.normal(size=(len(labels), N_CLASSES))
        present = bacc10_present(logits, labels, min_count=1)
        self.assertIsNotNone(present)
        self.assertIsNone(bacc10_from_logits(logits, labels))
        full_labels = np.array([cls for cls in range(50) for _ in range(2)], dtype=np.int64)
        full_logits = rng.normal(size=(len(full_labels), N_CLASSES))
        self.assertAlmostEqual(bacc10_present(full_logits, full_labels, min_count=1), bacc10_from_logits(full_logits, full_labels))
        declaration = metric_declaration(labels)
        self.assertEqual(declaration["primary_metric"], PRIMARY_METRIC)
        self.assertTrue(declaration["declared_before_scoring"])
        self.assertFalse(declaration["hit_rates_used_to_choose_support"])
        self.assertIn(29, declaration["classes_absent"])
        self.assertIn(46, declaration["classes_absent"])
        wide = choose_support({cls: 2 for cls in range(48)})
        self.assertFalse(wide["powered"])
        tight = choose_support({cls: 20000 for cls in range(20)})
        self.assertTrue(tight["powered"])
        self.assertLessEqual(tight["mdd"], BACC_EQUIVALENCE_MARGIN)

    def test_contrasts_resolve_by_exclusion_or_equivalence(self):
        self.assertTrue(contrast_resolved({"defined": True, "low": -0.001, "high": 0.001}, 0.02))
        self.assertTrue(contrast_resolved({"defined": True, "low": 0.05, "high": 0.2}, 0.02))
        self.assertFalse(contrast_resolved({"defined": True, "low": -0.5, "high": 0.4}, 0.02))
        self.assertFalse(contrast_resolved({"defined": False, "low": None, "high": None}, 0.02))
        resolved = {"contrasts": {"learned_state_minus_uniform": {"defined": True, "low": 0.0, "high": 0.0}}}
        decision = decision_from_contrasts(resolved, {"contrasts": {}})
        self.assertTrue(decision["decisive"])
        self.assertFalse(decision["effect_claimed"])
        self.assertFalse(decision["valid_scientific_claim"])

    def test_reduced_arm_drops_a_dead_column_and_is_lower_dimensional(self):
        rng = np.random.default_rng(0)
        n_train, n_val = 18, 200
        signal_train = rng.normal(size=n_train)
        signal_val = rng.normal(size=n_val)
        noise_train = rng.normal(size=(n_train, 4))
        noise_val = rng.normal(size=(n_val, 4))
        z_train = np.column_stack([signal_train * 2.5, noise_train, np.zeros(n_train)])
        z_val = np.column_stack([signal_val * 2.5, noise_val, np.zeros(n_val)])
        residual_train = np.sign(rng.normal(size=n_train)) * np.exp(0.7 * signal_train)
        residual_val = np.sign(rng.normal(size=n_val)) * np.exp(0.7 * signal_val)
        names = [
            "run_fraction", "artifact_rms", "bandpower", "day_rank", "day_distance", "log_train_window_count",
        ]
        reduced = fit_reduced_state(z_train, residual_train, names)
        learned = fit_noise(z_train, residual_train, "learned_state", feature_names=names, seed=0)
        self.assertNotIn("log_train_window_count", reduced["kept_feature_names"])
        self.assertEqual(len(reduced["slope"]), 1)
        self.assertEqual(reduced["ridge_penalty_per_row"], 1.0)
        self.assertEqual(reduced["learned_ridge_penalty_per_row"], 0.01)
        self.assertFalse(reduced["collapsed_to_constant"])
        reduced_nll = gaussian_nll(residual_val, reduced["mu"], log_variance_15(z_val, reduced))
        learned_nll = gaussian_nll(residual_val, learned["mu"], log_variance(z_val, learned))
        self.assertLess(reduced_nll, learned_nll)

    def test_undated_coefficient_rows_require_the_day_term_drop(self):
        dated = _rows("sub-0_ses-11_task-Sherlock1_run-2", 0, "2024-10-04", "train", n=6)
        other = _rows("sub-0_ses-12_task-Sherlock1_run-2", 0, "2024-10-04", "train", n=6, scale=1.2)
        undated = _rows("sub-2_ses-11_task-Sherlock1_run-1", 2, None, "train", n=6, scale=0.8)
        selection = _rows("sub-1_ses-11_task-Sherlock1_run-1", 1, None, "selection", n=4)
        with self.assertRaises(ProtocolError) as caught:
            assemble_features(dated + undated + selection)
        self.assertEqual(str(caught.exception), missing_acquisition_date("sub-2_ses-11_task-Sherlock1_run-1"))
        bundle = assemble_features(dated + other + undated + selection, allow_undated_train_when_days_inestimable=True)
        self.assertTrue(bundle["day_terms_inestimable"])
        self.assertEqual(bundle["state"]["train_days"], ["2024-10-04"])
        self.assertNotIn("day_rank", bundle["names"])
        self.assertFalse(bundle["state"]["dates_invented"])
        self.assertFalse(bundle["state"]["session_ordinal_used_as_day"])
        second = _rows("sub-0_ses-1_task-audio_run-1", 0, "2024-07-31", "train", n=4)
        with self.assertRaises(ProtocolError) as caught:
            assemble_features(dated + second + undated, allow_undated_train_when_days_inestimable=True)
        self.assertIn("refusing to invent day_rank", str(caught.exception))
        with self.assertRaises(ProtocolError):
            fit_equations(_rows("sub-0_a", 0, "2024-10-04", "train", n=4, quarter="Q4") + selection)
        with self.assertRaises(ProtocolError):
            fit_equations(_rows("sub-0_a", 0, "2024-10-04", "train", n=4, quarter="Q1") + selection)
        with self.assertRaises(ProtocolError):
            fit_equations(_rows("sub-3_ses-11_task-Sherlock1_run-1", 3, "2024-10-04", "train", n=4) + selection)

    def test_validation_stop_follows_the_controls(self):
        curve = [
            {"step": 200, "control_mean_ce": 1.0, "learned_state_ce": 1.0},
            {"step": 300, "control_mean_ce": 1.1, "learned_state_ce": 0.1},
            {"step": 400, "control_mean_ce": 1.2, "learned_state_ce": 0.05},
            {"step": 500, "control_mean_ce": 1.3, "learned_state_ce": 0.01},
        ]
        decision = validation_stop_15(curve, PARTIAL_SCHEDULE_15)
        self.assertTrue(decision["stop"])
        self.assertEqual(decision["best_step"], 200)
        self.assertEqual(decision["reason"], "validation_plateau")
        with self.assertRaises(ProtocolError):
            validation_stop_15([{"step": 1, "control_mean_ce": 1.0, "eval_ce": 1.0}], PARTIAL_SCHEDULE_15)

    def test_matched_arms_share_batches_and_freeze_precision(self):
        weight = np.zeros((3, N_CLASSES))
        bias = np.zeros(N_CLASSES)
        labels = np.array([0, 1, 2, 3] * 8, dtype=np.int64)
        embeddings = np.random.default_rng(0).normal(size=(len(labels), 3))
        frozen = {
            "constant": np.zeros(len(labels)),
            "random_covariate": np.full(len(labels), 2.0),
            "learned_state": np.linspace(-1, 1, len(labels)),
            "reduced_state": np.linspace(0.2, 0.4, len(labels)),
        }
        arms = initial_partial_arms_15(weight, bias)
        again = initial_partial_arms_15(weight, bias)
        matched_partial_steps_15(arms, embeddings, labels, frozen, steps=2, seed=4)
        matched_partial_steps_15(again, embeddings, labels, frozen, steps=2, seed=4)
        self.assertEqual(set(arms), set(PARTIAL_ARMS_15))
        self.assertTrue(np.allclose(arms["uniform"]["W"], again["uniform"]["W"]))
        self.assertFalse(np.allclose(arms["uniform"]["W"], arms["learned_state"]["W"]))
        self.assertFalse(np.allclose(arms["reduced_state"]["W"], arms["random_covariate"]["W"]))

    def test_timers_match_and_a_finite_scope_is_refused(self):
        self.assertIsNone(RUNTIME_CEILING_SEC)
        self.assertIsNone(INNER_CHUNK_DEADLINE_SEC)
        self.assertEqual(timers_coherent(None)["coherent"], True)
        self.assertEqual(timers_coherent("infinity")["inner_chunk_deadline_sec"], None)
        with self.assertRaises(ProtocolError) as caught:
            timers_coherent(600)
        self.assertIn("600", str(caught.exception))
        root = os.path.join(os.path.dirname(__file__), "..", "..")
        with open(os.path.join(root, "scripts", "sdr_uncertainty_strong_decoder_15.sh"), encoding="utf-8") as handle:
            script = handle.read()
        assert_timer_contract(script)
        self.assertIn("MemoryMax=4G", script)
        self.assertIn("MemorySwapMax=0", script)
        self.assertIn("CPUQuota=100%", script)
        self.assertIn("TasksMax=16", script)
        self.assertIn("sdr_lock_run.sh", script)
        self.assertIn("gpu_hide.sh", script)
        self.assertIn("libribrain-strong-decoder-noise", script)
        self.assertNotIn("RuntimeMaxSec", script)
        self.assertNotIn("while ", script)
        self.assertIn("strong_decoder_15_runner", script)
        subprocess.run(["bash", "-n", os.path.join(root, "scripts", "sdr_uncertainty_strong_decoder_15.sh")], check=True)
        self.assertTrue(measurement_complete_15(
            checkpoint_bytes=CHECKPOINT_BYTES, fit_rows=FIT_ROW_FLOOR, primary_defined=True, lock_held=True,
        ))
        self.assertFalse(measurement_complete_15(
            checkpoint_bytes=CHECKPOINT_BYTES, fit_rows=680, primary_defined=True, lock_held=True,
        ))
        self.assertFalse(measurement_complete_15(
            checkpoint_bytes=CHECKPOINT_BYTES, fit_rows=FIT_ROW_FLOOR, primary_defined=True, lock_held=True, fixture=True,
        ))

    def test_next_action_streams_one_recording_and_stops_below_the_floor(self):
        bases = [
            "sub-0_ses-11_task-Sherlock1_run-2",
            "sub-2_ses-11_task-Sherlock1_run-1",
            "sub-1_ses-11_task-Sherlock1_run-1",
            "sub-3_ses-11_task-Sherlock1_run-1",
            "sub-25_ses-11_task-Sherlock1_run-1",
            "sub-0_ses-1_task-audio_run-1",
        ]
        plan = plan_recordings(bases, {"sub-0_ses-11_task-Sherlock1_run-2": "2024-10-04"})
        common = dict(partial_steps=0, partial_stopped=False, score_frozen=False, score_partial=False, score_heldout=False)
        first = next_action_15(plan, done_bases=[], noise_fit=False, fit_rows=0, **common)
        self.assertEqual(first["stage"], "forward")
        self.assertEqual(first["base"], "sub-0_ses-11_task-Sherlock1_run-2")
        self.assertEqual(first["recordings_this_launch"], 1)
        pool_done = ["sub-0_ses-11_task-Sherlock1_run-2", "sub-2_ses-11_task-Sherlock1_run-1"]
        selection = next_action_15(plan, done_bases=pool_done, noise_fit=False, fit_rows=FIT_ROW_FLOOR, **common)
        self.assertEqual(selection["base"], "sub-1_ses-11_task-Sherlock1_run-1")
        ready = pool_done + ["sub-1_ses-11_task-Sherlock1_run-1"]
        fitted = next_action_15(plan, done_bases=ready, noise_fit=False, fit_rows=FIT_ROW_FLOOR, **common)
        self.assertEqual(fitted["stage"], "noise-fit")
        short = next_action_15(plan, done_bases=ready, noise_fit=False, fit_rows=680, **common)
        self.assertEqual(short["reason"], "fit_pool_exhausted_below_power_floor")
        self.assertFalse(short["measurement_complete"])
        empty = plan_recordings([])
        stopped = next_action_15(empty, done_bases=[], noise_fit=False, fit_rows=0, **common)
        self.assertEqual(stopped["reason"], "fit_pool_exhausted_below_power_floor")
        after_fit = next_action_15(plan, done_bases=ready, noise_fit=True, fit_rows=FIT_ROW_FLOOR, **common)
        self.assertEqual(after_fit["phase"], "eval")
        scored = ready + ["sub-3_ses-11_task-Sherlock1_run-1", "sub-25_ses-11_task-Sherlock1_run-1"]
        frozen = next_action_15(plan, done_bases=scored, noise_fit=True, fit_rows=FIT_ROW_FLOOR, **common)
        self.assertEqual(frozen["stage"], "score-frozen")
        training = next_action_15(
            plan, done_bases=scored, noise_fit=True, fit_rows=FIT_ROW_FLOOR,
            partial_steps=0, partial_stopped=False, score_frozen=True, score_partial=False, score_heldout=False,
        )
        self.assertEqual(training["base"], "sub-0_ses-1_task-audio_run-1")
        partial = next_action_15(
            plan, done_bases=scored + ["sub-0_ses-1_task-audio_run-1"], noise_fit=True, fit_rows=FIT_ROW_FLOOR,
            partial_steps=0, partial_stopped=False, score_frozen=True, score_partial=False, score_heldout=False,
        )
        self.assertEqual(partial["stage"], "partial-train")
        held = next_action_15(
            plan, done_bases=scored + ["sub-0_ses-1_task-audio_run-1"], noise_fit=True, fit_rows=FIT_ROW_FLOOR,
            partial_steps=2000, partial_stopped=True, score_frozen=True, score_partial=True, score_heldout=False,
        )
        self.assertEqual(held["stage"], "score-heldout-q4")
        done = next_action_15(
            plan, done_bases=scored + ["sub-0_ses-1_task-audio_run-1"], noise_fit=True, fit_rows=FIT_ROW_FLOOR,
            partial_steps=2000, partial_stopped=True, score_frozen=True, score_partial=True, score_heldout=True,
        )
        self.assertEqual(done["next_action"], "stop")
        self.assertEqual(done["reason"], "stages_complete")

    def test_failure_marker_retries_and_exposure_log_is_append_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            begin_chunk(tmp, "sub-2_ses-11_task-Sherlock1_run-1", "noise_pool")
            folder = os.path.join(tmp, "forward", "noise_pool")
            os.makedirs(folder, exist_ok=True)
            temporary = os.path.join(folder, ".sub-2_ses-11_task-Sherlock1_run-1.writing.npz")
            with open(temporary, "wb"):
                pass
            begin_chunk(tmp, "sub-0_ses-11_task-Sherlock1_run-2", "noise_pool")
            roles = [row["role"] for row in read_exposure(tmp)]
            self.assertIn("failure_detected", roles)
            self.assertFalse(os.path.exists(temporary))
            self.assertFalse(os.path.exists(os.path.join(folder, "sub-2_ses-11_task-Sherlock1_run-1.npz")))
            with open(os.path.join(tmp, "exposure_log.jsonl"), "rb") as handle:
                before = handle.read()
            append_exposure(tmp, {"role": "forward", "base": "sub-0_ses-11_task-Sherlock1_run-2", "phase": "noise_pool"})
            with open(os.path.join(tmp, "exposure_log.jsonl"), "rb") as handle:
                after = handle.read()
            self.assertTrue(after.startswith(before))

    def test_directory_fit_keeps_held_out_evidence_separate(self):
        with tempfile.TemporaryDirectory() as tmp:
            pool = os.path.join(tmp, "forward", "noise_pool")
            _q3_file(pool, "sub-0_ses-11_task-Sherlock1_run-2", 0, 900, seed=1)
            _q3_file(pool, "sub-2_ses-11_task-Sherlock1_run-1", 2, 900, seed=2)
            _write_forward(
                os.path.join(tmp, "forward", "selection"),
                "sub-1_ses-11_task-Sherlock1_run-1",
                1,
                np.ones(20, dtype=np.int64),
                np.linspace(10, 20, 20),
                seed=3,
            )
            dates = {"sub-0_ses-11_task-Sherlock1_run-2": "2024-10-04"}
            os.makedirs(os.path.join(tmp, "forward", "eval"), exist_ok=True)
            _write_forward(
                os.path.join(tmp, "forward", "eval"),
                "sub-3_ses-11_task-Sherlock1_run-1",
                3,
                np.zeros(4, dtype=np.int64),
                np.full(4, 60.0),
                seed=4,
            )
            with self.assertRaises(ProtocolError) as caught:
                fit_directory(tmp, dates)
            self.assertIn("eval", str(caught.exception))
            os.remove(os.path.join(tmp, "forward", "eval", "sub-3_ses-11_task-Sherlock1_run-1.npz"))
            payload = fit_directory(tmp, dates)
            self.assertGreaterEqual(payload["fit_rows"], FIT_ROW_FLOOR)
            self.assertEqual(payload["fit_rows"], 1800)
            self.assertTrue(payload["day_terms_inestimable"])
            self.assertEqual(payload["state"]["train_days"], ["2024-10-04"])
            self.assertNotIn("log_train_window_count", payload["reduced_state"]["kept_feature_names"])
            self.assertIn("reduced_state", payload["candidates"])
            self.assertEqual(set(payload["val_nll"]), set(CANDIDATES_15))
            self.assertFalse(payload["dates_invented"])
            self.assertFalse(payload["session_ordinal_used_as_day"])
            self.assertTrue(payload["q4_not_used_for_fit_or_selection"])
            self.assertFalse(payload["eval_files_read"])
            exposed = {row["role"] for row in read_exposure(tmp)}
            self.assertIn("coefficient_fit", exposed)
            self.assertIn("validation_selection", exposed)
            self.assertNotIn("score_frozen", exposed)
            _save_json(os.path.join(tmp, "noise", "fit.json"), payload)
            with self.assertRaises(ProtocolError):
                _save_json(os.path.join(tmp, "noise", "fit.json"), payload)
            labels = np.array([cls for cls in range(20) for _ in range(2)], dtype=np.int64)
            for base, subject, seed in (
                ("sub-3_ses-11_task-Sherlock1_run-1", 3, 5),
                ("sub-6_ses-12_task-Sherlock1_run-1", 6, 6),
            ):
                _write_forward(os.path.join(tmp, "forward", "eval"), base, subject, labels, np.full(len(labels), 40.0), seed=seed)
            frozen = score_frozen_directory(tmp, dates)
            self.assertIsNotNone(frozen["primary_bacc10_present"])
            self.assertTrue(frozen["primary_defined"])
            self.assertIsNone(frozen["primary_bacc10_50"])
            self.assertFalse(frozen["fifty_class_metric_defined"])
            self.assertTrue(frozen["frozen_primary_does_not_differ_by_arm"])
            self.assertEqual(frozen["primary_metric"], PRIMARY_METRIC)
            self.assertTrue(frozen["nll_contrasts"]["defined"])
            with open(os.path.join(tmp, "metric_declaration.json"), encoding="utf-8") as handle:
                declaration = json.load(handle)
            self.assertFalse(declaration["hit_rates_used_to_choose_support"])
            _save_json(os.path.join(tmp, "score_frozen.json"), frozen)
            weight = np.zeros((3, N_CLASSES))
            bias = np.zeros(N_CLASSES)
            os.makedirs(os.path.join(tmp, "partial"), exist_ok=True)
            best = {f"W_{name}": weight for name in PARTIAL_ARMS_15}
            best.update({f"b_{name}": bias for name in PARTIAL_ARMS_15})
            best["steps"] = np.int64(200)
            best["schedule"] = np.array(PARTIAL_SCHEDULE_15["name"])
            np.savez_compressed(os.path.join(tmp, "partial", "best_val.npz"), **best)
            state = {"steps": np.int64(200), "stopped": np.int64(1), "schedule": np.array(PARTIAL_SCHEDULE_15["name"])}
            np.savez_compressed(os.path.join(tmp, "partial", "state.npz"), **state)
            partial = score_partial_directory(tmp, dates)
            self.assertIsInstance(partial["learned_minus_uniform"], float)
            self.assertEqual(set(partial["primary_bacc10_present"]), set(PARTIAL_ARMS_15))
            self.assertFalse(partial["fifty_class_metric_defined"])
            self.assertFalse(partial["effect_claimed"])
            self.assertIn("learned_state_minus_uniform", partial["bacc_contrasts"]["contrasts"])
            self.assertIn("learned_state_minus_constant", partial["bacc_contrasts"]["contrasts"])
            self.assertIn("reduced_state_minus_random_covariate", partial["bacc_contrasts"]["contrasts"])
            heldout = score_heldout_directory(tmp, dates)
            self.assertGreater(heldout["heldout_rows"], 0)
            self.assertFalse(heldout["entered_coefficients"])
            self.assertFalse(heldout["entered_selection"])
            self.assertFalse(heldout["used_for_selection"])

    def test_dry_run_and_timing_pilot_do_not_read_meg_or_rewrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = os.environ.copy()
            env["LIBRIBRAIN_SDR_OUT"] = os.path.join(tmp, "sdr_out")
            env["LIBRIBRAIN_CACHE"] = os.path.join(tmp, "cache")
            env["LIBRIBRAIN_DATA"] = os.path.join(tmp, "data")
            old = {key: os.environ.get(key) for key in ("LIBRIBRAIN_SDR_OUT", "LIBRIBRAIN_CACHE", "LIBRIBRAIN_DATA")}
            os.environ.update(env)
            try:
                first = run_dry(os.getcwd())
                path = os.path.join(env["LIBRIBRAIN_SDR_OUT"], OUT_DIRNAME, "pilot", "dry_run.json")
                with open(path, "rb") as handle:
                    digest = hashlib.sha256(handle.read()).hexdigest()
                second = run_dry(os.getcwd())
                self.assertTrue(second["already_written"])
                self.assertFalse(second["rewritten"])
                with open(path, "rb") as handle:
                    self.assertEqual(hashlib.sha256(handle.read()).hexdigest(), digest)
                self.assertFalse(first["meg_read"])
                self.assertEqual(first["power"]["learned_state_row_floor"], 1745)
                self.assertIsNone(first["runtime_ceiling_sec"])
                self.assertIsNone(first["inner_chunk_deadline_sec"])
                pilot = run_timing_pilot(os.getcwd())
                self.assertTrue(pilot["synthetic"])
                self.assertFalse(pilot["meg_read"])
                self.assertFalse(pilot["measurement_complete"])
                self.assertNotIn("log_train_window_count", pilot["reduced_kept_feature_names"])
                self.assertGreaterEqual(pilot["elapsed_sec"], 0.0)
                status = subprocess.run(
                    [os.environ.get("PYTHON", "python3"), "-m", "campaigns.sdr_uncertainty.strong_decoder_15_runner", "dry-run"],
                    cwd=os.path.join(os.path.dirname(__file__), "..", ".."),
                    env={**env, "PYTHONPATH": os.path.join(os.path.dirname(__file__), "..", "..")},
                    text=True, capture_output=True, check=False,
                )
                self.assertEqual(status.returncode, 0, status.stderr)
                forbidden = dict(env)
                forbidden["SDR_FORBID_LIVE_COMPUTE"] = "1"
                blocked = subprocess.run(
                    [os.environ.get("PYTHON", "python3"), "-m", "campaigns.sdr_uncertainty.strong_decoder_15_runner", "next"],
                    cwd=os.path.join(os.path.dirname(__file__), "..", ".."),
                    env={**forbidden, "PYTHONPATH": os.path.join(os.path.dirname(__file__), "..", "..")},
                    text=True, capture_output=True, check=False,
                )
                self.assertEqual(blocked.returncode, 2)
                self.assertIn("refusing", blocked.stderr)
                with self.assertRaises(ProtocolError):
                    assert_output_dir(os.path.join(tmp, "holdout", OUT_DIRNAME), os.getcwd())
                with self.assertRaises(ProtocolError):
                    assert_output_dir(os.path.join(tmp, "strong_decoder_1_4"), os.getcwd())
            finally:
                for key, value in old.items():
                    if value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = value
        self.assertEqual(CANDIDATES, ("uniform", "constant", "random_covariate", "learned_state", "error_scaling"))
        self.assertIn("reduced_state", CANDIDATES_15)

    def test_live_next_requires_the_lock(self):
        env = os.environ.copy()
        env.pop("SDR_STRONG_LOCK", None)
        env["SDR_FORBID_LIVE_COMPUTE"] = "0"
        proc = subprocess.run(
            [os.environ.get("PYTHON", "python3"), "-m", "campaigns.sdr_uncertainty.strong_decoder_15_runner", "next"],
            cwd=os.path.join(os.path.dirname(__file__), "..", ".."),
            env=env, text=True, capture_output=True, check=False,
        )
        self.assertEqual(proc.returncode, 2)
        self.assertIn("scheduler self-lock", proc.stderr)
        self.assertEqual(main(["manifest"]), 0)


if __name__ == "__main__":
    unittest.main()
