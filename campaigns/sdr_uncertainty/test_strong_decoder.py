"""Strong-decoder amendment 1.4. Fixtures are not a training result and not a claim."""
from __future__ import annotations

import glob
import hashlib
import json
import os
import stat
import subprocess
import tempfile
import textwrap
import time
import unittest
import unittest.mock

import numpy as np

from campaigns.sdr_uncertainty.strong_decoder import (
    attest_checkpoint_dict,
    claim_from_decoder,
    is_strong_decoder,
    load_strong_decoder,
)
from campaigns.sdr_uncertainty.strong_decoder_runner import (
    _candidate_matrix,
    _merge_report,
    _rows_from_forward,
    _write_forward,
    assert_output_dir,
    fit_noise_directory,
    honesty,
    merge_honesty_sources,
    partial_train_directory,
    read_dates,
    run_next,
    run_timing,
    score_frozen_directory,
    score_partial_directory,
)
from campaigns.sdr_uncertainty.strong_forward import forward_recording, recording_covariates, slice_window
from campaigns.sdr_uncertainty.strong_noise import (
    CANDIDATES,
    CHECKPOINT_BYTES,
    DAY_TERMS,
    PARTIAL_ARMS,
    REJECTED_CANDIDATES,
    ProtocolError,
    arm_logits,
    assemble_features,
    assign_split,
    bacc10_50,
    bacc10_from_logits,
    decoder_residual,
    dlogits_ce,
    fit_noise,
    frozen_trunk_role,
    gaussian_nll,
    initial_partial_arms,
    log_variance,
    matched_partial_steps,
    SUBSTANTIVE_SCHEDULE,
    measurement_claim,
    measurement_status,
    missing_acquisition_date,
    next_sessions,
    per_row_ce,
    row_temperature,
    select_on_validation,
    session_bootstrap,
    step_head,
    temperature_changes_top10,
    timing_inputs,
    transform_with_state,
    vocab_view,
    weighted_nll,
)
from campaigns.sdr_uncertainty.strong_plan import classify_base, next_action, plan_order
from campaigns.sdr_uncertainty.scheduler_lock import chosen_ttl, inspect_text


def _fake_ckpt():
    def weight(out, inn):
        return np.zeros((out, inn))

    return {
        "enc": {"w": np.zeros((2000, 512))},
        "ctx": {"w": np.zeros((1000, 256))},
        "head_ce": {"weight": weight(50, 256)},
        "head_t5": {"weight": weight(1024, 256)},
        "logit_s": np.zeros(1),
        "A": np.zeros((1, 306, 306)),
        "subs": [0],
        "log_prior": np.zeros(50),
        "args": {"d": 128, "D": 256, "maxw": 12, "t5": 1.0, "ctx": 1, "rel": 0, "ctx_layers": 4},
    }


def _rows(base, subject, date, split, n=4, scale=1.0):
    rows = []
    for index in range(n):
        rows.append({
            "base": base,
            "subject": subject,
            "date": date,
            "split": split,
            "run_fraction": (index + 1) / (n + 1),
            "artifact_rms": scale * (1.0 + 0.1 * index),
            "bandpower": 0.2 + 0.01 * index,
            "group_windows": n,
        })
    return rows


class SplitAndNoiseTests(unittest.TestCase):
    def test_split_is_chronological_and_not_a_frozen_holdout(self):
        sessions = [{"date": f"2024-01-{day:02d}", "base": "run"} for day in range(2, 22)]
        sessions.append({"date": "2024-10-04", "base": "sherlock"})
        sessions.sort(key=lambda row: row["date"])
        out = assign_split(sessions)
        self.assertNotIn("2024-10-04", [row["date"] for row in out["test"]])
        self.assertLess(out["train"][-1]["date"], out["val"][0]["date"])
        self.assertLess(out["val"][-1]["date"], out["test"][0]["date"])
        self.assertFalse(out["eval_subjects_are_pristine_holdout"])
        self.assertFalse(out["valid_with_frozen_deep_c"])
        self.assertIn("other", out["frozen_trunk_fact"])
        self.assertEqual(out["transfer_label"], "historically_inspected_transfer")
        self.assertEqual(out["pristine_confirmatory_subjects"], [])
        subject0 = frozen_trunk_role(0, shared=False)
        self.assertFalse(subject0["frozen_holdout"])
        self.assertEqual(subject0["role"], "train_subject0_not_a_frozen_holdout")
        self.assertEqual(frozen_trunk_role(3, shared=True)["role"], "exploratory_untrained")
        self.assertFalse(frozen_trunk_role(3, shared=True)["pristine_confirmatory"])
        self.assertFalse(frozen_trunk_role(2, shared=True)["frozen_holdout"])

    def test_true_word_frequency_is_rejected_and_objectives_match(self):
        with self.assertRaises(ProtocolError):
            from campaigns.sdr_uncertainty.strong_noise import assert_inference_features
            assert_inference_features(["word_frequency"])
        from campaigns.sdr_uncertainty.strong_noise import assert_inference_features
        assert_inference_features(["artifact_rms", "bandpower", "day_rank", "run_fraction"])
        self.assertIn("true_label_word_frequency", REJECTED_CANDIDATES)
        rng = np.random.default_rng(0)
        z = rng.normal(size=(40, 4))
        energy = np.exp(0.2 * z[:, 0] + rng.normal(scale=0.05, size=40))
        first = fit_noise(z, energy, "learned_state")
        second = fit_noise(z, energy, "learned_state")
        self.assertTrue(np.allclose(first["slope"], second["slope"]))
        self.assertEqual(len(fit_noise(z, energy, "constant")["slope"]), 4)
        self.assertTrue(np.allclose(fit_noise(z, energy, "constant")["slope"], 0))
        random_a = fit_noise(z, energy, "random_covariate", seed=3)
        random_b = fit_noise(z, energy[::-1], "random_covariate", seed=3)
        self.assertFalse(np.allclose(random_a["slope"], random_b["slope"]))
        val = {name: weighted_nll(energy, z, fit_noise(z, energy, name, seed=1)) for name in CANDIDATES}
        chosen = select_on_validation(val)
        self.assertEqual(chosen, select_on_validation(dict(val)))
        self.assertIn(chosen, CANDIDATES)
        uniform = fit_noise(z, energy, "uniform")
        constant = fit_noise(z, energy, "constant")
        nll_u = weighted_nll(energy, z, uniform)
        nll_c = weighted_nll(energy, z, constant)
        self.assertAlmostEqual(nll_u, gaussian_nll(energy, uniform["mu"], np.zeros(len(energy))))
        self.assertAlmostEqual(nll_c, gaussian_nll(energy, constant["mu"], np.full(len(energy), constant["intercept"])))
        self.assertNotAlmostEqual(nll_u, float(np.mean(energy)))
        self.assertNotAlmostEqual(nll_u, nll_c)
        self.assertEqual(uniform["objective"], "gaussian_nll")
        self.assertEqual(constant["objective"], "gaussian_nll")

    def test_unseen_day_is_clipped_to_the_training_range(self):
        rows = []
        rows += _rows("sub-0_a", 0, "2024-02-01", "train")
        rows += _rows("sub-0_b", 0, "2024-03-01", "train", scale=1.4)
        rows += _rows("sub-0_c", 0, "2024-04-01", "train", scale=0.7)
        rows += _rows("sub-1_ses-11_task-Sherlock1_run-1", 1, "2030-01-01", "selection", scale=9.0)
        bundle = assemble_features(rows)
        again = transform_with_state(rows, bundle["state"])
        self.assertTrue(np.allclose(bundle["z"], again))
        train = bundle["train_index"]
        far = ~train
        self.assertTrue(np.allclose(bundle["z"][far, 0], bundle["z"][train, 0].max()))
        leaked = _rows("sub-3_ses-11_task-Sherlock1_run-1", 3, "2024-02-01", "train")
        with self.assertRaises(ProtocolError):
            assemble_features(rows[:4] + leaked)
        self.assertFalse(bundle["day_terms_inestimable"])
        self.assertEqual(bundle["day_terms_dropped"], [])
        self.assertEqual(bundle["state"]["day_terms_inestimable"], False)
        self.assertEqual(bundle["state"]["day_terms_dropped"], [])
        self.assertEqual(bundle["n_train_days"], 3)
        self.assertEqual(bundle["names"][0], "day_rank")
        self.assertIn("day_distance", bundle["names"])
        self.assertEqual(len(bundle["names"]), 6)

    def test_undated_broad_rows_drop_day_terms_and_subject0_fails_closed(self):
        rows = []
        rows += _rows("sub-0_a", 0, "2024-02-01", "train")
        rows += _rows("sub-0_b", 0, "2024-03-01", "train", scale=1.4)
        rows += _rows("sub-1_ses-11_task-Sherlock1_run-1", 1, None, "selection")
        rows += _rows("sub-5_ses-11_task-Sherlock1_run-1", 5, None, "selection")
        rows += _rows("sub-10_ses-11_task-Sherlock1_run-1", 10, "2024-09-01", "selection")
        bundle = assemble_features(rows)
        names = bundle["names"]
        day_idx = [names.index(name) for name in DAY_TERMS]
        train = bundle["train_index"]
        undated = np.array([row["date"] is None for row in rows])
        self.assertTrue(np.any(undated))
        self.assertTrue(np.all(bundle["z"][undated][:, day_idx] == 0.0))
        self.assertGreater(float(bundle["z"][train, day_idx[0]].max()), 0.0)
        self.assertFalse(np.allclose(bundle["z"][undated, names.index("run_fraction")], 0.0))
        dated_selection = ~train & ~undated
        self.assertTrue(np.allclose(bundle["z"][dated_selection, day_idx[0]], bundle["z"][train, day_idx[0]].max()))
        self.assertFalse(bundle["state"]["dates_invented"])
        self.assertFalse(bundle["state"]["session_ordinal_used_as_day"])
        self.assertEqual(bundle["state"]["undated_day_terms"], "dropped")
        again = transform_with_state(rows, bundle["state"])
        self.assertTrue(np.allclose(bundle["z"], again))
        residual = np.linspace(0.2, 1.8, len(rows))
        params = fit_noise(bundle["z"][train], residual[train], "learned_state", feature_names=names)
        muted = dict(params)
        slope = np.array(params["slope"], dtype=np.float64, copy=True)
        slope[day_idx] = 0.0
        muted["slope"] = slope
        self.assertTrue(np.allclose(
            log_variance(bundle["z"][undated], params),
            log_variance(bundle["z"][undated], muted),
        ))
        missing = _rows("sub-0_ses-11_task-Sherlock1_run-1", 0, None, "train")
        missing += _rows("sub-0_b", 0, "2024-03-01", "train")
        with self.assertRaises(ProtocolError) as caught:
            assemble_features(missing)
        self.assertEqual(
            str(caught.exception),
            missing_acquisition_date("sub-0_ses-11_task-Sherlock1_run-1"),
        )
        self.assertNotRegex(str(caught.exception), r"\d{4}-\d{2}-\d{2}")
        session_as_day = _rows("sub-1_ses-11_task-Sherlock1_run-1", 1, "11", "selection")
        with self.assertRaises(ProtocolError) as caught:
            assemble_features(rows[:8] + session_as_day)
        self.assertIn("not YYYY-MM-DD", str(caught.exception))

    def test_one_training_date_drops_inestimable_day_terms(self):
        rows = []
        rows += _rows("sub-0_ses-11_task-Sherlock1_run-2", 0, "2024-10-04", "train")
        rows += _rows("sub-0_ses-12_task-Sherlock1_run-2", 0, "2024-10-04", "train", scale=1.3)
        rows += _rows("sub-1_ses-11_task-Sherlock1_run-1", 1, None, "selection")
        rows += _rows("sub-5_ses-11_task-Sherlock1_run-1", 5, None, "selection", scale=1.2)
        rows += _rows("sub-10_ses-11_task-Sherlock1_run-1", 10, "2030-01-01", "selection", scale=2.0)
        bundle = assemble_features(rows)
        self.assertTrue(bundle["day_terms_inestimable"])
        self.assertEqual(bundle["day_terms_dropped"], ["day_rank", "day_distance"])
        self.assertEqual(bundle["state"]["train_days"], ["2024-10-04"])
        self.assertEqual(bundle["n_train_days"], 1)
        self.assertEqual(
            bundle["names"],
            ["run_fraction", "artifact_rms", "bandpower", "log_train_window_count"],
        )
        self.assertNotIn("day_rank", bundle["names"])
        self.assertNotIn("day_distance", bundle["names"])
        self.assertEqual(bundle["z"].shape[1], 4)
        self.assertFalse(bundle["state"]["dates_invented"])
        self.assertFalse(bundle["state"]["session_ordinal_used_as_day"])
        self.assertEqual(bundle["state"]["date_source"], "bids_scans_tsv_only")
        train = bundle["train_index"]
        self.assertGreater(float(np.std(bundle["z"][train, bundle["names"].index("run_fraction")])), 0.0)
        self.assertGreater(float(np.std(bundle["z"][train, bundle["names"].index("artifact_rms")])), 0.0)
        again = transform_with_state(rows, bundle["state"])
        self.assertTrue(np.allclose(bundle["z"], again))
        later = _rows("sub-0_ses-1_task-Sherlock1_run-1", 0, "2024-07-31", "train")
        held = transform_with_state(later, bundle["state"])
        self.assertEqual(held.shape[1], 4)
        self.assertTrue(np.isfinite(held).all())
        residual = np.linspace(0.2, 1.8, len(rows))
        for name in CANDIDATES:
            matrix, feat_names = _candidate_matrix(bundle["z"], bundle["names"], name)
            params = fit_noise(matrix[train], residual[train], name, feature_names=feat_names, seed=0)
            self.assertTrue(np.isfinite(weighted_nll(residual[~train], matrix[~train], params)))
            self.assertNotIn("day_rank", feat_names)
            self.assertNotIn("day_distance", feat_names)
        _matrix_scaling, scaling_names = _candidate_matrix(bundle["z"], bundle["names"], "error_scaling")
        self.assertEqual(scaling_names, ["log_train_window_count"])
        with self.assertRaises(ProtocolError):
            assemble_features(rows[:1])
        leaked_val = _rows("sub-1_ses-11_task-Sherlock1_run-1", 1, "2024-10-04", "train")
        leaked_eval = _rows("sub-3_ses-11_task-Sherlock1_run-1", 3, "2024-10-04", "train")
        for leaked in (leaked_val, leaked_eval):
            with self.assertRaises(ProtocolError) as caught:
                assemble_features(rows[:8] + leaked)
            self.assertIn("leaked into the noise fit", str(caught.exception))
        missing = _rows("sub-0_ses-11_task-Sherlock1_run-2", 0, None, "train")
        missing += _rows("sub-0_ses-12_task-Sherlock1_run-2", 0, "2024-10-04", "train")
        with self.assertRaises(ProtocolError) as caught:
            assemble_features(missing)
        self.assertEqual(
            str(caught.exception),
            missing_acquisition_date("sub-0_ses-11_task-Sherlock1_run-2"),
        )
        session_as_day = _rows("sub-1_ses-11_task-Sherlock1_run-1", 1, "11", "selection")
        with self.assertRaises(ProtocolError) as caught:
            assemble_features(rows[:8] + session_as_day)
        self.assertIn("not YYYY-MM-DD", str(caught.exception))

    def test_two_training_dates_keep_day_terms(self):
        rows = []
        rows += _rows("sub-0_ses-11_task-Sherlock1_run-2", 0, "2024-10-04", "train")
        rows += _rows("sub-0_ses-1_task-Sherlock1_run-1", 0, "2024-07-31", "train", scale=0.8)
        rows += _rows("sub-1_ses-11_task-Sherlock1_run-1", 1, None, "selection")
        bundle = assemble_features(rows)
        self.assertFalse(bundle["day_terms_inestimable"])
        self.assertEqual(bundle["day_terms_dropped"], [])
        self.assertEqual(bundle["state"]["train_days"], ["2024-07-31", "2024-10-04"])
        self.assertEqual(bundle["n_train_days"], 2)
        self.assertEqual(bundle["names"][0], "day_rank")
        self.assertEqual(bundle["names"][-1], "day_distance")
        self.assertEqual(bundle["z"].shape[1], 6)
        train = bundle["train_index"]
        day_rank = bundle["z"][train, bundle["names"].index("day_rank")]
        self.assertGreater(float(np.std(day_rank)), 0.0)
        self.assertTrue(np.isfinite(bundle["z"][:, bundle["names"].index("day_distance")]).all())
        undated = np.array([row["date"] is None for row in rows])
        day_idx = [bundle["names"].index(name) for name in DAY_TERMS]
        self.assertTrue(np.all(bundle["z"][undated][:, day_idx] == 0.0))
        self.assertFalse(np.allclose(bundle["z"][undated, bundle["names"].index("bandpower")], 0.0))
        again = transform_with_state(rows, bundle["state"])
        self.assertTrue(np.allclose(bundle["z"], again))
        self.assertFalse(bundle["state"]["dates_invented"])
        self.assertFalse(bundle["state"]["session_ordinal_used_as_day"])

    def test_session_interval_is_undefined_for_one_recording(self):
        wide = session_bootstrap(["a", "a", "b", "b"], [0.1, 0.2, 0.3, 0.4], seed=0, draws=50)
        self.assertTrue(wide["defined"])
        self.assertLess(wide["low"], wide["high"])
        single = session_bootstrap(["only", "only"], [0.2, 0.4])
        self.assertFalse(single["defined"])
        self.assertIsNone(bacc10_50({0: 1.0}))
        self.assertAlmostEqual(bacc10_50({c: 0.2 for c in range(50)}), 0.2)

    def test_timing_primary_is_zero_and_resume_is_deterministic(self):
        rel, meg = timing_inputs(np.array([1.0, 2.0]), np.ones(2), "zero")
        self.assertTrue(np.allclose(rel, 0))
        self.assertTrue(np.allclose(meg, 1))
        only, cleared = timing_inputs(np.array([1.0, 2.0]), np.ones(2), "timing_only")
        self.assertTrue(np.allclose(cleared, 0))
        self.assertFalse(np.allclose(only, 0))
        self.assertEqual(next_sessions(["s1", "s2", "s3"], ["s1"]), ["s2", "s3"])

    def test_temperature_does_not_change_top10_and_precision_does_change_the_head(self):
        logits = np.zeros((3, 50))
        logits[0, 0] = 3.0
        logits[0, 1] = 1.0
        logits[1, 4] = 4.0
        logits[1, 2] = 0.2
        logits[2, 7] = 5.0
        logits[2, 3] = 1.1
        log_var = np.array([-1.0, 0.5, 2.0])
        self.assertFalse(temperature_changes_top10(logits, log_var))
        self.assertFalse(np.allclose(row_temperature(logits, log_var), logits))
        labels = np.array([0, 1, 2])
        weight = np.zeros((3, 4))
        bias = np.zeros(4)
        embeddings = np.eye(3)
        plain, _bias_a = step_head(weight, bias, embeddings, labels, np.zeros(3), lr=1.0)
        weighted, _bias_b = step_head(weight, bias, embeddings, labels, np.array([0.0, 0.0, 4.0]), lr=1.0)
        self.assertFalse(np.allclose(plain, weighted))
        arms = initial_partial_arms(np.zeros((3, 4)), np.zeros(4), 2)
        features = np.array([[0.0, 0.0], [0.0, 0.0], [3.0, 0.0]])
        matched_partial_steps(arms, embeddings, labels, features, steps=15, lr=0.4, seed=0, batch=3)
        self.assertFalse(np.allclose(arms["uniform"]["W"], arms["learned_state"]["W"]))
        again = initial_partial_arms(np.zeros((3, 4)), np.zeros(4), 2)
        matched_partial_steps(again, embeddings, labels, features, steps=15, lr=0.4, seed=0, batch=3)
        self.assertTrue(np.allclose(arms["learned_state"]["W"], again["learned_state"]["W"]))
        self.assertEqual(set(arms), set(PARTIAL_ARMS))
        top = arm_logits(arms["uniform"], embeddings).argmax(axis=1)
        self.assertEqual(len(top), 3)

    def test_head_gradient_matches_a_finite_difference(self):
        rng = np.random.default_rng(1)
        weight = rng.normal(scale=0.05, size=(3, 4))
        bias = np.zeros(4)
        embeddings = rng.normal(size=(5, 3))
        labels = np.array([0, 1, 2, 3, 0])

        def loss(trial):
            return float(np.mean(per_row_ce(embeddings @ trial + bias, labels, smoothing=0.1)))

        base = loss(weight)
        nudged = weight.copy()
        nudged[0, 0] += 1e-6
        numeric = (loss(nudged) - base) / 1e-6
        analytic = embeddings.T @ dlogits_ce(embeddings @ weight + bias, labels, smoothing=0.1) / len(labels)
        self.assertAlmostEqual(analytic[0, 0], numeric, places=4)

    def test_code_path_is_v3_and_a_surrogate_cannot_claim(self):
        root = os.path.join(os.path.dirname(__file__), "..", "..")
        text = open(os.path.join(root, "src", "v3lib.py"), encoding="utf-8").read()
        train = open(os.path.join(root, "scripts", "train_v3.py"), encoding="utf-8").read()
        split = open(os.path.join(root, "src", "split2.py"), encoding="utf-8").read()
        self.assertIn("class Model", text)
        self.assertIn("nullcontext", text)
        self.assertIn("torch.cuda.is_available()", text)
        self.assertIn("head_t5", text)
        self.assertIn("SigLIP", train)
        self.assertIn("EVAL = (3, 6, 9, 12)", split)
        self.assertIn('np.isin(q, ["other", "Q1", "Q2"])', split)
        attested = attest_checkpoint_dict(_fake_ckpt())
        self.assertGreaterEqual(attested["parameter_count"], 1_000_000)
        self.assertEqual(attested["rel"], 0)
        tiny = _fake_ckpt()
        tiny["enc"] = {"w": np.zeros((2, 2))}
        tiny["ctx"] = None
        with self.assertRaises(ProtocolError):
            attest_checkpoint_dict(tiny)
        self.assertFalse(is_strong_decoder(object()))
        rejected = claim_from_decoder(object(), metric_defined=True)
        self.assertFalse(rejected["valid_scientific_claim"])
        self.assertFalse(rejected["effect_claimed"])
        self.assertFalse(rejected["frozen_trunk_subject0_days_are_holdout"])
        with self.assertRaises(FileNotFoundError):
            load_strong_decoder("/no/such/deep_c_ep6.pt")
        self.assertFalse(measurement_claim({"decoder": "src.v3lib.Model", "checkpoint_bytes": CHECKPOINT_BYTES}))
        engineering = {
            "decoder": "src.v3lib.Model",
            "checkpoint_bytes": CHECKPOINT_BYTES,
            "objective": "gaussian_nll",
            "candidates": CANDIDATES,
            "eval_subjects_in_train": [],
            "subject0_frozen_holdout": False,
            "temperature_applied_to_reported_logits": False,
            "in_sample_trunk_trained_residual": False,
            "eval_rows": 3,
            "fit_rows": 4,
            "selection_rows": 2,
            "lock_held": True,
        }
        status = measurement_status(engineering)
        self.assertTrue(status["measurement_complete"])
        self.assertFalse(status["valid_scientific_claim"])
        self.assertFalse(status["effect_claimed"])
        self.assertTrue(status["exploratory"])
        self.assertTrue(status["not_confirmatory"])
        self.assertTrue(status["independent_confirmation_required"])
        engineering["fixture"] = True
        self.assertFalse(measurement_status(engineering)["measurement_complete"])
        self.assertFalse(measurement_status(engineering)["valid_scientific_claim"])
        wrapped = honesty(valid_scientific_claim=True, effect_claimed=True, measurement_complete=True)
        self.assertFalse(wrapped["valid_scientific_claim"])
        self.assertFalse(wrapped["effect_claimed"])
        self.assertTrue(wrapped["not_confirmatory"])
        self.assertTrue(wrapped["independent_confirmation_required"])
        self.assertFalse(wrapped["numeric_precision"]["direct_score_comparison_to_leaderboard"])
        self.assertFalse(wrapped["full_retrain"]["launched"])
        self.assertFalse(wrapped["full_retrain"]["practical_under_cpu_4g_one_thread"])

    def test_plan_forwards_untrained_subjects_and_not_subject0_as_holdout(self):
        bases = [
            "sub-0_ses-01_task-audiobook_run-2",
            "sub-0_ses-01_task-audiobook_run-1",
            "sub-3_ses-11_task-Sherlock1_run-1",
            "sub-1_ses-11_task-Sherlock1_run-1",
            "sub-0_ses-11_task-Sherlock1_run-1",
        ]
        dates = {
            "sub-0_ses-01_task-audiobook_run-2": "2024-08-02",
            "sub-0_ses-01_task-audiobook_run-1": "2024-08-01",
        }
        order = plan_order(bases, dates)
        budgeted = [row for row in order if row["in_budget"]]
        self.assertEqual(budgeted[0]["phase"], "noise_pool")
        self.assertEqual(budgeted[0]["base"], "sub-0_ses-11_task-Sherlock1_run-1")
        self.assertIn("sub-1_ses-11_task-Sherlock1_run-1", [row["base"] for row in budgeted])
        self.assertIn("sub-3_ses-11_task-Sherlock1_run-1", [row["base"] for row in budgeted])
        train = [row["base"] for row in budgeted if row["phase"] == "train"]
        self.assertLess(train.index("sub-0_ses-01_task-audiobook_run-1"), train.index("sub-0_ses-01_task-audiobook_run-2"))
        self.assertEqual(classify_base("sub-0_ses-02_task-x_run-1")["phase"], "train")
        action = next_action(order, done_bases=[], noise_fit=False, partial_steps=0, score_frozen=False, score_partial=False)
        self.assertEqual(action["phase"], "noise_pool")
        done = [row["base"] for row in budgeted if row["phase"] in ("noise_pool", "selection")]
        action = next_action(order, done_bases=done, noise_fit=False, partial_steps=0, score_frozen=False, score_partial=False)
        self.assertEqual(action["stage"], "noise-fit")
        action = next_action(order, done_bases=done, noise_fit=True, partial_steps=0, score_frozen=False, score_partial=False)
        self.assertEqual(action["phase"], "eval")
        from campaigns.sdr_uncertainty.strong_noise import SUBSTANTIVE_SCHEDULE, validation_stop
        curve = [
            {"step": 100, "learned_ce": 1.0},
            {"step": 200, "learned_ce": 0.8},
            {"step": 300, "learned_ce": 0.9},
            {"step": 400, "learned_ce": 0.91},
            {"step": 500, "learned_ce": 0.92},
        ]
        decision = validation_stop(curve, SUBSTANTIVE_SCHEDULE)
        self.assertTrue(decision["stop"])
        self.assertEqual(decision["reason"], "validation_plateau")
        self.assertEqual(decision["best_step"], 200)
        with self.assertRaises(ProtocolError):
            validation_stop([{"step": 100, "learned_ce": 1.0, "eval_ce": 0.1}], SUBSTANTIVE_SCHEDULE)

    def test_forward_recording_calls_the_model_and_zeroes_timing(self):
        meg = np.zeros((306, 400), dtype=np.float32)
        meg[:, 100] = 4.0
        window, valid = slice_window(meg, 2.0, 0.0, 5.0)
        self.assertEqual(window.shape, (306, 150))
        self.assertAlmostEqual(float(window[0, 50]), 4.0)
        self.assertTrue(bool(valid[50]))

        class Fake:
            def __init__(self):
                self.modes = []

            def forward(self, x, v, am, gi, pad, rel):
                array = np.asarray(x)
                self.modes.append((array.shape, float(np.max(np.abs(np.asarray(rel)))), float(np.sum(np.abs(array)))))
                n = array.shape[0]
                return np.zeros((n, 4)), np.zeros((n, 50)), np.zeros((n, 2))

        events = {
            "onset_s": np.array([2.0, 2.4, 2.8]),
            "label": np.array([1, 2, 3]),
            "sentenceidx": np.array([0, 0, 0]),
            "sent_start_s": np.array([1.0, 1.0, 1.0]),
            "sent_end_s": np.array([3.5, 3.5, 3.5]),
        }
        model = Fake()
        primary = forward_recording(model, meg, events, np.eye(306, dtype=np.float32), timing="zero", maxw=12)
        self.assertEqual(primary["logits"].shape, (3, 50))
        self.assertEqual(primary["embedding"].shape, (3, 4))
        self.assertEqual(model.modes[-1][0], (3, 306, 150))
        self.assertEqual(model.modes[-1][1], 0.0)
        self.assertGreater(model.modes[-1][2], 0.0)
        forward_recording(model, meg, events, np.eye(306, dtype=np.float32), timing="timing_only", maxw=12)
        self.assertEqual(model.modes[-1][2], 0.0)
        self.assertGreater(model.modes[-1][1], 0.0)
        labels = np.arange(50)
        logits = np.eye(50) * 3.0
        self.assertAlmostEqual(bacc10_from_logits(logits, labels), 1.0)
        self.assertIsNone(bacc10_from_logits(primary["logits"], primary["label"]))

    def test_directory_fit_does_not_read_eval_and_refuses_bad_output_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            dates = {}
            specs = (
                ("noise_pool", "sub-0_ses-11_task-Sherlock1_run-1", 0, "2024-05-01", np.array([10.0, 60.0, 62.0, 90.0])),
                ("noise_pool", "sub-0_ses-12_task-Sherlock1_run-1", 0, "2024-06-01", np.array([12.0, 58.0, 61.0, 95.0])),
                ("selection", "sub-1_ses-11_task-Sherlock1_run-1", 1, "2024-07-01", np.array([20.0, 30.0, 40.0, 50.0])),
                ("eval", "sub-3_ses-11_task-Sherlock1_run-1", 3, "2024-08-01", np.array([15.0, 25.0, 35.0, 45.0])),
            )
            for phase, base, subject, date, timechapter in specs:
                folder = os.path.join(tmp, "forward", phase)
                os.makedirs(folder, exist_ok=True)
                n = len(timechapter)
                np.savez_compressed(
                    os.path.join(folder, base + ".npz"),
                    base=np.array(base),
                    subject=np.int64(subject),
                    embedding=np.zeros((n, 4), np.float32),
                    logits=np.zeros((n, 50), np.float32),
                    label=np.full(n, subject, np.int64),
                    run_fraction=np.linspace(0.1, 0.9, n),
                    artifact_rms=np.linspace(0.5, 1.5, n),
                    bandpower=np.linspace(0.1, 0.4, n),
                    timechapter=timechapter,
                    checkpoint_sha=np.array("abc"),
                    timing=np.array("zero"),
                )
                dates[base] = date
            fit = fit_noise_directory(tmp, dates)
            self.assertEqual(fit["eval_files_read"], False)
            self.assertFalse(fit["in_sample_trunk_trained_residual"])
            self.assertEqual(fit["residual_fit_population"], "subject0_shared_sherlock_Q3")
            self.assertFalse(fit["selection_rows_used_for_coefficients"])
            self.assertEqual(fit["eval_subjects_in_train"], [])
            self.assertEqual(fit["train_subjects"], [0])
            self.assertGreaterEqual(fit["fit_rows"], 2)
            self.assertEqual(set(fit["val_nll"]), set(CANDIDATES))
            os.makedirs(os.path.join(tmp, "noise"), exist_ok=True)
            with open(os.path.join(tmp, "noise", "fit.json"), "w", encoding="utf-8") as handle:
                json.dump(fit, handle)
            # Re-dump through the same serializer the runner uses: slopes may be lists already.
            score = score_frozen_directory(tmp, dates)
            self.assertIsNone(score["primary_bacc10_50"])
            self.assertNotIn("bacc10_50", score["eligible_metrics"])
            self.assertIn("gaussian_nll_of_cross_entropy", score["eligible_metrics"])
            self.assertIn("3", score["coverage"]["per_subject"])
            self.assertGreater(score["coverage"]["per_subject"]["3"]["n"], 0)
            self.assertLess(score["coverage"]["per_subject"]["3"]["n_classes_present"], 50)
            self.assertFalse(score["temperature_applied_to_reported_logits"])
            self.assertIn("uniform", score["eval_nll"])
            self.assertIn("constant", score["eval_nll"])
            root = os.path.join(os.path.dirname(__file__), "..", "..")
            with self.assertRaises(ProtocolError):
                assert_output_dir(os.path.join(root, "strong_decoder_1_4"), root)
            with self.assertRaises(ProtocolError):
                assert_output_dir(os.path.join(tmp, "holdout", "strong_decoder_1_4"), root)
            cache = os.path.join(tmp, "forward_cache")
            written = _write_forward(cache, "noise_pool", "sub-0_ses-11_task-Sherlock1_run-1", 0, {
                "embedding": np.zeros((2, 4), np.float32),
                "logits": np.zeros((2, 50), np.float32),
                "label": np.array([1, 2]),
                "run_fraction": np.array([0.1, 0.2]),
                "artifact_rms": np.array([1.0, 1.0]),
                "bandpower": np.array([0.2, 0.2]),
                "timechapter": np.array([60.0, 62.0]),
                "timing": "zero",
            }, "abc")
            self.assertTrue(written.endswith("sub-0_ses-11_task-Sherlock1_run-1.npz"))
            self.assertEqual(glob.glob(os.path.join(cache, "forward", "noise_pool", ".*.npz")), [])
            with np.load(written, allow_pickle=False) as handle:
                self.assertEqual(handle["logits"].shape, (2, 50))
            with self.assertRaises(ProtocolError):
                _write_forward(cache, "noise_pool", "sub-0_ses-11_task-Sherlock1_run-1", 0, {
                    "embedding": np.zeros((2, 4), np.float32),
                    "logits": np.zeros((2, 50), np.float32),
                    "label": np.array([1, 2]),
                    "run_fraction": np.array([0.1, 0.2]),
                    "artifact_rms": np.array([1.0, 1.0]),
                    "bandpower": np.array([0.2, 0.2]),
                    "timing": "zero",
                }, "abc")
            with self.assertRaises(ProtocolError):
                _write_forward(cache, "noise_pool", "sub-0_bad", 0, {
                    "embedding": np.zeros((1, 4), np.float32),
                    "logits": np.array([[np.nan] * 50], np.float32),
                    "label": np.array([0]),
                    "run_fraction": np.array([0.1]),
                    "artifact_rms": np.array([1.0]),
                    "bandpower": np.array([0.2]),
                    "timing": "zero",
                }, "abc")
            self.assertFalse(os.path.exists(os.path.join(cache, "forward", "noise_pool", "sub-0_bad.npz")))

    def test_shell_lock_is_released_and_gpu_hide_fails_closed(self):
        root = os.path.join(os.path.dirname(__file__), "..", "..")
        strong = open(os.path.join(root, "scripts", "sdr_uncertainty_strong_decoder.sh"), encoding="utf-8").read()
        lock = open(os.path.join(root, "scripts", "sdr_lock_run.sh"), encoding="utf-8").read()
        hide = open(os.path.join(root, "scripts", "gpu_hide.sh"), encoding="utf-8").read()
        self.assertEqual(lock.count("acquire_lock.sh"), 1)
        self.assertEqual(lock.count("release_lock.sh"), 1)
        self.assertNotIn("exec ", lock)
        self.assertNotIn("while ", lock)
        self.assertNotIn("while ", strong)
        self.assertNotIn("RuntimeMaxSec", strong)
        self.assertNotIn("RuntimeMaxSec", lock)
        self.assertIn("MemoryMax=4G", strong)
        self.assertIn("MemorySwapMax=0", strong)
        self.assertIn("TasksMax=16", strong)
        self.assertIn("gpu_hide.sh", strong)
        self.assertIn("sdr_lock_run.sh", strong)
        self.assertNotIn("exec systemd-run", strong)
        self.assertNotIn('exec "$ROOT/scripts/sdr_lock_run.sh"', strong)
        self.assertIn("LIBRIBRAIN_SDR_SSD_PREFIX", strong)
        self.assertIn("mount --bind /dev/null", hide)
        self.assertIn("must be 1", hide)
        for path in (
            "scripts/sdr_uncertainty_strong_decoder.sh",
            "scripts/sdr_lock_run.sh",
            "scripts/gpu_hide.sh",
        ):
            subprocess.run(["bash", "-n", os.path.join(root, path)], check=True)
        env = os.environ.copy()
        env["SDR_FORBID_LIVE_COMPUTE"] = "1"
        proc = subprocess.run(
            [os.environ.get("PYTHON", "python3"), "-m", "campaigns.sdr_uncertainty.strong_decoder_runner", "run"],
            cwd=root, env=env, text=True, capture_output=True, check=False,
        )
        self.assertEqual(proc.returncode, 2)
        self.assertIn("refusing", proc.stderr)

        with tempfile.TemporaryDirectory() as home:
            scheduler = os.path.join(home, "research", "empirical_scheduler")
            os.makedirs(scheduler)
            log = os.path.join(home, "lock_log")
            acquire = os.path.join(scheduler, "acquire_lock.sh")
            release = os.path.join(scheduler, "release_lock.sh")
            lock_root = os.path.join(home, "locks")
            os.makedirs(lock_root)
            open(acquire, "w", encoding="utf-8").write(textwrap.dedent(f"""\
                #!/bin/bash
                # acquire_lock.sh JOB [TTL]. Default TTL is ${{2:-700}} and is a label only.
                JOB="$1"
                TTL="${{2:-700}}"
                DIR="{lock_root}/$JOB"
                if [[ -d "$DIR" ]]; then
                  echo "busy" >&2
                  exit 1
                fi
                mkdir -p "$DIR"
                python3 - "$JOB" "$TTL" "$DIR/holder.json" "$$" <<'PY'
                import json, sys
                job, ttl, path, pid = sys.argv[1:]
                json.dump({{"pid": int(pid), "job_id": job, "ttl_label_seconds": int(ttl), "ttl_enforced": False}}, open(path, "w"))
                PY
                echo "acquire $JOB $TTL" >> "{log}"
                exit 0
            """))
            open(release, "w", encoding="utf-8").write(textwrap.dedent(f"""\
                #!/bin/bash
                JOB="$1"
                DIR="{lock_root}/$JOB"
                if [[ ! -f "$DIR/holder.json" ]]; then exit 1; fi
                python3 - "$JOB" "$DIR/holder.json" <<'PY'
                import json, sys
                job, path = sys.argv[1], sys.argv[2]
                if json.load(open(path))["job_id"] != job:
                    raise SystemExit(1)
                PY
                rm -rf "$DIR"
                echo "release $JOB" >> "{log}"
                exit 0
            """))
            os.chmod(acquire, os.stat(acquire).st_mode | stat.S_IEXEC)
            os.chmod(release, os.stat(release).st_mode | stat.S_IEXEC)
            missing_ssd = os.path.join(home, "missing-ssd-prefix")
            self.assertFalse(os.path.isdir(missing_ssd))
            env = os.environ.copy()
            env["HOME"] = home
            env["LIBRIBRAIN_SDR_SSD_PREFIX"] = missing_ssd
            env.pop("SDR_LOCK_HELD", None)
            env.pop("SDR_STRONG_LOCK", None)
            ran = subprocess.run(
                ["bash", os.path.join(root, "scripts", "sdr_uncertainty_strong_decoder.sh"), "next"],
                cwd=root, env=env, text=True, capture_output=True, check=False,
            )
            self.assertNotEqual(ran.returncode, 0)
            text = open(log, encoding="utf-8").read()
            self.assertIn("acquire libribrain-strong-decoder-noise 14400", text)
            self.assertNotIn("--heartbeat", text)
            self.assertNotIn("--ttl", text)
            self.assertIn("release libribrain-strong-decoder-noise", text)
            self.assertIn(missing_ssd, ran.stderr)
            self.assertFalse(os.path.isdir(os.path.join(lock_root, "libribrain-strong-decoder-noise")))
            holder_after = os.path.join(lock_root, "libribrain-strong-decoder-noise", "holder.json")
            self.assertFalse(os.path.exists(holder_after))

            spec = inspect_text(open(acquire, encoding="utf-8").read())
            self.assertEqual(spec["style"], "positional")
            self.assertFalse(spec["heartbeat"])
            self.assertFalse(spec["ttl_enforced"])
            with self.assertRaises(ProtocolError):
                inspect_text("#!/bin/bash\necho no-ttl\n")
            self.assertEqual(chosen_ttl(), 14400)

            held = subprocess.Popen(
                ["bash", os.path.join(root, "scripts", "sdr_lock_run.sh"), "libribrain-strong-decoder-noise", "--", "sleep", "30"],
                cwd=root, env=env, text=True,
            )
            holder = os.path.join(lock_root, "libribrain-strong-decoder-noise", "holder.json")
            for _ in range(50):
                if os.path.isfile(holder):
                    break
                time.sleep(0.05)
            self.assertTrue(os.path.isfile(holder))
            recorded = json.load(open(holder, encoding="utf-8"))
            self.assertEqual(recorded["ttl_label_seconds"], 14400)
            self.assertFalse(recorded["ttl_enforced"])
            self.assertFalse(os.path.exists(f"/proc/{recorded['pid']}"))
            busy = subprocess.run(
                ["bash", os.path.join(root, "scripts", "sdr_lock_run.sh"), "libribrain-strong-decoder-noise", "--", "true"],
                cwd=root, env=env, text=True, capture_output=True, check=False,
            )
            self.assertNotEqual(busy.returncode, 0)
            self.assertTrue(os.path.isfile(holder))
            self.assertEqual(open(log, encoding="utf-8").read().count("release "), 1)
            held.terminate()
            held.wait(timeout=10)
            self.assertFalse(os.path.isdir(os.path.join(lock_root, "libribrain-strong-decoder-noise")))
            self.assertEqual(open(log, encoding="utf-8").read().count("release "), 2)

        hide_env = os.environ.copy()
        hide_env.update({
            "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1", "BLIS_NUM_THREADS": "1",
            "LB_WORKERS": "1", "CUDA_VISIBLE_DEVICES": "",
        })
        ok = subprocess.run(["bash", os.path.join(root, "scripts", "gpu_hide.sh"), "--", "true"], env=hide_env, check=False)
        self.assertEqual(ok.returncode, 0)
        missing = dict(hide_env)
        del missing["OMP_NUM_THREADS"]
        bad = subprocess.run(
            ["bash", os.path.join(root, "scripts", "gpu_hide.sh"), "--", "true"],
            env=missing, text=True, capture_output=True, check=False,
        )
        self.assertEqual(bad.returncode, 2)
        self.assertIn("OMP_NUM_THREADS", bad.stderr)

    def _raised(self, path):
        try:
            inspect_text(open(path, encoding="utf-8").read())
        except ProtocolError as exc:
            return exc
        self.fail("inspect_text accepted a 700s default")

    def test_real_encoder_modules_forward_one_window(self):
        import torch
        from src.models import CtxEncoder, SentCtx

        torch.manual_seed(0)
        torch.set_num_threads(1)
        encoder = CtxEncoder(d=32, D=32, T=150, p=0.0)
        context = SentCtx(D=32, layers=2, maxw=4)
        head = torch.nn.Linear(32, 50)
        window = torch.zeros(3, 307, 150)
        window[:, :306, 40:80] = torch.randn(3, 306, 40)
        encoder.eval()
        context.eval()
        with torch.no_grad():
            embedding = encoder(window)
            self.assertEqual(tuple(embedding.shape), (3, 32))
            gathered = embedding.reshape(1, 3, 32)
            rel = torch.zeros(1, 3)
            pad = torch.zeros(1, 3, dtype=torch.bool)
            mixed = context(gathered, rel, pad)
            logits = head(mixed.reshape(3, 32))
        self.assertEqual(tuple(logits.shape), (3, 50))
        held = logits / torch.tensor([0.5, 2.0, 0.25])[:, None]
        self.assertTrue(torch.equal(logits.argmax(dim=1), held.argmax(dim=1)))

    def test_float64_onsets_run_through_the_real_context_layer(self):
        """The failed PC forward passed float64 relative onsets into a float32 SentCtx."""
        import torch
        from src.models import CtxEncoder, SentCtx

        torch.manual_seed(0)
        torch.set_num_threads(1)
        encoder = CtxEncoder(d=32, D=32, T=150, p=0.0).eval()
        context = SentCtx(D=32, layers=2, maxw=4).eval()
        head = torch.nn.Linear(32, 50).eval()
        self.assertEqual(context.time[0].weight.dtype, torch.float32)
        meg = torch.from_numpy(np.random.default_rng(1).normal(size=(3, 307, 150)).astype(np.float64))
        self.assertEqual(meg.dtype, torch.float64)
        sensors = meg[:, :306, :].float()
        alignment = torch.from_numpy(np.eye(306, dtype=np.float64)).to(dtype=sensors.dtype)
        mixed = torch.einsum("ij,bjt->bit", alignment, sensors)
        self.assertEqual(mixed.dtype, torch.float32)
        tokens_in = torch.cat([mixed, torch.ones(3, 1, 150)], dim=1)
        gather = torch.from_numpy(np.array([[0, 1, 2]], dtype=np.int64))
        mask = torch.from_numpy(np.array([[False, False, False]]))
        zeros = torch.from_numpy(np.zeros((1, 3), dtype=np.float64))
        onsets = torch.from_numpy(np.array([[0.0, 0.35, 1.2]], dtype=np.float64))
        self.assertEqual(gather.dtype, torch.int64)
        self.assertEqual(mask.dtype, torch.bool)
        self.assertEqual(zeros.dtype, torch.float64)
        with torch.no_grad():
            tokens = encoder(tokens_in)
            hidden = context(tokens[gather] * (~mask)[..., None], zeros, mask)
            logits = head(hidden.reshape(3, 32))
            timed = context(tokens[gather] * (~mask)[..., None], onsets, mask)
        self.assertEqual(tuple(logits.shape), (3, 50))
        self.assertEqual(tuple(hidden.shape), (1, 3, 32))
        self.assertTrue(torch.isfinite(logits).all())
        self.assertTrue(torch.isfinite(timed).all())
        self.assertEqual(context.time[0].weight.dtype, torch.float32)
        with self.assertRaises(TypeError):
            context(tokens[gather], zeros, torch.zeros(1, 3))
        root = os.path.join(os.path.dirname(__file__), "..", "..")
        source = open(os.path.join(root, "src", "v3lib.py"), encoding="utf-8").read()
        self.assertIn("context gather index must be an integer", source)
        self.assertIn("context pad mask must be bool", source)
        self.assertIn("Am.dtype != x.dtype", source)
        self.assertIn('torch.autocast("cuda", torch.bfloat16)', source)

    def test_cross_entropy_rejects_out_of_range_labels_without_clipping(self):
        logits = np.zeros((1, 50))
        logits[0, 49] = 4.0
        for bad in (-1, 50, 77):
            with self.assertRaises(ProtocolError) as caught:
                per_row_ce(logits, np.array([bad]))
            self.assertEqual(str(caught.exception), "label is outside the decoder classes")
            with self.assertRaises(ProtocolError) as caught:
                dlogits_ce(logits, np.array([bad]))
            self.assertEqual(str(caught.exception), "label is outside the decoder classes")
        on_last = float(per_row_ce(logits, np.array([49]))[0])
        on_first = float(per_row_ce(logits, np.array([0]))[0])
        self.assertLess(on_last, on_first)
        with self.assertRaises(ProtocolError):
            bacc10_from_logits(np.zeros((2, 50)), np.array([-1, 0]))

    def test_reported_top10_is_the_unweighted_class_mean(self):
        labels = np.array([0] * 30 + list(range(50)), dtype=np.int64)
        logits = np.zeros((len(labels), 50), dtype=np.float64)
        for index, label in enumerate(labels):
            if label < 10:
                logits[index, label] = 10.0
            else:
                logits[index, :10] = np.linspace(20.0, 11.0, 10)
                logits[index, label] = -1.0
        balanced = bacc10_from_logits(logits, labels)
        chosen = np.argpartition(-logits, kth=9, axis=1)[:, :10]
        row_rate = float(np.mean([label in chosen[index] for index, label in enumerate(labels)]))
        self.assertAlmostEqual(balanced, 0.2)
        self.assertAlmostEqual(row_rate, 0.5)
        self.assertNotAlmostEqual(balanced, row_rate)
        missing = labels[labels != 49]
        self.assertIsNone(bacc10_from_logits(logits[labels != 49], missing))

    def test_oov_rows_are_excluded_before_post_forward_stages(self):
        """The iteration-9 failure: cached forwards keep label -1, and per_row_ce rejects it.

        Noise fit, validation selection, the three head arms, eval, and coverage
        then share one in-vocabulary mask. The npz files are not rewritten.
        """
        width = 8
        rng = np.random.default_rng(0)

        def write_forward(folder, base, subject, labels, timechapter):
            os.makedirs(folder, exist_ok=True)
            labels = np.asarray(labels, dtype=np.int64)
            n = len(labels)
            path = os.path.join(folder, base + ".npz")
            np.savez_compressed(
                path,
                base=np.array(base),
                subject=np.int64(subject),
                embedding=(rng.normal(size=(n, width)) + subject).astype(np.float32),
                logits=np.zeros((n, 50), np.float32),
                label=labels,
                run_fraction=np.linspace(0.1, 0.9, n),
                artifact_rms=np.linspace(0.5, 1.5, n),
                bandpower=np.linspace(0.1, 0.4, n),
                timechapter=np.asarray(timechapter, dtype=np.float64),
                checkpoint_sha=np.array("abc"),
                timing=np.array("zero"),
            )
            return path

        def file_sha(path):
            digest = hashlib.sha256()
            with open(path, "rb") as handle:
                digest.update(handle.read())
            return digest.hexdigest()

        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "out")
            root = os.path.join(tmp, "repo")
            os.makedirs(root)
            dates = {}
            specs = [
                ("noise_pool", "sub-0_ses-11_task-Sherlock1_run-1", 0, "2024-05-01",
                 [-1, 0, -1, 1, 3], [10.0, 60.0, 61.0, 62.0, 90.0]),
                ("noise_pool", "sub-0_ses-12_task-Sherlock1_run-1", 0, "2024-06-01",
                 [-1, 2, 4, -1], [12.0, 58.0, 61.0, 95.0]),
                ("selection", "sub-1_ses-11_task-Sherlock1_run-1", 1, "2024-07-01",
                 [-1, 0, 1, 50, 2, -1], [20.0, 30.0, 40.0, 50.0, 60.0, 70.0]),
                ("selection", "sub-5_ses-11_task-Sherlock1_run-1", 5, "2024-07-02",
                 [-1, 0, 1, 50, 2, -1], [20.0, 30.0, 40.0, 50.0, 60.0, 70.0]),
                ("selection", "sub-10_ses-11_task-Sherlock1_run-1", 10, "2024-07-03",
                 [-1, 0, 1, 50, 2, -1], [20.0, 30.0, 40.0, 50.0, 60.0, 70.0]),
                ("train", "sub-0_ses-01_task-continuous_run-1", 0, "2024-04-01",
                 [-1, 0, 3, -1], [1.0, 2.0, 3.0, 4.0]),
            ]
            written = []
            for phase, base, subject, date, labels, times in specs:
                written.append(write_forward(os.path.join(out, "forward", phase), base, subject, labels, times))
                dates[base] = date
            eval_labels = np.array([-1, -1, 50, -1] + [0] * 30 + list(range(50)), dtype=np.int64)
            eval_logits = np.zeros((len(eval_labels), 50), np.float32)
            for index, label in enumerate(eval_labels):
                if label < 0 or label >= 50:
                    eval_logits[index, 0] = 3.0
                elif label < 10:
                    eval_logits[index, label] = 10.0
                else:
                    eval_logits[index, :10] = np.linspace(20.0, 11.0, 10)
                    eval_logits[index, label] = -1.0
            eval_base = "sub-3_ses-11_task-Sherlock1_run-1"
            eval_dir = os.path.join(out, "forward", "eval")
            os.makedirs(eval_dir, exist_ok=True)
            eval_path = os.path.join(eval_dir, eval_base + ".npz")
            np.savez_compressed(
                eval_path,
                base=np.array(eval_base),
                subject=np.int64(3),
                embedding=(rng.normal(size=(len(eval_labels), width)) + 3).astype(np.float32),
                logits=eval_logits,
                label=eval_labels,
                run_fraction=np.linspace(0.1, 0.9, len(eval_labels)),
                artifact_rms=np.linspace(0.4, 1.2, len(eval_labels)),
                bandpower=np.linspace(0.05, 0.3, len(eval_labels)),
                timechapter=np.linspace(1.0, 80.0, len(eval_labels)),
                checkpoint_sha=np.array("abc"),
                timing=np.array("zero"),
            )
            dates[eval_base] = "2024-08-01"
            written.append(eval_path)
            before = {path: file_sha(path) for path in written}
            with np.load(written[0], allow_pickle=False) as handle:
                with self.assertRaises(ProtocolError) as caught:
                    decoder_residual(handle["logits"], handle["label"])
                self.assertEqual(str(caught.exception), "label is outside the decoder classes")
            fit = fit_noise_directory(out, dates)
            self.assertEqual(fit["fit_rows"], 4)
            self.assertEqual(fit["selection_rows"], 9)
            self.assertFalse(fit["eval_files_read"])
            self.assertEqual(fit["oov_excluded"]["sub-0_ses-11_task-Sherlock1_run-1"], {
                "base": "sub-0_ses-11_task-Sherlock1_run-1",
                "rows": 5, "excluded_oov": 2, "in_vocab": 3, "used": 2,
            })
            self.assertEqual(fit["oov_excluded"]["sub-0_ses-12_task-Sherlock1_run-1"]["used"], 2)
            selection_counts = [
                fit["oov_excluded"][base]["excluded_oov"]
                for base in (
                    "sub-1_ses-11_task-Sherlock1_run-1",
                    "sub-5_ses-11_task-Sherlock1_run-1",
                    "sub-10_ses-11_task-Sherlock1_run-1",
                )
            ]
            self.assertEqual(selection_counts, [3, 3, 3])
            kept_labels = []
            for base in (
                "sub-1_ses-11_task-Sherlock1_run-1",
                "sub-5_ses-11_task-Sherlock1_run-1",
                "sub-10_ses-11_task-Sherlock1_run-1",
            ):
                with np.load(os.path.join(out, "forward", "selection", base + ".npz"), allow_pickle=False) as handle:
                    view = vocab_view(base, handle["label"], {"embedding": handle["embedding"]})
                    kept_labels.append(view["labels"])
                    self.assertEqual(int(np.sum(view["labels"] < 0)), 0)
                    self.assertEqual(int(np.sum(view["labels"] >= 50)), 0)
            self.assertTrue(np.array_equal(kept_labels[0], kept_labels[1]))
            self.assertTrue(np.array_equal(kept_labels[0], kept_labels[2]))
            os.makedirs(os.path.join(out, "noise"), exist_ok=True)
            with open(os.path.join(out, "noise", "fit.json"), "w", encoding="utf-8") as handle:
                json.dump(fit, handle)
            score = score_frozen_directory(out, dates)
            self.assertEqual(score["eval_rows"], 80)
            self.assertAlmostEqual(score["primary_bacc10_50"], 0.2)
            self.assertIn("bacc10_50", score["eligible_metrics"])
            self.assertEqual(score["coverage"]["per_subject"]["3"]["n"], 80)
            self.assertAlmostEqual(score["coverage"]["per_subject"]["3"]["bacc10_50"], 0.2)
            self.assertEqual(score["oov_excluded"][eval_base]["excluded_oov"], 4)
            self.assertEqual(score["oov_excluded"][eval_base]["in_vocab"], 80)
            filtered = eval_labels[(eval_labels >= 0) & (eval_labels < 50)]
            filtered_logits = eval_logits[(eval_labels >= 0) & (eval_labels < 50)]
            chosen = np.argpartition(-filtered_logits, kth=9, axis=1)[:, :10]
            row_rate = float(np.mean([label in chosen[index] for index, label in enumerate(filtered)]))
            self.assertAlmostEqual(row_rate, 0.5)
            self.assertNotAlmostEqual(score["primary_bacc10_50"], row_rate)
            claimed = honesty(**score)
            self.assertFalse(claimed["effect_claimed"])
            self.assertFalse(claimed["valid_scientific_claim"])
            schedule = {
                "name": "substantive_head_finetune",
                "max_steps": 2,
                "steps_per_launch": 2,
                "min_steps": 2,
                "patience_checks": None,
                "substantive": True,
            }
            with self.assertRaises(ProtocolError) as caught:
                partial_train_directory(out, root, 2, dates, schedule=schedule)
            self.assertIn("missing strong checkpoint", str(caught.exception))
            self.assertNotIn("label is outside", str(caught.exception))
            arms = initial_partial_arms(np.zeros((width, 50)), np.zeros(50), 6)
            state_dir = os.path.join(out, "partial")
            os.makedirs(state_dir, exist_ok=True)
            seeded = {
                "steps": np.int64(0),
                "stopped": np.int64(0),
                "stop_reason": np.array(""),
                "best_step": np.int64(0),
                "schedule": np.array(schedule["name"]),
                "init_sha": np.array("seed"),
                "checkpoint_sha": np.array("abc"),
            }
            for name, arm in arms.items():
                seeded[f"W_{name}"] = arm["W"]
                seeded[f"b_{name}"] = arm["b"]
                seeded[f"intercept_{name}"] = np.float64(arm["intercept"])
                seeded[f"slope_{name}"] = arm["slope"]
            np.savez_compressed(os.path.join(state_dir, "state.npz"), **seeded)
            trained = partial_train_directory(out, root, 2, dates, schedule=schedule)
            self.assertEqual(trained["train_rows_in_vocab"], 2)
            self.assertEqual(trained["selection_rows_in_vocab"], 9)
            self.assertTrue(trained["stopped"])
            self.assertEqual(trained["oov_excluded"]["sub-0_ses-01_task-continuous_run-1"]["excluded_oov"], 2)
            self.assertEqual(
                trained["oov_excluded"]["sub-1_ses-11_task-Sherlock1_run-1"]["excluded_oov"],
                fit["oov_excluded"]["sub-1_ses-11_task-Sherlock1_run-1"]["excluded_oov"],
            )
            self.assertTrue(np.isfinite(trained["validation_learned_ce"]))
            with open(os.path.join(state_dir, "val_curve.json"), encoding="utf-8") as handle:
                curve = json.load(handle)
            self.assertEqual(len({row["learned_ce"] for row in curve}), 1)
            val_emb = []
            val_y = []
            for base in (
                "sub-1_ses-11_task-Sherlock1_run-1",
                "sub-5_ses-11_task-Sherlock1_run-1",
                "sub-10_ses-11_task-Sherlock1_run-1",
            ):
                with np.load(os.path.join(out, "forward", "selection", base + ".npz"), allow_pickle=False) as handle:
                    view = vocab_view(base, handle["label"], {"embedding": handle["embedding"]})
                val_emb.append(view["arrays"]["embedding"])
                val_y.append(view["labels"])
            val_emb = np.concatenate(val_emb, axis=0)
            val_y = np.concatenate(val_y, axis=0)
            self.assertEqual(len(val_y), 9)
            with np.load(os.path.join(state_dir, "state.npz"), allow_pickle=False) as handle:
                recomputed = {}
                for name in PARTIAL_ARMS:
                    arm = {"W": np.asarray(handle[f"W_{name}"]), "b": np.asarray(handle[f"b_{name}"])}
                    recomputed[name] = float(np.mean(decoder_residual(arm_logits(arm, val_emb), val_y)))
            self.assertAlmostEqual(curve[-1]["learned_ce"], recomputed["learned_state"])
            self.assertAlmostEqual(curve[-1]["uniform_ce"], recomputed["uniform"])
            self.assertAlmostEqual(curve[-1]["random_ce"], recomputed["random_covariate"])
            partial = score_partial_directory(out)
            self.assertEqual(partial["eval_rows"], 80)
            self.assertTrue(partial["fifty_class_metric_defined"])
            self.assertFalse(partial["effect_claimed"])
            self.assertEqual(set(partial["primary_bacc10_50"]), set(PARTIAL_ARMS))
            self.assertEqual(partial["oov_excluded"][eval_base]["excluded_oov"], 4)
            with np.load(os.path.join(state_dir, "best_val.npz"), allow_pickle=False) as handle:
                for name in PARTIAL_ARMS:
                    arm = {"W": np.asarray(handle[f"W_{name}"]), "b": np.asarray(handle[f"b_{name}"])}
                    scored = bacc10_from_logits(arm_logits(arm, np.load(eval_path)["embedding"][(eval_labels >= 0) & (eval_labels < 50)]), filtered)
                    self.assertAlmostEqual(partial["primary_bacc10_50"][name], scored)
            after = {path: file_sha(path) for path in written}
            self.assertEqual(before, after)

    def _write_scan(self, root, dataset, subject, session, base, date):
        folder = os.path.join(root, dataset, f"sub-{subject}", f"ses-{session}")
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, f"sub-{subject}_ses-{session}_scans.tsv")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("filename\tacq_time\n")
            handle.write(f"meg/{base}_meg.fif\t{date}T12:00:00\n")
        return path

    def _write_cached_forward(self, folder, base, subject, labels, timechapter, checkpoint_sha="abc"):
        os.makedirs(folder, exist_ok=True)
        labels = np.asarray(labels, dtype=np.int64)
        n = len(labels)
        np.savez_compressed(
            os.path.join(folder, base + ".npz"),
            base=np.array(base),
            subject=np.int64(subject),
            embedding=np.zeros((n, 4), np.float32),
            logits=np.zeros((n, 50), np.float32),
            label=labels,
            run_fraction=np.linspace(0.1, 0.9, n),
            artifact_rms=np.linspace(0.5, 1.5, n),
            bandpower=np.linspace(0.1, 0.4, n),
            timechapter=np.asarray(timechapter, dtype=np.float64),
            checkpoint_sha=np.array(checkpoint_sha),
            timing=np.array("zero"),
        )

    def test_broad_selection_without_scans_resolves_and_holdout_is_not_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._write_scan(tmp, "Sherlock1", 0, "11", "sub-0_ses-11_task-Sherlock1_run-1", "2024-05-01")
            self._write_scan(tmp, "Sherlock1", 1, "11", "sub-1_ses-11_task-Sherlock1_run-1", "2024-07-01")
            hold = os.path.join(tmp, "COMPETITION_HOLDOUT", "sub-9", "ses-11")
            os.makedirs(hold, exist_ok=True)
            hold_scan = os.path.join(hold, "sub-9_ses-11_scans.tsv")
            with open(hold_scan, "w", encoding="utf-8") as handle:
                handle.write("filename\tacq_time\n")
                handle.write("meg/sub-9_ses-11_task-Sherlock1_run-1_meg.fif\t1999-01-01T00:00:00\n")
            opened = []
            real_open = open

            def tracking_open(path, *args, **kwargs):
                opened.append(str(path))
                return real_open(path, *args, **kwargs)

            with unittest.mock.patch("builtins.open", tracking_open):
                dates = read_dates(tmp)
            self.assertEqual(dates["sub-0_ses-11_task-Sherlock1_run-1"], "2024-05-01")
            self.assertEqual(dates["sub-1_ses-11_task-Sherlock1_run-1"], "2024-07-01")
            self.assertNotIn("1999-01-01", dates.values())
            self.assertNotIn("sub-9_ses-11_task-Sherlock1_run-1", dates)
            self.assertFalse(any("COMPETITION_HOLDOUT" in path for path in opened))
            out = os.path.join(tmp, "out")
            self._write_cached_forward(
                os.path.join(out, "forward", "noise_pool"),
                "sub-0_ses-11_task-Sherlock1_run-1", 0, [0, 1, 2, 3], [10.0, 60.0, 62.0, 90.0],
            )
            self._write_cached_forward(
                os.path.join(out, "forward", "noise_pool"),
                "sub-0_ses-12_task-Sherlock1_run-1", 0, [0, 1, 2, 3], [12.0, 58.0, 61.0, 95.0],
            )
            for subject in (1, 5, 10):
                base = f"sub-{subject}_ses-11_task-Sherlock1_run-1"
                self._write_cached_forward(
                    os.path.join(out, "forward", "selection"),
                    base, subject, [0, 1, 2, 3], [20.0, 30.0, 40.0, 50.0],
                )
            self._write_cached_forward(
                os.path.join(out, "forward", "eval"),
                "sub-3_ses-11_task-Sherlock1_run-1", 3, [0, 1, 2, 3], [15.0, 25.0, 35.0, 45.0],
            )
            fit_dates = {
                "sub-0_ses-11_task-Sherlock1_run-1": "2024-05-01",
                "sub-0_ses-12_task-Sherlock1_run-1": "2024-06-01",
            }
            fit = fit_noise_directory(out, fit_dates)
            self.assertEqual(fit["selection_subjects"], [1, 5, 10])
            self.assertGreater(fit["selection_rows"], 0)
            self.assertEqual(fit["day_terms_dropped_bases"], sorted(
                f"sub-{subject}_ses-11_task-Sherlock1_run-1" for subject in (1, 5, 10)
            ))
            self.assertFalse(fit["dates_invented"])
            self.assertFalse(fit["session_ordinal_used_as_day"])
            self.assertEqual(fit["amendment_1_4_2"], "1.4.2")
            self.assertEqual(fit["amendment_1_4_3"], "1.4.3")
            self.assertEqual(fit["undated_day_terms"], "dropped")
            self.assertFalse(fit["day_terms_inestimable"])
            self.assertEqual(fit["day_terms_dropped"], [])
            self.assertEqual(fit["state"]["n_train_days"], 2)
            self.assertEqual(fit["state"]["train_days"], ["2024-05-01", "2024-06-01"])
            self.assertEqual(
                fit["candidates"]["error_scaling"]["feature_names"],
                ["log_train_window_count", "day_distance"],
            )
            self.assertFalse(fit["eval_files_read"])
            self.assertEqual(set(fit["val_nll"]), set(CANDIDATES))
            for value in fit["val_nll"].values():
                self.assertTrue(np.isfinite(value))
            claimed = honesty(**fit)
            self.assertFalse(claimed["effect_claimed"])
            self.assertFalse(claimed["valid_scientific_claim"])
            self.assertFalse(claimed["dates_invented"])
            self.assertFalse(claimed["competition_holdout_read"])
            os.makedirs(os.path.join(out, "noise"), exist_ok=True)
            with open(os.path.join(out, "noise", "fit.json"), "w", encoding="utf-8") as handle:
                json.dump(fit, handle)
            score = score_frozen_directory(out, fit_dates)
            self.assertEqual(score["day_terms_dropped_bases"], ["sub-3_ses-11_task-Sherlock1_run-1"])
            self.assertFalse(score["day_terms_inestimable"])
            self.assertEqual(score["day_terms_dropped"], [])
            for value in score["eval_nll"].values():
                self.assertTrue(np.isfinite(value))
            self.assertFalse(honesty(**score)["effect_claimed"])
            with self.assertRaises(ProtocolError) as caught:
                fit_noise_directory(out, {"sub-0_ses-12_task-Sherlock1_run-1": "2024-06-01"})
            self.assertEqual(
                str(caught.exception),
                missing_acquisition_date("sub-0_ses-11_task-Sherlock1_run-1"),
            )
            loaded = _rows_from_forward(
                os.path.join(out, "forward", "selection", "sub-10_ses-11_task-Sherlock1_run-1.npz"),
                "selection",
                fit_dates,
            )
            self.assertIsNone(loaded["rows"][0]["date"])
            self.assertGreater(len(loaded["rows"]), 0)

    def test_shared_sherlock_one_calendar_day_fits_without_inventing_a_date(self):
        """ses-11 and ses-12 run-2 are both 2024-10-04. Clock time is not a second day.

        A non-shared subject-0 scan with another date stays out of the coefficient rows.
        """
        with tempfile.TemporaryDirectory() as tmp:
            self._write_scan_time(
                tmp, "Sherlock1", 0, "11",
                "sub-0_ses-11_task-Sherlock1_run-2", "2024-10-04T14:14:01.362854Z",
            )
            self._write_scan_time(
                tmp, "Sherlock1", 0, "12",
                "sub-0_ses-12_task-Sherlock1_run-2", "2024-10-04T14:54:33.544517Z",
            )
            self._write_scan_time(
                tmp, "Sherlock1", 0, "1",
                "sub-0_ses-1_task-Sherlock1_run-1", "2024-07-31T12:12:07.029345Z",
            )
            dates = read_dates(tmp)
            self.assertEqual(dates["sub-0_ses-11_task-Sherlock1_run-2"], "2024-10-04")
            self.assertEqual(dates["sub-0_ses-12_task-Sherlock1_run-2"], "2024-10-04")
            self.assertEqual(dates["sub-0_ses-1_task-Sherlock1_run-1"], "2024-07-31")
            out = os.path.join(tmp, "out")
            shared = (
                ("sub-0_ses-11_task-Sherlock1_run-2", [10.0, 60.0, 62.0, 90.0]),
                ("sub-0_ses-12_task-Sherlock1_run-2", [12.0, 58.0, 61.0, 95.0]),
            )
            written = []
            for base, times in shared:
                self._write_cached_forward(
                    os.path.join(out, "forward", "noise_pool"),
                    base, 0, [0, 1, 2, 3], times,
                )
                written.append(os.path.join(out, "forward", "noise_pool", base + ".npz"))
            self._write_cached_forward(
                os.path.join(out, "forward", "train"),
                "sub-0_ses-1_task-Sherlock1_run-1", 0, [0, 1, 2, 3], [5.0, 15.0, 25.0, 35.0],
            )
            for subject in (1, 5, 10):
                base = f"sub-{subject}_ses-11_task-Sherlock1_run-1"
                self._write_cached_forward(
                    os.path.join(out, "forward", "selection"),
                    base, subject, [0, 1, 2, 3], [20.0, 30.0, 40.0, 50.0],
                )
            eval_base = "sub-3_ses-11_task-Sherlock1_run-1"
            self._write_cached_forward(
                os.path.join(out, "forward", "eval"),
                eval_base, 3, [0, 1, 2, 3], [15.0, 25.0, 35.0, 45.0],
            )
            before = {path: self._sha(path) for path in written}
            fit = fit_noise_directory(out, dates)
            self.assertEqual(before, {path: self._sha(path) for path in written})
            self.assertEqual(fit["train_subjects"], [0])
            self.assertEqual(fit["eval_subjects_in_train"], [])
            self.assertEqual(fit["selection_subjects"], [1, 5, 10])
            self.assertFalse(fit["selection_rows_used_for_coefficients"])
            self.assertEqual(fit["residual_fit_population"], "subject0_shared_sherlock_Q3")
            self.assertTrue(fit["day_terms_inestimable"])
            self.assertEqual(fit["day_terms_dropped"], ["day_rank", "day_distance"])
            self.assertEqual(fit["state"]["train_days"], ["2024-10-04"])
            self.assertEqual(fit["state"]["n_train_days"], 1)
            self.assertNotIn("2024-07-31", fit["state"]["train_days"])
            self.assertEqual(
                fit["state"]["names"],
                ["run_fraction", "artifact_rms", "bandpower", "log_train_window_count"],
            )
            self.assertEqual(fit["candidates"]["error_scaling"]["feature_names"], ["log_train_window_count"])
            self.assertEqual(fit["candidates"]["learned_state"]["feature_names"], fit["state"]["names"])
            self.assertFalse(fit["dates_invented"])
            self.assertFalse(fit["session_ordinal_used_as_day"])
            self.assertFalse(fit["eval_files_read"])
            self.assertEqual(set(fit["val_nll"]), set(CANDIDATES))
            for value in fit["val_nll"].values():
                self.assertTrue(np.isfinite(value))
            claimed = honesty(**fit)
            self.assertFalse(claimed["effect_claimed"])
            self.assertFalse(claimed["valid_scientific_claim"])
            self.assertTrue(claimed["exploratory"])
            self.assertTrue(claimed["not_confirmatory"])
            self.assertFalse(claimed["dates_invented"])
            os.makedirs(os.path.join(out, "noise"), exist_ok=True)
            with open(os.path.join(out, "noise", "fit.json"), "w", encoding="utf-8") as handle:
                json.dump(fit, handle)
            score = score_frozen_directory(out, dates)
            self.assertTrue(score["day_terms_inestimable"])
            self.assertEqual(score["day_terms_dropped"], ["day_rank", "day_distance"])
            self.assertEqual(score["day_terms_dropped_bases"], [eval_base])
            for value in score["eval_nll"].values():
                self.assertTrue(np.isfinite(value))
            self.assertFalse(honesty(**score)["valid_scientific_claim"])

    def _write_scan_time(self, root, dataset, subject, session, base, acq_time):
        folder = os.path.join(root, dataset, f"sub-{subject}", f"ses-{session}")
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, f"sub-{subject}_ses-{session}_scans.tsv")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("filename\tacq_time\n")
            handle.write(f"meg/{base}_meg.fif\t{acq_time}\n")
        return path

    @staticmethod
    def _sha(path):
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            digest.update(handle.read())
        return digest.hexdigest()

    def test_honesty_merge_keeps_the_non_claim_value(self):
        agreed = merge_honesty_sources(
            {"not_confirmatory": True, "independent_confirmation_required": True, "eval_rows": 4},
            {"not_confirmatory": True, "independent_confirmation_required": True, "effect_claimed": False},
            checkpoint="ckpt",
        )
        self.assertEqual(agreed["checkpoint"], "ckpt")
        self.assertEqual(agreed["eval_rows"], 4)
        self.assertTrue(agreed["not_confirmatory"])
        disagreed = merge_honesty_sources(
            {
                "not_confirmatory": False,
                "effect_claimed": True,
                "exploratory": False,
                "pooled_with_libribrain": True,
                "independent_confirmation_required": False,
            },
            {
                "not_confirmatory": True,
                "effect_claimed": False,
                "exploratory": True,
                "pooled_with_libribrain": False,
                "independent_confirmation_required": True,
            },
        )
        self.assertTrue(disagreed["not_confirmatory"])
        self.assertFalse(disagreed["effect_claimed"])
        self.assertTrue(disagreed["exploratory"])
        self.assertFalse(disagreed["pooled_with_libribrain"])
        self.assertTrue(disagreed["independent_confirmation_required"])
        # The merge does not rewrite a value the sources already agree on.
        self.assertTrue(merge_honesty_sources(
            {"pooled_with_libribrain": True},
            {"pooled_with_libribrain": True},
        )["pooled_with_libribrain"])
        wrapped = honesty(
            {"pooled_with_libribrain": True, "not_confirmatory": False, "effect_claimed": True, "eval_rows": 4},
            {"pooled_with_libribrain": False, "not_confirmatory": True, "valid_scientific_claim": True, "eval_rows": 4},
        )
        self.assertFalse(wrapped["pooled_with_libribrain"])
        self.assertTrue(wrapped["not_confirmatory"])
        self.assertFalse(wrapped["effect_claimed"])
        self.assertFalse(wrapped["valid_scientific_claim"])
        self.assertTrue(wrapped["exploratory"])
        self.assertTrue(wrapped["independent_confirmation_required"])
        self.assertEqual(wrapped["eval_rows"], 4)
        self.assertEqual(wrapped["amendment_1_4_4"], "1.4.4")
        self.assertFalse(wrapped["competition_holdout_read"])
        self.assertFalse(wrapped["dates_invented"])
        self.assertFalse(wrapped["real_meg_scientific_validation"])
        smoke = honesty(
            {"steps": 2, "effect_claimed": True, "not_confirmatory": False, "schedule": "smoke"},
            stage="smoke",
            next_action="smoke_is_not_the_substantive_schedule",
        )
        self.assertEqual(smoke["stage"], "smoke")
        self.assertEqual(smoke["next_action"], "smoke_is_not_the_substantive_schedule")
        self.assertEqual(smoke["steps"], 2)
        self.assertFalse(smoke["effect_claimed"])
        self.assertTrue(smoke["not_confirmatory"])
        with self.assertRaises(ProtocolError) as caught:
            merge_honesty_sources({"primary_bacc10_50": 0.2}, {"primary_bacc10_50": 0.5})
        self.assertIn("primary_bacc10_50", str(caught.exception))
        with self.assertRaises(ProtocolError):
            merge_honesty_sources({"dates_invented": True}, {"dates_invented": False})
        with self.assertRaises(ProtocolError):
            merge_honesty_sources({"competition_holdout_read": False}, {"competition_holdout_read": True})
        with self.assertRaises(ProtocolError):
            merge_honesty_sources(
                {"temperature_applied_to_reported_logits": False},
                {"temperature_applied_to_reported_logits": True},
            )
        with self.assertRaises(ProtocolError):
            honesty({"not_confirmatory": "yes"}, {"not_confirmatory": True})
        with self.assertRaises(ProtocolError):
            honesty({"stage": "partial-train"}, stage="smoke")
        report = honesty(stage="partial-train")
        _merge_report(report, {
            "effect_claimed": True,
            "valid_scientific_claim": True,
            "not_confirmatory": False,
            "steps": 2,
        })
        self.assertEqual(report["stage"], "partial-train")
        self.assertFalse(report["effect_claimed"])
        self.assertFalse(report["valid_scientific_claim"])
        self.assertTrue(report["not_confirmatory"])
        self.assertEqual(report["steps"], 2)
        with self.assertRaises(ProtocolError):
            _merge_report(honesty(stage="smoke"), {"stage": "partial-train", "steps": 1})

    def test_run_next_score_frozen_and_later_stages_write_json(self):
        """The live crash: score and measurement_status both set not_confirmatory.

        Fixture forwards are already on disk, as they were after the eight
        successful forward launches. run_next then writes score_frozen.json,
        takes the partial-train branch, and writes score_partial.json.
        """

        class _Array:
            def __init__(self, value):
                self._value = value

            def detach(self):
                return self

            def cpu(self):
                return self

            def numpy(self):
                return self._value

        class _Head:
            def __init__(self):
                self.weight = _Array(np.zeros((50, 4), dtype=np.float64))
                self.bias = _Array(np.zeros(50, dtype=np.float64))

        class _Model:
            def __init__(self):
                self.head_ce = _Head()

        with tempfile.TemporaryDirectory() as tmp:
            data_root = os.path.join(tmp, "data")
            cache_cont = os.path.join(tmp, "cache", "cont")
            directory = os.path.join(tmp, "sdr_out", "strong_decoder_1_4")
            root = os.path.join(tmp, "repo")
            os.makedirs(cache_cont)
            os.makedirs(root)
            ckpt = os.path.join(tmp, "fixture-checkpoint.bin")
            with open(ckpt, "wb") as handle:
                handle.write(b"fixture-checkpoint-not-deep-c")
            sha = self._sha(ckpt)
            rows = (
                ("noise_pool", "sub-0_ses-11_task-Sherlock1_run-2", 0, "11",
                 [0, 1, 2, 3], [10.0, 60.0, 62.0, 90.0], "2024-10-04T14:14:01.362854Z"),
                ("noise_pool", "sub-0_ses-12_task-Sherlock1_run-2", 0, "12",
                 [0, 1, 2, 3], [12.0, 58.0, 61.0, 95.0], "2024-10-04T14:54:33.544517Z"),
                ("selection", "sub-1_ses-11_task-Sherlock1_run-1", 1, None,
                 [0, 1, 2, 3], [20.0, 30.0, 40.0, 50.0], None),
                ("selection", "sub-5_ses-11_task-Sherlock1_run-1", 5, None,
                 [0, 1, 2, 3], [20.0, 30.0, 40.0, 50.0], None),
                ("selection", "sub-10_ses-11_task-Sherlock1_run-1", 10, None,
                 [0, 1, 2, 3], [20.0, 30.0, 40.0, 50.0], None),
                ("eval", "sub-3_ses-11_task-Sherlock1_run-1", 3, None,
                 [0, 1, 2, 3], [15.0, 25.0, 35.0, 45.0], None),
                ("train", "sub-0_ses-1_task-Sherlock1_run-1", 0, "1",
                 [0, 1, 2, 3], [5.0, 15.0, 25.0, 35.0], "2024-07-31T12:12:07.029345Z"),
            )
            written = []
            for phase, base, subject, session, labels, times, acq in rows:
                if acq is not None:
                    self._write_scan_time(data_root, "Sherlock1", subject, session, base, acq)
                self._write_cached_forward(
                    os.path.join(directory, "forward", phase), base, subject, labels, times, sha,
                )
                written.append(os.path.join(directory, "forward", phase, base + ".npz"))
                np.save(os.path.join(cache_cont, base + ".npy"), np.zeros(2, np.float32))
                np.savez_compressed(os.path.join(cache_cont, base + "_ev.npz"), label=np.array([0]))
            dates = read_dates(data_root)
            self.assertEqual(dates["sub-0_ses-11_task-Sherlock1_run-2"], "2024-10-04")
            self.assertEqual(dates["sub-0_ses-12_task-Sherlock1_run-2"], "2024-10-04")
            fit = fit_noise_directory(directory, dates)
            self.assertTrue(fit["day_terms_inestimable"])
            self.assertEqual(fit["day_terms_dropped"], ["day_rank", "day_distance"])
            os.makedirs(os.path.join(directory, "noise"), exist_ok=True)
            fit_path = os.path.join(directory, "noise", "fit.json")
            with open(fit_path, "w", encoding="utf-8") as handle:
                json.dump(fit, handle)
            fit_sha = self._sha(fit_path)
            before = {path: self._sha(path) for path in written}
            score = score_frozen_directory(directory, dates)
            status = measurement_status({
                "decoder": "src.v3lib.Model",
                "checkpoint_bytes": os.path.getsize(ckpt),
                "objective": score["objective"],
                "candidates": score["candidates_scored"],
                "eval_subjects_in_train": score["eval_subjects_in_train"],
                "subject0_frozen_holdout": False,
                "temperature_applied_to_reported_logits": False,
                "in_sample_trunk_trained_residual": score.get("in_sample_trunk_trained_residual"),
                "eval_rows": score["eval_rows"],
                "fit_rows": score.get("fit_rows", 0),
                "selection_rows": score["selection_rows"],
                "lock_held": True,
                "fixture": False,
            })
            overlap = set(score) & set(status)
            self.assertIn("not_confirmatory", overlap)
            self.assertIn("independent_confirmation_required", overlap)
            with self.assertRaises(TypeError):
                honesty(**score, **status, checkpoint=ckpt)
            original = dict(SUBSTANTIVE_SCHEDULE)
            tiny = {
                "name": "substantive_head_finetune",
                "max_steps": 2,
                "steps_per_launch": 2,
                "min_steps": 2,
                "patience_checks": None,
                "substantive": True,
            }
            env = {
                "LIBRIBRAIN_SDR_OUT": os.path.join(tmp, "sdr_out"),
                "LIBRIBRAIN_CACHE": os.path.join(tmp, "cache"),
                "LIBRIBRAIN_DATA": data_root,
                "LB_STRONG_CKPT": ckpt,
            }
            stages = []
            try:
                SUBSTANTIVE_SCHEDULE.clear()
                SUBSTANTIVE_SCHEDULE.update(tiny)
                with unittest.mock.patch.dict(os.environ, env, clear=False):
                    with unittest.mock.patch(
                        "campaigns.sdr_uncertainty.strong_decoder_runner.load_strong_decoder",
                        return_value=_Model(),
                    ):
                        for _ in range(6):
                            report = run_next(root)
                            stages.append(report["stage"])
                            self.assertFalse(report["effect_claimed"])
                            self.assertFalse(report["valid_scientific_claim"])
                            self.assertTrue(report["not_confirmatory"])
                            self.assertTrue(report["exploratory"])
                            self.assertTrue(report["independent_confirmation_required"])
                            self.assertFalse(report["real_meg_scientific_validation"])
                            self.assertEqual(report["amendment_1_4_4"], "1.4.4")
                            if report["stage"] == "stop":
                                self.assertTrue(report["runs_completed"])
                                break
                        else:
                            self.fail(f"did not stop: {stages}")
            finally:
                SUBSTANTIVE_SCHEDULE.clear()
                SUBSTANTIVE_SCHEDULE.update(original)
            self.assertEqual(stages, ["score-frozen", "partial-train", "score-partial", "stop"])
            self.assertEqual(before, {path: self._sha(path) for path in written})
            self.assertEqual(fit_sha, self._sha(fit_path))
            with open(os.path.join(directory, "score_frozen.json"), encoding="utf-8") as handle:
                frozen = json.load(handle)
            with open(os.path.join(directory, "score_partial.json"), encoding="utf-8") as handle:
                partial = json.load(handle)
            for saved in (frozen, partial):
                self.assertFalse(saved["effect_claimed"])
                self.assertFalse(saved["valid_scientific_claim"])
                self.assertTrue(saved["not_confirmatory"])
                self.assertTrue(saved["exploratory"])
                self.assertTrue(saved["independent_confirmation_required"])
                self.assertFalse(saved["dates_invented"])
                self.assertFalse(saved["competition_holdout_read"])
                self.assertFalse(saved["real_meg_scientific_validation"])
                self.assertEqual(saved["amendment_1_4_4"], "1.4.4")
            self.assertEqual(frozen["checkpoint"], ckpt)
            self.assertTrue(os.path.isfile(os.path.join(directory, "partial", "state.npz")))
            self.assertTrue(os.path.isfile(os.path.join(directory, "partial", "best_val.npz")))
            self.assertTrue(os.path.isfile(os.path.join(directory, "partial", "val_curve.json")))
            self.assertTrue(frozen["day_terms_inestimable"])
            self.assertEqual(frozen["day_terms_dropped"], ["day_rank", "day_distance"])
            self.assertFalse(frozen["measurement_complete"])
            self.assertFalse(frozen["fifty_class_metric_defined"])
            self.assertIsNone(frozen["primary_bacc10_50"])
            self.assertNotIn("bacc10_50", frozen["eligible_metrics"])
            self.assertEqual(set(frozen["eval_nll"]), set(CANDIDATES))
            for value in frozen["eval_nll"].values():
                self.assertTrue(np.isfinite(value))
            self.assertEqual(frozen["eval_rows"], 4)
            self.assertEqual(frozen["oov_excluded"]["sub-3_ses-11_task-Sherlock1_run-1"]["excluded_oov"], 0)
            self.assertFalse(partial["fifty_class_metric_defined"])
            self.assertEqual(partial["steps"], 2)
            self.assertEqual(set(partial["primary_bacc10_50"]), {"uniform", "random_covariate", "learned_state"})
            self.assertTrue(partial["not_a_full_retrain"])
            self.assertEqual(SUBSTANTIVE_SCHEDULE["max_steps"], original["max_steps"])

    def test_undated_rows_with_dropped_day_terms_reach_partial_and_score(self):
        """The iter-18 blocker: day terms already dropped, then partial-train.

        Head-train rows have no scans.tsv date. Their stored bandpower is
        non-finite because the trunk zero-filled that sample and the summary
        did not. The npz is not rewritten. The noise fit is not refit.
        """

        class _Array:
            def __init__(self, value):
                self._value = value

            def detach(self):
                return self

            def cpu(self):
                return self

            def numpy(self):
                return self._value

        class _Head:
            def __init__(self):
                self.weight = _Array(np.zeros((50, 4), dtype=np.float64))
                self.bias = _Array(np.zeros(50, dtype=np.float64))

        class _Model:
            def __init__(self):
                self.head_ce = _Head()

        with tempfile.TemporaryDirectory() as tmp:
            data_root = os.path.join(tmp, "data")
            cache_cont = os.path.join(tmp, "cache", "cont")
            directory = os.path.join(tmp, "sdr_out", "strong_decoder_1_4")
            root = os.path.join(tmp, "repo")
            os.makedirs(cache_cont)
            os.makedirs(root)
            ckpt = os.path.join(tmp, "fixture-checkpoint.bin")
            with open(ckpt, "wb") as handle:
                handle.write(b"fixture-checkpoint-not-deep-c")
            sha = self._sha(ckpt)
            rows = (
                ("noise_pool", "sub-0_ses-11_task-Sherlock1_run-2", 0, "11",
                 [0, 1, 2, 3], [10.0, 60.0, 62.0, 90.0], "2024-10-04T14:14:01.362854Z"),
                ("noise_pool", "sub-0_ses-12_task-Sherlock1_run-2", 0, "12",
                 [0, 1, 2, 3], [12.0, 58.0, 61.0, 95.0], "2024-10-04T14:54:33.544517Z"),
                ("selection", "sub-1_ses-11_task-Sherlock1_run-1", 1, None,
                 [0, 1, 2, 3], [20.0, 30.0, 40.0, 50.0], None),
                ("selection", "sub-5_ses-11_task-Sherlock1_run-1", 5, None,
                 [0, 1, 2, 3], [20.0, 30.0, 40.0, 50.0], None),
                ("selection", "sub-10_ses-11_task-Sherlock1_run-1", 10, None,
                 [0, 1, 2, 3], [20.0, 30.0, 40.0, 50.0], None),
                ("eval", "sub-3_ses-11_task-Sherlock1_run-1", 3, None,
                 [0, 1, 2, 3], [15.0, 25.0, 35.0, 45.0], None),
                ("train", "sub-0_ses-1_task-Sherlock1_run-1", 0, None,
                 [0, 1, 2, 3], [2.0, 2.2, 2.4, 2.6], None),
            )
            train_base = "sub-0_ses-1_task-Sherlock1_run-1"
            meg = np.zeros((306, 500), dtype=np.float32)
            meg[0, 80] = np.nan
            meg[0, 100] = 5.0
            events = {
                "onset_s": np.array([2.0, 2.2, 2.4, 2.6]),
                "label": np.array([0, 1, 2, 3]),
                "sentenceidx": np.array([0, 0, 0, 0]),
                "sent_start_s": np.array([1.0, 1.0, 1.0, 1.0]),
                "sent_end_s": np.array([3.5, 3.5, 3.5, 3.5]),
            }
            fresh = recording_covariates(meg, events)
            self.assertTrue(np.isfinite(fresh["bandpower"]).all())
            self.assertGreater(float(np.max(fresh["artifact_rms"])), 0.0)
            written = []
            for phase, base, subject, session, labels, times, acq in rows:
                if acq is not None:
                    self._write_scan_time(data_root, "Sherlock1", subject, session, base, acq)
                n = len(labels)
                folder = os.path.join(directory, "forward", phase)
                os.makedirs(folder, exist_ok=True)
                band = np.linspace(0.1, 0.4, n)
                rms = np.linspace(0.5, 1.5, n)
                frac = np.linspace(0.1, 0.9, n)
                if base == train_base:
                    band = np.full(n, np.nan)
                np.savez_compressed(
                    os.path.join(folder, base + ".npz"),
                    base=np.array(base),
                    subject=np.int64(subject),
                    embedding=np.zeros((n, 4), np.float32),
                    logits=np.zeros((n, 50), np.float32),
                    label=np.asarray(labels, dtype=np.int64),
                    run_fraction=frac,
                    artifact_rms=rms,
                    bandpower=band,
                    timechapter=np.asarray(times, dtype=np.float64),
                    checkpoint_sha=np.array(sha),
                    timing=np.array("zero"),
                )
                written.append(os.path.join(folder, base + ".npz"))
                if base == train_base:
                    np.save(os.path.join(cache_cont, base + ".npy"), meg)
                    np.savez_compressed(os.path.join(cache_cont, base + "_ev.npz"), **events)
                else:
                    np.save(os.path.join(cache_cont, base + ".npy"), np.zeros(2, np.float32))
                    np.savez_compressed(os.path.join(cache_cont, base + "_ev.npz"), label=np.array([0]))
            dates = read_dates(data_root)
            self.assertNotIn(train_base, dates)
            fit = fit_noise_directory(directory, dates)
            self.assertTrue(fit["day_terms_inestimable"])
            self.assertEqual(fit["day_terms_dropped"], ["day_rank", "day_distance"])
            os.makedirs(os.path.join(directory, "noise"), exist_ok=True)
            fit_path = os.path.join(directory, "noise", "fit.json")
            with open(fit_path, "w", encoding="utf-8") as handle:
                json.dump(fit, handle)
            frozen_path = os.path.join(directory, "score_frozen.json")
            with open(frozen_path, "w", encoding="utf-8") as handle:
                json.dump({"kept": True}, handle)
            fit_sha = self._sha(fit_path)
            frozen_sha = self._sha(frozen_path)
            before = {path: self._sha(path) for path in written}
            env = {
                "LIBRIBRAIN_SDR_OUT": os.path.join(tmp, "sdr_out"),
                "LIBRIBRAIN_CACHE": os.path.join(tmp, "cache"),
                "LIBRIBRAIN_DATA": data_root,
                "LB_STRONG_CKPT": ckpt,
            }
            with unittest.mock.patch.dict(os.environ, env, clear=False):
                loaded = _rows_from_forward(
                    os.path.join(directory, "forward", "train", train_base + ".npz"),
                    "train",
                    dates,
                    require_acquisition_date=False,
                )
            self.assertIsNone(loaded["rows"][0]["date"])
            self.assertTrue(np.allclose(
                [row["artifact_rms"] for row in loaded["rows"]],
                np.linspace(0.5, 1.5, 4),
            ))
            self.assertTrue(np.allclose(
                [row["bandpower"] for row in loaded["rows"]],
                fresh["bandpower"],
            ))
            self.assertFalse(np.allclose(fresh["bandpower"], np.linspace(0.1, 0.4, 4)))
            with np.load(os.path.join(directory, "forward", "train", train_base + ".npz")) as handle:
                self.assertTrue(np.isnan(handle["bandpower"]).all())
            with self.assertRaises(ProtocolError) as caught:
                transform_with_state(
                    [{**row, "date": "11"} for row in loaded["rows"]],
                    fit["state"],
                )
            self.assertIn("not YYYY-MM-DD", str(caught.exception))
            self.assertNotIn("day_rank", str(caught.exception))
            original = dict(SUBSTANTIVE_SCHEDULE)
            tiny = {
                "name": "substantive_head_finetune",
                "max_steps": 2,
                "steps_per_launch": 2,
                "min_steps": 2,
                "patience_checks": None,
                "substantive": True,
            }
            stages = []

            def _fail_refit(*_args, **_kwargs):
                raise AssertionError("noise fit was refit")

            try:
                SUBSTANTIVE_SCHEDULE.clear()
                SUBSTANTIVE_SCHEDULE.update(tiny)
                with unittest.mock.patch.dict(os.environ, env, clear=False):
                    with unittest.mock.patch(
                        "campaigns.sdr_uncertainty.strong_decoder_runner.load_strong_decoder",
                        return_value=_Model(),
                    ):
                        with unittest.mock.patch(
                            "campaigns.sdr_uncertainty.strong_decoder_runner.assemble_features",
                            side_effect=_fail_refit,
                        ):
                            for _ in range(4):
                                report = run_next(root)
                                stages.append(report["stage"])
                                self.assertFalse(report["effect_claimed"])
                                self.assertFalse(report["valid_scientific_claim"])
                                self.assertTrue(report["not_confirmatory"])
                                self.assertTrue(report["exploratory"])
                                self.assertTrue(report["independent_confirmation_required"])
                                self.assertFalse(report["dates_invented"])
                                self.assertFalse(report["session_ordinal_used_as_day"])
                                self.assertEqual(report["amendment_1_4_4"], "1.4.4")
                                self.assertEqual(report["amendment_1_4_5"], "1.4.5")
                                if report["stage"] == "partial-train":
                                    self.assertTrue(report["day_terms_inestimable"])
                                    self.assertEqual(report["day_terms_dropped"], ["day_rank", "day_distance"])
                                    self.assertNotIn("day_rank", report["active_features"])
                                if report["stage"] == "stop":
                                    break
            finally:
                SUBSTANTIVE_SCHEDULE.clear()
                SUBSTANTIVE_SCHEDULE.update(original)
            self.assertEqual(stages, ["partial-train", "score-partial", "stop"])
            self.assertEqual(before, {path: self._sha(path) for path in written})
            self.assertEqual(fit_sha, self._sha(fit_path))
            self.assertEqual(frozen_sha, self._sha(frozen_path))
            with open(os.path.join(directory, "score_partial.json"), encoding="utf-8") as handle:
                partial = json.load(handle)
            self.assertTrue(partial["day_terms_inestimable"])
            self.assertEqual(partial["day_terms_dropped"], ["day_rank", "day_distance"])
            self.assertNotIn("day_rank", partial["active_features"])
            self.assertNotIn("day_distance", partial["active_features"])
            self.assertFalse(partial["dates_invented"])
            self.assertFalse(partial["effect_claimed"])
            self.assertTrue(partial["not_confirmatory"])
            self.assertTrue(partial["independent_confirmation_required"])
            self.assertEqual(partial["amendment_1_4_5"], "1.4.5")
            with np.load(os.path.join(directory, "forward", "train", train_base + ".npz")) as handle:
                self.assertTrue(np.isnan(handle["bandpower"]).all())

            def _fake_forward(_directory, _root, base, _phase, *, timing="zero", write=True):
                self.assertFalse(write)
                return {
                    "result": {
                        "logits": np.zeros((4, 50)),
                        "label": np.array([0, 1, 2, 3]),
                    }
                }

            with unittest.mock.patch.dict(os.environ, env, clear=False):
                with unittest.mock.patch(
                    "campaigns.sdr_uncertainty.strong_decoder_runner.read_dates",
                    side_effect=AssertionError("timing control re-read dates"),
                ):
                    with unittest.mock.patch(
                        "campaigns.sdr_uncertainty.strong_decoder_runner.execute_forward",
                        side_effect=_fake_forward,
                    ):
                        with unittest.mock.patch(
                            "campaigns.sdr_uncertainty.strong_decoder_runner.assemble_features",
                            side_effect=_fail_refit,
                        ):
                            timed = run_timing(root)
            self.assertTrue(timed["day_terms_inestimable"])
            self.assertEqual(timed["day_terms_dropped"], ["day_rank", "day_distance"])
            self.assertNotIn("day_rank", timed["active_features"])
            self.assertFalse(timed["dates_invented"])
            self.assertFalse(timed["session_ordinal_used_as_day"])
            self.assertFalse(timed["effect_claimed"])
            self.assertTrue(timed["not_confirmatory"])
            self.assertTrue(timed["independent_confirmation_required"])
            self.assertEqual(timed["primary_timing"], "zero")
            self.assertEqual(fit_sha, self._sha(fit_path))
            self.assertEqual(before, {path: self._sha(path) for path in written})

    def test_kept_day_terms_still_require_a_date_and_a_finite_sensor(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "out")
            self._write_cached_forward(
                os.path.join(out, "forward", "noise_pool"),
                "sub-0_ses-11_task-Sherlock1_run-2", 0, [0, 1, 2, 3], [10.0, 60.0, 62.0, 90.0],
            )
            self._write_cached_forward(
                os.path.join(out, "forward", "noise_pool"),
                "sub-0_ses-12_task-Sherlock1_run-2", 0, [0, 1, 2, 3], [12.0, 58.0, 61.0, 95.0],
            )
            for subject in (1, 5, 10):
                self._write_cached_forward(
                    os.path.join(out, "forward", "selection"),
                    f"sub-{subject}_ses-11_task-Sherlock1_run-1",
                    subject, [0, 1, 2, 3], [20.0, 30.0, 40.0, 50.0],
                )
            dates = {
                "sub-0_ses-11_task-Sherlock1_run-2": "2024-10-04",
                "sub-0_ses-12_task-Sherlock1_run-2": "2024-07-31",
            }
            fit = fit_noise_directory(out, dates)
            self.assertFalse(fit["day_terms_inestimable"])
            os.makedirs(os.path.join(out, "noise"), exist_ok=True)
            with open(os.path.join(out, "noise", "fit.json"), "w", encoding="utf-8") as handle:
                json.dump(fit, handle)
            train = "sub-0_ses-2_task-Sherlock1_run-1"
            self._write_cached_forward(
                os.path.join(out, "forward", "train"),
                train, 0, [0, 1, 2, 3], [1.0, 2.0, 3.0, 4.0],
            )
            with self.assertRaises(ProtocolError) as caught:
                partial_train_directory(out, tmp, 1, {}, schedule={
                    "name": "substantive_head_finetune",
                    "max_steps": 1,
                    "steps_per_launch": 1,
                    "min_steps": 1,
                    "patience_checks": None,
                    "substantive": True,
                })
            self.assertEqual(str(caught.exception), missing_acquisition_date(train))
            nan_out = os.path.join(tmp, "nan")
            self._write_cached_forward(
                os.path.join(nan_out, "forward", "noise_pool"),
                "sub-0_ses-11_task-Sherlock1_run-2", 0, [0, 1, 2, 3], [10.0, 60.0, 62.0, 90.0],
            )
            self._write_cached_forward(
                os.path.join(nan_out, "forward", "noise_pool"),
                "sub-0_ses-12_task-Sherlock1_run-2", 0, [0, 1, 2, 3], [12.0, 58.0, 61.0, 95.0],
            )
            for subject in (1, 5, 10):
                self._write_cached_forward(
                    os.path.join(nan_out, "forward", "selection"),
                    f"sub-{subject}_ses-11_task-Sherlock1_run-1",
                    subject, [0, 1, 2, 3], [20.0, 30.0, 40.0, 50.0],
                )
            one_day = {
                "sub-0_ses-11_task-Sherlock1_run-2": "2024-10-04",
                "sub-0_ses-12_task-Sherlock1_run-2": "2024-10-04",
            }
            dropped = fit_noise_directory(nan_out, one_day)
            self.assertTrue(dropped["day_terms_inestimable"])
            os.makedirs(os.path.join(nan_out, "noise"), exist_ok=True)
            with open(os.path.join(nan_out, "noise", "fit.json"), "w", encoding="utf-8") as handle:
                json.dump(dropped, handle)
            bare = "sub-0_ses-1_task-Sherlock1_run-1"
            n = 4
            folder = os.path.join(nan_out, "forward", "train")
            os.makedirs(folder, exist_ok=True)
            np.savez_compressed(
                os.path.join(folder, bare + ".npz"),
                base=np.array(bare),
                subject=np.int64(0),
                embedding=np.zeros((n, 4), np.float32),
                logits=np.zeros((n, 50), np.float32),
                label=np.arange(n),
                run_fraction=np.linspace(0.1, 0.9, n),
                artifact_rms=np.linspace(0.5, 1.5, n),
                bandpower=np.full(n, np.nan),
                timechapter=np.linspace(1.0, 4.0, n),
                checkpoint_sha=np.array("abc"),
                timing=np.array("zero"),
            )
            with unittest.mock.patch.dict(os.environ, {"LIBRIBRAIN_CACHE": os.path.join(tmp, "empty-cache")}, clear=False):
                with self.assertRaises(ProtocolError) as caught:
                    partial_train_directory(nan_out, tmp, 1, {}, schedule={
                        "name": "substantive_head_finetune",
                        "max_steps": 1,
                        "steps_per_launch": 1,
                        "min_steps": 1,
                        "patience_checks": None,
                        "substantive": True,
                    })
            self.assertEqual(str(caught.exception), "used feature bandpower is non-finite")
            with np.load(os.path.join(folder, bare + ".npz")) as handle:
                self.assertTrue(np.isnan(handle["bandpower"]).all())


if __name__ == "__main__":
    unittest.main()
