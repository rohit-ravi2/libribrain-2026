"""Strong-decoder amendment 1.4. Fixtures are not a training result and not a claim."""
from __future__ import annotations

import json
import os
import stat
import subprocess
import tempfile
import textwrap
import time
import unittest

import numpy as np

from campaigns.sdr_uncertainty.strong_decoder import (
    attest_checkpoint_dict,
    claim_from_decoder,
    is_strong_decoder,
    load_strong_decoder,
)
from campaigns.sdr_uncertainty.strong_decoder_runner import (
    assert_output_dir,
    fit_noise_directory,
    honesty,
    score_frozen_directory,
)
from campaigns.sdr_uncertainty.strong_forward import forward_recording, slice_window
from campaigns.sdr_uncertainty.strong_noise import (
    CANDIDATES,
    CHECKPOINT_BYTES,
    PARTIAL_ARMS,
    REJECTED_CANDIDATES,
    ProtocolError,
    arm_logits,
    assemble_features,
    assign_split,
    bacc10_50,
    bacc10_from_logits,
    dlogits_ce,
    fit_noise,
    frozen_trunk_role,
    gaussian_nll,
    initial_partial_arms,
    matched_partial_steps,
    measurement_claim,
    measurement_status,
    next_sessions,
    per_row_ce,
    row_temperature,
    select_on_validation,
    session_bootstrap,
    step_head,
    temperature_changes_top10,
    timing_inputs,
    transform_with_state,
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
        with self.assertRaises(ProtocolError):
            assemble_features(_rows("only", 0, "2024-02-01", "train") + _rows("same-day", 0, "2024-02-01", "train"))

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


if __name__ == "__main__":
    unittest.main()
