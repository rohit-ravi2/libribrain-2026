"""Strong-decoder noise prior. Fixtures are not a training result."""
from __future__ import annotations

import os
import subprocess
import unittest

import numpy as np

from campaigns.sdr_uncertainty.strong_decoder import (
    attest_checkpoint_dict,
    claim_from_decoder,
    is_strong_decoder,
    load_strong_decoder,
)
from campaigns.sdr_uncertainty.strong_noise import (
    CANDIDATES,
    REJECTED_CANDIDATES,
    ProtocolError,
    assert_inference_features,
    assign_split,
    bacc10_50,
    fit_noise,
    next_sessions,
    select_on_validation,
    session_bootstrap,
    timing_inputs,
    weighted_nll,
)


def _fake_ckpt():
    def weight(out, inn):
        return np.zeros((out, inn))

    enc = {"w": np.zeros((2000, 512))}
    return {
        "enc": enc,
        "ctx": {"w": np.zeros((1000, 256))},
        "head_ce": {"weight": weight(50, 256)},
        "head_t5": {"weight": weight(1024, 256)},
        "logit_s": np.zeros(1),
        "A": np.zeros((1, 306, 306)),
        "subs": [0],
        "log_prior": np.zeros(50),
        "args": {"d": 128, "D": 256, "maxw": 12, "t5": 1.0, "ctx": 1, "rel": 0, "ctx_layers": 4},
    }


class SplitAndNoiseTests(unittest.TestCase):
    def test_split_is_chronological_and_does_not_read_a_score(self):
        sessions = [{"date": f"2024-01-{day:02d}", "base": "run"} for day in range(2, 22)]
        sessions.append({"date": "2024-10-04", "base": "sherlock"})
        sessions.sort(key=lambda row: row["date"])
        out = assign_split(sessions)
        self.assertNotIn("2024-10-04", [row["date"] for row in out["test"]])
        self.assertLess(out["train"][-1]["date"], out["val"][0]["date"])
        self.assertLess(out["val"][-1]["date"], out["test"][0]["date"])
        self.assertFalse(out["eval_subjects_are_pristine_holdout"])
        self.assertEqual(out["transfer_label"], "historically_inspected_transfer")
        self.assertIn(3, out["historically_inspected_subjects"])

    def test_true_word_frequency_is_rejected_and_test_labels_do_not_change_the_fit(self):
        with self.assertRaises(ProtocolError):
            assert_inference_features(["word_frequency"])
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
        self.assertEqual(next_sessions(["s1", "s2", "s3"], ["s1"]), next_sessions(["s1", "s2", "s3"], ["s1"]))

    def test_code_path_is_v3_and_a_surrogate_cannot_claim(self):
        text = open(os.path.join(os.path.dirname(__file__), "..", "..", "src", "v3lib.py"), encoding="utf-8").read()
        train = open(os.path.join(os.path.dirname(__file__), "..", "..", "scripts", "train_v3.py"), encoding="utf-8").read()
        split = open(os.path.join(os.path.dirname(__file__), "..", "..", "src", "split2.py"), encoding="utf-8").read()
        self.assertIn("class Model", text)
        self.assertIn("head_t5", text)
        self.assertIn("SigLIP", train)
        self.assertIn("EVAL = (3, 6, 9, 12)", split)
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
        self.assertEqual(rejected["decoder_identity"], "rejected_not_v3lib.Model")
        self.assertFalse(claim_from_decoder(object(), metric_defined=False)["valid_scientific_claim"])
        with self.assertRaises(FileNotFoundError):
            load_strong_decoder("/no/such/deep_c_ep6.pt")

    def test_shell_takes_one_self_lock(self):
        root = os.path.join(os.path.dirname(__file__), "..", "..")
        script = open(os.path.join(root, "scripts", "sdr_uncertainty_strong_decoder.sh"), encoding="utf-8").read()
        self.assertEqual(script.count("acquire_lock.sh"), 1)
        self.assertEqual(script.count("release_lock.sh"), 1)
        self.assertNotIn("while ", script)
        self.assertNotIn("RuntimeMaxSec", script)
        self.assertIn("MemoryMax=4G", script)
        self.assertIn("MemorySwapMax=0", script)
        subprocess.run(["bash", "-n", os.path.join(root, "scripts", "sdr_uncertainty_strong_decoder.sh")], check=True)
        env = os.environ.copy()
        env["SDR_FORBID_LIVE_COMPUTE"] = "1"
        proc = subprocess.run(
            [os.environ.get("PYTHON", "python3"), "-m", "campaigns.sdr_uncertainty.strong_decoder_runner", "run"],
            cwd=root,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 2)
        self.assertIn("refusing", proc.stderr)


if __name__ == "__main__":
    unittest.main()
