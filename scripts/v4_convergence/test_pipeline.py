"""Meaningful numerical/protocol checks before any full paid cloud run."""
import argparse
import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from scripts.v4_convergence.core import (
    build, choose_validation, conditions, ema_update, gate_conditions, json_write,
    load_models, load_pack, plateau, sample, select_indices, target,
)
from scripts.v4_convergence.evaluate import test_jobs, validation_jobs

ROOT = Path(__file__).resolve().parents[2]
BASE = None
DATA = None
PROTOCOL = None


class ProtocolTests(unittest.TestCase):
    def test_pack_has_exact_images_echo_and_disjoint_patients(self):
        pids = []
        for split in ("train", "val", "test"):
            df = pd.read_csv(ROOT / f"data/processed/ax_t1_128/splits/{split}.csv")
            np.testing.assert_array_equal(DATA[f"{split}_echo"], df.echo_time_norm.to_numpy(np.float32))
            self.assertEqual(DATA[f"{split}_x"].dtype, np.float32)
            for i in (0, len(df) // 2, len(df) - 1):
                expected = np.clip(np.load(ROOT / df.iloc[i].image_path.replace("\\", "/")).astype(np.float32), 0, 1)
                np.testing.assert_array_equal(DATA[f"{split}_x"][i], expected)
            pids.append(set(DATA[f"{split}_pid"]))
        self.assertFalse(pids[0] & pids[1] or pids[0] & pids[2] or pids[1] & pids[2])

    def test_same_order_as_historical_selector(self):
        from scripts.run_explainability_batch import select_diverse_examples
        for split in ("val", "test"):
            df = pd.read_csv(ROOT / f"data/processed/ax_t1_128/splits/{split}.csv")
            expected = select_diverse_examples(df, 100, seed=42)
            idx = select_indices(DATA, split)
            np.testing.assert_array_equal(DATA[f"{split}_path"][idx], expected.image_path.to_numpy())

    def test_only_target_axis_changes(self):
        src = conditions(DATA, "val", select_indices(DATA, "val")[:4])
        for axis, key, other in (("manufacturer", "mfg", "fs"), ("field_strength", "fs", "mfg")):
            tgt = target(src, axis)
            np.testing.assert_array_equal(tgt[key], 1 - src[key])
            np.testing.assert_array_equal(tgt[other], src[other])
            np.testing.assert_array_equal(tgt["echo"], src["echo"])

    def test_gate_conditions_are_balanced_without_test_dependence(self):
        base = {k: v for k, v in DATA.items() if k.startswith("train_")}
        cond = gate_conditions(base, 100)
        for m in (0, 1):
            for f in (0, 1):
                self.assertEqual(int(((cond["mfg"] == m) & (cond["fs"] == f)).sum()), 25)

    def test_validation_constraints_and_no_test_selection(self):
        rows = [dict(axis="field_strength", t=100, w=3, ssim=.75, stability=1, hit=.2, gain=.1),
                dict(axis="field_strength", t=300, w=5, ssim=.3, stability=1, hit=1, gain=1)]
        chosen = choose_validation(rows, "field_strength", PROTOCOL)
        self.assertEqual(chosen["t"], 100)
        self.assertTrue(chosen["feasible"])
        rows[0]["ssim"] = .6
        self.assertFalse(choose_validation(rows, "field_strength", PROTOCOL)["feasible"])
        with self.assertRaises(ValueError):
            test_jobs(PROTOCOL, {})
        self.assertTrue(all(j["split"] == "val" for j in validation_jobs(PROTOCOL)))

    def test_immutable_output_and_plateau_rules(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "metric.json"
            json_write(p, {"value": 1})
            with self.assertRaises(FileExistsError):
                json_write(p, {"value": 2})
            self.assertEqual(json.loads(p.read_text())["value"], 1)
        rows = [dict(epoch=e, guidance=3, mean_mfg_hit=.5, mean_fs_hit=.5) for e in range(20, 111, 10)]
        self.assertFalse(plateau(rows, 90))
        self.assertTrue(plateau(rows, 110))

    def test_ema_is_separate_from_raw_and_moves_correctly(self):
        raw = torch.nn.Linear(2, 1)
        ema = copy.deepcopy(raw)
        before = ema.weight.detach().clone()
        with torch.no_grad():
            raw.weight.add_(2)
        ema_update(ema, raw, .75)
        torch.testing.assert_close(ema.weight, before + .5)
        torch.testing.assert_close(raw.weight, before + 2)

    def test_fid_translation_and_unbiased_mmd(self):
        from scripts.v4_convergence.analyze import fid, mmd
        from scipy.linalg import sqrtm
        rng = np.random.default_rng(13)
        x, y = rng.normal(size=(24, 8)), rng.normal(size=(31, 8))
        self.assertAlmostEqual(fid(x, x), 0., places=10)
        self.assertAlmostEqual(fid(x, x + 2), 8 * 4, places=10)
        a, b = np.cov(x, rowvar=False), np.cov(y, rowvar=False)
        expected = np.sum((x.mean(0) - y.mean(0)) ** 2) + np.trace(a + b - 2 * sqrtm(a @ b).real)
        self.assertAlmostEqual(fid(x, y), float(expected), places=9)
        self.assertAlmostEqual(mmd(x, y, 1.5), mmd(y, x, 1.5), places=12)

    def test_sampler_restores_checkpoint_without_changing_effective_weights(self):
        c, u, _, ck = load_models(ROOT / "checkpoints/diffusion/ax_t1_128/latest.pt", device="cpu")
        for name, tensor in u.state_dict().items():
            torch.testing.assert_close(tensor, ck["unet_state_dict"][name], rtol=0, atol=0)
        self.assertEqual(int(ck["epoch"]), 10)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA end-to-end probe needs a GPU")
    def test_real_validation_job_and_all_judge_loaders(self):
        from scripts.v4_convergence.evaluate import run_job
        from scripts.v4_convergence.core import judges, probabilities
        assets = BASE / "inputs/assets"
        with tempfile.TemporaryDirectory() as tmp:
            job = dict(kind="validation", split="val", axis="field_strength", t=3, w=5.,
                       name="unit_val_probe", n=4, offset=0)
            result = run_job(DATA, ROOT / "checkpoints/diffusion/ax_t1_128/latest.pt", assets, tmp, job, PROTOCOL)
            self.assertEqual(result["split"], "val")
            self.assertEqual(result["n"], 4)
            df = pd.read_csv(Path(tmp) / job["name"] / "metrics.csv")
            self.assertEqual(len(df), 12)
            self.assertTrue(np.isfinite(df[["ssim", "mad", "fs_gain"]]).all().all())
            with self.assertRaises(FileExistsError):
                run_job(DATA, ROOT / "checkpoints/diffusion/ax_t1_128/latest.pt", assets, tmp, job, PROTOCOL)
        for name, model in judges(assets).items():
            p = probabilities(model, DATA["val_x"][:2], original=name == "original_predictor")
            np.testing.assert_allclose(p["fs"].sum(1), 1, atol=1e-6)
        restricted = load_pack(BASE / "inputs/pack.npz", splits=("train",))
        self.assertTrue(all(k.startswith("train_") for k in restricted))

    def test_resume_and_ema_snapshots_keep_distinct_weights(self):
        ckpt = BASE / "remote/smoke/train/epoch_0011.pt"
        if not ckpt.exists():
            self.skipTest("Cloud smoke checkpoint not downloaded yet")
        c, u, _, ck = load_models(ckpt, device="cpu", raw=True)
        ec, eu, _, _ = load_models(ckpt, device="cpu")
        for name, tensor in u.state_dict().items():
            torch.testing.assert_close(tensor, ck["raw_unet_state_dict"][name], rtol=0, atol=0)
            torch.testing.assert_close(eu.state_dict()[name], ck["ema_unet_state_dict"][name], rtol=0, atol=0)
        self.assertTrue(any(not torch.equal(v, eu.state_dict()[k]) for k, v in u.state_dict().items()))
        self.assertIn("rng_cuda", ck)
        opt = torch.optim.AdamW(list(c.parameters()) + list(u.parameters()))
        opt.load_state_dict(ck["optimizer_state_dict"])
        self.assertTrue(len(opt.state) > 0)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA comparison needs a GPU")
    def test_sampler_matches_original_editor_and_preserves_rng_pairing(self):
        from src.inference.counterfactual_editor import CounterfactualEditor
        device = "cuda"
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        ed = CounterfactualEditor(device=device)
        c, u, sched, _ = load_models(ROOT / "checkpoints/diffusion/ax_t1_128/latest.pt")
        cond = conditions(DATA, "val", np.array([0, 1]))
        images = DATA["val_x"][:2]
        out = sample(c, u, sched, cond, [42, 43], 3, 5, images)
        for i in range(2):
            old = ed.generate_counterfactual(images[i], int(cond["mfg"][i]), int(cond["fs"][i]),
                                             float(cond["echo"][i]), 3, 5, 42 + i)["generated_01"]
            np.testing.assert_allclose(out[i], old, atol=3e-5, rtol=0)
        solo = sample(c, u, sched, {k: v[:1] for k, v in cond.items()}, [42], 3, 5, images[:1])
        np.testing.assert_allclose(solo[0], out[0], atol=3e-5, rtol=0)
        # Same seeded null call must be reproducible despite intervening calls.
        again = sample(c, u, sched, cond, [42, 43], 3, 5, images)
        np.testing.assert_array_equal(out, again)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run-id", required=True)
    args = p.parse_args()
    BASE = ROOT / "reports/v4_diffusion" / args.run_id
    DATA = load_pack(BASE / "inputs/pack.npz")
    PROTOCOL = json.loads((BASE / "protocol.json").read_text())
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(ProtocolTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    receipt = BASE / "local_preflight.json"
    attempt = 2
    while receipt.exists():
        receipt = BASE / f"local_preflight_{attempt:03d}.json"
        attempt += 1
    json_write(receipt, {"tests": result.testsRun, "failures": len(result.failures),
               "errors": len(result.errors), "skipped": len(result.skipped), "success": result.wasSuccessful()})
    raise SystemExit(0 if result.wasSuccessful() else 1)
