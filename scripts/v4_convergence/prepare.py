"""Prepare an immutable float32 data pack, config and preregistered protocol.

Run from the project root: python -m scripts.v4_convergence.prepare --run-id <id>
No cloud resources are used. Existing run folders are never overwritten.
"""
import argparse
import copy
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def prepare(run_id):
    if not run_id or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for c in run_id):
        raise ValueError("run-id must contain only lowercase letters, digits, underscores or hyphens")
    out = ROOT / "reports/v4_diffusion" / run_id
    out.mkdir(parents=True, exist_ok=False)
    (out / "README.md").write_text(
        "# B1/B3 run " + run_id + "\n\n"
        "Prepared by `python -m scripts.v4_convergence.prepare`. Preparation is not a training result.\n\n"
        "| File/folder | Producer and purpose |\n|---|---|\n"
        "| config.yaml, protocol.json | prepare.py: frozen configuration and selection protocol |\n"
        "| inputs/pack.npz, inputs/manifest.json | prepare.py: float32 images, labels, fixed EchoTime, identities and hashes |\n"
        "| inputs/assets/ | prepare.py: copies of original checkpoints used as inputs |\n"
        "| remote/ | modal_app.py fetch: cloud output snapshots, immutable training checkpoints, gate and B3 outputs |\n"
        "| smoke/ | modal_app.py: bounded preflight timing and checkpoint-resume check; not a scientific training run |\n"
        "| analysis/ | analyze.py: gate curves, patient-clustered CIs, paired tests, soundness, HF, FID/MMD |\n"
        "| local_preflight.json | test_pipeline.py: local numerical and protocol checks |\n"
        "| cloud_launch*.json, cloud_*.log | Modal launch metadata and captured CLI output |\n",
        encoding="utf-8")
    inputs = out / "inputs"
    inputs.mkdir()
    arrays, manifest, identities = {}, {}, {}
    for split in ("train", "val", "test"):
        path = ROOT / f"data/processed/ax_t1_128/splits/{split}.csv"
        df = pd.read_csv(path)
        if df.image_path.duplicated().any():
            raise ValueError(f"Duplicate image path in {split}")
        identities[split] = set(df.patient_id.astype(str))
        images = []
        for p in df.image_path:
            x = np.load(ROOT / p.replace("\\", "/")).astype(np.float32)
            if x.shape != (128, 128) or not np.isfinite(x).all():
                raise ValueError(f"Invalid image {p}")
            images.append(np.clip(x, 0, 1))
        arrays[f"{split}_x"] = np.stack(images)
        for suffix, col, dtype in (("mfg", "manufacturer_label", np.int64),
                                   ("fs", "field_strength_label", np.int64),
                                   ("echo", "echo_time_norm", np.float32),
                                   ("pid", "patient_id", str), ("path", "image_path", str)):
            arrays[f"{split}_{suffix}"] = df[col].to_numpy(dtype)
        if not np.isfinite(arrays[f"{split}_echo"]).all():
            raise ValueError("Nonfinite EchoTime")
        for axis in ("mfg", "fs"):
            if set(np.unique(arrays[f"{split}_{axis}"])) != {0, 1}:
                raise ValueError(f"Missing class: {split}/{axis}")
        manifest[split] = {"slices": len(df), "patients": len(identities[split]),
                           "csv_sha256": sha256(path), "image_dtype": "float32",
                           "joint_counts": {f"{m}_{f}": int(((df.manufacturer_label == m) &
                                (df.field_strength_label == f)).sum()) for m in (0, 1) for f in (0, 1)}}
        print(split, manifest[split], flush=True)
    if any(identities[a] & identities[b] for a, b in (("train", "val"), ("train", "test"), ("val", "test"))):
        raise ValueError("Patient overlap between splits")
    pack = inputs / "pack.npz"
    np.savez_compressed(pack, **arrays)
    manifest["pack_sha256"] = sha256(pack)
    manifest["pack_bytes"] = pack.stat().st_size
    assets = inputs / "assets"
    assets.mkdir()
    paths = {"initial.pt": ROOT / "checkpoints/diffusion/ax_t1_128/latest.pt",
             "predictor.pt": ROOT / "checkpoints/predictor/ax_t1_128/best.pt"}
    for family, infix in (("robust", ""), ("plain", "_plain")):
        for seed in range(3):
            paths[f"{family}{seed}.pt"] = ROOT / f"reports/v2_quickwins/robust_judge/judge_resnet18{infix}_seed{seed}.pt"
    manifest["assets"] = {}
    for name, source in paths.items():
        shutil.copyfile(source, assets / name)
        manifest["assets"][name] = {"source": source.relative_to(ROOT).as_posix(), "sha256": sha256(source)}
    cfg = copy.deepcopy(yaml.safe_load((ROOT / "configs/diffusion.yaml").read_text()))
    cfg["project"]["experiment_name"] = run_id
    cfg["training"].update(epochs=200, batch_size=32, ema_decay=0.999, save_every_epochs=10)
    cfg["paths"].update(checkpoint_dir=f"checkpoints/diffusion/ax_t1_128_v4/{run_id}",
                        latest_checkpoint=None, best_checkpoint=None,
                        log_dir=f"reports/v4_diffusion/{run_id}", sample_dir=f"reports/v4_diffusion/{run_id}/samples")
    (out / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    protocol = {
        "run_id": run_id, "version": 1, "start": "resume epoch-10 raw model and optimizer; initialise EMA from raw weights",
        "max_epoch": 200, "save_every": 10, "minimum_epoch_for_plateau": 100,
        "gate": {"n": 100, "seed": 730000, "guidance": [3.0, 5.0], "batch_size": 10,
                 "conditions": "25 per manufacturer x field-strength cell; train-cell median EchoTime",
                 "input": "pure standard normal noise; complete 1000-step ancestral chain",
                 "judges": ["robust0", "robust1", "robust2"],
                 "selection": "maximise min(mean_mfg_hit/0.90, mean_fs_hit/0.80), then mean of both hit rates, then earlier epoch and smaller guidance",
                 "threshold_mfg": 0.90, "threshold_fs": 0.80,
                 "plateau": "last five checkpoint best-guidance scores span <=0.02 and best score improves <=0.01 over previous five, after epoch>=100",
                 "weights": "EMA; original epoch-10 raw checkpoint graded with the identical gate first"},
        "validation": {"n": 100, "max_per_patient": 10, "seed": 42,
                       "t": [50, 100, 200, 300], "w": [1.0, 3.0, 5.0],
                       "judges": ["robust0", "robust1", "robust2"],
                       "min_ssim": {"field_strength": 0.70, "manufacturer": 0.50},
                       "min_other_axis_stability": 0.95,
                       "selection": "among feasible settings maximise mean judge hit, then gain, then SSIM; ties prefer lower t, then w",
                       "infeasible": "select maximum SSIM, then stability, then hit; explicitly flag failure to meet constraints"},
        "test": {"n": 100, "max_per_patient": 10, "selection_seed": 42, "seed_offsets": [0, 10000, 20000],
                 "selection": "same selector and ordered slices as phase12; no test information used for selection",
                 "fixed_comparators": {"field_strength": [100, 5.0], "manufacturer": [300, 5.0]},
                 "soundness": "null1 same seed, null2 +1000, null3 +2000, w0 same seed, cycle second seed +1000",
                 "stronger": {"field_strength": [100, 200, 300, 999], "manufacturer": [300, 500, 999]},
                 "stronger_n": 50, "joint": {"n": 50, "t": 300, "w": 5.0},
                 "histogram_reference": "target-class train image of another patient, fixed RNG; differs from historical test-reference baseline"},
        "statistics": {"bootstrap": 10000, "seed": 42, "primary": "patient-clustered CI",
                       "paired": "exact McNemar and Wilcoxon, Holm across five baselines; uncorrected patient sign test",
                       "fid": "20 draws; 3 real target images per generated; both predictor and robust0 features; validation target-minus-source check"},
        "compute": {"gpu": "A10G", "max_gpu_containers": 8, "one_app_at_a_time": True,
                    "budget_usd": None, "price_gpu_sec": 0.000306, "price_cpu_core_sec": 0.0000131,
                    "price_memory_gib_sec": 0.00000222,
                    "rate_source": "https://frontend.modal.com/pricing",
                    "note": "No full cloud run until the spend cap is authorised; smoke is separately bounded"},
        "pack_sha256": manifest["pack_sha256"],
    }
    (inputs / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    print("Prepared", out, "pack MB", pack.stat().st_size / 1e6)
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--run-id", required=True)
    prepare(p.parse_args().run_id)
