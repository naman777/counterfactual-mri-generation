import os
import sys
import json
import yaml
import torch
import inspect
import pandas as pd
import numpy as np
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def print_section(title: str):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def exists_check(path: str):
    p = ROOT / path
    status = "FOUND" if p.exists() else "MISSING"
    print(f"{status:8s} | {path}")
    return p.exists(), p


def safe_signature(obj, name: str):
    try:
        print(f"{name} signature:")
        print(inspect.signature(obj))
    except Exception as e:
        print(f"Could not inspect {name}: {e}")


def summarize_checkpoint(ckpt_path: Path):
    print(f"Loading checkpoint on CPU: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")

    print(f"Checkpoint type: {type(ckpt)}")

    if isinstance(ckpt, dict):
        print("Top-level checkpoint keys:")
        for k in ckpt.keys():
            print(f"  - {k}")

        possible_state_keys = [
            "model_state_dict",
            "unet_state_dict",
            "conditioner_state_dict",
            "conditioning_state_dict",
            "state_dict",
            "model",
            "unet",
            "conditioner",
            "conditioning",
        ]

        for key in possible_state_keys:
            if key in ckpt:
                val = ckpt[key]
                print(f"\nFound checkpoint field: {key}")
                print(f"Type: {type(val)}")
                if isinstance(val, dict):
                    print(f"Number of tensors/entries: {len(val)}")
                    sample_keys = list(val.keys())[:10]
                    print("First few keys:")
                    for sk in sample_keys:
                        tensor = val[sk]
                        if hasattr(tensor, "shape"):
                            print(f"  {sk}: shape={tuple(tensor.shape)}")
                        else:
                            print(f"  {sk}: type={type(tensor)}")

        # Print common metadata if present
        for meta_key in ["epoch", "global_step", "best_val_loss", "config", "args"]:
            if meta_key in ckpt:
                print(f"\nMetadata field `{meta_key}`:")
                value = ckpt[meta_key]
                if isinstance(value, (dict, list)):
                    print(json.dumps(value, indent=2, default=str)[:2000])
                else:
                    print(value)

    return ckpt


def main():
    print_section("PHASE 9 STEP 1: REPO COMPATIBILITY CHECK")
    print(f"Project root detected as: {ROOT}")

    print_section("1. Required file existence check")

    required_paths = [
        "configs/diffusion.yaml",
        "checkpoints/diffusion/ax_t1_128/latest.pt",
        "checkpoints/diffusion/ax_t1_128/best.pt",
        "checkpoints/predictor/ax_t1_128/best.pt",
        "data/processed/ax_t1_128/splits/test.csv",
        "data/processed/ax_t1_128/processed_metadata.csv",
        "data/processed/ax_t1_128/label_maps/manufacturer_to_id.json",
        "data/processed/ax_t1_128/label_maps/field_strength_to_id.json",
        "data/processed/ax_t1_128/label_maps/echo_time_stats.json",
        "src/models/conditioning.py",
        "src/models/diffusion_unet.py",
        "src/models/ddpm_scheduler.py",
        "src/inference/sample_diffusion.py",
    ]

    found = {}
    for path in required_paths:
        ok, p = exists_check(path)
        found[path] = p

    print_section("2. Import existing diffusion classes")

    try:
        from src.models.conditioning import AcquisitionConditionEmbedding
        print("FOUND    | AcquisitionConditionEmbedding")
        safe_signature(AcquisitionConditionEmbedding, "AcquisitionConditionEmbedding")
    except Exception as e:
        print(f"FAILED   | Could not import AcquisitionConditionEmbedding: {repr(e)}")

    try:
        from src.models.diffusion_unet import MRIConditionalUNet
        print("\nFOUND    | MRIConditionalUNet")
        safe_signature(MRIConditionalUNet, "MRIConditionalUNet")
    except Exception as e:
        print(f"\nFAILED   | Could not import MRIConditionalUNet: {repr(e)}")

    try:
        from src.models.ddpm_scheduler import DDPMScheduler
        print("\nFOUND    | DDPMScheduler")
        safe_signature(DDPMScheduler, "DDPMScheduler")

        scheduler_methods = [
            name for name, _ in inspect.getmembers(DDPMScheduler, predicate=inspect.isfunction)
            if not name.startswith("_")
        ]
        print("DDPMScheduler public methods:")
        for m in scheduler_methods:
            print(f"  - {m}")
    except Exception as e:
        print(f"\nFAILED   | Could not import DDPMScheduler: {repr(e)}")

    print_section("3. diffusion.yaml summary")

    diffusion_yaml = found["configs/diffusion.yaml"]
    if diffusion_yaml.exists():
        with open(diffusion_yaml, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        print("Top-level config keys:")
        if isinstance(cfg, dict):
            for k in cfg.keys():
                print(f"  - {k}")

            print("\nFull config preview:")
            print(yaml.dump(cfg, sort_keys=False)[:4000])
        else:
            print(f"Config is not a dict. Type: {type(cfg)}")
    else:
        print("diffusion.yaml missing, skipping config summary.")

    print_section("4. Diffusion latest.pt checkpoint summary")

    latest_ckpt = found["checkpoints/diffusion/ax_t1_128/latest.pt"]
    if latest_ckpt.exists():
        summarize_checkpoint(latest_ckpt)
    else:
        print("latest.pt missing, skipping checkpoint summary.")

    print_section("5. Test CSV and first image check")

    test_csv = found["data/processed/ax_t1_128/splits/test.csv"]
    if test_csv.exists():
        df = pd.read_csv(test_csv)
        print(f"test.csv shape: {df.shape}")
        print("Columns:")
        for c in df.columns:
            print(f"  - {c}")

        print("\nFirst row preview:")
        first = df.iloc[0].to_dict()
        print(json.dumps(first, indent=2, default=str)[:3000])

        image_col_candidates = ["image_path", "processed_path", "path", "npy_path", "slice_path"]
        image_col = None
        for c in image_col_candidates:
            if c in df.columns:
                image_col = c
                break

        if image_col is None:
            print("\nFAILED   | Could not find image path column.")
            print(f"Tried columns: {image_col_candidates}")
        else:
            raw_path = str(df.iloc[0][image_col])
            img_path = Path(raw_path)
            if not img_path.is_absolute():
                img_path = ROOT / raw_path

            print(f"\nDetected image column: {image_col}")
            print(f"First image path resolved as: {img_path}")

            if img_path.exists():
                arr = np.load(img_path)
                print("First image loaded successfully.")
                print(f"Shape: {arr.shape}")
                print(f"Dtype: {arr.dtype}")
                print(f"Min: {float(np.min(arr)):.6f}")
                print(f"Max: {float(np.max(arr)):.6f}")
                print(f"Mean: {float(np.mean(arr)):.6f}")
            else:
                print("FAILED   | First image path does not exist.")
    else:
        print("test.csv missing, skipping image check.")

    print_section("6. Label maps check")

    for label_path in [
        "data/processed/ax_t1_128/label_maps/manufacturer_to_id.json",
        "data/processed/ax_t1_128/label_maps/field_strength_to_id.json",
        "data/processed/ax_t1_128/label_maps/echo_time_stats.json",
    ]:
        p = ROOT / label_path
        print(f"\n{label_path}")
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            print(json.dumps(data, indent=2, default=str))
        else:
            print("MISSING")

    print_section("STEP 1 CHECK COMPLETE")
    print("Send me this full terminal output before we move to Step 2.")


if __name__ == "__main__":
    main()