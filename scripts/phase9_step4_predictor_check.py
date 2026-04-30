import os
import sys
import json
import yaml
import torch
import inspect
import importlib
import pandas as pd
import numpy as np
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def print_section(title: str):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def safe_read_text(path: Path, max_chars: int = 8000):
    if not path.exists():
        print(f"MISSING | {path}")
        return

    print(f"FOUND   | {path}")
    print("-" * 80)
    text = path.read_text(encoding="utf-8", errors="replace")
    print(text[:max_chars])
    if len(text) > max_chars:
        print("\n... FILE PREVIEW TRUNCATED ...")


def summarize_checkpoint(path: Path):
    print(f"Loading predictor checkpoint: {path}")
    ckpt = torch.load(path, map_location="cpu")

    print(f"Checkpoint type: {type(ckpt)}")

    if isinstance(ckpt, dict):
        print("Top-level keys:")
        for k in ckpt.keys():
            print(f"  - {k}")

        possible_state_keys = [
            "model_state_dict",
            "state_dict",
            "predictor_state_dict",
            "model",
            "net",
        ]

        for key in possible_state_keys:
            if key in ckpt:
                val = ckpt[key]
                print(f"\nFound state field: {key}")
                print(f"Type: {type(val)}")
                if isinstance(val, dict):
                    print(f"Number of entries: {len(val)}")
                    print("First few tensor keys:")
                    for sk in list(val.keys())[:20]:
                        tensor = val[sk]
                        if hasattr(tensor, "shape"):
                            print(f"  {sk}: shape={tuple(tensor.shape)}")
                        else:
                            print(f"  {sk}: type={type(tensor)}")

        for meta_key in [
            "epoch",
            "best_val_loss",
            "val_loss",
            "config",
            "metrics",
            "label_maps",
            "echo_time_stats",
        ]:
            if meta_key in ckpt:
                print(f"\nMetadata field `{meta_key}`:")
                value = ckpt[meta_key]
                if isinstance(value, (dict, list)):
                    print(json.dumps(value, indent=2, default=str)[:3000])
                else:
                    print(value)

    return ckpt


def inspect_predictor_module():
    print_section("2. Inspect src.models.predictor module")

    try:
        module = importlib.import_module("src.models.predictor")
        print("Imported module: src.models.predictor")
    except Exception as e:
        print(f"FAILED importing src.models.predictor: {repr(e)}")
        return

    classes = []
    functions = []

    for name, obj in inspect.getmembers(module):
        if inspect.isclass(obj) and obj.__module__ == module.__name__:
            classes.append((name, obj))
        if inspect.isfunction(obj) and obj.__module__ == module.__name__:
            functions.append((name, obj))

    print("\nClasses found:")
    if not classes:
        print("  No classes found.")
    for name, cls in classes:
        print(f"\n  Class: {name}")
        try:
            print(f"  Signature: {inspect.signature(cls)}")
        except Exception as e:
            print(f"  Could not inspect class signature: {e}")

        if hasattr(cls, "forward"):
            try:
                print(f"  Forward signature: {inspect.signature(cls.forward)}")
            except Exception as e:
                print(f"  Could not inspect forward signature: {e}")

    print("\nFunctions found:")
    if not functions:
        print("  No module-level functions found.")
    for name, fn in functions:
        try:
            print(f"  {name}{inspect.signature(fn)}")
        except Exception:
            print(f"  {name}")


def inspect_dataset_module():
    print_section("3. Inspect src.data.dataset module")

    try:
        module = importlib.import_module("src.data.dataset")
        print("Imported module: src.data.dataset")
    except Exception as e:
        print(f"FAILED importing src.data.dataset: {repr(e)}")
        return

    classes = []

    for name, obj in inspect.getmembers(module):
        if inspect.isclass(obj) and obj.__module__ == module.__name__:
            classes.append((name, obj))

    print("\nClasses found:")
    if not classes:
        print("  No classes found.")
    for name, cls in classes:
        print(f"\n  Class: {name}")
        try:
            print(f"  Signature: {inspect.signature(cls)}")
        except Exception as e:
            print(f"  Could not inspect class signature: {e}")

        if hasattr(cls, "__getitem__"):
            try:
                print(f"  __getitem__ signature: {inspect.signature(cls.__getitem__)}")
            except Exception as e:
                print(f"  Could not inspect __getitem__ signature: {e}")


def main():
    print_section("PHASE 9 STEP 4: PREDICTOR COMPATIBILITY CHECK")
    print(f"Project root: {ROOT}")

    required_files = [
        "configs/predictor.yaml",
        "checkpoints/predictor/ax_t1_128/best.pt",
        "src/models/predictor.py",
        "src/data/dataset.py",
        "scripts/evaluate_predictor.py",
        "data/processed/ax_t1_128/splits/test.csv",
    ]

    print_section("1. Required file existence check")
    for rel in required_files:
        p = ROOT / rel
        print(f"{'FOUND' if p.exists() else 'MISSING':8s} | {rel}")

    inspect_predictor_module()
    inspect_dataset_module()

    print_section("4. predictor.yaml preview")
    predictor_yaml = ROOT / "configs" / "predictor.yaml"
    if predictor_yaml.exists():
        with open(predictor_yaml, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        print("Top-level keys:")
        if isinstance(cfg, dict):
            for k in cfg.keys():
                print(f"  - {k}")
            print("\nFull config preview:")
            print(yaml.dump(cfg, sort_keys=False)[:4000])
        else:
            print(f"Config type: {type(cfg)}")
    else:
        print("configs/predictor.yaml missing.")

    print_section("5. Predictor checkpoint summary")
    ckpt_path = ROOT / "checkpoints" / "predictor" / "ax_t1_128" / "best.pt"
    if ckpt_path.exists():
        summarize_checkpoint(ckpt_path)
    else:
        print("Predictor checkpoint missing.")

    print_section("6. evaluate_predictor.py preview")
    safe_read_text(ROOT / "scripts" / "evaluate_predictor.py", max_chars=10000)

    print_section("7. First test image tensor check")
    test_csv = ROOT / "data" / "processed" / "ax_t1_128" / "splits" / "test.csv"
    if test_csv.exists():
        df = pd.read_csv(test_csv)
        row = df.iloc[0]
        img_path = ROOT / str(row["image_path"])
        arr = np.load(img_path).astype(np.float32)

        print(f"test.csv shape: {df.shape}")
        print(f"Image path: {img_path}")
        print(f"Image shape: {arr.shape}")
        print(f"Image dtype: {arr.dtype}")
        print(f"Image min/max/mean: {arr.min():.6f} / {arr.max():.6f} / {arr.mean():.6f}")

        x = torch.from_numpy(arr).float().unsqueeze(0).unsqueeze(0)
        print(f"Basic predictor input tensor shape would be: {tuple(x.shape)}")
        print(f"Basic predictor input tensor range: {float(x.min()):.6f} to {float(x.max()):.6f}")

        print("\nFirst row condition labels:")
        print(f"manufacturer={row['manufacturer']} | label={int(row['manufacturer_label'])}")
        print(f"field_strength={row['field_strength']} | label={int(row['field_strength_label'])}")
        print(f"echo_time={float(row['echo_time'])}")
        print(f"echo_time_norm={float(row['echo_time_norm'])}")

    print_section("STEP 4 CHECK COMPLETE")
    print("Send me this full output. After I verify it, I will give the predictor-validation code.")


if __name__ == "__main__":
    main()