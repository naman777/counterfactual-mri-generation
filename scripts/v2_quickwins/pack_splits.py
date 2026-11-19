"""Pack train/val/test slices + labels into one npz (float16) for remote training."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
out = Path(sys.argv[1])
arrs = {}
for split in ["train", "val", "test"]:
    df = pd.read_csv(ROOT / f"data/processed/ax_t1_128/splits/{split}.csv")
    imgs = np.stack([np.clip(np.load(ROOT / p.replace("\\", "/")).astype(np.float32), 0, 1)
                     for p in df["image_path"]]).astype(np.float16)
    arrs[f"{split}_x"] = imgs
    arrs[f"{split}_mfg"] = df["manufacturer_label"].to_numpy(np.int64)
    arrs[f"{split}_fs"] = df["field_strength_label"].to_numpy(np.int64)
    arrs[f"{split}_pid"] = df["patient_id"].to_numpy(str)
    print(split, imgs.shape)
np.savez_compressed(out, **arrs)
print("saved", out, out.stat().st_size / 1e6, "MB")
