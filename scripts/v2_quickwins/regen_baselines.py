"""Regenerate the five classical baselines exactly as run_phase12_baselines.py (same RNG order)
and save them as npz, verifying MAD against reports/phase12/baselines/baseline_results.csv."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.run_phase12_baselines import BASELINES, CONTROLS, find_reference_image  # noqa: E402
from scripts.run_explainability_batch import build_target_condition, select_diverse_examples  # noqa: E402

out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
seed = 42
test_df = pd.read_csv(ROOT / "data/processed/ax_t1_128/splits/test.csv")
examples = select_diverse_examples(test_df, 100, seed=seed)
rng = np.random.default_rng(seed)
ref = pd.read_csv(ROOT / "reports/phase12/baselines/baseline_results.csv")
for control in CONTROLS:  # includes echo_time so the RNG stream matches the original run
    for name, cfg in BASELINES.items():
        imgs = []
        for idx, (_, row) in enumerate(examples.iterrows()):
            orig = np.clip(np.load(ROOT / row["image_path"].replace("\\", "/")).astype(np.float32), 0, 1)
            _, tgt = build_target_condition(row, control)
            r = None
            if cfg["needs_reference"]:
                r = find_reference_image(test_df, tgt, str(row["patient_id"]), control, rng)
                if r is None:
                    r = orig.copy()
            imgs.append(cfg["fn"](orig, r, seed + idx))
        imgs = np.stack(imgs).astype(np.float32)
        if control == "echo_time":
            continue
        origs = np.stack([np.clip(np.load(ROOT / p.replace("\\", "/")).astype(np.float32), 0, 1) for p in examples.image_path])
        mad = np.abs(imgs - origs).mean((1, 2))
        refmad = ref[(ref.baseline == name) & (ref.control == control)].sort_values("example_id").mean_abs_diff.to_numpy()
        print(control, name, "max |MAD - saved MAD| =", float(np.abs(mad - refmad).max()))
        np.savez_compressed(out / f"{control}__{name}.npz", images=imgs,
                            image_path=examples.image_path.to_numpy(str), patient_id=examples.patient_id.to_numpy(str))
