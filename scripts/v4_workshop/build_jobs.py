"""Build edit-chain jobs for tasks 03 (soundness) and 05 (seed variance, joint edit).

Usage: python scripts/v4_workshop/build_jobs.py <out_dir>
Writes soundness_jobs.pkl, seeds_jobs.pkl, joint_jobs.pkl.

Seeds (per slice idx, all distinct from the main run's 42+idx except where pairing is intended):
  null k=1      42+idx         (same noise as the main target edit -> paired with it)
  null k=2,3    42+idx+1000, +2000
  uncond (w=0)  42+idx         (same noise as null k=1)
  cycle back    42+idx+1000    (same noise as null k=2), starting from the SAVED main counterfactual
  extra seeds   42+idx+10000, 42+idx+20000 (target edit at the operating point)
  joint edit    42+idx         (as scripts/run_multiaxis_evaluation.py)
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.run_explainability_batch import build_target_condition, select_diverse_examples  # noqa: E402

OPS = {  # operating points (paper v3) and saved main-run counterfactuals
    "field_strength": (100, 5.0, "reports/v2_quickwins/fs_t100_w5/field_strength"),
    "manufacturer": (300, 5.0, "reports/phase12/manufacturer"),
}


def load01(p):
    return np.clip(np.load(ROOT / p.replace("\\", "/")).astype(np.float32), 0, 1)


def step(c, t, w, seed):
    return {"m": c["manufacturer_label"], "f": c["field_strength_label"], "e": c["echo_time_norm"], "t": t, "w": w, "seed": seed}


def main(out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    test = pd.read_csv(ROOT / "data/processed/ax_t1_128/splits/test.csv")
    ex = select_diverse_examples(test, 100, seed=42)
    sound, seeds, joint = [], [], []
    for control, (t, w, run_dir) in OPS.items():
        res = pd.read_csv(ROOT / run_dir / f"{control}_results.csv")
        assert (res.image_path.to_numpy() == ex.image_path.to_numpy()).all()
        for idx, (_, row) in enumerate(ex.iterrows()):
            x0 = load01(row["image_path"])
            src, tgt = build_target_condition(row, control)
            s0 = 42 + idx
            sound.append({"key": f"{control}|null|{idx}", "image": x0,
                          "steps": [step(src, t, w, s0), step(src, t, w, s0 + 1000), step(src, t, w, s0 + 2000)]})
            sound.append({"key": f"{control}|uncond|{idx}", "image": x0, "steps": [step(src, t, 0.0, s0)]})
            cf = np.clip(np.load(ROOT / run_dir / "counterfactuals" / f"example_{idx:03d}_counterfactual.npy").astype(np.float32), 0, 1)
            sound.append({"key": f"{control}|cycle|{idx}", "image": cf, "steps": [step(src, t, w, s0 + 1000)]})
            for off in (10000, 20000):
                seeds.append({"key": f"{control}|seed{off}|{idx}", "image": x0, "steps": [step(tgt, t, w, s0 + off)]})
    for idx, (_, row) in enumerate(ex.iloc[:50].iterrows()):
        src, _ = build_target_condition(row, "field_strength")
        tgt = dict(src, manufacturer_label=1 - src["manufacturer_label"], field_strength_label=1 - src["field_strength_label"])
        joint.append({"key": f"joint|{idx}", "image": load01(row["image_path"]), "steps": [step(tgt, 300, 5.0, 42 + idx)]})
    for name, jobs in [("soundness", sound), ("seeds", seeds), ("joint", joint)]:
        (out / f"{name}_jobs.pkl").write_bytes(pickle.dumps(jobs))
        print(name, len(jobs), "jobs", sum(len(j["steps"]) for j in jobs), "edits")


if __name__ == "__main__":
    main(sys.argv[1])
