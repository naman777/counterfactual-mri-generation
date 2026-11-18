"""High-frequency (Laplacian) energy of field-strength edits vs. null edits vs. real slices of each class.

HF(x) = mean over pixels of (discrete Laplacian of x)^2, x in [0,1], 128x128.
Edits: reports/v2_quickwins/fs_t100_w5 (target condition, seed 42+idx); null and w=0 edits: reports/v4_soundness/raw.
Real reference: all 934 test slices by field-strength class (patient-clustered CIs).
Output: reports/v4_soundness/hf_energy_per_example.csv, hf_energy_summary.csv
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import laplace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.run_explainability_batch import select_diverse_examples  # noqa: E402
from scripts.v2_quickwins.bootstrap_ci import boot  # noqa: E402

OUT = ROOT / "reports/v4_soundness"
LAB = ["1.5T", "3.0T"]


def load01(p):
    return np.clip(np.load(ROOT / p.replace("\\", "/")).astype(np.float32), 0, 1)


def hf(a):
    return float(np.mean(laplace(a.astype(np.float32)) ** 2)) * 1e3


def main():
    test = pd.read_csv(ROOT / "data/processed/ax_t1_128/splits/test.csv")
    ex = select_diverse_examples(test, 100, seed=42)
    snd = pickle.loads((OUT / "raw/soundness_results.pkl").read_bytes())
    run = ROOT / "reports/v2_quickwins/fs_t100_w5/field_strength/counterfactuals"
    rows = []
    for i, (_, r) in enumerate(ex.iterrows()):
        rows.append({"example_id": f"example_{i:03d}", "patient_id": r.patient_id, "source": LAB[r.field_strength_label],
                     "hf_input": hf(load01(r.image_path)),
                     "hf_edit": hf(np.load(run / f"example_{i:03d}_counterfactual.npy")),
                     "hf_null": hf(snd[f"field_strength|null|{i}"][0]),
                     "hf_uncond": hf(snd[f"field_strength|uncond|{i}"][0])})
    d = pd.DataFrame(rows)
    d["ratio_edit_null"] = d.hf_edit / d.hf_null
    d.to_csv(OUT / "hf_energy_per_example.csv", index=False)
    summ = []
    for src, g in d.groupby("source"):
        for m in ["hf_input", "hf_edit", "hf_null", "hf_uncond", "ratio_edit_null"]:
            mean, plo, phi = boot(g[m].to_numpy(), g.patient_id.to_numpy())
            summ.append({"set": f"edited slices, source {src} (n={len(g)})", "metric": m, "mean": mean, "patient_lo": plo, "patient_hi": phi})
    real = test.assign(hf=[hf(load01(p)) for p in test.image_path])
    for lab, g in real.groupby("field_strength_label"):
        mean, plo, phi = boot(g.hf.to_numpy(), g.patient_id.to_numpy())
        summ.append({"set": f"real test slices {LAB[lab]} (n={len(g)}, {g.patient_id.nunique()} patients)", "metric": "hf",
                     "mean": mean, "patient_lo": plo, "patient_hi": phi})
    S = pd.DataFrame(summ)
    S.to_csv(OUT / "hf_energy_summary.csv", index=False)
    pd.set_option("display.width", 200)
    print(S.round(3).to_string())


if __name__ == "__main__":
    main()
