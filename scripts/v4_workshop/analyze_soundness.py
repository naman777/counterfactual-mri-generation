"""Task 03: composition and reversibility (Monteiro et al., ICLR 2023) at the paper's operating points.

Per slice x0 (100 test slices), per axis (FS t=100 w=5; Mfg t=300 w=5):
  cf      saved main counterfactual (target condition, seed s0)
  null_k  k-fold null edit (source condition), seeds s0, s0+1000, s0+2000  -> composition
  uncond  w=0 reconstruction, seed s0                                        -> reference for null_1
  cycle   edit of cf back to the source condition, seed s0+1000             -> reversibility; reference null_2
Paired references share the same noise, so differences isolate the condition.
Outputs: reports/v4_soundness/soundness_per_example.csv, soundness_summary.csv
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.explainability.engine import ExplainabilityEngine  # noqa: E402
from src.explainability.ssim_map import compute_local_ssim_map  # noqa: E402
from scripts.run_explainability_batch import select_diverse_examples  # noqa: E402
from scripts.v2_quickwins.bootstrap_ci import boot  # noqa: E402
from scripts.v2_quickwins.judge_model import load_judge, judge_probs  # noqa: E402

OUT = ROOT / "reports/v4_soundness"
RUNS = {"field_strength": "reports/v2_quickwins/fs_t100_w5/field_strength", "manufacturer": "reports/phase12/manufacturer"}
dev = "cuda" if torch.cuda.is_available() else "cpu"


def load01(p):
    return np.clip(np.load(ROOT / p.replace("\\", "/")).astype(np.float32), 0, 1)


def ssim(a, b):
    return float(compute_local_ssim_map(a, b)[0])


def main():
    res = pickle.loads((OUT / "raw/soundness_results.pkl").read_bytes())
    test = pd.read_csv(ROOT / "data/processed/ax_t1_128/splits/test.csv")
    ex = select_diverse_examples(test, 100, seed=42)
    engine = ExplainabilityEngine(predictor_checkpoint="checkpoints/predictor/ax_t1_128/best.pt",
                                  label_maps_dir="data/processed/ax_t1_128/label_maps/")
    judges = {s: load_judge(ROOT / f"reports/v2_quickwins/robust_judge/judge_resnet18_seed{s}.pt", dev) for s in [0, 1, 2]}
    rows = []
    for control, run in RUNS.items():
        lab = "field_strength_label" if control == "field_strength" else "manufacturer_label"
        head = "fs" if control == "field_strength" else "mfg"
        imgs = {"x0": [], "cf": [], "null1": [], "null2": [], "null3": [], "uncond": [], "cycle": []}
        for idx, (_, row) in enumerate(ex.iterrows()):
            x0 = load01(row["image_path"])
            n = [o.astype(np.float32) for o in res[f"{control}|null|{idx}"]]
            imgs["x0"].append(x0)
            imgs["cf"].append(np.clip(np.load(ROOT / run / "counterfactuals" / f"example_{idx:03d}_counterfactual.npy").astype(np.float32), 0, 1))
            imgs["null1"].append(n[0]); imgs["null2"].append(n[1]); imgs["null3"].append(n[2])
            imgs["uncond"].append(res[f"{control}|uncond|{idx}"][0].astype(np.float32))
            imgs["cycle"].append(res[f"{control}|cycle|{idx}"][0].astype(np.float32))
        src = ex[lab].to_numpy()
        # source-class agreement under each judge
        agree = {}
        for k in ["x0", "cf", "null1", "null3", "uncond", "cycle"]:
            arr = np.stack(imgs[k])
            preds = np.array([engine._run_predictor(a)[f"{head}_pred_label"] for a in arr])
            agree[f"orig_src_{k}"] = (preds == src).astype(float)
            for s, j in judges.items():
                pm, pf = judge_probs(j, arr, dev)
                p = pf if control == "field_strength" else pm
                agree[f"robust{s}_src_{k}"] = (p.argmax(1) == src).astype(float)
        for idx in range(len(ex)):
            x0 = imgs["x0"][idx]
            r = {"control": control, "example_id": f"example_{idx:03d}", "patient_id": ex.patient_id.iloc[idx],
                 "direction": ("1.5T->3.0T" if src[idx] == 0 else "3.0T->1.5T") if control == "field_strength"
                 else ("GE->SIEMENS" if src[idx] == 0 else "SIEMENS->GE")}
            for k in ["cf", "null1", "null2", "null3", "uncond", "cycle"]:
                r[f"ssim_{k}"] = ssim(x0, imgs[k][idx])
                r[f"mad_{k}"] = float(np.abs(x0 - imgs[k][idx]).mean())
            for k, v in agree.items():
                r[k] = v[idx]
            rows.append(r)
    df = pd.DataFrame(rows)
    df["comp_gap_null1_vs_cf"] = df.ssim_null1 - df.ssim_cf          # condition effect at fixed noise
    df["comp_gap_null1_vs_uncond"] = df.ssim_null1 - df.ssim_uncond  # does the source condition help reconstruction?
    df["rev_gap_null2_vs_cycle"] = df.ssim_null2 - df.ssim_cycle     # reversibility loss beyond two noise passes
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "soundness_per_example.csv", index=False)
    metrics = [c for c in df.columns if c.startswith(("ssim_", "mad_", "comp_gap", "rev_gap", "orig_src", "robust"))]
    summ = []
    for control, g in df.groupby("control", sort=False):
        for subset, h in [("all", g)] + list(g.groupby("direction")):
            for m in metrics:
                mean, lo, hi = boot(h[m].to_numpy())
                _, plo, phi = boot(h[m].to_numpy(), h.patient_id.to_numpy())
                summ.append({"control": control, "subset": subset, "n": len(h), "metric": m, "mean": mean,
                             "slice_lo": lo, "slice_hi": hi, "patient_lo": plo, "patient_hi": phi})
    S = pd.DataFrame(summ)
    S.to_csv(OUT / "soundness_summary.csv", index=False)
    pd.set_option("display.width", 220)
    print(S[S.subset == "all"].round(4).to_string())


if __name__ == "__main__":
    main()
