"""Grade the stronger-edit diagnostic (first 50 of the 100 slices) with the original predictor and both judge families."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.explainability.engine import ExplainabilityEngine  # noqa: E402
from scripts.v2_quickwins.judge_model import load_judge, judge_probs  # noqa: E402

dev = "cuda" if torch.cuda.is_available() else "cpu"
N = 50
test = pd.read_csv(ROOT / "data/processed/ax_t1_128/splits/test.csv").set_index("image_path")
engine = ExplainabilityEngine(predictor_checkpoint="checkpoints/predictor/ax_t1_128/best.pt",
                              label_maps_dir="data/processed/ax_t1_128/label_maps/")
judges = {f"{fam}{s}": load_judge(ROOT / f"reports/v2_quickwins/robust_judge/judge_resnet18{'' if fam == 'robust' else '_plain'}_seed{s}.pt", dev)
          for fam in ["robust", "plain"] for s in [0, 1, 2]}


def load01(p):
    return np.clip(np.load(ROOT / p.replace("\\", "/")).astype(np.float32), 0, 1)


sets = []
for control, csv, d, t, w in [
    ("field_strength", "reports/phase12/field_strength/field_strength_results.csv", "reports/phase12/field_strength/counterfactuals", 100, 3),
    ("field_strength", "reports/v2_quickwins/fs_t100_w5/field_strength/field_strength_results.csv", "reports/v2_quickwins/fs_t100_w5/field_strength/counterfactuals", 100, 5),
    ("manufacturer", "reports/phase12/manufacturer/manufacturer_results.csv", "reports/phase12/manufacturer/counterfactuals", 300, 5)]:
    df = pd.read_csv(ROOT / csv).iloc[:N]
    cfs = np.stack([np.clip(np.load(ROOT / d / f"{e}_counterfactual.npy"), 0, 1) for e in df.example_id])
    sets.append((control, t, w, df.image_path.to_numpy(), cfs))
for control, t, w in [("field_strength", 200, 5), ("field_strength", 300, 5), ("field_strength", 999, 5),
                      ("manufacturer", 500, 5), ("manufacturer", 999, 5)]:
    z = np.load(ROOT / f"reports/v2_quickwins/stronger_edits/{control}_t{t}_w{w}.npz")
    sets.append((control, t, w, z["image_path"], np.clip(z["images"], 0, 1)))

rows = []
for control, t, w, paths, cfs in sets:
    origs = np.stack([load01(p) for p in paths])
    src_m = test.loc[paths, "manufacturer_label"].to_numpy()
    src_f = test.loc[paths, "field_strength_label"].to_numpy()
    tgt_m = 1 - src_m if control == "manufacturer" else src_m
    tgt_f = 1 - src_f if control == "field_strength" else src_f
    ssim, ohit, ostab = [], [], []
    for i in range(len(paths)):
        sc = {"manufacturer_label": int(src_m[i]), "field_strength_label": int(src_f[i]), "echo_time_norm": float(test.loc[paths[i], "echo_time_norm"])}
        tc = dict(sc, manufacturer_label=int(tgt_m[i]), field_strength_label=int(tgt_f[i]))
        r = engine.compute_all(original=origs[i], counterfactual=cfs[i].astype(np.float32), source_condition=sc, target_condition=tc)
        ssim.append(r["global_ssim"])
        key = "fs" if control == "field_strength" else "mfg"
        ohit.append(float(r["target_hit"][f"{key}_hit"]))
        ostab.append(float(r["stability"]["mfg_stable" if key == "fs" else "fs_stable"]))
    row = {"control": control, "t_edit": t, "w": w, "n": len(paths), "ssim": float(np.mean(ssim)),
           "orig_predictor_hit": float(np.mean(ohit)), "orig_predictor_other_axis_stability": float(np.mean(ostab))}
    for name, j in judges.items():
        am, af = judge_probs(j, cfs, dev)
        hit = (af.argmax(1) == tgt_f) if control == "field_strength" else (am.argmax(1) == tgt_m)
        row[f"hit_{name}"] = float(hit.mean())
    for fam in ["robust", "plain"]:
        v = [row[f"hit_{fam}{s}"] for s in [0, 1, 2]]
        row[f"hit_{fam}_mean"], row[f"hit_{fam}_min"], row[f"hit_{fam}_max"] = float(np.mean(v)), float(min(v)), float(max(v))
    rows.append(row)
    print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items() if "seed" not in k})
out = ROOT / "reports/v2_quickwins/stronger_edits"
pd.DataFrame(rows).to_csv(out / "stronger_edits_graded.csv", index=False)
