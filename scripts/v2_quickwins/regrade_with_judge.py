"""Re-grade saved diffusion counterfactuals and regenerated baselines with the ResNet-18 judge(s)."""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.v2_quickwins.judge_model import load_judge, judge_probs  # noqa: E402
from scripts.v2_quickwins.bootstrap_ci import boot  # noqa: E402

JD = ROOT / "reports/v2_quickwins/robust_judge"
TAG = sys.argv[1] if len(sys.argv) > 1 else ""  # "" = robust judges, "_plain" = flip-only judges
OUT = ROOT / f"reports/v2_quickwins/regrade{TAG}"
SEEDS = [0, 1, 2]
dev = "cuda" if torch.cuda.is_available() else "cpu"

test = pd.read_csv(ROOT / "data/processed/ax_t1_128/splits/test.csv")
lab = test.set_index("image_path")[["manufacturer_label", "field_strength_label"]]


def load01(p):
    return np.clip(np.load(ROOT / p.replace("\\", "/")).astype(np.float32), 0, 1)


def diffusion_set(csv, cf_dir):
    df = pd.read_csv(ROOT / csv)
    origs = np.stack([load01(p) for p in df.image_path])
    cfs = np.stack([np.clip(np.load(ROOT / cf_dir / f"{e}_counterfactual.npy").astype(np.float32), 0, 1) for e in df.example_id])
    return df[["example_id", "patient_id", "image_path"]].copy(), origs, cfs


sets = {}
for name, csv, d, control in [
    ("diffusion_fs_t100_w3", "reports/phase12/field_strength/field_strength_results.csv", "reports/phase12/field_strength/counterfactuals", "field_strength"),
    ("diffusion_fs_t100_w5", "reports/v2_quickwins/fs_t100_w5/field_strength/field_strength_results.csv", "reports/v2_quickwins/fs_t100_w5/field_strength/counterfactuals", "field_strength"),
    ("diffusion_mfg_t300_w5", "reports/phase12/manufacturer/manufacturer_results.csv", "reports/phase12/manufacturer/counterfactuals", "manufacturer"),
]:
    meta, o, c = diffusion_set(csv, d)
    sets[name] = (control, meta, o, c)
for control in ["field_strength", "manufacturer"]:
    for b in ["identity", "contrast_shift", "gamma", "gaussian_noise", "histogram_matching"]:
        z = np.load(ROOT / f"reports/v2_quickwins/baselines_regen/{control}__{b}.npz")
        meta = pd.DataFrame({"example_id": [f"example_{i:03d}" for i in range(len(z['images']))],
                             "patient_id": z["patient_id"], "image_path": z["image_path"]})
        o = np.stack([load01(p) for p in meta.image_path])
        sets[f"{b}_{'fs' if control == 'field_strength' else 'mfg'}"] = (control, meta, o, z["images"])

judges = {s: load_judge(JD / f"judge_resnet18{TAG}_seed{s}.pt", dev) for s in SEEDS}
OUT.mkdir(parents=True, exist_ok=True)
summary, per_ex = [], []
for name, (control, meta, o, c) in sets.items():
    src_m = lab.loc[meta.image_path, "manufacturer_label"].to_numpy()
    src_f = lab.loc[meta.image_path, "field_strength_label"].to_numpy()
    tgt_m = 1 - src_m if control == "manufacturer" else src_m
    tgt_f = 1 - src_f if control == "field_strength" else src_f
    for s, j in judges.items():
        bm, bf = judge_probs(j, o, dev)
        am, af = judge_probs(j, c, dev)
        r = np.arange(len(o))
        if control == "field_strength":
            gain = af[r, tgt_f] - bf[r, tgt_f]
            hit = (af.argmax(1) == tgt_f).astype(float)
            stable = (am.argmax(1) == bm.argmax(1)).astype(float)  # manufacturer stability
            src_correct = (bf.argmax(1) == src_f).astype(float)
            direction = np.where(src_f == 0, "1.5T->3.0T", "3.0T->1.5T")
        else:
            gain = am[r, tgt_m] - bm[r, tgt_m]
            hit = (am.argmax(1) == tgt_m).astype(float)
            stable = (af.argmax(1) == bf.argmax(1)).astype(float)  # field-strength stability
            src_correct = (bm.argmax(1) == src_m).astype(float)
            direction = np.where(src_m == 0, "GE->SIEMENS", "SIEMENS->GE")
        df = meta.assign(set=name, control=control, judge_seed=s, direction=direction, gain=gain, hit=hit,
                         other_axis_stable=stable, judge_correct_on_source=src_correct)
        per_ex.append(df)
        for sub_name, mask in [("all", np.ones(len(df), bool))] + [(d, direction == d) for d in np.unique(direction)]:
            sub = df[mask]
            row = {"set": name, "control": control, "judge_seed": s, "subset": sub_name, "n": int(mask.sum()),
                   "judge_acc_on_source": float(sub.judge_correct_on_source.mean()),
                   "other_axis_stability": float(sub.other_axis_stable.mean())}
            for k in ["gain", "hit"]:
                m, lo, hi = boot(sub[k].to_numpy())
                _, plo, phi = boot(sub[k].to_numpy(), sub.patient_id.to_numpy())
                row.update({k: m, f"{k}_slice_lo": lo, f"{k}_slice_hi": hi, f"{k}_patient_lo": plo, f"{k}_patient_hi": phi})
            summary.append(row)
pd.concat(per_ex).to_csv(OUT / "regrade_per_example.csv", index=False)
S = pd.DataFrame(summary)
S.to_csv(OUT / "regrade_summary.csv", index=False)
agg = (S.groupby(["set", "subset"], sort=False)[["n", "gain", "hit", "other_axis_stability", "judge_acc_on_source"]]
       .agg(["mean", "min", "max"]))
agg.to_csv(OUT / "regrade_summary_across_seeds.csv")
pd.set_option("display.width", 250)
print(S[S.judge_seed == 0][["set", "subset", "n", "judge_acc_on_source", "gain", "gain_patient_lo", "gain_patient_hi", "hit", "hit_slice_lo", "hit_slice_hi", "hit_patient_lo", "hit_patient_hi", "other_axis_stability"]].round(3).to_string())
print(S[S.subset == "all"].pivot_table(index="set", columns="judge_seed", values="hit").round(3).to_string())
