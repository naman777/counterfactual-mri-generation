"""95% bootstrap CIs (slice-level and patient-clustered) for Phase-12 result CSVs."""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
B = 10_000
SEED = 42


def boot(vals, groups=None, b=B, seed=SEED):
    vals = np.asarray(vals, dtype=float)
    rng = np.random.default_rng(seed)
    if groups is None:
        idx = rng.integers(0, len(vals), size=(b, len(vals)))
        stats = vals[idx].mean(1)
    else:
        groups = np.asarray(groups)
        ug = np.unique(groups)
        sums = np.array([vals[groups == g].sum() for g in ug])
        cnts = np.array([(groups == g).sum() for g in ug])
        pick = rng.integers(0, len(ug), size=(b, len(ug)))
        stats = sums[pick].sum(1) / cnts[pick].sum(1)
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return float(vals.mean()), float(lo), float(hi)


def summarize(df, gain_col, hit_col):
    out = {"n_slices": int(len(df)), "n_patients": int(df.patient_id.nunique())}
    for name, col in [("ssim", "global_ssim"), ("mad", "mean_abs_diff"), ("gain", gain_col), ("hit", hit_col)]:
        v = df[col].astype(float).to_numpy()
        m, lo, hi = boot(v)
        _, plo, phi = boot(v, df.patient_id.to_numpy())
        out[name] = {"mean": m, "slice_ci": [lo, hi], "patient_ci": [plo, phi]}
    return out


def to_bool(s):
    return s.astype(str).str.lower().isin(["true", "1"]).astype(float)


if __name__ == "__main__":
    out = Path(sys.argv[1])
    test = pd.read_csv(ROOT / "data/processed/ax_t1_128/splits/test.csv")[["image_path", "field_strength_label", "manufacturer_label"]]
    runs = {
        "field_strength_t100_w3 (phase12)": ("reports/phase12/field_strength/field_strength_results.csv", "fs_gain", "fs_hit", "field_strength_label"),
        "field_strength_t100_w5 (v2_quickwins)": ("reports/v2_quickwins/fs_t100_w5/field_strength/field_strength_results.csv", "fs_gain", "fs_hit", "field_strength_label"),
        "manufacturer_t300_w5 (phase12)": ("reports/phase12/manufacturer/manufacturer_results.csv", "mfg_gain", "mfg_hit", "manufacturer_label"),
    }
    res = {}
    rows = []
    for name, (path, g, h, src_col) in runs.items():
        df = pd.read_csv(ROOT / path)
        for c in ["mfg_hit", "fs_hit", "mfg_stable", "fs_stable"]:
            df[c] = to_bool(df[c])
        df = df.merge(test, on="image_path", how="left")
        assert df[src_col].notna().all()
        r = {"all": summarize(df, g, h)}
        r["all"]["mfg_stability"] = float(df.mfg_stable.mean())
        r["all"]["fs_stability"] = float(df.fs_stable.mean())
        for lab, sub in df.groupby(src_col):
            r[f"source_{src_col}={lab}"] = summarize(sub, g, h)
        res[name] = r
        for part, s in r.items():
            for k in ["ssim", "mad", "gain", "hit"]:
                rows.append({"run": name, "subset": part, "n": s["n_slices"], "patients": s["n_patients"], "metric": k,
                             "mean": s[k]["mean"], "slice_lo": s[k]["slice_ci"][0], "slice_hi": s[k]["slice_ci"][1],
                             "patient_lo": s[k]["patient_ci"][0], "patient_hi": s[k]["patient_ci"][1]})
    out.mkdir(parents=True, exist_ok=True)
    (out / "bootstrap_ci.json").write_text(json.dumps(res, indent=2))
    pd.DataFrame(rows).to_csv(out / "bootstrap_ci.csv", index=False)
    pd.set_option("display.width", 250)
    print(pd.DataFrame(rows).round(4).to_string())
