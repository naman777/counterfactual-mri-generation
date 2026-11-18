"""Task 05.1: paired tests, diffusion vs. each baseline, on the same 100 slices, per judge.

Hits: exact McNemar (binomial test on discordant pairs). Gains: Wilcoxon signed-rank on paired differences.
Holm correction across the 5 baselines within each (axis, diffusion setting, judge).
Patient-level check: exact sign test on per-patient mean hit differences (16 patients).
Judges: original predictor, robust ResNet-18 seeds 0-2, flip-only ResNet-18 seeds 0-2.
Usage: python scripts/v4_workshop/paired_tests.py   -> reports/v4_stats/paired_tests.csv
"""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports/v4_stats"
BASELINES = ["identity", "contrast_shift", "gamma", "gaussian_noise", "histogram_matching"]


def tobool(s):
    return s.astype(str).str.lower().isin(["true", "1", "1.0"]).astype(float)


def original_predictor_table():
    rows = []
    for name, path, key in [("diffusion_fs_t100_w3", "reports/phase12/field_strength/field_strength_results.csv", "fs"),
                            ("diffusion_fs_t100_w5", "reports/v2_quickwins/fs_t100_w5/field_strength/field_strength_results.csv", "fs"),
                            ("diffusion_mfg_t300_w5", "reports/phase12/manufacturer/manufacturer_results.csv", "mfg")]:
        d = pd.read_csv(ROOT / path)
        rows.append(pd.DataFrame({"set": name, "example_id": d.example_id, "patient_id": d.patient_id,
                                  "gain": d[f"{key}_gain"], "hit": tobool(d[f"{key}_hit"])}))
    b = pd.read_csv(ROOT / "reports/phase12/baselines/baseline_results.csv")
    for control, key, suf in [("field_strength", "fs", "fs"), ("manufacturer", "mfg", "mfg")]:
        for bl in BASELINES:
            d = b[(b.control == control) & (b.baseline == bl)]
            rows.append(pd.DataFrame({"set": f"{bl}_{suf}", "example_id": d.example_id, "patient_id": d.patient_id,
                                      "gain": d[f"{key}_gain"], "hit": tobool(d[f"{key}_hit"])}))
    return pd.concat(rows).assign(judge="original_predictor")


def judge_tables():
    out = [original_predictor_table()]
    for fam, folder in [("robust", "regrade"), ("flip_only", "regrade_plain")]:
        d = pd.read_csv(ROOT / f"reports/v2_quickwins/{folder}/regrade_per_example.csv")
        d["judge"] = fam + "_seed" + d.judge_seed.astype(str)
        out.append(d[["set", "example_id", "patient_id", "gain", "hit", "judge"]])
    return pd.concat(out, ignore_index=True)


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def compare(a, b):
    m = a.merge(b, on=["example_id", "patient_id"], suffixes=("_d", "_b"))
    assert len(m) == 100
    n10 = int(((m.hit_d == 1) & (m.hit_b == 0)).sum())
    n01 = int(((m.hit_d == 0) & (m.hit_b == 1)).sum())
    p_mc = binomtest(min(n10, n01), n10 + n01, 0.5).pvalue if n10 + n01 > 0 else 1.0
    diff = (m.gain_d - m.gain_b).to_numpy()
    p_w = wilcoxon(diff).pvalue if np.any(diff != 0) else 1.0
    pm = m.groupby("patient_id").apply(lambda g: (g.hit_d - g.hit_b).mean())
    pos, neg = int((pm > 0).sum()), int((pm < 0).sum())
    p_sign = binomtest(min(pos, neg), pos + neg, 0.5).pvalue if pos + neg > 0 else 1.0
    return {"hit_diffusion": m.hit_d.mean(), "hit_baseline": m.hit_b.mean(), "n_diff_only": n10, "n_base_only": n01,
            "p_mcnemar": p_mc, "gain_diffusion": m.gain_d.mean(), "gain_baseline": m.gain_b.mean(),
            "median_gain_diff": float(np.median(diff)), "p_wilcoxon": p_w,
            "patients_diff_better": pos, "patients_base_better": neg, "p_sign_patient": p_sign}


def main():
    T = judge_tables()
    rows = []
    for diff_set, suf in [("diffusion_fs_t100_w5", "fs"), ("diffusion_fs_t100_w3", "fs"), ("diffusion_mfg_t300_w5", "mfg")]:
        for judge, g in T.groupby("judge"):
            a = g[g.set == diff_set]
            block = []
            for bl in BASELINES:
                r = compare(a, g[g.set == f"{bl}_{suf}"])
                block.append({"diffusion": diff_set, "baseline": bl, "judge": judge, **r})
            for key in ["p_mcnemar", "p_wilcoxon"]:
                adj = holm([x[key] for x in block])
                for x, v in zip(block, adj):
                    x[key + "_holm"] = v
            rows += block
    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "paired_tests.csv", index=False)
    show = df[df.judge.isin(["original_predictor", "robust_seed0", "flip_only_seed0"])]
    pd.set_option("display.width", 250)
    print(show[["diffusion", "baseline", "judge", "hit_diffusion", "hit_baseline", "n_diff_only", "n_base_only",
                "p_mcnemar_holm", "gain_diffusion", "gain_baseline", "p_wilcoxon_holm", "patients_diff_better",
                "patients_base_better", "p_sign_patient"]].round(4).to_string())
    # how many of the 3 seeds per family show diffusion significantly better / worse (Holm, alpha 0.05)
    df["family"] = df.judge.str.replace(r"_seed\d", "", regex=True)
    df["hit_sig"] = np.where(df.p_mcnemar_holm < 0.05, np.sign(df.n_diff_only - df.n_base_only), 0)
    df["gain_sig"] = np.where(df.p_wilcoxon_holm < 0.05, np.sign(df.median_gain_diff), 0)
    agg = df[df.family != "original_predictor"].groupby(["diffusion", "baseline", "family"]).agg(
        hit_better=("hit_sig", lambda s: int((s > 0).sum())), hit_worse=("hit_sig", lambda s: int((s < 0).sum())),
        gain_better=("gain_sig", lambda s: int((s > 0).sum())), gain_worse=("gain_sig", lambda s: int((s < 0).sum())))
    agg.to_csv(OUT / "paired_tests_seed_agreement.csv")
    print(agg.to_string())


if __name__ == "__main__":
    main()
