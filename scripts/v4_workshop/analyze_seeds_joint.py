"""Task 05.2-3: sampling-seed variance at the operating points, and re-grade of the joint (mfg+fs) edit.

Seeds: per slice, seed 42+idx (saved main run), 42+idx+10000, 42+idx+20000 (Modal, reports/v4_stats/raw/seeds_results.pkl).
Joint: 50 slices, t=300 w=5, both axes flipped, seed 42+idx (reports/v4_stats/raw/joint_results.pkl); compared with
reports/multiaxis/mfg_plus_fs/mfg_plus_fs_results.csv to check that the regeneration reproduces the original run.
Judges: original predictor, robust ResNet-18 seeds 0-2, flip-only ResNet-18 seeds 0-2.
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
from scripts.v2_quickwins.judge_model import load_judge, judge_probs  # noqa: E402

OUT = ROOT / "reports/v4_stats"
RUNS = {"field_strength": "reports/v2_quickwins/fs_t100_w5/field_strength", "manufacturer": "reports/phase12/manufacturer"}
dev = "cuda" if torch.cuda.is_available() else "cpu"


def load01(p):
    return np.clip(np.load(ROOT / p.replace("\\", "/")).astype(np.float32), 0, 1)


def grade(engine, judges, x0s, cfs, src_m, src_f, tgt_m, tgt_f):
    """Return dict judge -> (mfg_hit, fs_hit, mfg_gain, fs_gain) arrays, plus ssim array."""
    out = {}
    ob = [engine._run_predictor(x) for x in x0s]
    oa = [engine._run_predictor(x) for x in cfs]
    pm_b = np.array([[o["mfg_prob_GE"], o["mfg_prob_SIEMENS"]] for o in ob])
    pm_a = np.array([[o["mfg_prob_GE"], o["mfg_prob_SIEMENS"]] for o in oa])
    pf_b = np.array([[o["fs_prob_1p5T"], o["fs_prob_3p0T"]] for o in ob])
    pf_a = np.array([[o["fs_prob_1p5T"], o["fs_prob_3p0T"]] for o in oa])
    probs = {"original_predictor": (pm_b, pm_a, pf_b, pf_a)}
    for name, j in judges.items():
        mb, fb = judge_probs(j, x0s, dev)
        ma, fa = judge_probs(j, cfs, dev)
        probs[name] = (mb, ma, fb, fa)
    r = np.arange(len(x0s))
    for name, (mb, ma, fb, fa) in probs.items():
        out[name] = {"mfg_hit": (ma.argmax(1) == tgt_m).astype(float), "fs_hit": (fa.argmax(1) == tgt_f).astype(float),
                     "mfg_gain": ma[r, tgt_m] - mb[r, tgt_m], "fs_gain": fa[r, tgt_f] - fb[r, tgt_f],
                     "mfg_pred_same": (ma.argmax(1) == mb.argmax(1)).astype(float),
                     "fs_pred_same": (fa.argmax(1) == fb.argmax(1)).astype(float)}
    ssim = np.array([compute_local_ssim_map(a, b)[0] for a, b in zip(x0s, cfs)])
    return out, ssim


def main():
    test = pd.read_csv(ROOT / "data/processed/ax_t1_128/splits/test.csv")
    ex = select_diverse_examples(test, 100, seed=42)
    engine = ExplainabilityEngine(predictor_checkpoint="checkpoints/predictor/ax_t1_128/best.pt",
                                  label_maps_dir="data/processed/ax_t1_128/label_maps/")
    judges = {f"{fam}_seed{s}": load_judge(ROOT / f"reports/v2_quickwins/robust_judge/judge_resnet18{tag}_seed{s}.pt", dev)
              for fam, tag in [("robust", ""), ("flip_only", "_plain")] for s in [0, 1, 2]}
    seeds = pickle.loads((OUT / "raw/seeds_results.pkl").read_bytes())
    x0s = np.stack([load01(p) for p in ex.image_path])
    sm, sf = ex.manufacturer_label.to_numpy(), ex.field_strength_label.to_numpy()

    # ---- seed variance ----
    per, summ = [], []
    for control, run in RUNS.items():
        key = "fs" if control == "field_strength" else "mfg"
        tm, tf = (sm, 1 - sf) if control == "field_strength" else (1 - sm, sf)
        hits = {}
        for off in [0, 10000, 20000]:
            if off == 0:
                cfs = np.stack([np.clip(np.load(ROOT / run / "counterfactuals" / f"example_{i:03d}_counterfactual.npy").astype(np.float32), 0, 1) for i in range(100)])
            else:
                cfs = np.stack([seeds[f"{control}|seed{off}|{i}"][0].astype(np.float32) for i in range(100)])
            g, ssim = grade(engine, judges, x0s, cfs, sm, sf, tm, tf)
            for judge, d in g.items():
                hits[(judge, off)] = d[f"{key}_hit"]
                per.append(pd.DataFrame({"control": control, "seed_offset": off, "judge": judge, "example_id": ex.index.map(lambda i: f"example_{i:03d}"),
                                         "patient_id": ex.patient_id.to_numpy(), "ssim": ssim, "hit": d[f"{key}_hit"], "gain": d[f"{key}_gain"]}))
                summ.append({"control": control, "judge": judge, "seed_offset": off, "ssim": ssim.mean(),
                             "hit": d[f"{key}_hit"].mean(), "gain": d[f"{key}_gain"].mean()})
        for judge in g:
            H = np.stack([hits[(judge, o)] for o in [0, 10000, 20000]])
            summ.append({"control": control, "judge": judge, "seed_offset": "consistency",
                         "hit": H.mean(), "slices_same_verdict_all_3": float((H.min(0) == H.max(0)).mean()),
                         "slices_hit_any_seed": float(H.max(0).mean()), "slices_hit_all_seeds": float(H.min(0).mean())})
    pd.concat(per).to_csv(OUT / "seed_variance_per_example.csv", index=False)
    S = pd.DataFrame(summ)
    num = S[S.seed_offset != "consistency"].groupby(["control", "judge"], sort=False).agg(
        ssim_mean=("ssim", "mean"), ssim_sd=("ssim", "std"), hit_mean=("hit", "mean"), hit_sd=("hit", "std"),
        hit_min=("hit", "min"), hit_max=("hit", "max"), gain_mean=("gain", "mean"), gain_sd=("gain", "std"))
    cons = S[S.seed_offset == "consistency"].set_index(["control", "judge"])[["slices_same_verdict_all_3", "slices_hit_any_seed", "slices_hit_all_seeds"]]
    table = num.join(cons)
    S.to_csv(OUT / "seed_variance_long.csv", index=False)
    table.to_csv(OUT / "seed_variance_summary.csv")
    pd.set_option("display.width", 220)
    print(table.round(3).to_string())

    # ---- joint edit ----
    joint = pickle.loads((OUT / "raw/joint_results.pkl").read_bytes())
    ex50 = ex.iloc[:50]
    cfs = np.stack([joint[f"joint|{i}"][0].astype(np.float32) for i in range(50)])
    g, ssim = grade(engine, judges, x0s[:50], cfs, sm[:50], sf[:50], 1 - sm[:50], 1 - sf[:50])
    ref = pd.read_csv(ROOT / "reports/multiaxis/mfg_plus_fs/mfg_plus_fs_results.csv")
    rows = []
    for judge, d in g.items():
        rows.append({"judge": judge, "n": 50, "ssim": ssim.mean(), "mfg_hit": d["mfg_hit"].mean(), "fs_hit": d["fs_hit"].mean(),
                     "both_hit": (d["mfg_hit"] * d["fs_hit"]).mean(), "mfg_gain": d["mfg_gain"].mean(), "fs_gain": d["fs_gain"].mean()})
    J = pd.DataFrame(rows)
    J.to_csv(OUT / "multiaxis_regrade.csv", index=False)
    print(J.round(3).to_string())
    o = g["original_predictor"]
    print("reproduction check vs saved run: |SSIM diff| max", float(np.abs(ssim - ref.global_ssim.to_numpy()).max()),
          "| saved mfg/fs hit", ref.mfg_hit.mean(), ref.fs_hit.mean(), "| regenerated", o["mfg_hit"].mean(), o["fs_hit"].mean())


if __name__ == "__main__":
    main()
