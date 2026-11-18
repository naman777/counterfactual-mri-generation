"""FID / MMD of counterfactuals against REAL test slices of the TARGET condition.

Reference = real test slices whose label equals each item's target label, sampled so the
reference class mixture matches the counterfactual set (3 real per generated, 20 draws).
Compared sets: diffusion CFs, unedited sources ("no edit"), Gaussian noise, histogram matching,
and a real-vs-real floor (100 real target-class slices vs the rest of the pool).
Feature extractors: original predictor backbone (SmallCNN, 256-d) and ResNet-18 judge (512-d).
"""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.compute_fid_mmd import CNNFeatureExtractor, compute_fid  # noqa: E402
from scripts.v2_quickwins.judge_model import load_judge, judge_features  # noqa: E402

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT = ROOT / "reports/v2_quickwins/fid_mmd_target"
OUT.mkdir(parents=True, exist_ok=True)
R, FACTOR = 20, 3
test = pd.read_csv(ROOT / "data/processed/ax_t1_128/splits/test.csv")


def load01(p):
    return np.clip(np.load(ROOT / p.replace("\\", "/")).astype(np.float32), 0, 1)


def mmd2_unbiased(X, Y, bw):
    def k(A, B):
        d = (A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None] - 2 * A @ B.T
        return np.exp(-np.maximum(d, 0) / (2 * bw ** 2))
    n, m = len(X), len(Y)
    Kxx, Kyy, Kxy = k(X, X), k(Y, Y), k(X, Y)
    return float((Kxx.sum() - np.trace(Kxx)) / (n * (n - 1)) + (Kyy.sum() - np.trace(Kyy)) / (m * (m - 1)) - 2 * Kxy.mean())


pc = CNNFeatureExtractor(dev).to(dev).eval()
judge = load_judge(ROOT / "reports/v2_quickwins/robust_judge/judge_resnet18_seed0.pt", dev)


def feats(imgs, which):
    if which == "predictor_backbone":
        with torch.no_grad():
            return pc.extract(list(imgs))
    return judge_features(judge, imgs, dev)


all_imgs = np.stack([load01(p) for p in test.image_path])
val = pd.read_csv(ROOT / "data/processed/ax_t1_128/splits/val.csv")
val_imgs = np.stack([load01(p) for p in val.image_path])
results = []
for control, label_col, sets in [
    ("field_strength", "field_strength_label", {
        "diffusion_t100_w3": ("reports/phase12/field_strength/field_strength_results.csv", "reports/phase12/field_strength/counterfactuals"),
        "diffusion_t100_w5": ("reports/v2_quickwins/fs_t100_w5/field_strength/field_strength_results.csv", "reports/v2_quickwins/fs_t100_w5/field_strength/counterfactuals")}),
    ("manufacturer", "manufacturer_label", {
        "diffusion_t300_w5": ("reports/phase12/manufacturer/manufacturer_results.csv", "reports/phase12/manufacturer/counterfactuals")}),
]:
    first_csv = next(iter(sets.values()))[0]
    meta = pd.read_csv(ROOT / first_csv)
    src = test.set_index("image_path").loc[meta.image_path, label_col].to_numpy()
    tgt = 1 - src
    gen = {k: np.stack([np.clip(np.load(ROOT / d / f"{e}_counterfactual.npy"), 0, 1) for e in pd.read_csv(ROOT / c).example_id])
           for k, (c, d) in sets.items()}
    for k, (c, _) in sets.items():
        assert (pd.read_csv(ROOT / c).image_path == meta.image_path).all()
    gen["no_edit_source"] = np.stack([load01(p) for p in meta.image_path])
    for b in ["gaussian_noise", "histogram_matching"]:
        gen[f"baseline_{b}"] = np.load(ROOT / f"reports/v2_quickwins/baselines_regen/{control}__{b}.npz")["images"]
    pool_lab = test[label_col].to_numpy()
    counts = {c_: int((tgt == c_).sum()) for c_ in [0, 1]}
    for fx in ["predictor_backbone", "resnet18_judge"]:
        F_all = feats(all_imgs, fx)
        F_gen = {k: feats(v, fx) for k, v in gen.items()}
        bw = float(np.median(np.linalg.norm(F_all[:500, None] - F_all[None, :500], axis=-1)))
        rng = np.random.default_rng(0)
        acc = {k: {"fid": [], "mmd2": []} for k in list(F_gen) + ["real_vs_real_floor"]}
        for _ in range(R):
            ref_idx, floor_idx = [], []
            for c_, n_c in counts.items():
                pool = rng.permutation(np.where(pool_lab == c_)[0])
                floor_idx += list(pool[:n_c])
                ref_idx += list(pool[n_c:n_c + FACTOR * n_c])
            ref = F_all[ref_idx]
            for k, Fg in F_gen.items():
                acc[k]["fid"].append(compute_fid(ref, Fg))
                acc[k]["mmd2"].append(mmd2_unbiased(ref, Fg, bw))
            acc["real_vs_real_floor"]["fid"].append(compute_fid(ref, F_all[floor_idx]))
            acc["real_vs_real_floor"]["mmd2"].append(mmd2_unbiased(ref, F_all[floor_idx], bw))
        # Patient-disjoint check on the validation pool: distance to real TARGET-class vs real SOURCE-class slices.
        F_val = feats(val_imgs, fx)
        vlab = val[label_col].to_numpy()
        for refkind, lab_counts in [("val_target", counts), ("val_source", {c_: int((src == c_).sum()) for c_ in [0, 1]})]:
            vacc = {k: {"fid": [], "mmd2": []} for k in F_gen}
            for _ in range(R):
                idx = []
                for c_, n_c in lab_counts.items():
                    idx += list(rng.choice(np.where(vlab == c_)[0], size=FACTOR * n_c, replace=False))
                for k, Fg in F_gen.items():
                    vacc[k]["fid"].append(compute_fid(F_val[idx], Fg))
                    vacc[k]["mmd2"].append(mmd2_unbiased(F_val[idx], Fg, bw))
            for k, v in vacc.items():
                results.append({"control": control, "features": fx, "set": k, "reference": refkind, "n_gen": 100,
                                "n_ref": FACTOR * 100, "fid_mean": float(np.mean(v["fid"])), "fid_sd": float(np.std(v["fid"])),
                                "mmd2_mean": float(np.mean(v["mmd2"])), "mmd2_sd": float(np.std(v["mmd2"])),
                                "rbf_bandwidth": bw, "target_counts": lab_counts})
        for k, v in acc.items():
            results.append({"control": control, "features": fx, "set": k, "reference": "test_target", "n_gen": 100, "n_ref": FACTOR * 100,
                            "fid_mean": float(np.mean(v["fid"])), "fid_sd": float(np.std(v["fid"])),
                            "mmd2_mean": float(np.mean(v["mmd2"])), "mmd2_sd": float(np.std(v["mmd2"])),
                            "rbf_bandwidth": bw, "target_counts": counts})
df = pd.DataFrame(results)
df.to_csv(OUT / "fid_mmd_target.csv", index=False)
(OUT / "fid_mmd_target.json").write_text(json.dumps(results, indent=2))
pd.set_option("display.width", 250)
print(df.drop(columns=["target_counts", "n_gen", "n_ref", "rbf_bandwidth"]).round(4).to_string())
piv = df.pivot_table(index=["control", "features", "set"], columns="reference", values="fid_mean", sort=False)
piv["val_target_minus_source"] = piv["val_target"] - piv["val_source"]
piv.to_csv(OUT / "fid_target_vs_source_val.csv")
print(piv.round(3).to_string())
