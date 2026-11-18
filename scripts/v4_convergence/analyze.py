"""Read-only inputs -> new analysis folder: gate, B3 CIs/tests/soundness/FID.

Run after fetch: python -m scripts.v4_convergence.analyze --run-id <id>
Analysis refuses a completed B3 claim until the remote sampling marker exists.
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.ndimage import laplace
from scipy.stats import binomtest, wilcoxon

from scripts.v2_quickwins.bootstrap_ci import boot
from scripts.v4_convergence.core import AXES, judges, json_write, load_pack

ROOT = Path(__file__).resolve().parents[2]


def interval(v, pid):
    mean, lo, hi = boot(v)
    _, plo, phi = boot(v, pid)
    return dict(mean=mean, slice_lo=lo, slice_hi=hi, patient_lo=plo, patient_hi=phi)


def holm(values):
    values = np.asarray(values, float)
    order = np.argsort(values)
    result = np.empty_like(values)
    prev = 0
    for j, idx in enumerate(order):
        prev = max(prev, (len(values) - j) * values[idx])
        result[idx] = min(1, prev)
    return result


def paired_tests(table, out):
    rows = []
    for axis, key in AXES.items():
        main = table[(table.axis == axis) & (table.kind == "main") & (table.offset == 0)]
        bases = table[(table.axis == axis) & (table.kind == "baselines")]
        for judge, g in main.groupby("judge"):
            block = []
            for baseline, b in bases[bases.judge == judge].groupby("variant"):
                m = g.merge(b, on=["example_id", "patient_id"], suffixes=("_d", "_b"), validate="one_to_one")
                if len(m) != 100:
                    raise ValueError("Incomplete paired comparison")
                hd, hb = m[f"{key}_hit_d"], m[f"{key}_hit_b"]
                n10 = int(((hd == 1) & (hb == 0)).sum())
                n01 = int(((hd == 0) & (hb == 1)).sum())
                diff = m[f"{key}_gain_d"] - m[f"{key}_gain_b"]
                perpatient = (hd - hb).groupby(m.patient_id).mean()
                pos, neg = int((perpatient > 0).sum()), int((perpatient < 0).sum())
                block.append(dict(axis=axis, judge=judge, baseline=baseline, n=100,
                                  hit_diffusion=float(hd.mean()), hit_baseline=float(hb.mean()),
                                  n_diff_only=n10, n_base_only=n01,
                                  p_mcnemar=float(binomtest(min(n10, n01), n10 + n01).pvalue) if n10 + n01 else 1.,
                                  median_gain_diff=float(diff.median()),
                                  p_wilcoxon=float(wilcoxon(diff).pvalue) if (diff != 0).any() else 1.,
                                  patients_diff_better=pos, patients_base_better=neg,
                                  p_sign_patient=float(binomtest(min(pos, neg), pos + neg).pvalue) if pos + neg else 1.))
            if len(block) != 5:
                raise ValueError("Expected five baselines")
            for keyp in ("p_mcnemar", "p_wilcoxon"):
                for row, adjusted in zip(block, holm([row[keyp] for row in block])):
                    row[keyp + "_holm"] = float(adjusted)
            rows.extend(block)
    pd.DataFrame(rows).to_csv(out / "paired_tests.csv", index=False)


def fid(X, Y):
    """PSD-stable FID without silently discarding complex square roots."""
    if min(len(X), len(Y)) < 2:
        raise ValueError("FID requires at least two samples per set")
    X, Y = np.asarray(X, np.float64), np.asarray(Y, np.float64)
    # Nuclear norm of the cross-Gram matrix equals the covariance-root trace.
    # This avoids a singular 512x512 matrix square root with only 100 samples.
    xc = (X - X.mean(0)) / np.sqrt(len(X) - 1)
    yc = (Y - Y.mean(0)) / np.sqrt(len(Y) - 1)
    trace = np.linalg.svd(xc @ yc.T, compute_uv=False).sum()
    value = np.sum((X.mean(0) - Y.mean(0)) ** 2) + (xc * xc).sum() + (yc * yc).sum() - 2 * trace
    return float(max(value, 0))


def mmd(X, Y, bandwidth):
    def kernel(A, B):
        d = (A * A).sum(1)[:, None] + (B * B).sum(1)[None] - 2 * A @ B.T
        return np.exp(-np.maximum(d, 0) / (2 * bandwidth ** 2))
    xx, yy, xy = kernel(X, X), kernel(Y, Y), kernel(X, Y)
    n, k = len(X), len(Y)
    return float((xx.sum() - np.trace(xx)) / (n * (n - 1)) +
                 (yy.sum() - np.trace(yy)) / (k * (k - 1)) - 2 * xy.mean())


@torch.inference_mode()
def features(model, images, original):
    dev = next(model.parameters()).device
    result = []
    for i in range(0, len(images), 128):
        x = torch.as_tensor(images[i:i + 128], device=dev, dtype=torch.float32).unsqueeze(1)
        value = model.backbone(x) if original else model.features(x)
        result.append(value.cpu().numpy())
    return np.concatenate(result).astype(np.float64)


def distribution(data, full, out, assets):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    models = judges(assets, device)
    models = {"predictor_backbone": models["original_predictor"], "resnet18_judge": models["robust0"]}
    rows = []
    for feature_name, model in models.items():
        original = feature_name == "predictor_backbone"
        pool = {split: features(model, data[f"{split}_x"], original) for split in ("test", "val")}
        for axis, key in AXES.items():
            sets = {}
            idx = None
            for folder in (full / "test").iterdir():
                if not (folder / "job.json").exists():
                    continue
                job = json.loads((folder / "job.json").read_text())
                if job["axis"] != axis or job["kind"] not in ("main", "fixed", "baselines") or job["offset"] != 0:
                    continue
                with np.load(folder / "images.npz") as pack:
                    idx = pack["indices"]
                    for variant in (list(("identity", "contrast_shift", "gamma", "gaussian_noise", "histogram_matching"))
                                    if job["kind"] == "baselines" else ["cf"]):
                        sets[job["kind"] + "_" + variant] = features(model, pack[variant], original)
            if idx is None:
                raise ValueError("Missing test feature sets")
            src = data[f"test_{key}"][idx]
            tgt = 1 - src
            sets["no_edit_source"] = pool["test"][idx]
            subset = pool["test"][:500]
            sq = (subset * subset).sum(1)
            distances = np.sqrt(np.maximum(sq[:, None] + sq[None, :] - 2 * subset @ subset.T, 0))
            bw = float(np.median(distances))
            if bw <= 0:
                raise ValueError("Degenerate feature bandwidth")
            for reference in ("test_target", "val_target", "val_source"):
                split = "test" if reference == "test_target" else "val"
                labels = src if reference == "val_source" else tgt
                wanted = {c: int((labels == c).sum()) for c in (0, 1)}
                rng = np.random.default_rng(0)
                acc = {name: [] for name in sets}
                acc["real_vs_real_floor"] = []
                for draw in range(20):
                    refs, floor = [], []
                    for label, n in wanted.items():
                        candidates = rng.permutation(np.where(data[f"{split}_{key}"] == label)[0])
                        if len(candidates) < 4 * n:
                            raise ValueError(f"Reference pool insufficient: {axis}/{reference}/class{label}; no silent class-mixture truncation")
                        floor.extend(candidates[:n])
                        refs.extend(candidates[n:4 * n])
                    ref = pool[split][refs]
                    for name, feat in {**sets, "real_vs_real_floor": pool[split][floor]}.items():
                        acc[name].append((fid(ref, feat), mmd(ref, feat, bw)))
                for name, vals in acc.items():
                    values = np.array(vals)
                    rows.append(dict(axis=axis, features=feature_name, reference=reference, set=name,
                                     n_gen=100, n_ref=300, draws=20, rbf_bandwidth=bw,
                                     fid_mean=float(values[:, 0].mean()), fid_sd=float(values[:, 0].std()),
                                     mmd2_mean=float(values[:, 1].mean()), mmd2_sd=float(values[:, 1].std())))
    pd.DataFrame(rows).to_csv(out / "fid_mmd_target.csv", index=False)


def analyze(run_id, skip_fid=False, base_override=None):
    base = Path(base_override) if base_override else ROOT / "reports/v4_diffusion" / run_id
    full = base / "full" if base_override else base / "remote/full"
    out = base / "analysis" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out.mkdir(parents=True, exist_ok=False)
    (out / "README.md").write_text(
        "# Analysis outputs\n\nAll files made by `python -m scripts.v4_convergence.analyze`.\n"
        "`gate_history.csv` and `gate_curve.*`: B1 gate values copied from cloud JSON.\n"
        "`per_example.csv`, `summary_ci.csv`: B3 judge scores and slice/patient-clustered bootstrap intervals.\n"
        "`paired_tests.csv`: matched-slice tests with Holm correction and patient sign check.\n"
        "`soundness_gaps_ci.csv`, `hf_energy_ci.csv`: same-noise controls and Laplacian energy.\n"
        "`seed_variance.csv`, `quality_scores.csv`: sampling variance and judge-dependent summary scores.\n"
        "`fid_mmd_target.csv`: target-reference FID/MMD in two feature spaces.\n"
        "`status.json`: actual completion state; files absent until their inputs exist.\n", encoding="utf-8")
    history = full / "gate_history.json"
    if history.exists():
        gates = json.loads(history.read_text())
        pd.DataFrame([{k: v for k, v in g.items() if k != "by_judge"} for g in gates]).to_csv(out / "gate_history.csv", index=False)
        fig, ax = plt.subplots(1, 2, figsize=(9, 3.5))
        for i, axis in enumerate(("mfg", "fs")):
            for w in (3., 5.):
                g = sorted([v for v in gates if v["guidance"] == w], key=lambda r: r["epoch"])
                ax[i].plot([v["epoch"] for v in g], [100 * v[f"mean_{axis}_hit"] for v in g], marker=".", label=f"w={w:g}")
            ax[i].axhline(90 if axis == "mfg" else 80, ls="--", color="gray")
            ax[i].set(xlabel="Epoch", ylabel="Robust-judge hit (%)", title=axis, ylim=(0, 102))
            ax[i].legend()
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(out / f"gate_curve.{ext}", dpi=180)
        plt.close(fig)
    if not (full / "complete.json").exists():
        json_write(out / "status.json", {"b3_complete": False, "reason": "Remote sampling complete.json is absent"})
        print("Partial B1 analysis only:", out)
        return out
    data = load_pack(base / "inputs/pack.npz")
    tables = []
    for folder in (full / "test").iterdir():
        job = json.loads((folder / "job.json").read_text())
        table = pd.read_csv(folder / "metrics.csv")
        tables.append(table.assign(kind=job["kind"], job=job["name"]))
    table = pd.concat(tables, ignore_index=True)
    table.to_csv(out / "per_example.csv", index=False)
    rows, quality = [], []
    for (job, variant, judge), g in table.groupby(["job", "variant", "judge"]):
        groups = [("all", g)]
        axis = g.axis.iloc[0]
        key = AXES.get(axis)
        if key:
            groups += [(f"source_{v}", x) for v, x in g.groupby(f"{key}_source")]
        for subset, group in groups:
            for metric in ("ssim", "mad", "mfg_hit", "fs_hit", "both_hit", "mfg_gain", "fs_gain", "mfg_stable", "fs_stable", "mfg_source_agree", "fs_source_agree"):
                rows.append(dict(job=job, variant=variant, judge=judge, subset=subset, metric=metric,
                                 n=len(group), patients=group.patient_id.nunique(),
                                 **interval(group[metric].to_numpy(), group.patient_id.to_numpy())))
        if key:
            other = "fs" if key == "mfg" else "mfg"
            cqs = (g.ssim.clip(0, 1) + ((g[f"{key}_gain"] + 1) / 2).clip(0, 1) + g[f"{other}_stable"]) / 3
            quality.append(dict(job=job, variant=variant, judge=judge, cqs=float(cqs.mean()),
                                cadi=float(g[f"{key}_hit"].mean() * g[f"{key}_gain"].mean() / g.mad.mean()) if g.mad.mean() else 0.))
    pd.DataFrame(rows).to_csv(out / "summary_ci.csv", index=False)
    pd.DataFrame(quality).to_csv(out / "quality_scores.csv", index=False)
    paired_tests(table, out)
    gaps, hf = [], []
    for (axis, judge), g in table[table.kind == "soundness"].groupby(["axis", "judge"]):
        subsets = [("all", g)] + [(f"source_{value}", sub) for value, sub in g.groupby(f"{AXES[axis]}_source")]
        for subset, sub in subsets:
            piv = sub.pivot(index=["example_id", "patient_id"], columns="variant", values="ssim")
            for name, a, b in (("null1_minus_cf", "null1", "cf"), ("null1_minus_uncond", "null1", "uncond"),
                                ("null2_minus_cycle", "null2", "cycle")):
                diff = piv[a] - piv[b]
                gaps.append(dict(axis=axis, judge=judge, subset=subset, metric=name,
                                 **interval(diff.to_numpy(), diff.index.get_level_values("patient_id").to_numpy())))
        if axis == "field_strength" and judge == "original_predictor":
            for source, sub in g.groupby("fs_source"):
                p = sub.pivot(index=["example_id", "patient_id"], columns="variant", values="hf")
                ratios = p.cf / p.null1
                hf.append(dict(set=f"FS source {source}", metric="hf_cf_over_null", **interval(ratios.to_numpy(), ratios.index.get_level_values("patient_id").to_numpy())))
    for label in (0, 1):
        idx = np.where(data["test_fs"] == label)[0]
        energy = np.array([float(np.mean(laplace(x) ** 2)) * 1000 for x in data["test_x"][idx]])
        hf.append(dict(set=f"real_test_fs_{label}", metric="hf_x1000", **interval(energy, data["test_pid"][idx])))
    pd.DataFrame(gaps).to_csv(out / "soundness_gaps_ci.csv", index=False)
    pd.DataFrame(hf).to_csv(out / "hf_energy_ci.csv", index=False)
    variance = []
    for (axis, judge), g in table[table.kind == "main"].groupby(["axis", "judge"]):
        key = AXES[axis]
        means = g.groupby("offset")[[f"{key}_hit", f"{key}_gain", "ssim"]].mean()
        consistency = g.pivot(index="example_id", columns="offset", values=f"{key}_hit")
        variance.append(dict(axis=axis, judge=judge, hit_mean=float(means[f"{key}_hit"].mean()),
                             hit_sd=float(means[f"{key}_hit"].std()), gain_mean=float(means[f"{key}_gain"].mean()),
                             gain_sd=float(means[f"{key}_gain"].std()), all_seeds_hit=float((consistency.sum(1) == 3).mean())))
    pd.DataFrame(variance).to_csv(out / "seed_variance.csv", index=False)
    if not skip_fid:
        distribution(data, full, out, base / "inputs/assets")
    json_write(out / "status.json", {"b3_complete": not skip_fid, "sampling_complete": True,
               "fid_complete": not skip_fid, "paper_unchanged": True})
    print("Analysis:", out)
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--run-id", required=True)
    p.add_argument("--skip-fid", action="store_true")
    a = p.parse_args()
    analyze(a.run_id, a.skip_fid)
