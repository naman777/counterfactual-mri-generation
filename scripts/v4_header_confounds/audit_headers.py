"""Exploratory DICOM-header confound audit for the ax t1 series (original 103 + held-out 64 patients).

python scripts/v4_header_confounds/audit_headers.py --run-id v4_header_confounds_20261017_01

Read-only: header reads use stop_before_pixels; HF energy is computed from the existing
preprocessed 128x128 .npy files. Nothing under data/, checkpoints/ or older reports/ is written.

HF(x) = 1e3 * mean over pixels of (scipy.ndimage.laplace(x))^2, x clipped to [0,1]; the same
definition as scripts/v4_workshop/analyze_hf_energy.py (reports/v4_soundness/).
Outputs go to reports/<run-id>/ (see the README written alongside).
"""
import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
from scipy.ndimage import laplace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.data.dicom_parser import safe_get  # noqa: E402
from scripts.v2_quickwins.bootstrap_ci import B, SEED, boot  # noqa: E402

HELDOUT_RUN = "preflight_20261016_01"
FS = {0: "1.5T", 1: "3.0T"}
MFG = {0: "GE", 1: "Siemens"}
# Header fields read verbatim (the task list plus protocol fields that affect resolution or noise).
CATEGORICAL = ["ReceiveCoilName", "TransmitCoilName", "SoftwareVersions", "ManufacturerModelName",
               "BitsStored", "PixelRepresentation", "InPlanePhaseEncodingDirection", "ScanningSequence",
               "SequenceVariant", "ScanOptions", "MRAcquisitionType", "AcquisitionMatrix",
               "PercentPhaseFieldOfView", "PatientPosition"]
NUMERIC = ["Rows", "Columns", "SliceThickness", "SpacingBetweenSlices", "RepetitionTime", "EchoTime",
           "FlipAngle", "PixelBandwidth", "EchoTrainLength", "NumberOfAverages", "ImagingFrequency"]
HASHED = ["InstitutionName", "StationName", "DeviceSerialNumber", "InstitutionalDepartmentName"]
# ImagingFrequency is the Larmor frequency, i.e. field strength itself: a positive control, not a confound.
POSITIVE_CONTROLS = {"ImagingFrequency"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def hash_id(value):
    if value is None or str(value).strip() == "":
        return "missing"
    return "h_" + hashlib.sha256(str(value).strip().encode()).hexdigest()[:12]


def num(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def hf(a):
    return float(np.mean(laplace(a.astype(np.float32)) ** 2)) * 1e3


def load_slices():
    parts = [pd.read_csv(ROOT / f"data/processed/ax_t1_128/splits/{s}.csv").assign(cohort="original", split=s)
             for s in ["train", "val", "test"]]
    new = pd.read_csv(ROOT / "data/processed/ax_t1_128_heldout" / HELDOUT_RUN / "processed_metadata.csv")
    parts.append(new.assign(cohort="heldout", split=new.judge_split))
    cols = ["cohort", "split", "patient_id", "sop_uid", "manufacturer_label", "field_strength_label",
            "image_path", "slice_path"]
    return pd.concat([p[cols] for p in parts], ignore_index=True)


def read_header(path):
    ds = pydicom.dcmread(str(path), stop_before_pixels=True)
    row = {f: safe_get(ds, f) for f in CATEGORICAL + NUMERIC}
    for f in HASHED:
        row[f + "_hash"] = hash_id(getattr(ds, f, None))
    ps = getattr(ds, "PixelSpacing", None)
    row["PixelSpacing_row"] = num(ps[0]) if ps is not None else float("nan")
    row["PixelSpacing_col"] = num(ps[1]) if ps is not None else float("nan")
    row["header_manufacturer"] = safe_get(ds, "Manufacturer")
    row["header_field_strength"] = num(getattr(ds, "MagneticFieldStrength", None))
    row["header_sop_uid"] = str(ds.SOPInstanceUID)
    return row


def derive(frame):
    for f in CATEGORICAL:  # keep absent/empty tags visible in groupbys and CSVs
        frame[f] = frame[f].map(lambda v: "(absent)" if v is None or (isinstance(v, float) and np.isnan(v))
                                else ("(empty)" if str(v).strip() == "" else v))
    for f in NUMERIC:
        frame[f + "_num"] = frame[f].map(num)
    frame["PixelSpacing_mm"] = frame.PixelSpacing_row.round(4)
    frame["FOV_row_mm"] = (frame.Rows_num * frame.PixelSpacing_row).round(1)
    frame["native_matrix"] = frame.Rows.astype(str) + "x" + frame.Columns.astype(str)
    frame["downsample_factor"] = frame.Rows_num / 128.0
    # AcquisitionMatrix = [freq rows, freq cols, phase rows, phase cols]; nonzero pair = acquired matrix.
    acq = frame.AcquisitionMatrix.map(lambda s: [int(v) for v in re.findall(r"\d+", s)] if isinstance(s, str) else [])
    frame["acq_freq"] = acq.map(lambda v: max(v[0], v[1]) if len(v) == 4 else np.nan)
    frame["acq_phase"] = acq.map(lambda v: max(v[2], v[3]) if len(v) == 4 else np.nan)
    frame["acq_pixel_freq_mm"] = (frame.FOV_row_mm / frame.acq_freq).round(3)
    frame["acq_pixel_phase_mm"] = (frame.FOV_row_mm / frame.acq_phase).round(3)
    frame["recon_over_acq_phase"] = (frame.Rows_num / frame.acq_phase).round(3)
    frame["output_pixel_mm"] = (frame.FOV_row_mm / 128.0).round(3)
    return frame


DERIVED_NUMERIC = ["PixelSpacing_mm", "FOV_row_mm", "downsample_factor", "acq_freq", "acq_phase",
                   "acq_pixel_freq_mm", "acq_pixel_phase_mm", "recon_over_acq_phase", "output_pixel_mm"]
SEP_CATEGORICAL = CATEGORICAL + [f + "_hash" for f in HASHED] + ["native_matrix"]
SEP_NUMERIC = [f + "_num" for f in NUMERIC] + DERIVED_NUMERIC


def patient_table(frame):
    keys = ["cohort", "split", "patient_id", "manufacturer_label", "field_strength_label"]
    fields = SEP_CATEGORICAL + SEP_NUMERIC
    agg = {f: (f, lambda v: " | ".join(sorted(set(map(str, v))))) for f in fields}
    pat = frame.groupby(keys).agg(slices=("sop_uid", "size"), hf_mean=("hf", "mean"), fg_frac_mean=("fg_frac", "mean"),
                                  hf_fg_mean=("hf_fg", "mean"), hf_bg_mean=("hf_bg", "mean"), **agg).reset_index()
    for f in SEP_NUMERIC:  # patient-level numeric value: median over that patient's slices
        pat[f + "_median"] = frame.groupby("patient_id")[f].median().reindex(pat.patient_id).to_numpy()
    pat["varying_fields"] = pat.apply(lambda r: ";".join(f for f in fields if " | " in r[f]), axis=1)
    return pat


def value_counts(frame):
    rows = []
    for f in SEP_CATEGORICAL + [f + "_num" for f in NUMERIC] + ["PixelSpacing_mm"]:
        g = frame.groupby(["cohort", "manufacturer_label", "field_strength_label", f]).agg(
            patients=("patient_id", "nunique"), slices=("sop_uid", "size")).reset_index()
        for _, r in g.iterrows():
            rows.append({"field": f, "value": r[f], "cohort": r.cohort, "manufacturer": MFG[r.manufacturer_label],
                         "field_strength": FS[r.field_strength_label], "patients": r.patients, "slices": r.slices})
    return pd.DataFrame(rows)


def numeric_summary(pat):
    rows = []
    for f in SEP_NUMERIC + ["hf_mean", "hf_fg_mean", "hf_bg_mean", "fg_frac_mean"]:
        col = f + "_median" if f in SEP_NUMERIC else f
        for cohort in ["original", "heldout", "combined"]:
            p = pat if cohort == "combined" else pat[pat.cohort.eq(cohort)]
            for (m, s), g in p.groupby(["manufacturer_label", "field_strength_label"]):
                v = g[col].dropna()
                rows.append({"field": f, "cohort": cohort, "manufacturer": MFG[m], "field_strength": FS[s],
                             "patients": len(g), "patients_with_value": len(v),
                             "patient_median": v.median() if len(v) else np.nan,
                             "patient_min": v.min() if len(v) else np.nan, "patient_max": v.max() if len(v) else np.nan})
    return pd.DataFrame(rows)


def balanced(y, pred):
    return float(np.mean([np.mean(pred[y == c] == c) for c in np.unique(y)]))


def lookup_lopo(values, y):
    """Value -> majority class of the OTHER patients with that value; unseen values -> majority of all others."""
    pred, unseen = np.empty_like(y), 0
    for i in range(len(y)):
        keep = np.arange(len(y)) != i
        same = keep & (values == values[i])
        pool = y[same] if same.any() else y[keep]
        unseen += int(not same.any())
        pred[i] = int(np.mean(pool) > 0.5) if np.mean(pool) != 0.5 else int(np.mean(y[keep]) > 0.5)
    return pred, unseen


def lookup_insample(values, y):
    pred = np.empty_like(y)
    for v in np.unique(values):
        m = values == v
        pred[m] = int(np.mean(y[m]) > 0.5)
    return pred


def best_threshold(x, y):
    best = (-1.0, None, None)
    u = np.unique(x)
    for t in (u[:-1] + u[1:]) / 2 if len(u) > 1 else []:
        for sign in (1, -1):
            pred = (sign * (x - t) > 0).astype(int)
            score = balanced(y, pred)
            if score > best[0]:
                best = (score, t, sign)
    return best


def threshold_lopo(x, y):
    pred = np.empty_like(y)
    for i in range(len(y)):
        keep = np.arange(len(y)) != i
        _, t, sign = best_threshold(x[keep], y[keep])
        pred[i] = int(np.mean(y[keep]) > 0.5) if t is None else int(sign * (x[i] - t) > 0)
    return pred


def separability(pat):
    rows = []
    pops = {"original": pat[pat.cohort.eq("original")], "heldout": pat[pat.cohort.eq("heldout")], "combined": pat}
    for pop_name, p0 in pops.items():
        for target, other in (("field_strength_label", "manufacturer_label"), ("manufacturer_label", "field_strength_label")):
            strata = [("all", p0)] + [(f"{'GE' if v == 0 else 'Siemens'} only" if other == "manufacturer_label"
                                       else f"{FS[v]} only", g) for v, g in p0.groupby(other)]
            for stratum, p in strata:
                y = p[target].to_numpy().astype(int)
                if len(np.unique(y)) < 2:
                    continue
                for f in SEP_CATEGORICAL + SEP_NUMERIC:
                    values = p[f].astype(str).to_numpy()
                    lo, unseen = lookup_lopo(values, y)
                    ins = lookup_insample(values, y)
                    by_class = [set(values[y == c]) for c in (0, 1)]
                    row = {"population": pop_name, "target": target.replace("_label", ""), "stratum": stratum, "field": f,
                           "positive_control": f.replace("_num", "") in POSITIVE_CONTROLS,
                           "patients": len(y), "patients_class0": int((y == 0).sum()), "patients_class1": int((y == 1).sum()),
                           "distinct_values": len(set(values)), "values_disjoint_between_classes": not (by_class[0] & by_class[1]),
                           "lookup_insample_balanced_acc": balanced(y, ins),
                           "lookup_lopo_acc": float(np.mean(lo == y)), "lookup_lopo_balanced_acc": balanced(y, lo),
                           "lookup_lopo_unseen_patients": unseen}
                    if f in SEP_NUMERIC:
                        x = p[f + "_median"].to_numpy(dtype=float)
                        ok = np.isfinite(x)
                        if ok.sum() == len(x) and len(np.unique(x)) > 1:
                            score, t, sign = best_threshold(x, y)
                            row.update({"threshold_insample_balanced_acc": score, "threshold": t,
                                        "threshold_rule": f"class1 if {f} {'>' if sign == 1 else '<'} {t:g}",
                                        "threshold_lopo_balanced_acc": balanced(y, threshold_lopo(x, y))})
                    rows.append(row)
    return pd.DataFrame(rows)


def group_hf(frame):
    groupings = {
        "cohort x manufacturer x field_strength": ["cohort", "manufacturer", "field_strength"],
        "cohort x split x manufacturer x field_strength": ["cohort", "split", "manufacturer", "field_strength"],
        "cohort x field_strength": ["cohort", "field_strength"],
        "native_matrix": ["native_matrix"],
        "native_matrix x manufacturer x field_strength": ["native_matrix", "manufacturer", "field_strength"],
        "pixel_spacing_mm": ["PixelSpacing_mm"],
        "pixel_spacing_mm x manufacturer x field_strength": ["PixelSpacing_mm", "manufacturer", "field_strength"],
        "acquisition_matrix x manufacturer x field_strength": ["AcquisitionMatrix", "manufacturer", "field_strength"],
        "model x field_strength": ["ManufacturerModelName", "field_strength"],
        "slice_thickness_mm x manufacturer x field_strength": ["SliceThickness_mm", "manufacturer", "field_strength"],
    }
    rows = []
    for name, keys in groupings.items():
        for vals, g in frame.groupby(keys):
            vals = vals if isinstance(vals, tuple) else (vals,)
            out = {"grouping": name, "group": " / ".join(map(str, vals)), "patients": g.patient_id.nunique(), "slices": len(g)}
            for m in ["hf", "hf_fg", "hf_bg", "fg_frac"]:
                mean, lo, hi = boot(g[m].to_numpy(), g.patient_id.to_numpy())
                out.update({f"{m}_mean": mean, f"{m}_patient_lo": lo, f"{m}_patient_hi": hi})
            rows.append(out)
    return pd.DataFrame(rows)


def diff_boot(a, b, metric="hf", seed=SEED):
    """Slice-weighted mean(b) - mean(a); patients resampled independently within each group."""
    rng = np.random.default_rng(seed)
    stats = []
    sums = []
    for g in (a, b):
        s = g.groupby("patient_id")[metric].agg(["sum", "size"])
        sums.append((s["sum"].to_numpy(), s["size"].to_numpy()))
    picks = [rng.integers(0, len(s[0]), size=(B, len(s[0]))) for s in sums]
    means = [s[0][p].sum(1) / s[1][p].sum(1) for s, p in zip(sums, picks)]
    stats = means[1] - means[0]
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return float(b[metric].mean() - a[metric].mean()), float(lo), float(hi)


def fs_differences(frame):
    """3.0T minus 1.5T HF within strata (the generator conditions on manufacturer as well)."""
    strata = {"all": frame}
    for c in ["original", "heldout"]:
        f = frame[frame.cohort.eq(c)]
        strata[f"{c}"] = f
        for m in ["GE", "Siemens"]:
            strata[f"{c} / {m}"] = f[f.manufacturer.eq(m)]
    tr = frame[frame.cohort.eq("original") & frame.split.eq("train")]
    strata["original train"] = tr
    for m in ["GE", "Siemens"]:
        strata[f"original train / {m}"] = tr[tr.manufacturer.eq(m)]
    for m in ["GE", "Siemens"]:
        strata[f"combined / {m}"] = frame[frame.manufacturer.eq(m)]
    for (m, nm), g in frame.groupby(["manufacturer", "native_matrix"]):
        strata[f"combined / {m} / matrix {nm}"] = g
    rows = []
    for name, g in strata.items():
        a, b = g[g.field_strength.eq("1.5T")], g[g.field_strength.eq("3.0T")]
        row = {"stratum": name, "patients_1.5T": a.patient_id.nunique(), "patients_3.0T": b.patient_id.nunique(),
               "slices_1.5T": len(a), "slices_3.0T": len(b)}
        for m in ["hf", "hf_fg", "hf_bg"]:
            if len(a) and len(b):
                d, lo, hi = diff_boot(a, b, m)
                row.update({f"{m}_1.5T": a[m].mean(), f"{m}_3.0T": b[m].mean(), f"{m}_diff_3.0T_minus_1.5T": d,
                            f"{m}_diff_patient_lo": lo, f"{m}_diff_patient_hi": hi,
                            f"{m}_ratio_3.0T_over_1.5T": b[m].mean() / a[m].mean()})
        rows.append(row)
    return pd.DataFrame(rows)


def patient_regression(pat, seed=SEED):
    """OLS on patient-mean HF; CIs from 10k patient bootstrap. Exploratory, descriptive."""
    p = pat.copy()
    p["is_3T"] = p.field_strength_label
    p["is_siemens"] = p.manufacturer_label
    p["is_heldout"] = p.cohort.eq("heldout").astype(int)
    p["log2_downsample"] = np.log2(p.downsample_factor_median)
    p["output_pixel_mm"] = p.output_pixel_mm_median
    p["slice_thickness_mm"] = p.SliceThickness_num_median
    p["acq_pixel_phase_mm"] = p.acq_pixel_phase_mm_median
    p["fg_frac"] = p.fg_frac_mean
    models = {
        "fs_only": ["is_3T"],
        "fs+mfg+cohort": ["is_3T", "is_siemens", "is_heldout"],
        "fs+mfg+cohort+log2_downsample": ["is_3T", "is_siemens", "is_heldout", "log2_downsample"],
        "fs+mfg+cohort+log2_downsample+output_pixel_mm": ["is_3T", "is_siemens", "is_heldout", "log2_downsample", "output_pixel_mm"],
        "log2_downsample+output_pixel_mm": ["log2_downsample", "output_pixel_mm"],
        "fs+mfg+cohort+resampling+slice_thickness+acq_pixel_phase": ["is_3T", "is_siemens", "is_heldout", "log2_downsample",
                                                                     "output_pixel_mm", "slice_thickness_mm", "acq_pixel_phase_mm"],
        "fs+mfg+cohort+resampling+fg_frac": ["is_3T", "is_siemens", "is_heldout", "log2_downsample", "output_pixel_mm", "fg_frac"],
    }
    rng = np.random.default_rng(seed)
    rows = []
    for target in ["hf_mean", "hf_fg_mean", "hf_bg_mean"]:
        y = p[target].to_numpy()
        for name, cols in models.items():
            X = np.column_stack([np.ones(len(p))] + [p[c].to_numpy(dtype=float) for c in cols])
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
            r2 = 1 - np.sum((y - X @ beta) ** 2) / np.sum((y - y.mean()) ** 2)
            draws = []
            for _ in range(B):
                i = rng.integers(0, len(y), len(y))
                draws.append(np.linalg.lstsq(X[i], y[i], rcond=None)[0])
            lo, hi = np.percentile(np.array(draws), [2.5, 97.5], axis=0)
            for k, c in enumerate(["intercept"] + cols):
                rows.append({"outcome": target, "model": name, "term": c, "coef": beta[k], "patient_lo": lo[k],
                             "patient_hi": hi[k], "r2": r2, "patients": len(y), "rank": int(np.linalg.matrix_rank(X))})
    return pd.DataFrame(rows)


def edit_hf_regions():
    """Epoch-110 B3 field-strength edits (t=100, w=5, seed 0) and same-noise null edits, split into the
    SOURCE slice's foreground (x > 0.05) and background. Read-only use of saved arrays."""
    base = ROOT / "reports/v4_diffusion/b1_20261016_01"
    with np.load(base / "remote/full/test/soundness_field_strength_t100_w5_s0/images.npz") as z:
        pack = {k: z[k] for k in z.files}
    with np.load(base / "inputs/pack.npz", allow_pickle=False) as z:
        pid = z["test_pid"][pack["indices"]]
        assert np.array_equal(z["test_x"][pack["indices"]], pack["source_images"])
    rows = []
    for i in range(len(pid)):
        fg = pack["source_images"][i] > 0.05
        row = {"example_id": f"example_{i:03d}", "patient_id": str(pid[i]), "source": FS[int(pack["source_fs"][i])],
               "direction": "to 3.0T" if pack["source_fs"][i] == 0 else "to 1.5T", "fg_frac": float(fg.mean())}
        for v in ["source_images", "cf", "null1"]:
            lap2 = laplace(np.clip(pack[v][i].astype(np.float32), 0, 1)) ** 2
            name = "input" if v == "source_images" else v
            row.update({f"hf_{name}": float(lap2.mean()) * 1e3, f"hf_fg_{name}": float(lap2[fg].mean()) * 1e3,
                        f"hf_bg_{name}": float(lap2[~fg].mean()) * 1e3})
        rows.append(row)
    d = pd.DataFrame(rows)
    for r in ["", "fg_", "bg_"]:
        d[f"ratio_{r}cf_over_null"] = d[f"hf_{r}cf"] / d[f"hf_{r}null1"]
    # Fraction of the whole-image Laplacian^2 change (cf - null) located in foreground pixels.
    d["fg_share_of_cf_minus_null"] = (d.fg_frac * (d.hf_fg_cf - d.hf_fg_null1)) / (d.hf_cf - d.hf_null1)
    summ = []
    for direction, g in d.groupby("direction"):
        for m in [c for c in d.columns if c.startswith(("hf_", "ratio_", "fg_"))]:
            mean, lo, hi = boot(g[m].to_numpy(), g.patient_id.to_numpy())
            summ.append({"direction": direction, "examples": len(g), "patients": g.patient_id.nunique(), "metric": m,
                         "mean": mean, "patient_lo": lo, "patient_hi": hi})
    return d, pd.DataFrame(summ)


def echo_overlap(pat):
    """Exact EchoTime values shared between field-strength classes, per vendor. EchoTime is a fixed
    conditioning input (kept at the source value in edits), so disjoint values mean the condition
    vector already carries field-strength identity."""
    rows = []
    pops = {"original train (diffusion training data)": pat[pat.cohort.eq("original") & pat.split.eq("train")],
            "original all": pat[pat.cohort.eq("original")], "heldout all": pat[pat.cohort.eq("heldout")], "combined": pat}
    for name, p in pops.items():
        for m, g in p.groupby("manufacturer_label"):
            te = {s: g[g.field_strength_label.eq(s)].EchoTime_num_median.round(3) for s in (0, 1)}
            shared = set(te[0]) & set(te[1])
            rows.append({"population": name, "manufacturer": MFG[m], "patients_1.5T": len(te[0]), "patients_3.0T": len(te[1]),
                         "distinct_TE_1.5T": te[0].nunique(), "distinct_TE_3.0T": te[1].nunique(),
                         "shared_exact_TE_values": len(shared), "shared_values": " | ".join(map(str, sorted(shared))),
                         "patients_1.5T_with_shared_TE": int(te[0].isin(shared).sum()),
                         "patients_3.0T_with_shared_TE": int(te[1].isin(shared).sum()),
                         "TE_range_1.5T": f"{te[0].min():g}-{te[0].max():g}", "TE_range_3.0T": f"{te[1].min():g}-{te[1].max():g}"})
    return pd.DataFrame(rows)


def separability_top(sep, cutoff=0.95):
    s = sep.copy()
    s["best_lopo_balanced_acc"] = s[["lookup_lopo_balanced_acc", "threshold_lopo_balanced_acc"]].max(axis=1)
    s = s[~s.positive_control & (s.best_lopo_balanced_acc >= cutoff)]
    return s.sort_values(["population", "target", "stratum", "best_lopo_balanced_acc"], ascending=[True, True, True, False])[
        ["population", "target", "stratum", "patients", "patients_class0", "patients_class1", "field", "distinct_values",
         "values_disjoint_between_classes", "lookup_lopo_balanced_acc", "lookup_lopo_unseen_patients",
         "threshold_lopo_balanced_acc", "threshold_rule", "best_lopo_balanced_acc"]]


def main(run_id, header_cache=None):
    if not run_id or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for c in run_id):
        raise ValueError("Invalid run ID")
    out = ROOT / "reports" / run_id
    out.mkdir(parents=True, exist_ok=False)
    sl = load_slices()
    cache = Path(header_cache) if header_cache else None
    if cache and cache.exists():  # optional scratch cache of the raw header rows (re-runs only)
        headers = pd.read_pickle(cache)
    else:
        rows = []
        for i, path in enumerate(sl.slice_path):
            rows.append(read_header(ROOT / path if not Path(path).is_absolute() else path))
            if (i + 1) % 2000 == 0:
                print("headers", i + 1, flush=True)
        headers = pd.DataFrame(rows)
        if cache:
            headers.to_pickle(cache)
    frame = pd.concat([sl, headers], axis=1)
    assert (frame.header_sop_uid == frame.sop_uid.astype(str)).all()
    frame = derive(frame)
    hfs = []
    for p in frame.image_path:
        x = np.clip(np.load(ROOT / p.replace("\\", "/")).astype(np.float32), 0, 1)
        lap2 = laplace(x) ** 2
        fg = x > 0.05  # same threshold as preprocess.compute_foreground_fraction
        hfs.append((float(lap2.mean()) * 1e3, float(lap2[fg].mean()) * 1e3 if fg.any() else np.nan,
                    float(lap2[~fg].mean()) * 1e3 if (~fg).any() else np.nan, float(fg.mean())))
    frame[["hf", "hf_fg", "hf_bg", "fg_frac"]] = np.array(hfs)
    frame["SliceThickness_mm"] = frame.SliceThickness_num.round(2)
    # The whole-image HF must equal the v4_soundness hf() exactly.
    assert np.allclose(frame.hf[:50], [hf(np.clip(np.load(ROOT / p.replace("\\", "/")).astype(np.float32), 0, 1))
                                       for p in frame.image_path[:50]])
    frame["manufacturer"] = frame.manufacturer_label.map(MFG)
    frame["field_strength"] = frame.field_strength_label.map(FS)
    label_check = {
        "manufacturer_header_matches_label": bool((frame.header_manufacturer.str.upper().str.contains("SIEMENS").astype(int) == frame.manufacturer_label).all()),
        "field_strength_header_matches_label": bool(((frame.header_field_strength > 2.25).astype(int) == frame.field_strength_label).all()),
    }
    frame.drop(columns=["header_sop_uid"]).to_csv(out / "slice_headers.csv", index=False)
    pat = patient_table(frame)
    pat.to_csv(out / "patient_headers.csv", index=False)
    value_counts(frame).to_csv(out / "field_value_counts.csv", index=False)
    numeric_summary(pat).to_csv(out / "numeric_fields_by_class.csv", index=False)
    sep = separability(pat)
    sep.to_csv(out / "separability.csv", index=False)
    separability_top(sep).to_csv(out / "separability_top.csv", index=False)
    echo_overlap(pat).to_csv(out / "echo_time_overlap.csv", index=False)
    group_hf(frame).to_csv(out / "hf_by_group.csv", index=False)
    fs_differences(frame).to_csv(out / "hf_fs_differences.csv", index=False)
    patient_regression(pat).to_csv(out / "hf_patient_regression.csv", index=False)
    edits, edit_summary = edit_hf_regions()
    edits.to_csv(out / "edit_hf_regions_per_example.csv", index=False)
    edit_summary.to_csv(out / "edit_hf_regions_summary.csv", index=False)
    cells = frame.groupby(["cohort", "manufacturer", "field_strength"]).agg(
        patients=("patient_id", "nunique"), slices=("sop_uid", "size")).reset_index()
    summary = {
        "exploratory_header_audit": True, "model_evaluated": False, "images_modified": False,
        "hf_definition": "1e3 * mean((scipy.ndimage.laplace(x))**2), x = preprocessed 128x128 slice clipped to [0,1] (as reports/v4_soundness)",
        "hf_fg_bg_definition": "same Laplacian^2 averaged over pixels with x > 0.05 (fg) or x <= 0.05 (bg); supplementary",
        "bootstrap": {"resamples": B, "seed": SEED, "unit": "patient"},
        "patients": {c: int(frame[frame.cohort.eq(c)].patient_id.nunique()) for c in ["original", "heldout"]},
        "slices": {c: int(frame.cohort.eq(c).sum()) for c in ["original", "heldout"]},
        "cells": cells.to_dict(orient="records"),
        "patients_with_any_varying_field": int(pat.varying_fields.ne("").sum()),
        "varying_fields_seen": sorted({f for v in pat.varying_fields for f in v.split(";") if f}),
        **label_check,
        "source_sha256": sha(__file__),
        "input_sha256": {str(p): sha(ROOT / p) for p in
                         [f"data/processed/ax_t1_128/splits/{s}.csv" for s in ["train", "val", "test"]]
                         + [f"data/processed/ax_t1_128_heldout/{HELDOUT_RUN}/processed_metadata.csv"]},
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--header-cache", default=None, help="optional pickle path outside reports/ for re-runs")
    args = ap.parse_args()
    main(args.run_id, args.header_cache)
