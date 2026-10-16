"""Read-only adapters for the test manifest and prespecified B3 result files."""
import base64
import csv
import io
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
TEST = ROOT / "data/processed/ax_t1_128/splits/test.csv"
B3 = ROOT / "reports/v4_diffusion/b1_20261016_01/remote/full/test"
B2 = ROOT / "reports/v4_heldout_judge/b2_local_20261016_01"
ANALYSIS = ROOT / "reports/v4_diffusion/b1_20261016_01/remote/analysis/20261016T153744Z"


def relative(path):
    return Path(path).relative_to(ROOT).as_posix()


@lru_cache(maxsize=16)
def csv_rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


@lru_cache(maxsize=1)
def slices():
    return [dict(id=str(i), patient=r["patient_id"], manufacturer=int(r["manufacturer_label"]),
                 field_strength=int(r["field_strength_label"]), echo=float(r["echo_time_norm"]),
                 echo_ms=float(r["echo_time"]), instance=int(float(r["instance_number"])),
                 path=r["image_path"].replace("\\", "/"), source=relative(TEST))
            for i, r in enumerate(csv_rows(TEST))]


def slice_info(slice_id):
    if slice_id.startswith("upload_"):
        from api.uploads import UPLOADS
        if slice_id not in UPLOADS:
            raise ValueError("Upload expired; upload the source again")
        return UPLOADS[slice_id][0].copy()
    if not slice_id.isdigit() or not 0 <= int(slice_id) < len(slices()):
        raise ValueError("Unknown test slice")
    return slices()[int(slice_id)].copy()


def source_image(slice_id):
    if slice_id.startswith("upload_"):
        from api.uploads import UPLOADS
        slice_info(slice_id)
        return UPLOADS[slice_id][1].copy()
    return np.load(ROOT / slice_info(slice_id)["path"], allow_pickle=False).astype(np.float32)


def png_bytes(arr, heat=False):
    arr = np.clip(arr, 0, 1)
    if heat:
        arr = np.stack([np.clip(arr * 2, 0, 1), np.clip(arr * 2 - .5, 0, 1), np.clip(arr * 3 - 2, 0, 1)], -1)
    output = io.BytesIO()
    Image.fromarray(np.round(arr * 255).astype("uint8")).save(output, format="PNG")
    return output.getvalue()


def image_url(arr, heat=False):
    return "data:image/png;base64," + base64.b64encode(png_bytes(arr, heat)).decode()


@lru_cache(maxsize=6)
def pack(set_name):
    if set_name not in evidence_sets():
        raise ValueError("Unknown evidence set")
    with np.load(B3 / set_name / "images.npz", allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def evidence_sets():
    if not B3.exists():
        return []
    return sorted(p.name for p in B3.iterdir() if (p / "images.npz").exists())


def evidence_rows(set_name, variant="cf"):
    z = pack(set_name)
    if variant not in z or z[variant].ndim != 3:
        raise ValueError("Unknown image variant")
    saved = csv_rows(B3 / set_name / "metrics.csv")
    held = csv_rows(B2 / "regrade_b3/per_slice.csv")
    axis = "both" if set_name.startswith("joint_") else "manufacturer" if "manufacturer" in set_name else "field_strength"
    key = "mfg" if axis == "manufacturer" else "fs"
    result = []
    for i, test_index in enumerate(z["indices"]):
        example = f"example_{i:03d}"
        original = next((r for r in saved if r["example_id"] == example and r["variant"] == variant and r["judge"] == "original_predictor"), None)
        independent = [r for r in held if r["set"] == set_name and r["example_id"] == example
                       and r["variant"] == variant and r["family"] == "orientation_robust"]
        result.append(dict(index=i, slice=slice_info(str(test_index)),
                           ssim=float(original["ssim"]) if original else None,
                           original_hit=bool(float(original["both_hit" if axis == "both" else f"{key}_hit"])) if original else None,
                           confirmed=sum(int(float(r["hit"])) for r in independent),
                           judges=[dict(name=f"Held-out judge {r['seed']}", hit=bool(float(r["hit"])), gain=float(r["gain"])) for r in independent],
                           direction="joint" if axis == "both" else f"{int(z['source_' + key][i])}_to_{1-int(z['source_' + key][i])}",
                           sources=[relative(B3 / set_name / "images.npz"), relative(B3 / set_name / "metrics.csv"), relative(B2 / "regrade_b3/per_slice.csv")]))
    return result


def headline_results(axis):
    """Select exact rows documented in research-paper-v5/numbers.md. No frontend constants."""
    setting = "field_strength_t100_w5" if axis == "field_strength" else "manufacturer_t300_w3"
    result = []
    path = ANALYSIS / "summary_ci.csv"
    for r in csv_rows(path):
        if r["subset"] != "all" or r["judge"] != "original_predictor" or r["metric"] != ("fs_hit" if axis == "field_strength" else "mfg_hit"):
            continue
        if r["job"] not in {f"main_{setting}_s0", f"baselines_{setting}_s0", f"soundness_{setting}_s0"}:
            continue
        if r["job"].startswith("soundness") and r["variant"] != "null1":
            continue
        result.append(dict(model="Epoch 110", judge="Original predictor", method=r["variant"],
                           mean=float(r["mean"]), lo=float(r["patient_lo"]), hi=float(r["patient_hi"]),
                           n=int(r["n"]), patients=int(r["patients"]), source=relative(path)))
    for version, file in [("Epoch 110", "regrade_b3/summary.csv"), ("Epoch 10", "regrade_old/summary.csv")]:
        path = B2 / file
        for r in csv_rows(path):
            if r["family"] != "orientation_robust" or r["subset"] != "all" or r["axis"] != axis:
                continue
            allowed = {f"main_{setting}_s0", f"baselines_{setting}_s0", f"soundness_{setting}_s0"} if version == "Epoch 110" else {"old_fs_t100_w5" if axis == "field_strength" else "old_mfg_t300_w5"}
            if r["set"] not in allowed or (r["set"].startswith("soundness") and r["variant"] != "null1"):
                continue
            result.append(dict(model=version, judge=f"Held-out judge {r['seed']}", method=r["variant"],
                               mean=float(r["hit_mean"]), lo=float(r["hit_patient_lo"]), hi=float(r["hit_patient_hi"]),
                               n=int(r["n"]), patients=int(r["patients"]), source=relative(path)))
    return result


def tradeoff(axis):
    setting = "field_strength_t100_w5_s0" if axis == "field_strength" else "manufacturer_t300_w3_s0"
    name = "main_" + setting
    path = B3 / name / "metrics.csv"
    key = "fs" if axis == "field_strength" else "mfg"
    originals = {r["example_id"]: r for r in csv_rows(path) if r["judge"] == "original_predictor" and r["variant"] == "cf"}
    rows = [dict(judge="Original predictor", ssim=float(r["ssim"]), gain=float(r[f"{key}_gain"]), patient=r["patient_id"], source=relative(path)) for r in originals.values()]
    held_path = B2 / "regrade_b3/per_slice.csv"
    for r in csv_rows(held_path):
        if r["family"] == "orientation_robust" and r["set"] == name and r["variant"] == "cf":
            rows.append(dict(judge=f"Held-out judge {r['seed']}", ssim=float(originals[r["example_id"]]["ssim"]), gain=float(r["gain"]), patient=r["patient_id"], source=relative(held_path), ssim_source=relative(path)))
    return rows
