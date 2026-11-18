"""Gate, validation search and B3 jobs. Test work requires a frozen val choice."""
import itertools
import time
from pathlib import Path

import numpy as np
from scipy.ndimage import laplace
from skimage.metrics import structural_similarity

from scripts.v4_convergence.core import (
    AXES, conditions, gate_conditions, gate_rank, json_write, judges, load_models,
    probabilities, sample, select_indices, target,
)


def metric_rows(images, originals, src, tgt, models, meta):
    rows = []
    ssim = [float(structural_similarity(a, b, data_range=1.0, win_size=7)) for a, b in zip(originals, images)]
    mad = np.abs(images - originals).mean((1, 2))
    hf = np.array([
        float(np.mean(laplace(x) ** 2)) * 1000 for x in images])
    for name, model in models.items():
        before = probabilities(model, originals, name == "original_predictor")
        after = probabilities(model, images, name == "original_predictor")
        for i in range(len(images)):
            row = {**meta[i], "judge": name, "ssim": ssim[i], "mad": float(mad[i]), "hf": float(hf[i])}
            for k in ("mfg", "fs"):
                row[f"{k}_target"] = int(tgt[k][i])
                row[f"{k}_source"] = int(src[k][i])
                row[f"{k}_gain"] = float(after[k][i, int(tgt[k][i])] - before[k][i, int(tgt[k][i])])
                row[f"{k}_hit"] = int(after[k][i].argmax() == tgt[k][i])
                row[f"{k}_stable"] = int(after[k][i].argmax() == before[k][i].argmax())
                row[f"{k}_source_agree"] = int(after[k][i].argmax() == src[k][i])
                row[f"{k}_p_before"] = float(before[k][i, int(tgt[k][i])])
                row[f"{k}_p_after"] = float(after[k][i, int(tgt[k][i])])
            row["both_hit"] = row["mfg_hit"] * row["fs_hit"]
            rows.append(row)
    return rows


def gate(data, checkpoint, assets, output, protocol, smoke=False):
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    c, u, sched, ck = load_models(checkpoint)
    rule = protocol["gate"]
    n = 8 if smoke else rule["n"]
    cond = gate_conditions(data, n)
    models = judges(assets, robust_only=True)
    rows = []
    start = time.perf_counter()
    for w in ([3.0] if smoke else rule["guidance"]):
        pieces = []
        for i in range(0, n, rule["batch_size"]):
            end = min(n, i + rule["batch_size"])
            pieces.append(sample(c, u, sched, {k: v[i:end] for k, v in cond.items()},
                                 np.arange(rule["seed"] + i, rule["seed"] + end), 999, w))
        images = np.concatenate(pieces)
        predictions = {}
        for name, model in models.items():
            probs = probabilities(model, images)
            predictions[name] = {k: float((probs[k].argmax(1) == cond[k]).mean()) for k in ("mfg", "fs")}
            np.savez_compressed(out / f"epoch_{ck['epoch']:04d}_w{w:g}_{name}_probs.npz", **probs)
        row = {"epoch": int(ck["epoch"]), "global_step": int(ck["global_step"]),
               "checkpoint": str(checkpoint), "guidance": w, "n": n, "by_judge": predictions,
               "mean_mfg_hit": float(np.mean([p["mfg"] for p in predictions.values()])),
               "mean_fs_hit": float(np.mean([p["fs"] for p in predictions.values()])),
               "smoke": smoke}
        row["score"] = gate_rank(row)[0]
        row["passed"] = row["mean_mfg_hit"] >= rule["threshold_mfg"] and row["mean_fs_hit"] >= rule["threshold_fs"]
        np.savez_compressed(out / f"epoch_{ck['epoch']:04d}_w{w:g}_images.npz", images=images, **cond)
        json_write(out / f"epoch_{ck['epoch']:04d}_w{w:g}.json", row)
        rows.append(row)
        print("GATE", row, flush=True)
    return {"rows": rows, "seconds": time.perf_counter() - start}


def validation_jobs(protocol):
    return [{"kind": "validation", "split": "val", "axis": axis, "t": t, "w": w,
             "name": f"val_{axis}_t{t}_w{w:g}", "n": protocol["validation"]["n"], "offset": 0}
            for axis in AXES for t, w in itertools.product(protocol["validation"]["t"], protocol["validation"]["w"])]


def test_jobs(protocol, selected):
    if set(selected) != set(AXES) or any(r["selection_split"] != "val" for r in selected.values()):
        raise ValueError("B3 requires frozen validation choices for both axes")
    result = []
    def add(kind, axis, t, w, n=100, offset=0):
        result.append({"kind": kind, "split": "test", "axis": axis, "t": int(t), "w": float(w),
                       "name": f"{kind}_{axis}_t{t}_w{w:g}_s{offset}", "n": n, "offset": offset})
    for axis in AXES:
        t, w = selected[axis]["t"], selected[axis]["w"]
        for offset in protocol["test"]["seed_offsets"]:
            add("main", axis, t, w, offset=offset)
        ft, fw = protocol["test"]["fixed_comparators"][axis]
        if (ft, fw) != (t, w):
            add("fixed", axis, ft, fw)
        for kind in ("soundness", "baselines"):
            add(kind, axis, t, w)
        for st in protocol["test"]["stronger"][axis]:
            add("stronger", axis, st, 5.0, n=50)
    add("joint", "both", 300, 5.0, n=50)
    return result


def run_job(data, checkpoint, assets, output, job, protocol):
    import pandas as pd
    from src.inference.baselines import make_all_baselines
    from scripts.v4_convergence.core import digest
    out = Path(output) / job["name"]
    out.mkdir(parents=True, exist_ok=False)
    json_write(out / "job.json", {**job, "checkpoint_sha256": digest(checkpoint)})
    split, axis, n = job["split"], job["axis"], job["n"]
    idx = select_indices(data, split, 100)[:n]
    src = conditions(data, split, idx)
    tgt = target(src, axis) if axis != "both" else {k: (1 - v if k != "echo" else v.copy()) for k, v in src.items()}
    images = data[f"{split}_x"][idx]
    seeds = np.arange(42, 42 + n) + job["offset"]
    meta = [{"example_id": f"example_{i:03d}", "patient_id": str(data[f"{split}_pid"][j]),
             "image_path": str(data[f"{split}_path"][j]), "axis": axis, "split": split,
             "t": job["t"], "w": job["w"], "offset": job["offset"]} for i, j in enumerate(idx)]
    models = judges(assets, robust_only=split == "val")
    c, u, sched, _ = load_models(checkpoint)
    def edit(x, cond, seed, guidance):
        return np.concatenate([sample(c, u, sched, {k: v[i:i + 10] for k, v in cond.items()},
                                     seed[i:i + 10], job["t"], guidance, x[i:i + 10]) for i in range(0, n, 10)])
    start = time.perf_counter()
    outputs = {}
    if job["kind"] == "baselines":
        rng = np.random.default_rng(42)
        for i in range(n):
            label = AXES[axis]
            candidates = np.where(data[f"train_{label}"] == tgt[label][i])[0]
            refidx = int(rng.choice(candidates))
            ref = data["train_x"][refidx]
            for name, x in make_all_baselines(images[i], ref, 42 + i).items():
                outputs.setdefault(name, []).append(x)
        outputs = {k: np.stack(v) for k, v in outputs.items()}
    else:
        cf = edit(images, tgt, seeds, job["w"])
        outputs["cf"] = cf
        if job["kind"] == "soundness":
            outputs["null1"] = edit(images, src, seeds, job["w"])
            outputs["null2"] = edit(outputs["null1"], src, seeds + 1000, job["w"])
            outputs["null3"] = edit(outputs["null2"], src, seeds + 2000, job["w"])
            outputs["uncond"] = edit(images, src, seeds, 0.0)
            outputs["cycle"] = edit(cf, src, seeds + 1000, job["w"])
    rows = []
    for kind, x in outputs.items():
        # Keep target labels fixed across controls: source agreement is a separate column.
        rows.extend(metric_rows(x, images, src, tgt, models, [{**m, "variant": kind} for m in meta]))
    np.savez_compressed(out / "images.npz", **outputs, source_images=images, source_mfg=src["mfg"],
                        source_fs=src["fs"], source_echo=src["echo"], indices=idx)
    pd.DataFrame(rows).to_csv(out / "metrics.csv", index=False)
    result = {"name": job["name"], "kind": job["kind"], "axis": axis, "split": split,
              "t": job["t"], "w": job["w"], "n": n, "seconds": time.perf_counter() - start}
    if split == "val":
        table = pd.DataFrame(rows)
        key = AXES[axis]
        other = "fs" if key == "mfg" else "mfg"
        result.update(ssim=float(table.ssim.mean()), hit=float(table[f"{key}_hit"].mean()),
                      gain=float(table[f"{key}_gain"].mean()), stability=float(table[f"{other}_stable"].mean()))
    json_write(out / "result.json", result)
    print("JOB", result, flush=True)
    return result
