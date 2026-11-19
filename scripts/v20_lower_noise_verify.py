"""Check the completed pilot against its raw images and frozen request grid."""

import json

import numpy as np
import pandas as pd

from scripts.v8_generator_dev.io import digest, json_write, verify_inputs
from scripts.v10_detail.metrics import measure
from scripts.v20_lower_noise import BASE


def verify(run_id="laptop_20261118_01"):
    folder = BASE / run_id
    protocol = json.loads((folder / "protocol.json").read_text())
    complete = json.loads((folder / "complete.json").read_text())
    verify_inputs(json.loads((folder / "inputs.json").read_text()))
    if not complete["complete"]:
        raise ValueError("Pilot is incomplete")
    frame = pd.read_csv(folder / "per_image.csv", keep_default_na=False, na_values=[""])
    intervals = pd.read_csv(folder / "paired_intervals.csv", keep_default_na=False, na_values=[""])
    expected_cases = {(idx, step) for idx in protocol["dataset_indices"] for step in protocol["timesteps"]}
    grouped = frame.groupby(["dataset_index", "timestep"])
    if set(grouped.groups) != expected_cases or len(frame) != complete["grades"]:
        raise ValueError("Saved grades do not match the frozen grid")

    judge_names = [f"heldout_orientation_robust_seed{i}" for i in range(3)]
    expected_rows = {(judge, arm) for judge in judge_names for arm in ("identity", "null", "target")}
    metrics = [f"{region}_{suffix}" for region in ("interior", "boundary")
               for suffix in ("mad", "rank_mad", "hf_residual_mad")]
    sources = {}
    for (idx, timestep), rows in grouped:
        if len(rows) != len(expected_rows) or set(zip(rows.judge, rows.variant)) != expected_rows:
            raise ValueError("Missing or duplicate judge/arm grades")
        by_judge = rows.pivot(index="judge", columns="variant", values="target_hit")
        adjusted = rows[rows.variant == "target"].set_index("judge").paired_hit_delta
        pd.testing.assert_series_equal(by_judge.target - by_judge["null"], adjusted, check_names=False)
        if (rows.actual_sampling_seed != protocol["generation_seed"] + idx).any():
            raise ValueError("Generation seed is inconsistent")
        if not rows.source_echo_time_norm.equals(rows.target_echo_time_norm):
            raise ValueError("EchoTime changed")
        paths = rows[["array_file", "array_sha256"]].drop_duplicates()
        if len(paths) != 1:
            raise ValueError("Grade rows refer to multiple arrays")
        name, expected_hash = paths.iloc[0]
        path = folder / name
        if digest(path) != expected_hash:
            raise ValueError(f"Image hash mismatch for {name}")
        with np.load(path, allow_pickle=False) as images:
            source = images["identity"]
            if idx in sources:
                np.testing.assert_array_equal(source, sources[idx])
            sources[idx] = source
            for arm in ("identity", "null", "target"):
                image = images[arm]
                if image.shape != (128, 128) or image.dtype != np.float32 or not np.isfinite(image).all():
                    raise ValueError(f"Invalid array in {name}")
                recomputed = measure(source, image)
                saved = rows[rows.variant == arm]
                for metric in metrics:
                    np.testing.assert_allclose(saved[metric], recomputed[metric], atol=1e-12, rtol=0, equal_nan=True)

    for row in intervals.itertuples():
        judge = row.judge if pd.notna(row.judge) else judge_names[0]
        keep = (frame.judge == judge) & (frame.variant == row.arm)
        if row.direction != "all":
            keep &= frame.direction == row.direction
        paired = frame.loc[keep].pivot(index="patient_id", columns="timestep", values=row.metric)
        values = paired[row.timestep] - paired[protocol["timesteps"][0]]
        if len(values) != row.patients or values.count() != row.complete_patients:
            raise ValueError("Patient coverage differs from the interval table")
        expected = values.mean() if values.notna().all() else np.nan
        np.testing.assert_allclose(row.delta, expected, atol=1e-12, rtol=0, equal_nan=True)

    out = BASE / f"{run_id}_verification"
    out.mkdir(exist_ok=False)
    result = dict(verified_array_groups=len(grouped), verified_grade_rows=len(frame),
                  verified_contrast_point_estimates=len(intervals))
    json_write(out / "verification.json", result)
    (out / "README.md").write_text(
        "Producer: `python -m scripts.v20_lower_noise_verify`. `verification.json` records checks of "
        "the requested patient/setting grid, array hashes, regional metrics, paired judge hit arithmetic, "
        "fixed EchoTime and contrast point estimates. The verifier uses saved judge predictions.\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    verify()
