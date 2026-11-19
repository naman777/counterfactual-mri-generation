"""Run and audit the three primary profiles as a bounded laptop pilot stage."""
import argparse
import json
import subprocess
import sys

import numpy as np
import pandas as pd

from scripts.v8_generator_dev.full_protocol import select_setting
from scripts.v8_generator_dev.io import ROOT, csv_write, digest, json_write, project_path
from scripts.v10_detail.analyse import crossed_interval
from scripts.v15_shared_residual.inputs import PREPARED


def analyse(prepared, protocol, children, output):
    frames = []
    evidence = {}
    for seed, child in children.items():
        received = json.loads((child / "complete.json").read_text())
        if received["engineering"] or received["cases"] != protocol["cases_per_profile"]:
            raise ValueError("A primary profile did not complete")
        meta = json.loads((child / "protocol.json").read_text())
        if meta["plan_sha256"] != digest(prepared / "protocol.json") or meta["route"]["training_seed"] != seed:
            raise ValueError("The primary profile used a different design")
        table = pd.read_csv(child / "per_image.csv", keep_default_na=False, na_values=[""])
        expected = pd.MultiIndex.from_product([protocol["dataset_indices"], protocol["generation_seeds"],
            ["ordinary_cfg","shared_residual"], ["identity","null","target"],
            [f"heldout_orientation_robust_seed{j}" for j in range(3)]])
        keys = ["dataset_index","generation_seed","sampler","variant","judge"]
        found = pd.MultiIndex.from_frame(table[keys])
        if found.has_duplicates or set(found) != set(expected):
            raise ValueError("Missing or repeated primary case, sampler, control or judge")
        if not table.source_echo_time_norm.eq(table.target_echo_time_norm).all():
            raise ValueError("Source EchoTime changed")
        if not table.actual_sampling_seed.eq(table.dataset_index + table.generation_seed).all():
            raise ValueError("Sampling seed changed")
        for relative, part in table.groupby("array_file"):
            path = (child / relative).resolve()
            if not path.is_relative_to(child) or not part.array_sha256.eq(digest(path)).all():
                raise ValueError("Saved image identity changed")
            evidence[path.relative_to(ROOT).as_posix()] = digest(path)
        for name in ("per_image.csv","protocol.json","inputs.json","complete.json"):
            path = child / name
            evidence[path.relative_to(ROOT).as_posix()] = digest(path)
        frames.append(table)
    frame = pd.concat(frames, ignore_index=True)
    target = frame[frame.variant == "target"].copy()
    keys = ["patient_id","dataset_index","training_seed","generation_seed","judge",
            "direction","condition_support_status","qc_warning"]
    baseline = target[target.sampler == "ordinary_cfg"]
    candidate = target[target.sampler == "shared_residual"]
    paired = candidate.merge(baseline, on=keys, suffixes=("_candidate","_ordinary"),
                             how="outer", validate="one_to_one", indicator=True)
    if not paired._merge.eq("both").all():
        raise ValueError("Primary pair does not match")
    design = protocol["primary"]
    reference = json.loads((PREPARED / "protocol.json").read_text())["reference_protocol"]
    settings = dict(training_seeds=protocol["training_seeds"], evaluation=reference["evaluation"],
        analysis={key: design[key] for key in ("bootstrap_seed","bootstrap_resamples","interval_quantiles")})
    checks = []
    for judge, part in paired.groupby("judge"):
        part = part.copy()
        part["difference"] = part.paired_hit_delta_candidate - part.paired_hit_delta_ordinary
        interval = crossed_interval(part, "difference", settings, adjusted=True)
        replica = part.groupby("training_seed").difference.mean()
        checks.append(dict(judge=judge, metric="target_minus_null_hit", **interval,
            replica_means={str(k):float(v) for k,v in replica.items()},
            meets_margin=bool(interval["ci_low"] > design["editing_lower_bound_above"]
                              and (replica >= design["every_replica_editing_at_least"]).all()),
            positive_lower_bound=bool(interval["ci_low"] > 0)))
    image_rows = paired[paired.judge == checks[0]["judge"]].copy()
    image_rows["difference"] = image_rows.interior_mad_candidate - image_rows.interior_mad_ordinary
    if np.isfinite(image_rows.difference).all():
        interval = crossed_interval(image_rows, "difference", settings, adjusted=True)
        checks.append(dict(metric="target_interior_mad", **interval,
                           meets_margin=bool(interval["ci_high"] < design["target_interior_mad_upper_bound_below"])))
    else:
        checks.append(dict(metric="target_interior_mad", meets_margin=False, unavailable=True))
    support = {str(seed):select_setting(rows, "manufacturer", reference)
               for seed, rows in candidate.groupby("training_seed")}
    csv_write(output / "paired_primary_cases.csv", paired.drop(columns="_merge"))
    json_write(output / "primary_intervals.json", dict(contrasts=checks, original_support_checks=support,
        complete_matrix_evaluated=False, advancement_assessed=False))
    json_write(output / "inputs.json", evidence)
    text = ["# Shared residual sampler primary profiles", "",
        "Both samplers used the same laptop, fine-guide weights, validation sources and noise. "
        "Each named judge's difference is the shared-residual target-minus-null hit change "
        "relative to ordinary classifier-free guidance. Intervals use the declared four-member family.", "",
        "| Judge or image measure | Difference [simultaneous CI] | Margin met |", "|---|---:|---|"]
    for row in checks:
        name = row.get("judge", "Target interior mean absolute distance")
        if row.get("unavailable"):
            value = "Unavailable"
        else:
            multiplier = 100 if "judge" in row else 1
            value = f'{row["estimate"]*multiplier:+.4f} [{row["ci_low"]*multiplier:+.4f}, {row["ci_high"]*multiplier:+.4f}]'
            if "judge" in row:
                value += " pp"
        text.append(f"| {name} | {value} | {row['meets_margin']} |")
    text.extend(["", "This stage covers the declared primary fine-guide comparison on reused development "
                 "patients. The broader guide-control matrix and clinical anatomy assessment remain pending."])
    (output / "RESULTS.md").write_text("\n".join(text)+"\n", encoding="utf-8")
    json_write(output / "complete.json", dict(complete=True, completed_profiles=len(children),
        matched_primary_cases=len(image_rows), named_control_grades=len(frame), full_matrix=False,
        advancement_assessed=False, cloud_spend_usd=0))


def run(prepared, run_id):
    prepared = project_path(prepared)
    protocol = json.loads((prepared / "protocol.json").read_text())
    if not run_id.replace("_", "").isalnum():
        raise ValueError("Use a fresh simple run identifier")
    output = ROOT / "reports/v15_shared_residual" / run_id
    output.mkdir(parents=True, exist_ok=False)
    (output / "README.md").write_text(
        "Producer is scripts.v15_shared_residual.primary. plan.json binds the primary-stage source and "
        "full frozen design. Subprocess logs record each profile, with original arrays and grades in "
        "the corresponding sibling folders. paired_primary_cases.csv and primary_intervals.json report "
        "the paired result. inputs.json binds the verified image and grade hashes. RESULTS.md interprets "
        "the primary profiles. complete.json marks this stage. failed.json preserves any execution error.\n",
        encoding="utf-8")
    children = {seed: output.parent / f"{run_id}_{seed}" for seed in protocol["training_seeds"]}
    json_write(output / "plan.json", dict(prepared=prepared.relative_to(ROOT).as_posix(),
        protocol_sha256=digest(prepared / "protocol.json"), producer_sha256=digest(__file__),
        profiles={str(seed):child.relative_to(ROOT).as_posix() for seed,child in children.items()}))
    try:
        for seed, child in children.items():
            command = [sys.executable, "-u", "-m", "scripts.v15_shared_residual.pilot", "--prepared",
                       str(prepared), "--profile", f"seed{seed}__coarse_fine__correct", "--run-id", child.name]
            with (output / f"seed{seed}.log").open("x", encoding="utf-8") as log:
                subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                               timeout=protocol["max_profile_seconds"]+120, check=True)
        analyse(prepared, protocol, children, output)
    except Exception as error:
        json_write(output / "failed.json", dict(complete=False, reason=str(error)))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--prepared", required=True)
    parser.add_argument("--run-id", required=True)
    run(**vars(parser.parse_args()))
