"""Verify Kaggle pilot arrays, grade them locally and compute the frozen primary intervals."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from scripts.v2_quickwins.judge_model import load_judge
from scripts.v7_geometry.support import request_support
from scripts.v8_generator_dev.data import CanonicalDataset
from scripts.v8_generator_dev.full_protocol import select_setting
from scripts.v8_generator_dev.full_train import deterministic_runtime
from scripts.v8_generator_dev.io import ROOT, csv_write, digest, json_write
from scripts.v10_detail.analyse import crossed_interval
from scripts.v10_detail.metrics import warning_for_index
from scripts.v15_shared_residual.pilot import grade
from scripts.v16_kaggle_prepare import BASE, PLAN, PREPARED


def review(download):
    download = Path(download).resolve()
    receipt = json.loads((download/"receipt.json").read_text())
    protocol = json.loads((PLAN/"protocol.json").read_text())
    point = protocol["point"]
    expected = {(seed, generation, case) for seed in protocol["training_seeds"]
        for generation in protocol["generation_seeds"] for case in range(len(protocol["dataset_indices"]))}
    if (receipt["mode"] != "pilot" or receipt["study_plan_sha256"] != digest(PLAN/"protocol.json")
        or {profile["training_seed"] for profile in receipt["result"]} != set(protocol["training_seeds"])
        or len(receipt["result"]) != 3 or not all(r["cases"] == 45 and r["frozen_weights"] for r in receipt["result"])):
        raise ValueError("Kaggle receipt does not complete the frozen primary matrix")
    deterministic_runtime()
    reference = json.loads((PREPARED/"protocol.json").read_text())["reference_protocol"]
    val = CanonicalDataset(reference["view"], "val")
    train = CanonicalDataset(reference["view"], "train")
    judges = {}
    ledger = json.loads((PREPARED/"inputs.json").read_text())
    for j, relative in enumerate(reference["evaluation"]["judge_paths"]):
        path = ROOT/relative
        if ledger[relative] != digest(path):
            raise ValueError("Frozen judge identity changed")
        judges[f"heldout_orientation_robust_seed{j}"] = load_judge(path, "cuda").eval().requires_grad_(False)
    thresholds = json.loads((PREPARED/"mask_thresholds.json").read_text())["thresholds"]["noise_aware_v1"]
    evidence = {path.relative_to(ROOT).as_posix():digest(path) for path in (download/"receipt.json",
        PLAN/"protocol.json", BASE/"pilot_20261118_01/dataset/pilot-input.json", Path(__file__))}
    case_inputs = []
    for index in protocol["dataset_indices"]:
        item = val[index]
        manufacturer, field, echo = item["manufacturer_label"], item["field_strength_label"], float(item["echo_time_norm"])
        source = val.image_01(index)
        case_inputs.append((item, source, request_support(train.frame,1-manufacturer,field,echo,5),
                            warning_for_index(source,index,thresholds)))
    grades = []
    seen = set()
    methods = ("ordinary_cfg","shared_residual")
    for training in protocol["training_seeds"]:
        journal = download/f"seed{training}.jsonl"
        evidence[journal.relative_to(ROOT).as_posix()] = digest(journal)
        for line in journal.read_text().splitlines():
            record = json.loads(line)
            case, generation = record["case"], record["generation_seed"]
            key = (training,generation,case)
            if record["training_seed"] != training or key in seen or key not in expected:
                raise ValueError("Unexpected or duplicate remote case")
            seen.add(key)
            index = protocol["dataset_indices"][case]
            if record["actual_sampling_seed"] != generation+index:
                raise ValueError("Sampling seed changed")
            path = download/f"seed{training}_g{generation}_case{case:02d}.npz"
            if path.name != record["array_file"] or digest(path) != record["array_sha256"]:
                raise ValueError("Saved output hash mismatch")
            evidence[path.relative_to(ROOT).as_posix()] = record["array_sha256"]
            item, source, support, warning = case_inputs[case]
            manufacturer = item["manufacturer_label"]
            echo = float(item["echo_time_norm"])
            request = dict(case=case, dataset_index=index, patient_id=item["patient_id"], training_seed=training,
                generation_seed=generation, actual_sampling_seed=generation+index, axis="manufacturer",
                direction=f"{manufacturer}_to_{1-manufacturer}", target_manufacturer=1-manufacturer,
                target_field_strength=item["field_strength_label"], source_echo_time_norm=echo,
                target_echo_time_norm=echo, **point, **support, **warning)
            with np.load(path, allow_pickle=False) as arrays:
                if set(arrays.files) != {f"{method}_{variant}" for method in methods for variant in ("null","target")}:
                    raise ValueError("Saved paired outputs changed")
                for method in methods:
                    images = {"identity":source,"null":arrays[f"{method}_null"],"target":arrays[f"{method}_target"]}
                    if any(a.shape!=(128,128) or a.dtype!=np.float32 or not np.isfinite(a).all()
                           or a.min()<0 or a.max()>1 for a in images.values()):
                        raise ValueError("Invalid saved MRI output")
                    grades.extend(grade(images, SimpleNamespace(val=val), index, 1-manufacturer, judges,
                                        dict(request,sampler=method)))
        print("Regraded", training, flush=True)
    if seen != expected:
        raise ValueError("Missing remote case in frozen design")
    out = BASE/"primary_review_20261118_01"
    out.mkdir(exist_ok=False)
    (out/"README.md").write_text("Producer is scripts.v16_kaggle_review. per_image.csv contains the local grading of "
        "each downloaded pair. primary_intervals.json applies the frozen primary margins and support checks. "
        "RESULTS.md interprets the three profiles. inputs.json verifies the download and design identities.\n",
        encoding="utf-8")
    frame = pd.DataFrame(grades)
    csv_write(out/"per_image.csv",frame)
    primary = protocol["primary"]
    settings = dict(training_seeds=protocol["training_seeds"], evaluation=reference["evaluation"],
        analysis={field:primary[field] for field in ("bootstrap_seed","bootstrap_resamples","interval_quantiles")})
    target = frame[frame.variant=="target"]
    ordinary = target[target.sampler==methods[0]]
    candidate = target[target.sampler==methods[1]]
    keys = ["patient_id","dataset_index","training_seed","generation_seed","judge"]
    pairs = candidate.merge(ordinary,on=keys,how="inner",suffixes=("_new","_base"),validate="one_to_one")
    if len(pairs) != len(expected)*len(judges):
        raise ValueError("Matched graded output incomplete")
    contrasts = []
    for judge, rows in pairs.groupby("judge"):
        rows = rows.assign(difference=rows.paired_hit_delta_new - rows.paired_hit_delta_base)
        result = crossed_interval(rows,"difference",settings,adjusted=True)
        replica = rows.groupby("training_seed").difference.mean()
        contrasts.append(dict(judge=judge, metric="paired_hit_delta", **result,
            replica_estimates={str(k):float(v) for k,v in replica.items()},
            passes=bool(result["ci_low"]>primary["editing_lower_bound_above"] and
                        (replica>=primary["every_replica_editing_at_least"]).all()),
            positive_lower_bound=bool(result["ci_low"]>0)))
    image = pairs[pairs.judge == sorted(judges)[0]].copy()
    image["difference"] = image.interior_mad_new-image.interior_mad_base
    if image.difference.notna().all():
        result = crossed_interval(image,"difference",settings,adjusted=True)
        contrasts.append(dict(metric="target_interior_mad",**result,
                              passes=result["ci_high"]<primary["target_interior_mad_upper_bound_below"]))
    else:
        contrasts.append(dict(metric="target_interior_mad",passes=False,unavailable=True))
    support = {str(seed):select_setting(rows,"manufacturer",reference)
               for seed,rows in candidate.groupby("training_seed")}
    merits_wider_control_run = (all(c["passes"] for c in contrasts)
        and any(c.get("positive_lower_bound",False) for c in contrasts)
        and all(value["selected"] is not None for value in support.values()))
    json_write(out/"primary_intervals.json",dict(contrasts=contrasts,original_support_checks=support,
        primary_supports_wider_control_run=bool(merits_wider_control_run),full_matrix_evaluated=False))
    lines = ["# Kaggle manufacturer pilot on the primary profiles","",
        f"Locally verified {len(seen)} paired cases and graded {len(frame)} image and judge combinations. "
        "Both samplers ran at the frozen setting with the same weights, source conditions and noise.","",
        "| Measure | Shared residual minus ordinary CFG [simultaneous CI] | Margin met |","|---|---:|---|"]
    for row in contrasts:
        name = row.get("judge","Target interior mean absolute distance")
        if row.get("unavailable"):
            interval = "Unavailable"
        else:
            factor = 100 if "judge" in row else 1
            interval = f'{factor*row["estimate"]:+.4f} [{factor*row["ci_low"]:+.4f}, {factor*row["ci_high"]:+.4f}]'
            if "judge" in row:
                interval += " pp"
        lines.append(f"| {name} | {interval} | {row['passes']} |")
    lines += ["", f"Primary evidence supports a wider control evaluation: **{merits_wider_control_run}**.",
        "The primary profiles use the existing development cohort. Clinical anatomy and lesion preservation "
        "require separate review.", ""]
    (out/"RESULTS.md").write_text("\n".join(lines),encoding="utf-8")
    json_write(out/"inputs.json",evidence)
    json_write(out/"complete.json",dict(complete=True,verified_cases=len(seen),grades=len(frame),
        primary_supports_wider_control_run=bool(merits_wider_control_run)))
    print(out,flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--download",required=True)
    review(**vars(parser.parse_args()))
