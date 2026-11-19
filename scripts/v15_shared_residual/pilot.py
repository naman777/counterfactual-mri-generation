"""Generate one complete matched local profile at the frozen manufacturer setting."""
import argparse
import json
import time

import numpy as np
import pandas as pd
import torch

from scripts.v2_quickwins.judge_model import load_judge
from scripts.v7_geometry.support import request_support
from scripts.v8_generator_dev.adapter_pilot import state_digest
from scripts.v8_generator_dev.data import CanonicalDataset
from scripts.v8_generator_dev.evaluate_full import predict_judge, source_native
from scripts.v8_generator_dev.io import ROOT, csv_write, digest, json_write, project_path, verify_inputs
from scripts.v8_generator_dev.train_smoke import sample_wiring
from scripts.v10_detail.metrics import measure, warning_for_index
from scripts.v13_guide_dropout.evaluate import json_safe
from scripts.v15_shared_residual.inputs import FrozenInputs
from scripts.v15_shared_residual.sampler import sample


def runtime():
    return dict(gpu=torch.cuda.get_device_name(), torch=str(torch.__version__), cuda=torch.version.cuda,
                cudnn=torch.backends.cudnn.version(), tf32=False, autocast=False, batch_size=1)


def grade(images, inputs, index, target, judges, metadata):
    names = ("identity", "null", "target")
    stack = source_native(np.stack([images[name] for name in names]), inputs.val.plans[index])
    metrics = {name: measure(images["identity"], images[name]) for name in names}
    for judge_name, judge in judges.items():
        probabilities = predict_judge(judge, stack, "cuda")
        decisions = probabilities["mfg"].argmax(1)
        strength = probabilities["fs"].argmax(1)
        hits = (decisions == target).astype(int)
        for position, variant in enumerate(names):
            yield dict(metadata, judge=judge_name, variant=variant, target_hit=int(hits[position]),
                       target_probability=float(probabilities["mfg"][position, target]),
                       paired_hit_delta=int(hits[position]-hits[1]),
                       other_axis_stable=int(strength[position] == strength[0]), **metrics[variant])


def run(prepared, profile, run_id, *, engineering=False):
    prepared = project_path(prepared)
    ledger = json.loads((prepared / "inputs.json").read_text())
    verify_inputs(ledger)
    protocol = json.loads((prepared / "protocol.json").read_text())
    if profile not in protocol["profiles"]:
        raise ValueError("Unknown declared profile")
    if not run_id.replace("_", "").isalnum():
        raise ValueError("Use a simple fresh run identifier")
    route = protocol["profiles"][profile]
    started = time.monotonic()
    inputs = FrozenInputs()
    if inputs.indices != protocol["dataset_indices"] or runtime() != protocol["runtime"]:
        raise ValueError("Laptop runtime or frozen source selection changed")
    point = protocol["point"] if not engineering else dict(timestep=3, guidance=protocol["point"]["guidance"])
    mode, control = route["mode"], route["control"]
    model = inputs.load_copy("guide_dropout", route["training_seed"], mode)
    evaluation = inputs.reference["evaluation"]
    frozen = (inputs.parent, inputs.conditioner, inputs.scheduler, model.adapter)
    before = [state_digest(module) for module in frozen]
    train = CanonicalDataset(inputs.reference["view"], "train")
    judges = {f"heldout_orientation_robust_seed{j}": load_judge(inputs.require(path), "cuda").eval().requires_grad_(False)
              for j, path in enumerate(evaluation["judge_paths"])}
    thresholds = inputs.read_json("reports/v13_guide_dropout/preparation_20261118_01/mask_thresholds.json")
    thresholds = thresholds["thresholds"]["noise_aware_v1"]
    output = ROOT / "reports/v15_shared_residual" / run_id
    output.mkdir(parents=True, exist_ok=False)
    (output / "arrays").mkdir()
    (output / "README.md").write_text(
        "Producer is scripts.v15_shared_residual.pilot. protocol.json fixes the route and engineering scope. "
        "arrays/ holds each matched source, ordinary CFG and shared-residual output. journal.jsonl preserves "
        "completed requests and all grades if interrupted. per_image.csv contains every named judge's "
        "identity, null and target grades for both samplers. inputs.json binds the actual source/code hashes. "
        "complete.json marks completion of this profile only.\n", encoding="utf-8")
    json_write(output / "protocol.json", dict(profile=profile, plan_sha256=digest(prepared / "protocol.json"),
        route=route, point=point, runtime=runtime(), engineering=engineering))
    grades = []
    cases = 0
    indices = inputs.indices[:1] if engineering else inputs.indices
    generations = protocol["generation_seeds"][:1] if engineering else protocol["generation_seeds"]
    with (output / "journal.jsonl").open("x", encoding="utf-8") as journal:
        for generation in generations:
            for index in indices:
                if time.monotonic() - started >= protocol["max_profile_seconds"]:
                    raise TimeoutError("Local profile time limit reached")
                item = inputs.source(index)
                source = inputs.val.image_01(index)
                source_label, field = item["manufacturer_label"], item["field_strength_label"]
                echo = float(item["echo_time_norm"])
                target = 1 - source_label
                guide = inputs.bound(model, index, mode, control)
                support = request_support(train.frame, target, field, echo, 5)
                warning = warning_for_index(source, index, thresholds)
                guide_index = inputs.indices[(inputs.indices.index(index)+1) % len(inputs.indices)] if control == "wrong" else index
                request = dict(case=cases, patient_id=item["patient_id"], dataset_index=index,
                               axis="manufacturer", direction=f"{source_label}_to_{target}",
                               training_seed=route["training_seed"], generation_seed=generation,
                               actual_sampling_seed=generation+index, mode=mode, guide_control=control,
                               guide_dataset_index=guide_index, guide_patient_id=inputs.val.frame.iloc[guide_index].patient_id,
                               source_echo_time_norm=echo, target_echo_time_norm=echo, target_field_strength=field,
                               target_manufacturer=target, **point, **support, **warning)
                arrays = {"source": source}
                case_grades = []
                for method in ("ordinary_cfg", "shared_residual"):
                    edits = {"identity": source}
                    for variant, label in (("null", source_label), ("target", target)):
                        args = inputs.arguments(item, label, generation, point["timestep"], point["guidance"])
                        if method == "ordinary_cfg":
                            edit = sample_wiring(inputs.conditioner, guide, *args)[0, 0]
                        else:
                            edit = sample(inputs.conditioner, inputs.parent, guide, *args)[0, 0]
                            if control == "absent":
                                np.testing.assert_array_equal(edit, arrays[f"ordinary_cfg_{variant}"])
                        edits[variant] = arrays[f"{method}_{variant}"] = edit
                    case_grades.extend(grade(edits, inputs, index, target, judges, dict(request, sampler=method)))
                name = f"arrays/case_{cases:05d}.npz"
                with (output / name).open("xb") as handle:
                    np.savez_compressed(handle, **arrays)
                array_hash = digest(output / name)
                for row in case_grades:
                    row.update(array_file=name, array_sha256=array_hash)
                grades.extend(case_grades)
                journal.write(json.dumps(json_safe(case_grades), allow_nan=False) + "\n")
                journal.flush()
                cases += 1
                print("PROFILE", profile, f"{cases}/{len(indices)*len(generations)}", flush=True)
    if [state_digest(module) for module in frozen] != before:
        raise ValueError("Frozen state changed during evaluation")
    verify_inputs(ledger)
    verify_inputs(inputs.paths)
    csv_write(output / "per_image.csv", pd.DataFrame(grades))
    json_write(output / "inputs.json", dict(ledger, **inputs.paths))
    json_write(output / "complete.json", dict(complete=True, profile=profile, cases=cases,
        grades=len(grades), engineering=engineering, seconds=time.monotonic()-started))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    for flag in ("prepared", "profile", "run-id"):
        parser.add_argument("--"+flag, required=True)
    parser.add_argument("--engineering", action="store_true")
    run(**vars(parser.parse_args()))
