"""Real checkpoint trajectory and runtime checks. Benchmark outputs are engineering evidence."""
import argparse
import hashlib
import json
import time
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from scripts.v8_generator_dev.adapter_pilot import state_digest
from scripts.v8_generator_dev.io import ROOT, csv_write, digest, json_write, project_path
from scripts.v8_generator_dev.train_smoke import sample_wiring
from scripts.v15_shared_residual.inputs import CONTROLS, FrozenInputs
from scripts.v15_shared_residual.sampler import sample

REPORTS = ROOT / "reports/v15_shared_residual"
PREVIOUS_RESULTS = ROOT / "reports/v13_guide_dropout/results_20261118_01/complete.json"


def with_noise_trace(operation, *args):
    """Observe real initial/reverse draws without changing the sampler API or RNG."""
    randn = torch.randn
    draws = []

    def record(*shape, **kwargs):
        value = randn(*shape, **kwargs)
        draws.append(hashlib.sha256(value.cpu().numpy().tobytes()).hexdigest())
        return value

    with patch.object(torch, "randn", record):
        image = operation(*args)
    return image, draws


def engineering(inputs, output):
    rows = []
    index = inputs.indices[0]
    item = inputs.source(index)
    label = item["manufacturer_label"]
    generation = inputs.reference["evaluation"]["selection_generation_seed"]
    before = state_digest(inputs.parent), state_digest(inputs.conditioner), state_digest(inputs.scheduler)
    for study in inputs.receipts:
        for seed in inputs.reference["training_seeds"]:
            for mode in inputs.protocol["modes"]:
                model = inputs.load_copy(study, seed, mode)
                adapter_before = state_digest(model.adapter)
                args = inputs.arguments(item, label, generation, 3, 3.0)
                parent_image, parent_trace = with_noise_trace(sample_wiring, inputs.conditioner, inputs.parent, *args)
                direct = sample(inputs.conditioner, inputs.parent, None, *args)
                np.testing.assert_array_equal(direct, parent_image)
                outputs = {}
                for control in CONTROLS:
                    guided = inputs.bound(model, index, mode, control)
                    regular_zero = sample_wiring(inputs.conditioner, guided, *args[:-1], 0.0)
                    candidate_zero = sample(inputs.conditioner, inputs.parent, guided, *args[:-1], 0.0)
                    np.testing.assert_array_equal(regular_zero, candidate_zero)
                    null, trace = with_noise_trace(sample, inputs.conditioner, inputs.parent, guided, *args)
                    target_args = inputs.arguments(item, 1 - label, generation, 3, 3.0)
                    target, target_trace = with_noise_trace(sample, inputs.conditioner, inputs.parent, guided, *target_args)
                    if trace != parent_trace or target_trace != trace or len(trace) != 4:
                        raise ValueError("Initial or reverse noise changed")
                    if control == "absent":
                        np.testing.assert_array_equal(null, parent_image)
                        np.testing.assert_array_equal(target, sample_wiring(inputs.conditioner, inputs.parent, *target_args))
                    outputs[control] = null
                    rows.append(dict(study=study, training_seed=seed, mode=mode, guide_control=control,
                                     timestep=3, noise_draw_sha256=trace, zero_guidance_exact=True,
                                     absent_parent_exact=control == "absent", labels_preserved=True))
                difference = float(np.max(np.abs(outputs["zero"] - outputs["absent"])))
                if difference <= 0:
                    raise ValueError("Zero-input and disabled adapter outputs collapsed")
                if state_digest(model.adapter) != adapter_before:
                    raise ValueError("Adapter parameters changed")
                for row in rows[-len(CONTROLS):]:
                    row["zero_vs_absent_max_abs"] = difference
                print("CHECKED", study, seed, mode, flush=True)
    after = state_digest(inputs.parent), state_digest(inputs.conditioner), state_digest(inputs.scheduler)
    if after != before:
        raise ValueError("Frozen model, conditioner or scheduler changed")
    csv_write(output / "endpoints.csv", pd.DataFrame(rows))
    return rows


def benchmark(inputs, output):
    evaluation = inputs.reference["evaluation"]
    seed = inputs.reference["training_seeds"][0]
    generation = evaluation["selection_generation_seed"]
    point = evaluation["fixed_settings"]["manufacturer"]
    model = inputs.load_copy("guide_dropout", seed, "coarse_fine")
    receipt = inputs.read_json(PREVIOUS_RESULTS)
    profile = receipt["evaluations"][f"seed{seed}__coarse_fine__manufacturer__correct"]
    cached_home = project_path(profile)
    requests = pd.read_csv(inputs.require(cached_home / "requests.csv"))
    # First source in each direction, plus the first flagged source. These rules
    # use metadata only and are independent of benchmark edits and grades.
    source_rows = requests[requests.generation_seed == generation]
    selection = source_rows.drop_duplicates("direction").dataset_index.astype(int).tolist()
    warnings = source_rows[source_rows.qc_warning.isin([True, 1, "True", "1"])]
    if warnings.empty or len(selection) != 2:
        raise ValueError("Both directions and a warning source are required")
    selection = list(dict.fromkeys([*selection, int(warnings.dataset_index.iloc[0])]))
    records = []
    for index in selection:
        item = inputs.source(index)
        guide = inputs.bound(model, index, "coarse_fine", "correct")
        row = source_rows[source_rows.dataset_index == index].iloc[0]
        archive = inputs.require(cached_home / row.array_file, row.array_sha256)
        with np.load(archive, allow_pickle=False) as data:
            cached = {kind: data[kind].copy() for kind in ("source", "null", "target")}
        np.testing.assert_array_equal(cached["source"], inputs.val.image_01(index))
        for variant, label in (("null", item["manufacturer_label"]), ("target", 1-item["manufacturer_label"])):
            args = inputs.arguments(item, label, generation, point["timestep"], point["guidance"])
            outputs = {}
            for method, operation, models in (
                ("local_reference", sample_wiring, (inputs.conditioner, guide)),
                ("shared_residual", sample, (inputs.conditioner, inputs.parent, guide)),
            ):
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                started = time.perf_counter()
                image = operation(*models, *args)[0, 0]
                torch.cuda.synchronize()
                seconds = time.perf_counter() - started
                records.append(dict(dataset_index=index, patient_id=item["patient_id"], direction=row.direction,
                                    qc_warning=bool(row.qc_warning), training_seed=seed, generation_seed=generation,
                                    actual_seed=generation+index, variant=variant, method=method,
                                    timestep=point["timestep"], guidance=point["guidance"], seconds=seconds,
                                    peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                                    cloud_reference_max_abs=float(np.max(np.abs(image-cached[variant]))) if method == "local_reference" else None,
                                    cloud_reference_mad=float(np.mean(np.abs(image-cached[variant]))) if method == "local_reference" else None,
                                    cloud_reference_exact=bool(np.array_equal(image,cached[variant])) if method == "local_reference" else None))
                outputs[method] = image
                print("TIMED", index, variant, method, round(seconds, 3), flush=True)
            with (output / f"sample_{index}_{variant}.npz").open("xb") as handle:
                np.savez_compressed(handle, source=cached["source"], cloud_reference=cached[variant], **outputs)
    csv_write(output / "benchmark.csv", pd.DataFrame(records))
    return records


def run(run_id):
    if not run_id.replace("_", "").isalnum():
        raise ValueError("Use a fresh simple report identifier")
    output = REPORTS / run_id
    output.mkdir(parents=True, exist_ok=False)
    (output / "README.md").write_text(
        "Producer is scripts.v15_shared_residual.check. endpoints.csv records each short-trajectory check and "
        "noise trace. benchmark.csv records synchronised laptop timing and cached A10 replay differences. "
        "sample_*.npz contains fixed engineering examples for reference and timing. runtime.json and inputs.json "
        "bind the execution environment and used data. complete.json records completion of the engineering scope.\n",
        encoding="utf-8")
    inputs = FrozenInputs()
    json_write(output / "runtime.json", dict(gpu=torch.cuda.get_device_name(), torch=str(torch.__version__),
        cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(), tf32=False, autocast=False, batch_size=1))
    routes = engineering(inputs, output)
    records = benchmark(inputs, output)
    for path in (ROOT / "scripts/v15_shared_residual/sampler.py", ROOT / "scripts/v15_shared_residual/inputs.py",
                 project_path(__file__), ROOT / "scripts/v8_generator_dev/train_smoke.py",
                 ROOT / "scripts/v14_dropout_audit/shared_residual.py"):
        inputs.require(path)
    json_write(output / "inputs.json", inputs.paths)
    json_write(output / "complete.json", dict(complete=True, engineering_routes=len(routes),
        benchmark_trajectories=len(records), local_only=True, frozen_weights_verified=True,
        exact_noise_and_endpoint_checks=True, full_evaluation=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--run-id", required=True)
    run(**vars(parser.parse_args()))
