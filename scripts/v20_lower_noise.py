"""Paired manufacturer-only development pilot with one frozen adapter."""

import argparse
import json
import time

import numpy as np
import pandas as pd

from scripts.v2_quickwins.judge_model import load_judge
from scripts.v8_generator_dev.adapter_pilot import state_digest
from scripts.v8_generator_dev.io import ROOT, csv_write, digest, json_write, verify_inputs
from scripts.v8_generator_dev.train_smoke import sample_wiring
from scripts.v13_guide_dropout.evaluate import json_safe
from scripts.v15_shared_residual.inputs import FrozenInputs
from scripts.v15_shared_residual.pilot import grade, runtime


BASE = ROOT / "reports/v20_anatomy_preservation"
DESIGN = ROOT / "reports/v19_hybrid/preparation_20261118_01/protocol.json"
REMOTE = ROOT / "reports/v16_kaggle/execution_20261118_01/download"
REFERENCE = ROOT / "reports/v16_kaggle/primary_review_20261118_01"
TIMESTEPS = (300, 200, 100)
REGIONS = ("interior", "boundary")


def run(run_id):
    out = BASE / run_id
    out.mkdir(parents=True, exist_ok=False)
    (out / "arrays").mkdir()
    (out / "README.md").write_text(
        "Producer: `python -m scripts.v20_lower_noise`. `protocol.json` freezes the design. "
        "`journal.jsonl`, `arrays/`, `per_image.csv` and `replays.json` preserve grades, "
        "images and baseline verification. `paired_intervals.csv` and `RESULTS.md` summarise "
        "paired patient effects. `inputs.json` and `complete.json` record provenance and closure.\n",
        encoding="utf-8",
    )
    frozen_design = json.loads(DESIGN.read_text())
    seed = frozen_design["training_seeds"][0]
    generation = frozen_design["generation_seeds"][0]
    indices = frozen_design["dataset_indices"]
    guidance = frozen_design["point"]["guidance"]
    protocol = dict(
        scope="Manufacturer development pilot, one fixed training and generation seed",
        training_seed=seed, generation_seed=generation, dataset_indices=indices,
        timesteps=TIMESTEPS, guidance=guidance, runtime=runtime(),
        baseline="Verified Kaggle ordinary CFG arrays, with one laptop target/null replay per direction",
        sentinel_pixel_atol=1e-4, sentinel_probability_atol=1e-5,
        rng="Same initial noise draw for each case at all settings and both requests. Within each timestep, "
            "target and null share the complete reverse stream. Across timesteps the stream is "
            "restarted from the same seed, with draws aligned by iteration from the start, not absolute step.",
        outcomes="Null and target source discrepancy (interior/boundary MAD, rank MAD and HF residual MAD); "
                 "each judge's target-minus-null hits, target hits and null hits. All warnings retained.",
        bootstrap_seed=583600, bootstrap_replicates=10000,
        decision="Descriptive pilot. Positive follow-up requires lower interior and boundary MAD in both arms "
                 "and point losses of at most 5 pp in each judge's target-minus-null effect, with intervals "
                 "shown. This point screen is for prioritisation, not a noninferiority conclusion.",
    )
    json_write(out / "protocol.json", protocol)

    inputs = FrozenInputs()
    if inputs.indices != indices or frozen_design["point"]["timestep"] != TIMESTEPS[0]:
        raise ValueError("Baseline plan mismatch")
    verify_inputs(json.loads((REFERENCE / "inputs.json").read_text()))
    base = pd.read_csv(inputs.require(REFERENCE / "per_image.csv"), keep_default_na=False, na_values=[""])
    base = base[(base.sampler == "ordinary_cfg") & (base.training_seed == seed)
                & (base.generation_seed == generation)].copy()
    request_columns = ["case", "dataset_index", "patient_id", "direction", "target_manufacturer",
                       "target_field_strength", "source_echo_time_norm", "target_echo_time_norm",
                       "condition_support_status", "qc_warning", "qc_reasons"]
    requests = base[request_columns].drop_duplicates().set_index("dataset_index", drop=False)
    if set(requests.index) != set(indices) or not requests.index.is_unique or len(base) != 9 * len(indices):
        raise ValueError("Baseline grade selection is incomplete")
    journal = inputs.require(REMOTE / f"seed{seed}.jsonl")
    cached = {row["case"]: row for row in map(json.loads, journal.read_text().splitlines())
              if row["generation_seed"] == generation}
    model = inputs.load_copy("guide_dropout", seed, "coarse_fine")
    judges = {f"heldout_orientation_robust_seed{i}": load_judge(inputs.require(path), "cuda").eval().requires_grad_(False)
              for i, path in enumerate(inputs.reference["evaluation"]["judge_paths"])}
    frozen = (inputs.conditioner, inputs.parent, inputs.scheduler, model.adapter)
    state = [state_digest(module) for module in frozen]
    sentinels = set(requests.groupby("direction", sort=False).dataset_index.first())
    if len(sentinels) != 2:
        raise ValueError("Missing manufacturer direction")

    rows, replays = [], []
    started = time.monotonic()
    with (out / "journal.jsonl").open("x", encoding="utf-8") as log:
        for index in indices:
            request = requests.loc[index].to_dict()
            item = inputs.source(index)
            original = item["manufacturer_label"]
            target = 1 - original
            echo = float(item["echo_time_norm"])
            if (target != request["target_manufacturer"] or item["patient_id"] != request["patient_id"]
                    or item["field_strength_label"] != request["target_field_strength"]
                    or not np.allclose([request["source_echo_time_norm"], request["target_echo_time_norm"]],
                                       echo, atol=1e-7, rtol=0)):
                raise ValueError("Source conditions differ from cached requests")
            source = inputs.val.image_01(index)
            bound = inputs.bound(model, index, "coarse_fine", "correct")
            record = cached[request["case"]]
            if record["actual_sampling_seed"] != generation + index:
                raise ValueError("Cached noise seed changed")
            path = inputs.require(REMOTE / record["array_file"], record["array_sha256"])
            with np.load(path, allow_pickle=False) as archive:
                baseline = {name: archive[f"ordinary_cfg_{name}"] for name in ("null", "target")}
            for step in TIMESTEPS:
                images = {"identity": source}
                if step == TIMESTEPS[0] and index not in sentinels:
                    images.update(baseline)
                else:
                    for name, label in (("null", original), ("target", target)):
                        args = inputs.arguments(item, label, generation, step, guidance)
                        images[name] = sample_wiring(inputs.conditioner, bound, *args)[0, 0]
                if step == TIMESTEPS[0] and index in sentinels:
                    for name, reference in baseline.items():
                        np.testing.assert_allclose(images[name], reference, atol=protocol["sentinel_pixel_atol"], rtol=0)
                        replays.append(dict(index=index, control=name, max_abs=float(np.abs(images[name] - reference).max())))
                    local = pd.DataFrame(grade(images, inputs, index, target, judges, {}))
                    saved = base[base.dataset_index == index].set_index(["judge", "variant"]).sort_index()
                    local = local.set_index(["judge", "variant"]).sort_index()
                    pd.testing.assert_index_equal(saved.index, local.index)
                    np.testing.assert_array_equal(local.target_hit, saved.target_hit)
                    np.testing.assert_allclose(local.target_probability, saved.target_probability,
                                               atol=protocol["sentinel_probability_atol"], rtol=0)
                    images.update(baseline)
                metadata = dict(request, timestep=step, guidance=guidance, training_seed=seed,
                                generation_seed=generation, actual_sampling_seed=generation + index,
                                runtime_arm="archived_kaggle" if step == TIMESTEPS[0] else "laptop_cuda")
                filename = out / "arrays" / f"index{index}_t{step}.npz"
                with filename.open("xb") as handle:
                    np.savez_compressed(handle, **images)
                grades = list(grade(images, inputs, index, target, judges, dict(
                    metadata, array_file=filename.relative_to(out).as_posix(), array_sha256=digest(filename))))
                log.write(json.dumps(json_safe(grades), allow_nan=False) + "\n")
                log.flush()
                rows.extend(grades)
            print(f"Compared {len(rows) // 27}/{len(indices)} patients", flush=True)

    if state != [state_digest(module) for module in frozen] or len(replays) != 4:
        raise ValueError("Frozen state or replay checks incomplete")
    frame = pd.DataFrame(rows)
    if len(frame) != len(indices) * len(TIMESTEPS) * 9:
        raise ValueError("Incomplete grading")
    csv_write(out / "per_image.csv", frame)
    json_write(out / "replays.json", replays)
    for path in (DESIGN, REFERENCE / "inputs.json", __file__, ROOT / "scripts/v8_generator_dev/train_smoke.py",
                 ROOT / "scripts/v10_detail/metrics.py", ROOT / "scripts/v15_shared_residual/pilot.py"):
        inputs.require(path)
    verify_inputs(inputs.paths)
    json_write(out / "inputs.json", inputs.paths)
    summarise(out, frame, protocol)
    json_write(out / "complete.json", dict(complete=True, patients=len(indices), grades=len(frame),
               reference_replays=len(replays), seconds=time.monotonic() - started))


def summarise(out, frame, protocol):
    rng = np.random.default_rng(protocol["bootstrap_seed"])
    comparisons = []
    judges = sorted(frame.judge.unique())
    groups = ["all", *sorted(frame.direction.unique())]
    for group in groups:
        data = frame if group == "all" else frame[frame.direction == group]
        patients = sorted(data.patient_id.unique())
        draws = rng.integers(0, len(patients), (protocol["bootstrap_replicates"], len(patients)))

        def add(part, column, label, arm, judge):
            pivot = part.pivot(index="patient_id", columns="timestep", values=column).reindex(patients)
            for step in TIMESTEPS[1:]:
                delta = (pivot[step] - pivot[TIMESTEPS[0]]).to_numpy(dtype=float)
                complete = np.isfinite(delta).all()
                ci = np.quantile(delta[draws].mean(axis=1), [.025, .975]) if complete else (None, None)
                comparisons.append(dict(direction=group, timestep=step, arm=arm, judge=judge, metric=label,
                    patients=len(patients), complete_patients=int(np.isfinite(delta).sum()),
                    baseline=float(pivot[TIMESTEPS[0]].mean()) if complete else None,
                    candidate=float(pivot[step].mean()) if complete else None,
                    delta=float(delta.mean()) if complete else None, ci_low=ci[0], ci_high=ci[1]))

        for arm in ("null", "target"):
            for region in REGIONS:
                part = data[(data.variant == arm) & (data.judge == judges[0])]
                for metric in ("mad", "rank_mad", "hf_residual_mad"):
                    column = f"{region}_{metric}"
                    add(part, column, column, arm, "")
        for judge in judges:
            selected = data[data.judge == judge]
            for arm in ("null", "target"):
                part = selected[selected.variant == arm]
                add(part, "target_hit", "target_hit", arm, judge)
                if arm == "target":
                    add(part, "paired_hit_delta", "paired_hit_delta", arm, judge)
    table = pd.DataFrame(comparisons)
    csv_write(out / "paired_intervals.csv", table)
    primary = table[(table.direction == "all") & table.metric.isin(["interior_mad", "boundary_mad", "paired_hit_delta"])]
    lines = ["# Lower-noise manufacturer pilot", "",
             "Ordinary CFG at timesteps 200 and 100 compared with its frozen timestep-300 baseline. "
             "One adapter and generation seed were fixed before sampling. All selected development patients "
             "and source warnings are included. Baseline outcomes are archival Kaggle images verified by four laptop "
             "sentinel replays. Values are changes from baseline with descriptive 95% paired patient intervals.", "",
             "| Timestep | Outcome | Arm | Change [95% CI] |", "|---:|---|---|---:|"]
    for row in primary.sort_values(["timestep", "metric", "arm", "judge"], ascending=[False, True, True, True]).itertuples():
        scale = 100 if row.judge else 1
        unit = " pp" if row.judge else ""
        label = f"{row.judge} target minus null" if row.judge else row.metric.replace("_", " ")
        fmt = f"{scale * row.delta:+.4f} [{scale * row.ci_low:+.4f}, {scale * row.ci_high:+.4f}]{unit}"
        lines.append(f"| {row.timestep} | {label} | {row.arm} | {fmt if pd.notna(row.delta) else 'Unavailable'} |")
    lines.append("")
    for step in TIMESTEPS[1:]:
        part = primary[primary.timestep == step]
        better = bool((part[part.judge == ""].delta < 0).all())
        editing = bool((part[part.judge != ""].delta >= -.05).all())
        lines.append(f"Timestep {step}: lower MAD in both regions and arms **{better}**. "
                     f"Every judge within the descriptive editing point tolerance **{editing}**.")
    lines += ["", "Direction-specific intervals and additional texture proxies are in `paired_intervals.csv`. "
              "Regional metrics use source-intensity masks. Clinical anatomy and lesion preservation remain unverified. "
              "Training-replica and generation-seed variability require a separate comparison.", ""]
    (out / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--run-id", default="laptop_20261118_01")
    run(**vars(parser.parse_args()))
