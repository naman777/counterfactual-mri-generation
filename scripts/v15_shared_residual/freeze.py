"""Freeze the paired local pilot after measuring the engineering workload."""
import argparse
import json
import math

import pandas as pd

from scripts.v8_generator_dev.io import ROOT, digest, json_write, project_path, verify_inputs
from scripts.v15_shared_residual.inputs import CONTROLS, PREPARED


def freeze(engineering, run_id):
    engineering = project_path(engineering)
    if not run_id.replace("_", "").isalnum():
        raise ValueError("Use a fresh simple report identifier")
    completed = json.loads((engineering / "complete.json").read_text())
    if not completed["exact_noise_and_endpoint_checks"]:
        raise ValueError("Completed trajectory checks required")
    evidence = json.loads((engineering / "inputs.json").read_text())
    verify_inputs(evidence)
    design = json.loads((PREPARED / "protocol.json").read_text())
    reference = design["reference_protocol"]
    evaluation = reference["evaluation"]
    selection = pd.read_csv(PREPARED / "validation_selection.csv")
    generations = [evaluation["selection_generation_seed"], *evaluation["confirmation_generation_seeds"]]
    seeds = reference["training_seeds"]
    profiles = {f"seed{seed}__{mode}__{control}": dict(training_seed=seed, mode=mode, control=control)
                for seed in seeds for mode in design["modes"] for control in CONTROLS}
    bench = pd.read_csv(engineering / "benchmark.csv")
    times = bench.groupby("method").seconds.agg(["median", "max"])
    worst_pair = 2 * times["max"].sum()
    typical_case = 2 * times["median"].sum()
    cases_per_profile = len(selection) * len(generations)
    estimate = len(profiles) * cases_per_profile * typical_case
    max_profile_seconds = math.ceil((1.25 * cases_per_profile * worst_pair + 120) / 60) * 60
    out = ROOT / "reports/v15_shared_residual" / run_id
    out.mkdir(parents=True, exist_ok=False)
    (out / "README.md").write_text("Producer is scripts.v15_shared_residual.freeze. protocol.json fixes the "
        "matched laptop design, study decisions and per-profile time ceiling. workload.json records the "
        "benchmark projection. inputs.json binds code, model, data and benchmark identities. PLAN.md "
        "explains scope and execution. complete.json records preparation.\n", encoding="utf-8")
    runtime = json.loads((engineering / "runtime.json").read_text())
    protocol = dict(format="local_shared_residual_pilot_v1", point=evaluation["fixed_settings"]["manufacturer"],
        axis="manufacturer", runtime=runtime, dataset_indices=selection.dataset_index.astype(int).tolist(),
        generation_seeds=generations, training_seeds=seeds, profiles=profiles, cases_per_profile=cases_per_profile,
        primary=dict(mode="coarse_fine", control="correct", comparison="shared_residual_minus_ordinary_cfg",
            population="All declared patients and matched seeds", bootstrap_seed=583595, bootstrap_resamples=10000,
            interval_quantiles=[.00625,.99375], simultaneous_requirements=4,
            editing_lower_bound_above=-.05, every_replica_editing_at_least=-.05,
            target_interior_mad_upper_bound_below=.005,
            superiority_rule="At least one named judge editing lower interval strictly above zero, all safeguards pass",
            support_and_stability="Apply original manufacturer safeguards per training replica",
            unknown_primary_metrics="Advancement unavailable"),
        cached_reference_pooled=False, paired_local_reference=True, one_profile_at_a_time=True,
        max_profile_seconds=max_profile_seconds, paid_compute=False)
    json_write(out / "protocol.json", protocol)
    local = bench[bench.method == "local_reference"]
    json_write(out / "workload.json", dict(timed_routes=len(bench), profile_count=len(profiles),
        paired_cases=len(profiles)*cases_per_profile, trajectories=4*len(profiles)*cases_per_profile,
        median_ordinary_seconds=float(times.loc["local_reference","median"]),
        median_shared_seconds=float(times.loc["shared_residual","median"]),
        median_projection_hours=float(estimate/3600), max_profile_seconds=max_profile_seconds,
        max_projected_worker_hours=len(profiles)*max_profile_seconds/3600,
        peak_allocated_bytes=int(bench.peak_allocated_bytes.max()),
        cloud_replays_exact=int(local.cloud_reference_exact.eq(True).sum()),
        cloud_replays=len(local), max_cloud_laptop_abs=float(local.cloud_reference_max_abs.max()),
        max_cloud_laptop_mad=float(local.cloud_reference_mad.max())))
    rel = out.relative_to(ROOT).as_posix()
    lines = [
        "# Paired laptop pilot for shared guide residuals", "",
        f"The complete manufacturer design has {len(profiles)} profiles and {len(profiles)*cases_per_profile:,} paired "
        "cases. It includes every frozen source, both guide modes, all six guide controls and all training, "
        "generation and judge seeds. Source EchoTime and field strength are fixed. Each profile samples "
        "ordinary classifier-free guidance (CFG) and the candidate with identical seeds, weights, source guide "
        "and operating point. Both use the same laptop runtime. Cached A10 images are historical references only.", "",
        f"Measured batch-one sampling projects **{estimate/3600:.1f} laptop GPU hours** for the complete matrix "
        f"before grading and thermal variation. The profile time ceiling is {max_profile_seconds/60:.0f} minutes. "
        "Run one profile at a time. Each completed profile has its own immutable evidence. A timed-out "
        "profile retains its journal. Repeat it with a fresh output ID.", "",
        "The primary comparison uses the fine model and correct guide. The protocol binds four simultaneous "
        "intervals over patients and replicas, one for each named judge's target-minus-null edit change "
        "and one for target interior image distance. All editing and image safeguards, the original "
        "support and stability checks, and at least one strictly positive editing lower bound are required "
        "for research advancement. All alternate guide controls and strata remain visible. Clinical "
        "anatomy and lesion preservation require separate assessment.", "",
        "Example complete profile invocation from the project directory", "", "    "
        f"python -m scripts.v15_shared_residual.pilot --prepared {rel} "
        "--profile seed583301__coarse_fine__correct --run-id local_seed583301_fine_correct_01", "",
    ]
    (out / "PLAN.md").write_text("\n".join(lines), encoding="utf-8")
    input_paths = list((ROOT / "scripts/v15_shared_residual").glob("*.py"))
    input_paths += [engineering / name for name in ("complete.json","runtime.json","inputs.json","benchmark.csv","endpoints.csv")]
    input_paths += [out / "protocol.json", out / "workload.json", PREPARED / "mask_thresholds.json"]
    input_paths += [project_path(path) for path in evaluation["judge_paths"]]
    # Reuse checked source hashes from the prior package for the numerical dependencies.
    prior_inputs = json.loads((PREPARED / "inputs.json").read_text())
    old_code = {name: value for name, value in prior_inputs.items()
                if name in ("scripts/v10_detail/metrics.py", "scripts/v7_geometry/support.py",
                            "scripts/v8_generator_dev/evaluate_full.py", "scripts/v8_generator_dev/full_train.py",
                            "scripts/v8_generator_dev/data.py", "scripts/v8_generator_dev/detail_candidates.py",
                            "scripts/v6_structure/model.py", "scripts/v2_quickwins/judge_model.py",
                            "src/models/diffusion_unet.py", "src/models/ddpm_scheduler.py",
                            "src/models/conditioning.py")}
    verify_inputs(old_code)
    actual = {path.relative_to(ROOT).as_posix(): digest(path) for path in input_paths}
    json_write(out / "inputs.json", {**evidence, **old_code, **actual})
    json_write(out / "complete.json", dict(complete=True, frozen_profile_plan=True,
        expected_cases=len(profiles)*cases_per_profile, full_pilot_executed=False, paid_compute=False))
    print("LAPTOP PILOT FROZEN", out, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--engineering", required=True)
    parser.add_argument("--run-id", required=True)
    freeze(**vars(parser.parse_args()))
