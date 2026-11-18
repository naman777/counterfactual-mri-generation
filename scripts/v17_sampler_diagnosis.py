"""Decompose the finished Kaggle sampler comparison using the saved grades."""
import argparse
import json

import numpy as np
import pandas as pd

from scripts.v8_generator_dev.io import ROOT, csv_write, digest, json_write, verify_inputs
from scripts.v10_detail.analyse import crossed_interval
from scripts.v11_guide_audit.analyse import CrossedBootstrap
from scripts.v14_dropout_audit.common import KEYS, META, METRICS, decompose, paired, strata


REVIEW = ROOT / "reports/v16_kaggle/primary_review_20261118_01"
PLAN = ROOT / "reports/v15_shared_residual/pilot_plan_20261118_01/protocol.json"
REFERENCE = ROOT / "reports/v13_guide_dropout/preparation_20261118_01/protocol.json"
BASELINE, CANDIDATE = "ordinary_cfg", "shared_residual"
HITS = ("target_hit", "null_hit", "paired_hit_delta")


def diagnose(run_id):
    if not run_id.replace("_", "").isalnum():
        raise ValueError("Use a fresh simple run identifier")
    output = ROOT / "reports/v17_sampler_diagnosis" / run_id
    output.mkdir(parents=True, exist_ok=False)
    (output / "README.md").write_text(
        "Producer is scripts.v17_sampler_diagnosis. protocol.json fixes the post-hoc comparison. "
        "case_differences.csv contains all matched cases with both samplers' hit outcomes. "
        "stratum_intervals.csv and per_replica_intervals.csv contain descriptive paired intervals. "
        "patient_contributions.csv and leave_one_patient_out.csv show all source effects. "
        "reproduction.json checks the frozen primary differences and intervals. "
        "RESULTS.md interprets the decomposition, and findings.json contains its supporting counts.\n",
        encoding="utf-8")
    original = json.loads((REVIEW / "complete.json").read_text())
    if not original["complete"]:
        raise ValueError("The original paired review must be complete")
    ledger = json.loads((REVIEW / "inputs.json").read_text())
    verify_inputs(ledger)
    plan = json.loads(PLAN.read_text())
    frozen = json.loads((REVIEW / "primary_intervals.json").read_text())
    bootstrap = dict(training_seeds=plan["training_seeds"], generation_seeds=plan["generation_seeds"],
                     bootstrap_seed=plan["primary"]["bootstrap_seed"],
                     bootstrap_resamples=plan["primary"]["bootstrap_resamples"],
                     interactions_family=dict(quantiles=[.025, .975]))
    json_write(output / "protocol.json", dict(post_hoc=True, frozen_result_reproduced_before_interpretation=True,
        comparison="shared_residual minus ordinary_cfg, correct guide and frozen manufacturer operating point",
        populations=["all", "direction", "target condition support", "source QC warning"],
        metrics=METRICS, uncertainty="descriptive 95% crossed-patient/training-replica percentile intervals",
        leave_one_out="Influence sensitivity only. Retain all patients in the primary population.",
        bootstrap=bootstrap, new_sampling=False))
    grades = pd.read_csv(REVIEW / "per_image.csv", keep_default_na=False, na_values=[""], float_precision="round_trip")
    if len(grades) != original["grades"] or set(grades.sampler) != {BASELINE, CANDIDATE} or set(grades.axis) != {"manufacturer"}:
        raise ValueError("Incomplete paired grade input")
    if not grades.source_echo_time_norm.eq(grades.target_echo_time_norm).all():
        raise ValueError("Source EchoTime changed")
    cases = decompose(grades.assign(study=grades.sampler, mode="coarse_fine", guide_control="correct"))
    dimensions = [KEYS[0], *KEYS[1:]]
    expected = pd.MultiIndex.from_product([plan["dataset_indices"], plan["training_seeds"],
                                          plan["generation_seeds"]], names=["dataset_index","training_seed","generation_seed"])
    judges = sorted(frozen_row["judge"] for frozen_row in frozen["contrasts"] if "judge" in frozen_row)
    if sorted(cases.judge.unique()) != judges:
        raise ValueError("Named judge set differs from the frozen review")
    for _, group in cases.groupby(["study", "judge"]):
        index = pd.MultiIndex.from_frame(group[list(expected.names)])
        if index.has_duplicates or set(index) != set(expected):
            raise ValueError("Incomplete patient-by-replica-by-seed matrix")
    paired_cases = paired(cases, [(CANDIDATE,"coarse_fine","correct",1),(BASELINE,"coarse_fine","correct",-1)])
    for label, engine in (("ordinary", BASELINE), ("shared", CANDIDATE)):
        arm = cases[cases.study == engine][dimensions + list(HITS)]
        paired_cases = paired_cases.merge(arm.rename(columns={metric:f"{label}_{metric}" for metric in HITS}),
                                          on=dimensions, validate="one_to_one")
    if len(paired_cases) != original["verified_cases"]*len(judges):
        raise ValueError("Paired rows were lost")
    np.testing.assert_array_equal(paired_cases.paired_hit_delta, paired_cases.target_hit-paired_cases.null_hit)
    # Check the direct comparison against the original family-adjusted intervals.
    evaluation = json.loads(REFERENCE.read_text())["reference_protocol"]["evaluation"]
    original_settings = dict(training_seeds=plan["training_seeds"], evaluation=evaluation,
        analysis={field:plan["primary"][field] for field in ("bootstrap_seed","bootstrap_resamples","interval_quantiles")})
    reproduced = []
    for row in frozen["contrasts"]:
        judge = row.get("judge", judges[0])
        part = paired_cases[paired_cases.judge == judge]
        metric = row["metric"]
        result = crossed_interval(part, metric, original_settings, adjusted=True)
        fields = ("estimate","ci_low","ci_high")
        np.testing.assert_allclose([result[f] for f in fields],[row[f] for f in fields],rtol=0,atol=1e-13)
        reproduced.append(dict(metric=metric,judge=judge,**result))
    json_write(output / "reproduction.json", reproduced)
    csv_write(output / "case_differences.csv", paired_cases)
    boot = CrossedBootstrap(bootstrap)
    summaries, replicas, influence = [], [], []
    for judge, rows in paired_cases.groupby("judge"):
        for name, subset in strata(rows):
            summary, replica = boot.intervals(subset, METRICS)
            summaries.extend(dict(judge=judge,stratum=name,**entry) for entry in summary)
            replicas.extend(dict(judge=judge,stratum=name,**entry) for entry in replica)
        for patient in sorted(rows.patient_id.unique()):
            comparison = rows.loc[rows.patient_id != patient,"paired_hit_delta"]
            influence.append(dict(judge=judge, omitted_patient_id=patient, remaining_mean=float(comparison.mean()),
                                  all_patients_mean=float(rows.paired_hit_delta.mean())))
    intervals = pd.DataFrame(summaries)
    csv_write(output / "stratum_intervals.csv", intervals)
    csv_write(output / "per_replica_intervals.csv", pd.DataFrame(replicas))
    csv_write(output / "leave_one_patient_out.csv", pd.DataFrame(influence))
    patients = paired_cases.groupby(["judge","patient_id","direction","condition_support_status","qc_warning"],
                                   as_index=False)[list(HITS)].sum()
    csv_write(output / "patient_contributions.csv", patients)
    findings = []
    for judge, rows in paired_cases.groupby("judge"):
        people = patients[patients.judge == judge]
        losses = people[people.paired_hit_delta < 0].sort_values(["paired_hit_delta","patient_id"])
        influences = [row["remaining_mean"] for row in influence if row["judge"] == judge]
        findings.append(dict(judge=judge,
            target_losses=int((rows.target_hit == -1).sum()), target_gains=int((rows.target_hit == 1).sum()),
            new_null_hits=int((rows.null_hit == 1).sum()), removed_null_hits=int((rows.null_hit == -1).sum()),
            net_outcomes={metric:int(rows[metric].sum()) for metric in HITS},
            net_loss_patients=losses[["patient_id","direction","condition_support_status","paired_hit_delta"]].to_dict("records"),
            leave_one_patient_out_range_pp=[100*min(influences),100*max(influences)]))
    json_write(output / "findings.json", findings)
    human = ["# Why the shared-residual sampler missed the editing safeguards", "",
        f"All {original['verified_cases']} saved pairs and three independent judges were included. The frozen "
        "primary estimates and four simultaneous intervals were reproduced. The breakdown below is exploratory.", "",
        "## Target edits and unchanged-label controls", "",
        "| Judge | Target hits lost / gained | New / removed null hits | Net editing outcomes |",
        "|---|---:|---:|---:|"]
    for row in findings:
        human.append(f"| {row['judge']} | {row['target_losses']} / {row['target_gains']} | "
                     f"{row['new_null_hits']} / {row['removed_null_hits']} | {row['net_outcomes']['paired_hit_delta']:+d} |")
    human += ["", "A null hit means that the judge predicts the target manufacturer under a request that "
        "keeps the original manufacturer. Net editing outcomes are the target-hit change minus the null-hit change.",
        "", "## Scanner direction and support", "",
        "| Named judge | Stratum | Patients | Editing difference [descriptive 95% interval] |",
        "|---|---|---:|---:|"]
    names = {"all":"All patients","direction:0_to_1":"GE → Siemens","direction:1_to_0":"Siemens → GE",
             "warning:True":"QC warning","warning:False":"No QC warning"}
    for row in intervals[intervals.metric == "paired_hit_delta"].itertuples():
        label = names.get(row.stratum,row.stratum.removeprefix("support:").replace("_"," "))
        human.append(f"| {row.judge} | {label} | {row.patients} | "
                     f"{100*row.estimate:+.2f} [{100*row.ci_low:+.2f}, {100*row.ci_high:+.2f}] pp |")
    human += ["", "## Patient influence", ""]
    for row in findings:
        losses = ", ".join(f"{entry['patient_id']} ({entry['paired_hit_delta']:+d})" for entry in row["net_loss_patients"])
        low, high = row["leave_one_patient_out_range_pp"]
        human.append(f"- {row['judge']} net loss sources are {losses or 'none'}. "
                     f"Leaving out one patient at a time gives point estimates from {low:+.2f} to {high:+.2f} pp.")
    human += ["", "The candidate still misses the frozen editing safeguards. The patient and support "
        "breakdown explains where the loss occurred on this reused development cohort. A score-level "
        "mechanism would require paired examination of the sampler along the actual reverse trajectory. "
        "Source-image differences alone cannot determine which features changed the judge decisions.", ""]
    (output / "RESULTS.md").write_text("\n".join(human), encoding="utf-8")
    used = [PLAN, REFERENCE, REVIEW/"complete.json", REVIEW/"inputs.json", REVIEW/"per_image.csv",
            REVIEW/"primary_intervals.json", ROOT/"scripts/v11_guide_audit/analyse.py",
            ROOT/"scripts/v14_dropout_audit/common.py", ROOT/"scripts/v10_detail/analyse.py",
            ROOT/"scripts/v17_sampler_diagnosis.py"]
    json_write(output / "inputs.json", {path.relative_to(ROOT).as_posix():digest(path) for path in used})
    json_write(output / "complete.json", dict(complete=True, paired_cases=original["verified_cases"],
        paired_case_judge_rows=len(paired_cases), frozen_intervals_reproduced=len(reproduced),
        new_sampling=False, all_source_patients_retained=True, exploratory=True))
    print(output, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--run-id", required=True)
    diagnose(**vars(parser.parse_args()))
