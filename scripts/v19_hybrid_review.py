"""Verify hybrid outputs against the frozen Kaggle baseline and grade locally."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from scripts.v2_quickwins.judge_model import load_judge
from scripts.v8_generator_dev.data import CanonicalDataset
from scripts.v8_generator_dev.full_protocol import select_setting
from scripts.v8_generator_dev.full_train import deterministic_runtime
from scripts.v8_generator_dev.io import ROOT,csv_write,digest,json_write,verify_inputs
from scripts.v10_detail.analyse import crossed_interval
from scripts.v15_shared_residual.pilot import grade
from scripts.v15_shared_residual.inputs import PREPARED as ORIGINAL_PREPARED
from scripts.v19_hybrid_prepare import BASE,PREPARED,REMOTE


def review(download):
    download=Path(download).resolve()
    design=json.loads((PREPARED/"protocol.json").read_text())
    verify_inputs(json.loads((PREPARED/"inputs.json").read_text()))
    receipt=json.loads((download/"receipt.json").read_text())
    if (receipt["protocol_sha256"]!=digest(PREPARED/"protocol.json") or receipt["mode"]!="hybrid_pilot"
        or receipt["switch_step"]!=design["switch_step"] or receipt["runtime"]!=design["expected_runtime"]
        or len(receipt["result"])!=3 or {r["training_seed"] for r in receipt["result"]}!=set(design["training_seeds"])
        or not all(r["cases"]==45 and r["frozen_weights"] for r in receipt["result"])):
        raise ValueError("Hybrid receipt differs from the frozen plan")
    original=ROOT/"reports/v16_kaggle/primary_review_20261118_01"
    verify_inputs(json.loads((original/"inputs.json").read_text()))
    old=pd.read_csv(original/"per_image.csv",keep_default_na=False,na_values=[""],float_precision="round_trip")
    old=old[old.sampler=="ordinary_cfg"].copy()
    metadata=["case","dataset_index","patient_id","training_seed","generation_seed","actual_sampling_seed",
              "axis","direction","target_manufacturer","target_field_strength","source_echo_time_norm",
              "target_echo_time_norm","timestep","guidance","condition_support_status","qc_warning","qc_reasons"]
    requests=old[metadata].drop_duplicates().set_index(["training_seed","generation_seed","case"],drop=False)
    if not requests.index.is_unique or len(requests)!=135:
        raise ValueError("Original baseline conditions incomplete")
    reference=json.loads((ORIGINAL_PREPARED/"protocol.json").read_text())["reference_protocol"]
    ledger=json.loads((ORIGINAL_PREPARED/"inputs.json").read_text())
    deterministic_runtime()
    val=CanonicalDataset(reference["view"],"val")
    judges={}
    for index,path in enumerate(reference["evaluation"]["judge_paths"]):
        if digest(ROOT/path)!=ledger[path]: raise ValueError("Frozen judge changed")
        judges[f"heldout_orientation_robust_seed{index}"]=load_judge(ROOT/path,"cuda").eval().requires_grad_(False)
    sources={i:val.image_01(i) for i in design["dataset_indices"]}
    grades=[]; seen=set(); sentinels=[]; evidence={}
    for seed in design["training_seeds"]:
        journal=download/f"seed{seed}.jsonl"
        evidence[journal.relative_to(ROOT).as_posix()]=digest(journal)
        for record in map(json.loads,journal.read_text().splitlines()):
            generation,case=record["generation_seed"],record["case"]
            key=(seed,generation,case)
            if key in seen or record["training_seed"]!=seed or key not in requests.index:
                raise ValueError("Unexpected or duplicate output")
            seen.add(key)
            request=requests.loc[key].to_dict()
            index=request["dataset_index"]
            if index!=design["dataset_indices"][case] or record["actual_sampling_seed"]!=generation+index:
                raise ValueError("Source or actual sampling seed changed")
            name=f"seed{seed}_g{generation}_case{case:02d}.npz"
            path=download/name
            if record["array_file"]!=name or digest(path)!=record["array_sha256"]:
                raise ValueError("Output hash mismatch")
            evidence[path.relative_to(ROOT).as_posix()]=record["array_sha256"]
            expect_sentinel=generation==design["generation_seeds"][0] and case==0
            if record["ordinary_replay"]!=expect_sentinel:
                raise ValueError("Unexpected baseline replay layout")
            with np.load(path,allow_pickle=False) as images:
                arms={"identity":sources[index],"null":images["hybrid_null"],"target":images["hybrid_target"]}
                if any(a.shape!=(128,128) or a.dtype!=np.float32 or not np.isfinite(a).all()
                        or a.min()<0 or a.max()>1 for a in arms.values()):
                    raise ValueError("Invalid hybrid array")
                if expect_sentinel:
                    with np.load(REMOTE/name,allow_pickle=False) as baseline:
                        for constant in ("null","target"):
                            a,b=images[f"ordinary_cfg_{constant}"],baseline[f"ordinary_cfg_{constant}"]
                            np.testing.assert_allclose(a,b,atol=1e-6,rtol=0)
                            sentinels.append(dict(seed=seed,control=constant,max_abs=float(np.max(np.abs(a-b)))))
                grades.extend(grade(arms,SimpleNamespace(val=val),index,request["target_manufacturer"],judges,
                                    dict(request,sampler="hybrid",array_file=name)))
        print("Regraded hybrid",seed,flush=True)
    if seen!=set(requests.index) or len(sentinels)!=6 or len(grades)!=len(old):
        raise ValueError("Incomplete paired grading or baseline replay")
    out=BASE/"review_20261118_01"
    out.mkdir(exist_ok=False)
    (out/"README.md").write_text("Producer is scripts.v19_hybrid_review. Hybrid grades are recalculated locally "
        "and paired with frozen ordinary CFG grades. baseline_replays.json records the three-profile "
        "sentinels. primary_intervals.json records the predeclared decision. RESULTS.md gives the outcome.\n",
        encoding="utf-8")
    frame=pd.DataFrame(grades)
    csv_write(out/"hybrid_grades.csv",frame)
    json_write(out/"baseline_replays.json",sentinels)
    primary=design["primary"]
    settings=dict(training_seeds=design["training_seeds"],evaluation=reference["evaluation"],
        analysis={f:primary[f] for f in ("bootstrap_seed","bootstrap_resamples","interval_quantiles")})
    candidate=frame[frame.variant=="target"]
    keys=["patient_id","dataset_index","training_seed","generation_seed","judge"]
    pairs=candidate.merge(old[old.variant=="target"],on=keys,suffixes=("_new","_base"),validate="one_to_one")
    if len(pairs)!=135*len(judges):raise ValueError("Matched judge rows incomplete")
    outcomes=[]
    for judge,part in pairs.groupby("judge"):
        part=part.assign(difference=part.paired_hit_delta_new-part.paired_hit_delta_base)
        interval=crossed_interval(part,"difference",settings,adjusted=True)
        replicas=part.groupby("training_seed").difference.mean()
        outcomes.append(dict(judge=judge, metric="paired_hit_delta", **interval,
            replica_estimates={str(k):float(v) for k,v in replicas.items()},
            passes=bool(interval["ci_low"]>primary["editing_lower_bound_above"]
                and (replicas>=primary["every_replica_editing_at_least"]).all()),
            positive_lower_bound=bool(interval["ci_low"]>0)))
    image=pairs[pairs.judge==sorted(judges)[0]].copy()
    image["difference"]=image.interior_mad_new-image.interior_mad_base
    if image.difference.notna().all():
        interval=crossed_interval(image,"difference",settings,adjusted=True)
        outcomes.append(dict(metric="target_interior_mad",**interval,
            passes=bool(interval["ci_high"]<primary["target_interior_mad_upper_bound_below"])))
    else:
        outcomes.append(dict(metric="target_interior_mad",unavailable=True,passes=False))
    support={str(seed):select_setting(rows,"manufacturer",reference) for seed,rows in candidate.groupby("training_seed")}
    advance=(all(r["passes"] for r in outcomes) and any(r.get("positive_lower_bound",False) for r in outcomes)
             and all(s["selected"] is not None for s in support.values()))
    json_write(out/"primary_intervals.json",dict(contrasts=outcomes,support=support,wider_run_supported=advance))
    text=["# Early-switch hybrid manufacturer pilot","",
          "The hybrid switched to ordinary classifier-free guidance at timestep 225. All 135 target/null "
          "pairs were compared with frozen ordinary CFG on the same development cases. All six ordinary "
          "replays passed the pixel comparison. Intervals are simultaneous within the four planned requirements.","",
          "| Measure | Hybrid minus ordinary [simultaneous CI] | Margin met |","|---|---:|---|"]
    for result in outcomes:
        label=result.get("judge","Target interior mean absolute distance")
        difference="Unavailable"
        if not result.get("unavailable"):
            factor=100 if "judge" in result else 1
            difference=f'{factor*result["estimate"]:+.4f} [{factor*result["ci_low"]:+.4f}, {factor*result["ci_high"]:+.4f}]'
            if factor==100:difference+=" pp"
        text.append(f"| {label} | {difference} | {result['passes']} |")
    text+=["",f"The primary evidence supports a wider evaluation: **{advance}**.",
           "This pilot reuses development patients and provides no independent or clinical preservation outcome.",""]
    (out/"RESULTS.md").write_text("\n".join(text),encoding="utf-8")
    for path in (original/"per_image.csv",original/"inputs.json", download/"receipt.json",
                 PREPARED/"protocol.json",Path(__file__)):
        evidence[path.relative_to(ROOT).as_posix()]=digest(path)
    json_write(out/"inputs.json",evidence)
    json_write(out/"complete.json",dict(complete=True,verified_case_pairs=len(seen),
        hybrid_grades=len(grades),wider_run_supported=advance,fresh_test=False))
    print(out,flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument("--download",required=True)
    review(**vars(parser.parse_args()))
