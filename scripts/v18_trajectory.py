"""Replay the original samplers with forward hooks at fixed reverse steps."""
import argparse
import json
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import torch

from scripts.v2_quickwins.judge_model import load_judge
from scripts.v8_generator_dev.adapter_pilot import state_digest
from scripts.v8_generator_dev.evaluate_full import predict_judge, source_native
from scripts.v8_generator_dev.io import ROOT, csv_write, digest, json_write, verify_inputs
from scripts.v8_generator_dev.train_smoke import sample_wiring
from scripts.v14_dropout_audit.shared_residual import parent_response_score
from scripts.v15_shared_residual.inputs import FrozenInputs
from scripts.v15_shared_residual.sampler import sample

BASE = ROOT/"reports/v18_trajectory"
DIAGNOSIS = ROOT/"reports/v17_sampler_diagnosis/diagnosis_20261118_01"
REMOTE = ROOT/"reports/v16_kaggle/execution_20261118_01/download"
TIMES = (300, 250, 200, 150, 100, 50, 25, 0)


def select_cases(pairs):
    selected = []
    for patient, judge in (("Breast_MRI_866",0),("Breast_MRI_812",1)):
        rows = pairs[pairs.judge == f"heldout_orientation_robust_seed{judge}"]
        focus = rows[(rows.patient_id==patient)&(rows.target_hit==-1)&(rows.null_hit==0)]
        focus = focus.sort_values(["training_seed","generation_seed"]).iloc[0]
        matches = rows[(rows.patient_id!=patient)&(rows.direction==focus.direction)
            &(rows.condition_support_status==focus.condition_support_status)
            &(rows.training_seed==focus.training_seed)&(rows.generation_seed==focus.generation_seed)
            &(rows.ordinary_target_hit==1)&(rows.shared_target_hit==1)&(rows.null_hit==0)]
        agreement = pairs.groupby(["dataset_index","training_seed","generation_seed"])[["target_hit","null_hit"]].apply(
            lambda pair: bool(pair.eq(0).all().all()))
        matches = matches[[bool(agreement.loc[r.dataset_index,r.training_seed,r.generation_seed])
                           for r in matches.itertuples()]]
        control = matches.sort_values("dataset_index").iloc[0]
        for role,row in (("loss",focus),("agreement",control)):
            selected.append(dict(role=role, matched_failure=patient, selecting_judge=judge,
                patient_id=row.patient_id, dataset_index=int(row.dataset_index), direction=row.direction,
                support=row.condition_support_status, training_seed=int(row.training_seed),
                generation_seed=int(row.generation_seed)))
    return selected


def hooked_sample(inputs, guided, method, arguments):
    observed = defaultdict(list)

    def hook(name):
        def save(_model, args, output):
            step = int(args[1][0])
            if step in TIMES:
                observed[step].append((name, args[0].detach().clone(), output.detach().clone()))
        return save

    handles = [guided.register_forward_hook(hook("guided")),inputs.parent.register_forward_hook(hook("parent"))]
    try:
        if method=="ordinary_cfg":
            result = sample_wiring(inputs.conditioner,guided,*arguments)[0,0]
        else:
            result = sample(inputs.conditioner,inputs.parent,guided,*arguments)[0,0]
    finally:
        for handle in handles:
            handle.remove()
    reconstructed = {}
    with torch.inference_mode():
        for step in TIMES:
            calls = observed[step]
            names = [name for name,_,_ in calls]
            if not calls or any(not torch.equal(calls[0][1],state) for _,state,_ in calls):
                raise ValueError("Unexpected state mismatch within scorer")
            state = calls[0][1]
            if method=="ordinary_cfg" and names==["guided","guided"]:
                null,conditional = (call[2] for call in calls)
                epsilon = null+arguments[-1]*(conditional-null)
            elif method=="shared_residual" and names==["guided","parent","parent"]:
                guided_null,null,conditional = (call[2] for call in calls)
                epsilon = parent_response_score(guided_null,conditional,null,arguments[-1])
            else:
                raise ValueError("Unexpected sampler forward order")
            abar = inputs.scheduler.alphas_cumprod[step]
            estimate = (state-torch.sqrt(1-abar)*epsilon)/torch.sqrt(abar)
            reconstructed[step] = dict(estimate=((estimate.clamp(-1,1)+1)/2)[0,0].cpu().numpy(),
                state=state[0,0].cpu().numpy(), epsilon=epsilon[0,0].cpu().numpy())
    return result,reconstructed


def run(run_id):
    verify_inputs(json.loads((DIAGNOSIS/"inputs.json").read_text()))
    pairs=pd.read_csv(DIAGNOSIS/"case_differences.csv")
    selected=select_cases(pairs)
    out=BASE/run_id
    out.mkdir(parents=True,exist_ok=False)
    (out/"arrays").mkdir()
    (out/"README.md").write_text("Producer is scripts.v18_trajectory. protocol.json records the selected failure "
        "and matched agreement cases. trajectories.csv records per-judge probabilities on reconstructed clean-image "
        "estimates and Kaggle endpoint differences. arrays/ holds captured states, scores and reconstructions. "
        "Inputs and outputs are bound by hashes. Exploratory case replay, with classifiers applied outside their "
        "validated input distribution at intermediate times.\n",encoding="utf-8")
    json_write(out/"protocol.json",dict(cases=selected,timesteps=TIMES,point=dict(timestep=300,guidance=3.),
        max_laptop_seconds=480,max_endpoint_pixel_difference=1e-4,new_cloud_calls=0,
        pairing_rule="Earliest saved target-only loss for the designated judge. Match seeds, direction and support with an all-judge agreement source successful for the designated judge. Break ties by dataset index."))
    inputs=FrozenInputs()
    judge_paths=inputs.reference["evaluation"]["judge_paths"]
    judges=[load_judge(inputs.require(path),"cuda").eval().requires_grad_(False) for path in judge_paths]
    journals={}
    for seed in {row["training_seed"] for row in selected}:
        path=inputs.require(REMOTE/f"seed{seed}.jsonl")
        journals[seed]={(row["generation_seed"],row["case"]):row for row in map(json.loads,path.read_text().splitlines())}
    weights_before=[state_digest(module) for module in (inputs.parent,inputs.conditioner,inputs.scheduler)]
    records, outputs = [], {}
    start=time.monotonic()
    for row in selected:
        seed,generation,index=(row[key] for key in ("training_seed","generation_seed","dataset_index"))
        item=inputs.source(index)
        model=inputs.load_copy("guide_dropout",seed,"coarse_fine")
        adapter_before=state_digest(model.adapter)
        guide=inputs.bound(model,index,"coarse_fine","correct")
        recorded=journals[seed][generation,inputs.indices.index(index)]
        cache=inputs.require(REMOTE/recorded["array_file"],recorded["array_sha256"])
        original=item["manufacturer_label"]
        traces={}
        with np.load(cache,allow_pickle=False) as cached:
            for method in ("ordinary_cfg","shared_residual"):
                for request,label in (("null",original),("target",1-original)):
                    if time.monotonic()-start>480:
                        raise TimeoutError("Laptop replay time limit")
                    args=inputs.arguments(item,label,generation,300,3.)
                    final,trajectory=hooked_sample(inputs,guide,method,args)
                    key=f"{method}_{request}"
                    difference=float(np.max(np.abs(final-cached[key])))
                    if difference>1e-4:
                        raise ValueError(f"Laptop output diverged from saved Kaggle output: {difference:g}")
                    payload={f"{name}_{step}":value for step,values in trajectory.items() for name,value in values.items()}
                    payload["final"]=final
                    filename=out/"arrays"/f"{index}_{seed}_{generation}_{key}.npz"
                    with filename.open("xb") as handle:np.savez_compressed(handle,**payload)
                    outputs[filename.relative_to(ROOT).as_posix()]=digest(filename)
                    traces[key]=trajectory
                    images=source_native([trajectory[t]["estimate"] for t in TIMES]+[final,cached[key]],inputs.val.plans[index])
                    for judge_id,judge in enumerate(judges):
                        probability=predict_judge(judge,images,"cuda")["mfg"]
                        if probability[-2].argmax()!=probability[-1].argmax():
                            raise ValueError("Laptop endpoint changed a recorded judge decision")
                        for t,values in zip((*TIMES,-1),probability):
                            records.append(dict(row,method=method,request=request,timestep=t,
                                judge=f"heldout_orientation_robust_seed{judge_id}",target_probability=float(values[1-original]),
                                target_hit=bool(values.argmax()==1-original),max_endpoint_pixel_difference=difference))
                    print("REPLAYED",index,method,request,flush=True)
        first=[trace[300]["state"] for trace in traces.values()]
        for state in first[1:]:np.testing.assert_array_equal(first[0],state)
        if state_digest(model.adapter)!=adapter_before:
            raise ValueError("Adapter changed during replay")
    if weights_before!=[state_digest(module) for module in (inputs.parent,inputs.conditioner,inputs.scheduler)]:
        raise ValueError("Frozen model state changed")
    frame=pd.DataFrame(records)
    csv_write(out/"trajectories.csv",frame)
    csv_write(out/"selection.csv",pd.DataFrame(selected))
    for path in (DIAGNOSIS/"case_differences.csv", DIAGNOSIS/"inputs.json", ROOT/"scripts/v18_trajectory.py",
                 ROOT/"scripts/v15_shared_residual/sampler.py", ROOT/"scripts/v8_generator_dev/train_smoke.py"):
        inputs.require(path)
    json_write(out/"inputs.json",inputs.paths)
    json_write(out/"outputs.json",outputs)
    json_write(out/"complete.json",dict(complete=True,cases=len(selected),endpoint_replays=len(outputs),
        judge_observations=len(records),seconds=time.monotonic()-start,max_endpoint_pixel_difference=float(
        frame.max_endpoint_pixel_difference.max()),all_endpoint_decisions_match_kaggle=True,
        paired_initial_states_exact=True,frozen_state_verified=True))


if __name__=="__main__":
    cli=argparse.ArgumentParser(__doc__)
    cli.add_argument("--run-id",required=True)
    run(**vars(cli.parse_args()))
