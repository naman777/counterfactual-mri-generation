"""Verify the local traced endpoints and describe target/null evolution."""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.v8_generator_dev.io import ROOT, csv_write, digest, json_write, project_path, verify_inputs
from scripts.v18_trajectory import BASE, DIAGNOSIS, TIMES

METHODS = ("ordinary_cfg", "shared_residual")


def report(replay, run_id):
    replay=project_path(replay)
    if not run_id.replace("_","").isalnum():
        raise ValueError("Use a fresh simple run identifier")
    if not json.loads((replay/"complete.json").read_text())["complete"]:
        raise ValueError("Completed replay required")
    for ledger in ("inputs.json","outputs.json"):
        verify_inputs(json.loads((replay/ledger).read_text()))
    frame=pd.read_csv(replay/"trajectories.csv",keep_default_na=False)
    pairs=pd.read_csv(DIAGNOSIS/"case_differences.csv")
    selection=pd.read_csv(replay/"selection.csv")
    final=frame[frame.timestep==-1]
    checked=0
    for row in final.itertuples():
        pair=pairs[(pairs.dataset_index==row.dataset_index)&(pairs.training_seed==row.training_seed)&
                   (pairs.generation_seed==row.generation_seed)&(pairs.judge==row.judge)]
        if len(pair)!=1:
            raise ValueError("Expected one original case/judge")
        arm="ordinary" if row.method==METHODS[0] else "shared"
        if row.target_hit != bool(pair.iloc[0][f"{arm}_{row.request}_hit"]):
            raise ValueError("Traced endpoint disagrees with frozen original grade")
        checked+=1
    out=BASE/run_id
    out.mkdir(parents=True,exist_ok=False)
    (out/"README.md").write_text("Producer is scripts.v18_trajectory_report_v2. RESULTS.md describes the exploratory "
        "replay. probability_pairs.csv records every judge and step. matched_state_distances.csv compares "
        "states and scores for the same seed and request. trajectories.png plots the selecting judge for each "
        "matching pair. inputs.json binds the replay evidence and plotting source.\n",encoding="utf-8")
    keys=["role","matched_failure","patient_id","dataset_index","training_seed","generation_seed","direction","judge","timestep"]
    pivot=frame.pivot(index=keys,columns=["method","request"],values="target_probability")
    if pivot.isna().any().any():
        raise ValueError("Missing probability in matched trajectory")
    pivot.columns=[f"{method}_{request}" for method,request in pivot.columns]
    pivot=pivot.reset_index()
    pivot["target_probability_difference"]=pivot.shared_residual_target-pivot.ordinary_cfg_target
    pivot["null_probability_difference"]=pivot.shared_residual_null-pivot.ordinary_cfg_null
    pivot["target_minus_null_gap_difference"]=pivot.target_probability_difference-pivot.null_probability_difference
    csv_write(out/"probability_pairs.csv",pivot)
    distances=[]
    fig,axes=plt.subplots(2,2,figsize=(12.8,8),layout="constrained",sharey=True)
    lines=["# Where the two sampler trajectories diverge","",
           f"Reproduced all {checked} frozen endpoint judge decisions from the selected remote outputs. "
           "The laptop replay uses the original sampler functions with forward hooks. Both arms started at "
           "the same noisy state for every matched request.","",
           "The plots grade reconstructed clean-image estimates at fixed denoising steps. The judges were "
           "evaluated on final images in the original study. Intermediate scores here are diagnostic proxies. "
           "The cases were selected after seeing the final outcomes, so this comparison cannot estimate a "
           "population effect.","",
           "| Source | Role | Selecting judge | Target probability, ordinary → shared | Null probability, ordinary → shared |",
           "|---|---|---|---:|---:|"]
    x=np.arange(len(TIMES)+1)
    for row,ax in zip(selection.itertuples(),axes.flatten()):
        judge=f"heldout_orientation_robust_seed{row.selecting_judge}"
        subset=pivot[(pivot.dataset_index==row.dataset_index)&(pivot.training_seed==row.training_seed)&
                     (pivot.generation_seed==row.generation_seed)&(pivot.judge==judge)].set_index("timestep")
        subset=subset.reindex((*TIMES,-1))
        end=subset.loc[-1]
        lines.append(f"| {row.patient_id} | {row.role} | {judge} | {end.ordinary_cfg_target:.3f} → "
                     f"{end.shared_residual_target:.3f} | {end.ordinary_cfg_null:.3f} → {end.shared_residual_null:.3f} |")
        for method,color,legend in ((METHODS[0],"#245e7b","Ordinary CFG"),(METHODS[1],"#b0482f","Shared residual")):
            for request,style in (("target","-"),("null","--")):
                ax.plot(x,subset[f"{method}_{request}"],style,color=color,marker="o",
                        markersize=3,label=f"{legend}, {request}")
        ax.axhline(.5,color="#777777",linewidth=.8)
        ax.set_ylim(-.02,1.02)
        ax.set_xticks(x,[*map(str,TIMES),"Final"],fontsize=8)
        ax.set_title(f"{row.patient_id} · {row.role} · robust judge {row.selecting_judge}",fontsize=10)
        ax.set_xlabel("Reverse timestep")
        ax.set_ylabel("Target manufacturer probability")
        ax.grid(alpha=.16)
        for request in ("null","target"):
            prefix=f"{row.dataset_index}_{row.training_seed}_{row.generation_seed}"
            files=[replay/"arrays"/f"{prefix}_{method}_{request}.npz" for method in METHODS]
            with np.load(files[0],allow_pickle=False) as a,np.load(files[1],allow_pickle=False) as b:
                for step in TIMES:
                    result={name+"_rms_difference":float(np.sqrt(np.mean(
                        (a[f"{name}_{step}"].astype(np.float64)-b[f"{name}_{step}"].astype(np.float64))**2)))
                        for name in ("state","epsilon","estimate")}
                    distances.append(dict(dataset_index=row.dataset_index,role=row.role,request=request,timestep=step,**result))
    axes[0,1].legend(loc="best",fontsize=7)
    fig.savefig(out/"trajectories.png",dpi=160)
    plt.close(fig)
    lines += ["","See probability_pairs.csv for all named judges and captured steps. "
        "The same-state score comparison at the first step and the differences at later, sampler-specific "
        "states are both retained. Observing a divergence does not isolate a causal image feature.",""]
    (out/"RESULTS.md").write_text("\n".join(lines),encoding="utf-8")
    csv_write(out/"matched_state_distances.csv",pd.DataFrame(distances))
    paths=[replay/name for name in ("complete.json","trajectories.csv","selection.csv","inputs.json","outputs.json")]
    paths.extend((DIAGNOSIS/"case_differences.csv",project_path(__file__)))
    json_write(out/"inputs.json",{path.relative_to(ROOT).as_posix():digest(path) for path in paths})
    json_write(out/"complete.json",dict(complete=True,frozen_endpoint_grades_reproduced=checked,
        paired_probability_rows=len(pivot),pairs=len(selection),exploratory=True))


if __name__=="__main__":
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument("--replay",required=True)
    parser.add_argument("--run-id",required=True)
    report(**vars(parser.parse_args()))
