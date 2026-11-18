"""Compare the two score rules on each saved noisy state."""
import argparse
import json

import numpy as np
import pandas as pd
import torch

from scripts.v2_quickwins.judge_model import load_judge
from scripts.v8_generator_dev.evaluate_full import predict_judge, source_native
from scripts.v8_generator_dev.io import ROOT, csv_write, digest, json_write, project_path, verify_inputs
from scripts.v14_dropout_audit.shared_residual import parent_response_score
from scripts.v15_shared_residual.inputs import FrozenInputs
from scripts.v18_trajectory import BASE, TIMES


def score_images(inputs, guided, noisy, step, condition, unconditional):
    times = torch.tensor([step], device=noisy.device)
    guided_null = guided(noisy, times, unconditional)
    ordinary = guided_null + 3 * (guided(noisy, times, condition) - guided_null)
    parent_null = inputs.parent(noisy, times, unconditional)
    shared = parent_response_score(guided_null, inputs.parent(noisy,times,condition), parent_null, 3.)
    abar = inputs.scheduler.alphas_cumprod[step]
    return [(((noisy-torch.sqrt(1-abar)*score)/torch.sqrt(abar)).clamp(-1,1)+1)[0,0].cpu().numpy()/2
            for score in (ordinary,shared)]


@torch.inference_mode()
def run(replay, run_id):
    replay=project_path(replay)
    if not run_id.replace("_","").isalnum():
        raise ValueError("Use a fresh simple run identifier")
    for ledger in ("inputs.json","outputs.json"):
        verify_inputs(json.loads((replay/ledger).read_text()))
    out=BASE/run_id
    out.mkdir(parents=True,exist_ok=False)
    (out/"README.md").write_text("Producer is scripts.v18_common_state. decomposition.csv records both judge "
        "probabilities at each crossed score rule and saved noisy state, and the symmetric arithmetic "
        "decomposition into score-at-fixed-state and accumulated-state terms. complete.json gives the "
        "original-score replay discrepancy. These selected-case diagnostic terms are not causal effects.\n",
        encoding="utf-8")
    inputs=FrozenInputs()
    judges=[load_judge(inputs.require(path),"cuda").eval().requires_grad_(False)
            for path in inputs.reference["evaluation"]["judge_paths"]]
    rows=[]
    max_error=0.
    selected=pd.read_csv(replay/"selection.csv")
    for case in selected.itertuples(index=False):
        index,seed,generation=case.dataset_index,case.training_seed,case.generation_seed
        item=inputs.source(index)
        model=inputs.load_copy("guide_dropout",seed,"coarse_fine")
        guided=inputs.bound(model,index,"coarse_fine","correct")
        target=1-item["manufacturer_label"]
        for request,label in (("null",1-target),("target",target)):
            values=inputs.arguments(item,label,generation,300,3.)
            condition=inputs.conditioner(*values[2:5])
            unconditional=inputs.conditioner(*values[2:5],force_unconditional=True)
            prefix=f"{index}_{seed}_{generation}"
            images=[]
            for arm,state_method in enumerate(("ordinary_cfg","shared_residual")):
                path=replay/"arrays"/f"{prefix}_{state_method}_{request}.npz"
                with np.load(path,allow_pickle=False) as data:
                    for step in TIMES:
                        noisy=torch.from_numpy(data[f"state_{step}"])[None,None].cuda()
                        candidates=score_images(inputs,guided,noisy,step,condition,unconditional)
                        difference=float(np.max(np.abs(candidates[arm]-data[f"estimate_{step}"])))
                        max_error=max(max_error,difference)
                        if difference>2e-6:
                            raise ValueError("Saved-state rescoring differs from original")
                        images.extend(candidates)
            native=source_native(images,inputs.val.plans[index])
            for j,judge in enumerate(judges):
                probs=predict_judge(judge,native,"cuda")["mfg"][:,target].reshape(2,len(TIMES),2)
                for i,step in enumerate(TIMES):
                    oo,on=map(float,probs[0,i])
                    no,nn=map(float,probs[1,i])
                    score=((on-oo)+(nn-no))/2
                    state=((no-oo)+(nn-on))/2
                    rows.append(dict(dataset_index=index,patient_id=case.patient_id, role=case.role,request=request,
                        training_seed=seed,generation_seed=generation,timestep=step,
                        judge=f"heldout_orientation_robust_seed{j}",
                        old_state_old_score=oo,old_state_new_score=on,new_state_old_score=no,new_state_new_score=nn,
                        score_at_fixed_state_term=score,accumulated_state_term=state,total_difference=nn-oo))
    csv_write(out/"decomposition.csv",pd.DataFrame(rows))
    for path in (replay/"outputs.json",replay/"selection.csv",project_path(__file__)):
        inputs.require(path)
    json_write(out/"inputs.json",inputs.paths)
    json_write(out/"complete.json",dict(complete=True,rows=len(rows),max_original_reconstruction_error=max_error))


if __name__=="__main__":
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument("--replay",required=True)
    parser.add_argument("--run-id",required=True)
    run(**vars(parser.parse_args()))
