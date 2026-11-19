"""Freeze one hybrid schedule and verify its endpoints before generating pilot cases."""
import ast
import inspect
import json
import time

import numpy as np
import torch

from scripts.v8_generator_dev.adapter_pilot import state_digest
from scripts.v8_generator_dev.io import ROOT, digest, json_write, verify_inputs
from scripts.v8_generator_dev.train_smoke import sample_wiring
from scripts.v15_shared_residual.check import with_noise_trace
from scripts.v15_shared_residual.inputs import FrozenInputs
from scripts.v15_shared_residual.sampler import sample
from scripts.v16_kaggle_prepare import DATASET, OWNER, PLAN as PRIOR_PLAN
from scripts.v19_hybrid import hybrid

BASE=ROOT/"reports/v19_hybrid"
PREPARED=BASE/"preparation_20261118_01"
PRIOR=ROOT/"reports/v16_kaggle/pilot_20261118_01"
REMOTE=ROOT/"reports/v16_kaggle/execution_20261118_01/download"
KERNEL=f"{OWNER}/capstone-hybrid-sampler-20261118-01"


def prepare():
    verify_inputs(json.loads((PRIOR/"inputs.json").read_text()))
    PREPARED.mkdir(parents=True,exist_ok=False)
    (PREPARED/"kernel").mkdir()
    (PREPARED/"README.md").write_text(
        "Producer is scripts.v19_hybrid_prepare. protocol.json freezes the single switch point, existing paired "
        "matrix and four decision intervals before any hybrid evaluation. endpoints.json verifies exact local "
        "endpoints, noise and weight identity. kernel/ contains the private Kaggle job. inputs.json binds "
        "the protocol and source package.\n",encoding="utf-8")
    old=json.loads((PRIOR_PLAN/"protocol.json").read_text())
    runtime=json.loads((REMOTE/"receipt.json").read_text())["runtime"]
    design=dict(format="predeclared_hybrid_v1",switch_step=225,point=old["point"],
        rationale="Use shared scoring for the first quarter of the reverse schedule. Fix this early switch "
                  "by step fraction before generating hybrid outcomes. The earlier four-case replay cannot optimise a switch.",
        training_seeds=old["training_seeds"],generation_seeds=old["generation_seeds"],
        dataset_indices=old["dataset_indices"],primary=old["primary"],expected_runtime=runtime,
        max_kaggle_seconds=4800,previous_inputs_dataset=DATASET,
        decision="Reuse the 135 development cases and the four primary requirements. "
                 "A wider study requires all margins, original support/stability, and a positive lower bound "
                 "for at least one named judge. This stage includes no independent cohort.")
    json_write(PREPARED/"protocol.json",design)
    inputs=FrozenInputs()
    endpoint_rows=[]
    original_state=[state_digest(m) for m in (inputs.parent,inputs.conditioner,inputs.scheduler)]
    for seed in design["training_seeds"]:
        model=inputs.load_copy("guide_dropout",seed,"coarse_fine")
        adapter_before=state_digest(model.adapter)
        index=inputs.indices[0]
        item=inputs.source(index)
        guide=inputs.bound(model,index,"coarse_fine","correct")
        for label in (item["manufacturer_label"],1-item["manufacturer_label"]):
            args=inputs.arguments(item,label,design["generation_seeds"][0],3,3.)
            ordinary,noise=with_noise_trace(sample_wiring,inputs.conditioner,guide,*args)
            shared,shared_noise=with_noise_trace(sample,inputs.conditioner,inputs.parent,guide,*args)
            first,first_noise=with_noise_trace(hybrid,inputs.conditioner,inputs.parent,guide,*args,3)
            last,last_noise=with_noise_trace(hybrid,inputs.conditioner,inputs.parent,guide,*args,None)
            middle,middle_noise=with_noise_trace(hybrid,inputs.conditioner,inputs.parent,guide,*args,2)
            np.testing.assert_array_equal(ordinary,first)
            np.testing.assert_array_equal(shared,last)
            if not (noise==shared_noise==first_noise==last_noise==middle_noise):
                raise ValueError("Sampler noise draw stream changed")
            endpoint_rows.append(dict(training_seed=seed,label=label,immediate_switch_exact=True,
                switch_disabled_exact=True,mixed_noise_exact=True,mixed_endpoint_finite=bool(np.isfinite(middle).all()),
                noise_draw_sha256=noise))
        if state_digest(model.adapter)!=adapter_before:
            raise ValueError("Frozen fine adapter changed")
    if [state_digest(m) for m in (inputs.parent,inputs.conditioner,inputs.scheduler)]!=original_state:
        raise ValueError("Frozen parent state changed")
    json_write(PREPARED/"endpoints.json",dict(rows=endpoint_rows,frozen_state_verified=True))
    old_kernel=(PRIOR/"kernel/sampler_job.py").read_text(encoding="utf-8")
    prefix=old_kernel.split('"""Worker body appended to the verified sampler and model definitions."""',1)[0]
    old_worker=(ROOT/"scripts/v16_kaggle_worker.py").read_text(encoding="utf-8")
    definitions={node.name:ast.get_source_segment(old_worker,node) for node in ast.parse(old_worker).body if isinstance(node,ast.FunctionDef)}
    worker=ROOT/"scripts/v19_hybrid_worker.py"
    patch=dict(switch_step=225,protocol_sha256=digest(PREPARED/"protocol.json"),expected_runtime=runtime)
    combined="\n\n".join([prefix,"from pathlib import Path","import json, time, platform",
        "from concurrent.futures import ThreadPoolExecutor","SETTINGS.update("+repr(patch)+")",
        definitions["runtime"],definitions["build_cpu"],inspect.getsource(hybrid),worker.read_text(encoding="utf-8")])
    compile(combined,"hybrid_job.py","exec")
    job=PREPARED/"kernel/hybrid_job.py"
    job.write_text(combined,encoding="utf-8",newline="\n")
    json_write(PREPARED/"kernel/kernel-metadata.json",dict(id=KERNEL,title="Capstone Hybrid Sampler 20261118 01",
        code_file=job.name,language="python",kernel_type="script",is_private=True,enable_gpu=True,
        enable_internet=False,dataset_sources=[DATASET],competition_sources=[],kernel_sources=[]))
    paths=(PREPARED/"protocol.json",PREPARED/"endpoints.json",job,PREPARED/"kernel/kernel-metadata.json",
           worker,ROOT/"scripts/v19_hybrid.py",PRIOR/"kernel/sampler_job.py",REMOTE/"receipt.json",
           PRIOR/"dataset/pilot-input.json")
    json_write(PREPARED/"inputs.json",{value.relative_to(ROOT).as_posix():digest(value) for value in paths})
    print("Hybrid plan frozen and six endpoint pairs verified",flush=True)


if __name__=="__main__":
    prepare()
