"""Run the frozen schedule on the already attached inference bundle."""


def hybrid_case(models, row, source, seed, device, sentinel):
    conditioner,parent,scheduler,model=models
    image=torch.from_numpy(source.copy())[None,None].to(device)
    guide=BoundStructureModel(model,multiscale_guide(image,"coarse_fine"),torch.ones(1,device=device)).eval()
    point=SETTINGS["point"]
    outputs={}
    for variant, request in (("null",row["manufacturer"]),("target",1-row["manufacturer"])):
        args=(scheduler,image*2-1,torch.tensor([request],device=device),
              torch.tensor([row["field_strength"]],device=device),
              torch.tensor([row["echo_time_norm"]],device=device),
              seed,point["timestep"],point["guidance"])
        outputs[f"hybrid_{variant}"]=hybrid(conditioner,parent,guide,*args,SETTINGS["switch_step"])[0,0]
        if sentinel:
            outputs[f"ordinary_cfg_{variant}"]=sample_wiring(conditioner,guide,*args)[0,0]
    return outputs


def profile(index, seed, weights, dataset, sources, manifest, output):
    device=torch.device(f"cuda:{index}")
    adapter=torch.load(dataset/f"seed{seed}.pt",map_location="cpu",weights_only=True)
    models=[m.to(device) for m in build_cpu(SETTINGS["config"],weights,adapter)]
    frozen=[state_digest(m) for m in models]
    count=0
    started=time.monotonic()
    with torch.inference_mode(), (output/f"seed{seed}.jsonl").open("x",encoding="utf-8") as log:
        for generation in manifest["generation_seeds"]:
            for case,row in enumerate(manifest["conditions"]):
                actual_seed=generation+row["dataset_index"]
                sentinel=count==0
                result=hybrid_case(models,row,sources[case],actual_seed,device,sentinel)
                name=f"seed{seed}_g{generation}_case{case:02d}.npz"
                with (output/name).open("xb") as handle:
                    np.savez_compressed(handle,**result)
                log.write(json.dumps(dict(training_seed=seed,generation_seed=generation,case=case,
                    actual_sampling_seed=actual_seed,array_file=name,
                    array_sha256=hashlib.sha256((output/name).read_bytes()).hexdigest(),
                    ordinary_replay=sentinel))+"\n")
                log.flush()
                count+=1
                if count%5==0: print(f"GPU {index} seed {seed}: {count} cases",flush=True)
    if [state_digest(m) for m in models]!=frozen:
        raise ValueError("Frozen inference weights changed")
    return dict(training_seed=seed,cases=count,device=index,seconds=time.monotonic()-started,frozen_weights=True)


def main():
    deterministic_runtime()
    observed=runtime()
    if observed!=SETTINGS["expected_runtime"]:
        raise ValueError("Kaggle runtime changed from the ordinary baseline")
    matches=list(Path("/kaggle/input").rglob("pilot-input.json"))
    if len(matches)!=1:
        raise ValueError("Expected one attached original inference bundle")
    dataset=matches[0].parent
    if hashlib.sha256(matches[0].read_bytes()).hexdigest()!=SETTINGS["input_manifest_sha256"]:
        raise ValueError("Input manifest differs from original pilot")
    manifest=json.loads(matches[0].read_text())
    for name,sha in manifest["files"].items():
        if hashlib.sha256((dataset/name).read_bytes()).hexdigest()!=sha:
            raise ValueError("Attached input changed")
    weights=torch.load(dataset/"parent.pt",map_location="cpu",weights_only=True)
    sources=np.load(dataset/"validation.npy",allow_pickle=False)
    if sources.shape!=(15,128,128) or sources.dtype!=np.float32 or not np.isfinite(sources).all():
        raise ValueError("Unexpected canonical inputs")
    output=Path("/kaggle/working")
    started=time.monotonic()

    def lane(index):
        return [profile(index,seed,weights,dataset,sources,manifest,output)
                for seed in manifest["training_seeds"][index::2]]

    with ThreadPoolExecutor(max_workers=2) as pool:
        results=[item for group in pool.map(lane,range(2)) for item in group]
    receipt=dict(mode="hybrid_pilot",runtime=observed,protocol_sha256=SETTINGS["protocol_sha256"],
        switch_step=SETTINGS["switch_step"],wall_seconds=time.monotonic()-started,result=results)
    with (output/"receipt.json").open("x",encoding="utf-8") as handle:
        json.dump(receipt,handle,indent=2)
    print(json.dumps(receipt),flush=True)


if __name__=="__main__":
    main()
