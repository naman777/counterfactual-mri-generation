"""Worker body appended to the verified sampler and model definitions."""

import json
import platform
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def runtime():
    return dict(python=platform.python_version(), torch=str(torch.__version__),
                cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
                gpu_names=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
                tf32=False, autocast=False, batch_size=1)


def run_case(models, source, manufacturer, field, echo, seed, point, device):
    conditioner, parent, scheduler, guided_model = models
    image01 = torch.from_numpy(np.array(source, copy=True))[None, None].to(device)
    guide = multiscale_guide(image01, "coarse_fine")
    guide = BoundStructureModel(guided_model, guide, torch.ones(1, device=device)).eval()
    values = {}
    times = {}
    for method in ("ordinary_cfg", "shared_residual"):
        for variant, requested in (("null", manufacturer), ("target", 1-manufacturer)):
            args = (scheduler, image01*2-1, torch.tensor([requested], device=device),
                    torch.tensor([field], device=device), torch.tensor([echo], device=device),
                    seed, point["timestep"], point["guidance"])
            torch.cuda.synchronize(device)
            start = time.perf_counter()
            if method == "ordinary_cfg":
                edit = sample_wiring(conditioner, guide, *args)
            else:
                edit = sample(conditioner, parent, guide, *args)
            torch.cuda.synchronize(device)
            key = method + "_" + variant
            times[key] = time.perf_counter() - start
            values[key] = edit[0, 0]
    return values, times


def build_cpu(config, weights=None, adapter=None):
    conditioner, parent, scheduler = build_models(config, "additive", "cpu")
    if weights is not None:
        conditioner.load_state_dict(weights["conditioner"], strict=True)
        parent.load_state_dict(weights["parent"], strict=True)
    guided = DetailConditionedUNet(parent, SETTINGS["adapter_width"])
    if adapter is not None:
        guided.adapter.load_state_dict(adapter, strict=True)
    else:
        # Exercise the guide path with deterministic synthetic, nonzero adapter outputs.
        with torch.no_grad():
            for layer in guided.adapter.outputs:
                layer.bias.fill_(.001)
    return [module.eval().requires_grad_(False) for module in (conditioner, parent, scheduler, guided)]


def bench_device(index, models):
    device = torch.device(f"cuda:{index}")
    models = [module.to(device) for module in models]
    source = np.random.default_rng(991).random((128, 128), dtype=np.float32)
    with torch.inference_mode():
        _, warmup = run_case(models, source, 0, 0, .4, 701,
                             dict(timestep=3, guidance=3.), device)
        torch.cuda.reset_peak_memory_stats(device)
        times = []
        for seed in (701, 702):
            _, sample_times = run_case(models, source, 0, 0, .4, seed, SETTINGS["point"], device)
            times.append(sample_times)
    return dict(device=index, warmup=warmup, paired_seconds=times,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(device))


def run_profile(index, profile_seed, models, sources, rows, generations, point, output):
    device = torch.device(f"cuda:{index}")
    models = [module.to(device) for module in models]
    fingerprints = [state_digest(module) for module in models]
    records = []
    started = time.monotonic()
    with torch.inference_mode(), (output/f"seed{profile_seed}.jsonl").open("x", encoding="utf-8") as journal:
        for generation in generations:
            for case, row in enumerate(rows):
                seed = generation + row["dataset_index"]
                values, times = run_case(models, sources[case], row["manufacturer"], row["field_strength"],
                                         row["echo_time_norm"], seed, point, device)
                name = f"seed{profile_seed}_g{generation}_case{case:02d}.npz"
                with (output/name).open("xb") as handle:
                    np.savez_compressed(handle, **values)
                record = dict(training_seed=profile_seed, generation_seed=generation, case=case,
                              actual_sampling_seed=seed, array_file=name,
                              array_sha256=hashlib.sha256((output/name).read_bytes()).hexdigest(),
                              timing_seconds=times)
                records.append(record)
                journal.write(json.dumps(record)+"\n")
                journal.flush()
                if len(records) % 5 == 0:
                    print(f"GPU {index} seed {profile_seed}: {len(records)} cases", flush=True)
    if fingerprints != [state_digest(module) for module in models]:
        raise ValueError("Inference state changed")
    return dict(device=index, training_seed=profile_seed, cases=len(records),
                seconds=time.monotonic()-started, frozen_weights=True)


def pilot(output):
    dataset = Path(SETTINGS["dataset_path"])
    if not dataset.is_dir():
        matches = list(Path("/kaggle/input").rglob("pilot-input.json"))
        if len(matches) != 1:
            raise ValueError("Expected one attached pilot dataset")
        dataset = matches[0].parent
    manifest_path = dataset/"pilot-input.json"
    if hashlib.sha256(manifest_path.read_bytes()).hexdigest() != SETTINGS["input_manifest_sha256"]:
        raise ValueError("Attached pilot dataset manifest changed")
    manifest = json.loads(manifest_path.read_text())
    for name, expected in manifest["files"].items():
        if hashlib.sha256((dataset/name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Attached asset changed: {name}")
    weights = torch.load(dataset/"parent.pt", map_location="cpu", weights_only=True)
    sources = np.load(dataset/"validation.npy", allow_pickle=False)
    if sources.shape != (15,128,128) or sources.dtype != np.float32:
        raise ValueError("Expected all 15 canonical validation sources")
    if not np.isfinite(sources).all() or sources.min() < 0 or sources.max() > 1:
        raise ValueError("Invalid source array")
    seeds = manifest["training_seeds"]
    profiles = {seed:build_cpu(SETTINGS["config"], weights,
        torch.load(dataset/f"seed{seed}.pt", map_location="cpu", weights_only=True)) for seed in seeds}

    def lane(device):
        results = []
        for seed in seeds[device::2]:
            results.append(run_profile(device, seed, profiles[seed], sources, manifest["conditions"],
                                       manifest["generation_seeds"], SETTINGS["point"], output))
        return results

    with ThreadPoolExecutor(max_workers=2) as pool:
        return [result for lane_results in pool.map(lane, range(2)) for result in lane_results]


def main():
    deterministic_runtime()
    if torch.cuda.device_count() != 2 or not all("T4" in torch.cuda.get_device_name(i) for i in range(2)):
        raise RuntimeError("Expected two T4 GPUs")
    output = Path("/kaggle/working")
    started = time.monotonic()
    if SETTINGS["mode"] == "benchmark":
        torch.manual_seed(991)
        models = [build_cpu(SETTINGS["config"]) for _ in range(2)]
        with ThreadPoolExecutor(max_workers=2) as pool:
            result = list(pool.map(bench_device, range(2), models))
    else:
        result = pilot(output)
    receipt = dict(mode=SETTINGS["mode"], runtime=runtime(), wall_seconds=time.monotonic()-started,
                   study_plan_sha256=SETTINGS["study_plan_sha256"], result=result)
    with (output/"receipt.json").open("x", encoding="utf-8") as handle:
        json.dump(receipt, handle, indent=2)
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
