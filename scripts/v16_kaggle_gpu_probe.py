"""Prepare a private, synthetic Kaggle GPU probe for a signed-in account."""
import argparse
import json
from pathlib import Path
import re

from scripts.v8_generator_dev.io import ROOT, digest, json_write


PROBE = """
import json
from pathlib import Path
import platform
import time

import torch

result = {
    "python": platform.python_version(),
    "torch": str(torch.__version__),
    "torch_cuda": torch.version.cuda,
    "gpu_count": torch.cuda.device_count(),
    "devices": [],
}
for index in range(torch.cuda.device_count()):
    device = torch.device(f"cuda:{index}")
    generator = torch.Generator(device=device).manual_seed(701)
    x = torch.randn((1, 32, 128, 128), device=device, generator=generator)
    weight = torch.randn((32, 32, 3, 3), device=device, generator=generator)
    with torch.inference_mode():
        torch.nn.functional.conv2d(x, weight, padding=1)
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        for _ in range(24):
            y = torch.nn.functional.conv2d(x, weight, padding=1)
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
    result["devices"].append({
        "index": index,
        "name": torch.cuda.get_device_name(device),
        "total_memory_bytes": torch.cuda.get_device_properties(device).total_memory,
        "seconds_for_24_synthetic_convolutions": elapsed,
        "finite": bool(torch.isfinite(y).all()),
    })
result["cuda_check_passed"] = bool(result["devices"]) and all(d["finite"] for d in result["devices"])
with Path("/kaggle/working/gpu_probe.json").open("x", encoding="utf-8") as handle:
    json.dump(result, handle, indent=2)
print(json.dumps(result), flush=True)
if not result["cuda_check_passed"]:
    raise RuntimeError("Kaggle did not provide a working CUDA device")
"""


def prepare(owner, run_id):
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", owner):
        raise ValueError("Use the signed-in Kaggle account slug")
    if not re.fullmatch(r"[a-z0-9_]+", run_id):
        raise ValueError("Use a simple fresh run identifier")
    out = ROOT / "reports/v16_kaggle" / run_id
    out.mkdir(parents=True, exist_ok=False)
    kernel = out / "kernel"
    kernel.mkdir()
    source = kernel / "gpu_probe.py"
    compile(PROBE, source.name, "exec")
    source.write_text(PROBE.lstrip(), encoding="utf-8")
    metadata = kernel / "kernel-metadata.json"
    slug = "capstone-gpu-probe-" + run_id.replace("_", "-")
    json_write(metadata, dict(id=f"{owner}/{slug}", title=slug.replace("-", " ").title(),
        code_file=source.name, language="python", kernel_type="script", is_private=True,
        enable_gpu=True, enable_internet=False, dataset_sources=[], competition_sources=[], kernel_sources=[]))
    (out / "README.md").write_text(
        "Producer is scripts.v16_kaggle_gpu_probe. kernel/ contains a private script and Kaggle metadata "
        "for a brief synthetic CUDA check. inputs.json records the staged hashes. The probe uses random "
        "tensors and no patient data or model weights. The remote gpu_probe.json records device availability. "
        "Use a 300-second execution limit. This timing does not estimate the capstone sampler throughput.",
        encoding="utf-8")
    json_write(out / "inputs.json", {f.relative_to(ROOT).as_posix():digest(f) for f in (source, metadata, Path(__file__))})
    print(kernel)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--owner", required=True)
    parser.add_argument("--run-id", required=True)
    prepare(**vars(parser.parse_args()))
