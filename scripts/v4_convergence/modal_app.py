"""One-app B1 -> B3 controller, bounded smoke run and append-only fetch.

Examples from the project root:
 python -m modal run scripts/v4_convergence/modal_app.py --run-id b1_20261016_01 --stage smoke
 python -m modal run --detach scripts/v4_convergence/modal_app.py --run-id <id> --stage full --budget-usd <approved cap>
 python -m scripts.v4_convergence.modal_app --fetch <id>

The controller runs remotely, so the sequence survives a local disconnect.
Only this app owns GPU work; all its stages share one pool of <=8 containers.
"""
import json
import os
import sys
import time
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[2] if modal.is_local() else Path("/root/project")
app = modal.App("capstone-v4-convergence")
volume = modal.Volume.from_name("capstone-judge-data", create_if_missing=False)
GPU = os.environ.get("CAPSTONE_MODAL_GPU", "A10G")
if GPU not in ("A10G", "L40S"):
    raise ValueError("Supported GPUs are A10G and L40S")
image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "torch==2.5.1", "torchvision==0.20.1", "numpy==2.1.3", "pandas==2.2.3",
    "scipy==1.14.1", "scikit-image==0.24.0", "pyyaml==6.0.2", "pillow", "matplotlib==3.9.2"
).env({"PYTHONPATH": "/root/project", "OPENBLAS_NUM_THREADS": "2", "OMP_NUM_THREADS": "2",
       "CAPSTONE_MODAL_GPU": GPU})
if modal.is_local():
    image = image.add_local_dir(ROOT / "src", "/root/project/src").add_local_dir(
        ROOT / "scripts", "/root/project/scripts", ignore=["**/__pycache__/**"])

# GPU, two physical CPU cores (conservative), 16 GiB, plus 20% safety allowance.
GPU_RATE = ({"A10G": .000306, "L40S": .000542}[GPU] + 2 * 0.0000131 + 16 * 0.00000222) * 1.2
CONTROLLER_RATE = (0.0000131 + 2 * 0.00000222) * 1.2
WORKER_TIMEOUT = 1800


def remote_root(run_id):
    if not run_id or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for c in run_id):
        raise ValueError("Invalid run id")
    return Path("/data/v4_convergence") / run_id


@app.function(image=image, gpu=GPU, cpu=2, memory=16384, volumes={"/data": volume},
              max_containers=8, timeout=WORKER_TIMEOUT, scaledown_window=2, retries=0)
def worker(run_id: str, operation: str, checkpoint: str, subdir: str, end_epoch: int = 0, job: dict = {}):
    sys.path.insert(0, "/root/project")
    import torch
    from scripts.v4_convergence.core import json_write, load_pack
    from scripts.v4_convergence.train import train_block
    from scripts.v4_convergence.evaluate import gate, run_job
    volume.reload()
    base = remote_root(run_id)
    cfg = json.loads((base / "config.json").read_text())
    protocol = json.loads((base / "protocol.json").read_text())
    splits = None
    if operation in ("train", "smoke_train") or (operation == "job" and job.get("split") == "val"):
        splits = ("train", "val")
    elif operation in ("gate", "smoke_gate"):
        splits = ("train",)
    data = load_pack(base / "inputs/pack.npz", splits=splits)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    start = time.perf_counter()
    assets = base / "inputs/assets"
    out = base / subdir
    if operation == "benchmark":
        import numpy as np
        from scripts.v4_convergence.core import gate_conditions, load_models, sample
        c, u, scheduler, _ = load_models(checkpoint)
        cond = gate_conditions(data, 100)
        times = []
        for n in (10, 25, 50):
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            then = time.perf_counter()
            generated = sample(c, u, scheduler, {k: v[:n] for k, v in cond.items()},
                               np.arange(730000, 730000 + n), 100, 3.0)
            torch.cuda.synchronize()
            seconds = time.perf_counter() - then
            times.append({"batch": n, "seconds_101_steps": seconds,
                          "image_steps_per_second": n * 101 / seconds,
                          "max_gpu_memory_bytes": torch.cuda.max_memory_allocated()})
        result = {"timings": times, "scientific_run": False, "gpu_name": torch.cuda.get_device_name()}
        json_write(out / "timings.json", result)
    elif operation == "analysis":
        from scripts.v4_convergence.analyze import analyze
        analysis_path = analyze(run_id, base_override=base)
        status = json.loads((analysis_path / "status.json").read_text())
        if not status["b3_complete"]:
            raise RuntimeError("B3 analysis did not complete")
        result = {"analysis_path": str(analysis_path), **status}
    elif operation in ("train", "smoke_train"):
        result = train_block(data, cfg, checkpoint, out, end_epoch,
                             max_steps=12 if operation == "smoke_train" else None)
    elif operation in ("gate", "smoke_gate"):
        result = gate(data, checkpoint, assets, out, protocol, smoke=operation == "smoke_gate")
    elif operation == "job":
        result = run_job(data, checkpoint, assets, out, job, protocol)
    else:
        raise ValueError(operation)
    result["worker_elapsed_seconds"] = time.perf_counter() - start
    volume.commit()
    return result


@app.function(image=image, cpu=.25, memory=2048, volumes={"/data": volume}, timeout=24 * 3600,
              max_containers=1, retries=0, scaledown_window=2)
def controller(run_id: str, stage: str, budget_usd: float):
    sys.path.insert(0, "/root/project")
    from scripts.v4_convergence.core import choose_validation, gate_rank, json_write, plateau, loss_plateau
    from scripts.v4_convergence.evaluate import test_jobs, validation_jobs
    volume.reload()
    base = remote_root(run_id)
    protocol = json.loads((base / "protocol.json").read_text())
    addendum = base / "protocol_addendum_v2.json"
    if addendum.exists():
        protocol["addendum_v2"] = json.loads(addendum.read_text())
    protocol["compute"] = {**protocol["compute"], "gpu": GPU, "budget_usd": budget_usd,
                           "price_gpu_sec": {"A10G": .000306, "L40S": .000542}[GPU],
                           "startup_seconds_allowance": 30,
                           "note": "Actual launch settings supersede preparation-time hardware defaults"}
    if stage not in ("smoke", "full") or (stage == "full" and budget_usd <= 0):
        raise ValueError("Full run requires a positive authorised compute cap")
    root = base / stage
    root.mkdir(parents=True, exist_ok=True)
    ledger_path = root / "ledger"
    ledger_path.mkdir(exist_ok=True)
    prior = [json.loads(p.read_text()) for p in sorted(ledger_path.glob("*.json"))]
    sessions = root / "controller_sessions"
    sessions.mkdir(exist_ok=True)
    spent = sum(r["conservative_usd"] for r in prior) + sum(
        json.loads(p.read_text())["conservative_usd"] for p in sessions.glob("*.json"))
    started = time.perf_counter()
    serial = len(prior)
    def remaining():
        return budget_usd - spent - (time.perf_counter() - started) * CONTROLLER_RATE
    def call(operation, checkpoint, subdir, end_epoch=0, job=None):
        nonlocal spent, serial
        receipt = root / "completed" / (subdir.replace("/", "_") + ("_" + job["name"] if job else "") + ".json")
        if receipt.exists():
            return json.loads(receipt.read_text())["result"]
        # Reserve worst-case GPU time, including conservative startup overhead.
        reserve = (WORKER_TIMEOUT + 30) * GPU_RATE
        if stage == "full" and remaining() < reserve:
            raise RuntimeError(f"Compute cap reached before {operation}; remaining estimate ${remaining():.2f}, reservation ${reserve:.2f}. Outputs preserved.")
        json_write(root / "reservations" / f"{serial:05d}.json", {"operation": operation,
                   "path": subdir, "worst_case_usd": reserve})
        volume.commit()
        try:
            result = worker.remote(run_id, operation, str(checkpoint), subdir, end_epoch, job or {})
        except Exception:
            volume.reload()
            spent += reserve
            json_write(ledger_path / f"{serial:05d}.json", {"operation": operation, "path": subdir,
                       "conservative_usd": reserve, "failed": True,
                       "note": "Failed attempt charged at its reserved upper estimate before any retry"})
            serial += 1
            volume.commit()
            raise
        volume.reload()
        charge = (result["worker_elapsed_seconds"] + 30) * GPU_RATE
        spent += charge
        json_write(ledger_path / f"{serial:05d}.json", {"operation": operation, "path": subdir,
                   "worker_seconds": result["worker_elapsed_seconds"], "conservative_usd": charge})
        serial += 1
        json_write(receipt, {"result": result})
        volume.commit()
        return result
    initial = base / "inputs/assets/initial.pt"
    try:
        if stage == "smoke":
            result = call("smoke_train", initial, "smoke/train", end_epoch=11)
            probe = call("smoke_gate", initial, "smoke/gate")
            json_write(root / "summary.json", {"training": result, "gate": probe,
                       "conservative_usd": spent, "scientific_run": False})
            volume.commit()
            return {"status": "smoke_complete", "estimated_usd": spent}
        json_write(root / f"launch_{serial:05d}.json", {"budget_usd": budget_usd,
                   "protocol": protocol, "gpu_rate_with_margin": GPU_RATE,
                   "gpu": GPU,
                   "note": "Estimate includes memory/CPU and startup margin; not a billing API hard limit"})
        gates = []
        checkpoint = initial
        stop_reason = "maximum_epoch"
        validation_losses = []
        for end_epoch in range(20, protocol["max_epoch"] + 1, 10):
            result = call("train", checkpoint, f"full/train/to_{end_epoch:04d}", end_epoch=end_epoch)
            checkpoint = Path(result["checkpoint"])
            validation_losses.append(result["rows"][-1]["ema_val_loss"])
            if end_epoch == 20:
                gates.extend(call("gate", initial, "full/gate_initial")["rows"])
            gates.extend(call("gate", checkpoint, f"full/gate_{end_epoch:04d}")["rows"])
            if plateau(gates, end_epoch) and loss_plateau(validation_losses):
                stop_reason = "prespecified_gate_and_validation_loss_plateau_after_epoch_100"
                break
        best = max(gates, key=gate_rank)
        selection_file = root / "selected_checkpoint.json"
        if not selection_file.exists():
            json_write(selection_file, {**best, "stop_reason": stop_reason,
                       "last_epoch": end_epoch, "passed_gate": best["passed"],
                       "selection_data": "fixed synthetic gate, no test images or labels"})
            json_write(root / "gate_history.json", gates)
            volume.commit()
        checkpoint = best["checkpoint"]
        # Test access is postponed until every validation setting is complete.
        validation = [call("job", checkpoint, "full/validation", job=j) for j in validation_jobs(protocol)]
        selected = {axis: choose_validation(validation, axis, protocol) for axis in ("manufacturer", "field_strength")}
        if not (root / "selected_operating_points.json").exists():
            json_write(root / "selected_operating_points.json", selected)
            volume.commit()
        results = [call("job", checkpoint, "full/test", job=j) for j in test_jobs(protocol, selected)]
        if not (root / "complete.json").exists():
            json_write(root / "complete.json", {"status": "sampling_complete", "selected_checkpoint": best,
                       "operating_points": selected, "test_jobs": results, "conservative_usd": spent,
                       "analysis_remaining": "automatic analysis follows; sampling alone is not B3 completion"})
            volume.commit()
        analysis = call("analysis", checkpoint, "full/analysis")
        json_write(root / "research_complete.json", {"status": "b1_b3_complete", "analysis": analysis,
                   "estimated_usd": spent, "selected_epoch": best["epoch"], "gate_passed": best["passed"],
                   "stop_reason": stop_reason})
        volume.commit()
        return {"status": "b1_b3_complete", "estimated_usd": spent, "selected_epoch": best["epoch"]}
    except Exception as error:
        json_write(root / f"interrupted_{serial:05d}_{int(time.time())}.json", {
            "error": str(error), "conservative_usd": spent, "stage": stage,
            "gate_rows": locals().get("gates", []), "validation_losses": locals().get("validation_losses", []),
            "note": "Completed stage receipts and checkpoints preserved. Retry only after reviewing the cause."})
        volume.commit()
        raise
    finally:
        json_write(sessions / f"{time.time_ns()}.json", {
            "elapsed_seconds": time.perf_counter() - started,
            "conservative_usd": (time.perf_counter() - started) * CONTROLLER_RATE})
        volume.commit()


def upload(run_id):
    import yaml
    base = ROOT / "reports/v4_diffusion" / run_id
    remote = f"/v4_convergence/{run_id}"
    # Check collision before uploading. Never use force=True on this shared volume.
    try:
        entries = volume.listdir(remote)
    except (FileNotFoundError, modal.exception.NotFoundError):
        entries = []
    if entries:
        print("Remote inputs already exist; verifying manifest hash, upload skipped.")
        remote_manifest = b"".join(volume.read_file(remote + "/inputs/manifest.json"))
        if json.loads(remote_manifest)["pack_sha256"] != json.loads((base / "inputs/manifest.json").read_text())["pack_sha256"]:
            raise ValueError("Remote/local pack mismatch")
        addendum = base / "protocol_addendum_v2.json"
        if addendum.exists() and not any(e.path.endswith("protocol_addendum_v2.json") for e in entries):
            with volume.batch_upload(force=False) as batch:
                batch.put_file(addendum, remote + "/protocol_addendum_v2.json")
        return
    cfg = yaml.safe_load((base / "config.yaml").read_text())
    config_json = base / "inputs/config.json"
    if not config_json.exists():
        config_json.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    with volume.batch_upload(force=False) as batch:
        batch.put_directory(base / "inputs", remote + "/inputs")
        batch.put_file(base / "protocol.json", remote + "/protocol.json")
        batch.put_file(config_json, remote + "/config.json")
        if (base / "protocol_addendum_v2.json").exists():
            batch.put_file(base / "protocol_addendum_v2.json", remote + "/protocol_addendum_v2.json")


def fetch(run_id):
    from scripts.v4_convergence.core import digest
    base = ROOT / "reports/v4_diffusion" / run_id / "remote"
    prefix = f"v4_convergence/{run_id}/"
    count = 0
    for entry in volume.iterdir("/" + prefix, recursive=True):
        name = entry.path.lstrip("/")
        if entry.type.name != "FILE" or "/inputs/" in name:
            continue
        relative = name.removeprefix(prefix)
        if relative == name or ".." in Path(relative).parts:
            raise ValueError("Unexpected remote path")
        target = base / relative
        if target.exists():
            continue  # fetched snapshots are immutable; do not overwrite
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_suffix(target.suffix + ".partial")
        with partial.open("wb") as f:
            for chunk in volume.read_file(entry.path):
                f.write(chunk)
        partial.rename(target)
        count += 1
    selection = base / "full/selected_checkpoint.json"
    if selection.exists():
        info = json.loads(selection.read_text())
        relative = Path(info["checkpoint"]).relative_to(Path("/data") / prefix)
        src = (ROOT / "reports/v4_diffusion" / run_id / relative) if relative.parts[0] == "inputs" else base / relative
        dest = ROOT / "checkpoints/diffusion/ax_t1_128_v4" / run_id / f"selected_epoch_{info['epoch']:04d}.pt"
        if not dest.exists():
            import shutil
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dest)
        print("Selected checkpoint", dest, "sha256", digest(dest))
    print("Fetched", count, "new files to", base)


@app.local_entrypoint()
def main(run_id: str, stage: str = "smoke", budget_usd: float = 0.0):
    upload(run_id)
    print("Run", run_id, "stage", stage, "budget", budget_usd, flush=True)
    if stage == "benchmark":
        result = worker.remote(run_id, "benchmark", (remote_root(run_id) / "inputs/assets/initial.pt").as_posix(), f"benchmark_{GPU.lower()}")
    else:
        result = controller.remote(run_id, stage, budget_usd)
    print(result)
    fetch(run_id)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--fetch", required=True)
    fetch(p.parse_args().fetch)
