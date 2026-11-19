"""Submit the authorised private pilot and follow through to the local review."""
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

from scripts.v8_generator_dev.io import ROOT, digest, json_write, verify_inputs
from scripts.v16_kaggle_prepare import BASE, PLAN


def run():
    cli = Path(os.environ["LOCALAPPDATA"])/"CodexTools/kaggle-cli/Scripts/kaggle.exe"
    staged = BASE/"pilot_20261118_01"
    benchmark = BASE/"benchmark_20261118_01/receipt.json"
    data = json.loads(benchmark.read_text())
    preflight = json.loads((staged/"local_preflight.json").read_text())
    protocol = json.loads((PLAN/"protocol.json").read_text())
    metadata = json.loads((staged/"kernel/kernel-metadata.json").read_text())
    manifest = staged/"dataset/pilot-input.json"
    if (data["mode"]!="benchmark" or data["study_plan_sha256"]!=digest(PLAN/"protocol.json")
        or data["runtime"]["gpu_names"]!=["Tesla T4"]*2 or not preflight["complete"]
        or preflight["input_manifest_sha256"]!=digest(manifest) or not metadata["is_private"]):
        raise ValueError("Unexpected benchmark or staging evidence")
    verify_inputs(json.loads((staged/"inputs.json").read_text()))
    cases = protocol["cases_per_profile"]
    timings = {lane["device"]:[sum(values.values()) for values in lane["paired_seconds"]] for lane in data["result"]}
    wall_estimate = max(2*cases*statistics.median(timings[0]), cases*statistics.median(timings[1]))
    timeout = math.ceil((2*cases*max(timings[0]) * 1.35 + 240) / 60)*60
    out = BASE/"execution_20261118_01"
    out.mkdir(exist_ok=False)
    (out/"README.md").write_text("Producer is scripts.v16_kaggle_run. launch_plan.json binds the benchmark, "
        "runtime limit and versioned kernel. push.txt and status.jsonl record submission and progress. "
        "download/ contains original Kaggle results. Local grading and primary intervals are written to "
        "the sibling primary_review_20261118_01 folder. complete.json or failed.json records the outcome.\n",
        encoding="utf-8")
    ref = metadata["id"]+"/1"
    json_write(out/"launch_plan.json",dict(kernel=ref,estimated_gpu_seconds=wall_estimate,timeout_seconds=timeout,
        cases=3*cases,benchmark_sha256=digest(benchmark),manifest_sha256=digest(manifest),
        private_data=True,private_kernel=True,gpu_count=2,per_case_seconds=timings))
    try:
        submit = subprocess.run([str(cli),"kernels","push","-p",str(staged/"kernel"),
            "--accelerator","NvidiaTeslaT4","--timeout",str(timeout)],cwd=ROOT,check=False,
            capture_output=True,text=True,timeout=120)
        (out/"push.txt").write_text(submit.stdout+submit.stderr,encoding="utf-8")
        submit.check_returncode()
        if "version 1 successfully pushed" not in submit.stdout:
            raise ValueError("Kaggle did not report the expected first version")
        print(submit.stdout.strip(),flush=True)
        limit = time.monotonic()+timeout+3600
        with (out/"status.jsonl").open("x",encoding="utf-8") as log:
            last=None
            while time.monotonic()<limit:
                response = subprocess.run([str(cli),"kernels","status",ref],check=True,
                    capture_output=True,text=True,timeout=60)
                status = response.stdout.strip()
                log.write(json.dumps(dict(time_utc=time.time(),status=status))+"\n")
                log.flush()
                if status!=last:
                    print(status,flush=True)
                    last=status
                if "KernelWorkerStatus.COMPLETE" in status:
                    break
                if any(f"KernelWorkerStatus.{term}" in status for term in ("ERROR","CANCELLED","TIMEOUT","FAILED")):
                    raise RuntimeError(status)
                time.sleep(30)
            else:
                raise TimeoutError("Kaggle status monitor exceeded its local deadline")
        target=out/"download"
        target.mkdir()
        (target/"README.md").write_text("Produced by the Kaggle CLI output command for the submitted version. "
            "The worker receipt and three journals identify the matched image arrays. See the sibling local review for grading.\n",
            encoding="utf-8")
        with (out/"download.log").open("x",encoding="utf-8") as log:
            subprocess.run([str(cli),"kernels","output",ref,"-p",str(target),"--page-size","200",
                "--file-pattern",r"^(receipt\.json|seed\d+\.jsonl|seed\d+_g\d+_case\d+\.npz)$"],check=True,cwd=ROOT,
                stdout=log,stderr=subprocess.STDOUT,timeout=1800)
        receipt=json.loads((target/"receipt.json").read_text())
        if receipt["runtime"]!=data["runtime"]:
            raise ValueError("Pilot ran on a different runtime from the dual-GPU benchmark")
        with (out/"review.log").open("x",encoding="utf-8") as log:
            subprocess.run([sys.executable,"-u","-m","scripts.v16_kaggle_review","--download",str(target)],
                check=True,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=1800)
        json_write(out/"complete.json",dict(complete=True,downloaded=True,locally_reviewed=True))
    except Exception as error:
        json_write(out/"failed.json",dict(complete=False,error=str(error)))
        raise


if __name__ == "__main__":
    run()
