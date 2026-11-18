"""Execute the authorised private hybrid notebook and retrieve only its versioned results."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from scripts.v8_generator_dev.io import ROOT,digest,json_write,verify_inputs
from scripts.v19_hybrid_prepare import BASE,PREPARED,KERNEL


def run():
    verify_inputs(json.loads((PREPARED/"inputs.json").read_text()))
    endpoints=json.loads((PREPARED/"endpoints.json").read_text())
    if len(endpoints["rows"])!=6 or not endpoints["frozen_state_verified"]:
        raise ValueError("Endpoint verification missing")
    design=json.loads((PREPARED/"protocol.json").read_text())
    cli=Path(os.environ["LOCALAPPDATA"])/"CodexTools/kaggle-cli/Scripts/kaggle.exe"
    out=BASE/"execution_20261118_01"
    out.mkdir(exist_ok=False)
    (out/"README.md").write_text("Producer is scripts.v19_hybrid_run. launch.json binds the private kernel and "
        "runtime cap. push.txt and status.jsonl record submission and monitoring. download/ contains versioned "
        "outputs. review.log records local analysis. complete.json or failed.json records the outcome.\n",
        encoding="utf-8")
    ref=KERNEL+"/1"
    cap=design["max_kaggle_seconds"]
    json_write(out/"launch.json",dict(kernel=ref,timeout_seconds=cap,
        preparation_sha256=digest(PREPARED/"inputs.json"),benchmark_projection_minutes=52))
    try:
        resp=subprocess.run([str(cli),"kernels","push","-p",str(PREPARED/"kernel"),
            "--accelerator","NvidiaTeslaT4","--timeout",str(cap)],capture_output=True,text=True,cwd=ROOT,timeout=120)
        (out/"push.txt").write_text(resp.stdout+resp.stderr,encoding="utf-8")
        resp.check_returncode()
        if "version 1 successfully pushed" not in resp.stdout:
            raise ValueError("Unexpected notebook version")
        print(resp.stdout.strip(),flush=True)
        limit=time.monotonic()+cap+1800
        with (out/"status.jsonl").open("x",encoding="utf-8") as log:
            last=None
            while time.monotonic()<limit:
                status=subprocess.run([str(cli),"kernels","status",ref],check=True,capture_output=True,
                    text=True,timeout=60).stdout.strip()
                log.write(json.dumps(dict(time_utc=time.time(),status=status))+"\n")
                log.flush()
                if status!=last:
                    print(status,flush=True)
                    last=status
                if "KernelWorkerStatus.COMPLETE" in status:break
                if any(f"KernelWorkerStatus.{state}" in status for state in ("ERROR","CANCELLED","FAILED","TIMEOUT")):
                    raise RuntimeError(status)
                time.sleep(30)
            else: raise TimeoutError("Local polling limit reached")
        download=out/"download"
        download.mkdir()
        (download/"README.md").write_text("Producer is Kaggle CLI output for the version in launch.json. "
            "The receipt and journals bind the original hybrid arrays and ordinary replay sentinels.\n",encoding="utf-8")
        with (out/"download.log").open("x",encoding="utf-8") as log:
            subprocess.run([str(cli),"kernels","output",ref,"-p",str(download),"--page-size","200",
                "--file-pattern",r"^(receipt\.json|seed\d+\.jsonl|seed\d+_g\d+_case\d+\.npz)$"],
                check=True,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=1800)
        with (out/"review.log").open("x",encoding="utf-8") as log:
            subprocess.run([sys.executable,"-u","-m","scripts.v19_hybrid_review","--download",str(download)],
                           check=True,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=1800)
        json_write(out/"complete.json",dict(complete=True,downloaded=True,locally_reviewed=True))
    except Exception as error:
        json_write(out/"failed.json",dict(complete=False,error=str(error)))
        raise


if __name__=="__main__":
    run()
