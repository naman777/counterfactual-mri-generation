"""Generic Modal runner for chains of SDEdit+CFG edits with the project's CounterfactualEditor.

A job is {"key": str, "image": float32[H,W] in [0,1], "steps": [{"m","f","e","t","w","seed"}, ...]}.
Each step edits the output of the previous step; every intermediate output is returned.

Usage (from counterfactual-mri-generation/):
    python -m modal run scripts/v4_workshop/modal_edit_chains.py --jobs <jobs.pkl> --out <results.pkl>
The diffusion checkpoint must be on the volume as /diffusion_latest.pt (uploaded by v2_quickwins/modal_stronger_edits.py);
pass --upload-ckpt to (re)upload checkpoints/diffusion/ax_t1_128/latest.pt.
GPU cap: env MODAL_MAX_GPUS (default 8; workspace limit is 10 GPUs total, so run one app at a time).
"""
import os
import pickle
import sys
from pathlib import Path

import modal

_here = Path(__file__).resolve()
ROOT = _here.parents[2] if len(_here.parents) > 2 else _here.parent
app = modal.App("capstone-edit-chains")
image = modal.Image.debian_slim(python_version="3.12").pip_install("torch==2.5.1", "numpy<2.2", "pyyaml", "pillow")
if modal.is_local():
    image = image.add_local_dir(ROOT / "src", "/root/proj/src").add_local_dir(ROOT / "configs", "/root/proj/configs")
vol = modal.Volume.from_name("capstone-judge-data", create_if_missing=True)
# Workspace limit is 10 GPUs in total: keep one app below it and never run two apps at once.
MAX_GPUS = int(os.environ.get("MODAL_MAX_GPUS", "8"))


@app.cls(image=image, gpu="A10G", volumes={"/data": vol}, timeout=3600, max_containers=MAX_GPUS)
class ChainEditor:
    @modal.enter()
    def load(self):
        sys.path.insert(0, "/root/proj")
        from src.inference.counterfactual_editor import CounterfactualEditor
        self.ed = CounterfactualEditor(config_path="/root/proj/configs/diffusion.yaml",
                                       checkpoint_path="/data/diffusion_latest.pt", device="cuda")

    @modal.method()
    def run(self, job):
        import numpy as np
        x = np.asarray(job["image"], np.float32)
        outs = []
        for s in job["steps"]:
            r = self.ed.generate_counterfactual(image_01=x, target_manufacturer_label=int(s["m"]),
                                                target_field_strength_label=int(s["f"]),
                                                target_echo_time_norm=float(s["e"]), edit_timestep=int(s["t"]),
                                                guidance_scale=float(s["w"]), seed=int(s["seed"]))
            x = np.clip(r["generated_01"].astype(np.float32), 0, 1)
            outs.append(x.astype(np.float16))
        return job["key"], outs


@app.local_entrypoint()
def main(jobs: str, out: str, upload_ckpt: bool = False):
    if upload_ckpt:
        with vol.batch_upload(force=True) as b:
            b.put_file(str(ROOT / "checkpoints/diffusion/ax_t1_128/latest.pt"), "/diffusion_latest.pt")
    job_list = pickle.loads(Path(jobs).read_bytes())
    print(f"{len(job_list)} jobs, {sum(len(j['steps']) for j in job_list)} edits")
    res = {}
    for i, (key, outs) in enumerate(ChainEditor().run.map(job_list, order_outputs=False)):
        res[key] = outs
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(job_list)} done")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_bytes(pickle.dumps(res))
    print("saved", out, len(res))
