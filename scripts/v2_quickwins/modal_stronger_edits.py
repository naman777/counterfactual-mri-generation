"""Diagnostic: stronger diffusion edits (incl. t=999 ~ pure conditional generation) on Modal GPUs.

Same 100-slice selection (seed 42, <=10/patient); first --n slices; per-example seed 42+idx as in Phase 12.
Usage: python -m modal run scripts/v2_quickwins/modal_stronger_edits.py --out reports/v2_quickwins/stronger_edits
"""
import sys
from pathlib import Path

import modal

_here = Path(__file__).resolve()
ROOT = _here.parents[2] if len(_here.parents) > 2 else _here.parent  # local repo root; unused in the container
app = modal.App("capstone-stronger-edits")
image = modal.Image.debian_slim(python_version="3.12").pip_install("torch==2.5.1", "numpy<2.2", "pyyaml", "pillow")
if modal.is_local():
    image = image.add_local_dir(ROOT / "src", "/root/proj/src").add_local_dir(ROOT / "configs", "/root/proj/configs")
vol = modal.Volume.from_name("capstone-judge-data", create_if_missing=True)


@app.cls(image=image, gpu="A10G", volumes={"/data": vol}, timeout=3600, max_containers=8)  # workspace limit: 10 GPUs total
class Editor:
    @modal.enter()
    def load(self):
        sys.path.insert(0, "/root/proj")
        from src.inference.counterfactual_editor import CounterfactualEditor
        self.ed = CounterfactualEditor(config_path="/root/proj/configs/diffusion.yaml",
                                       checkpoint_path="/data/diffusion_latest.pt", device="cuda")

    @modal.method()
    def edit(self, job):
        import numpy as np
        key, img, m, f, e, t, w, seed = job
        r = self.ed.generate_counterfactual(image_01=np.asarray(img, np.float32), target_manufacturer_label=m,
                                            target_field_strength_label=f, target_echo_time_norm=e,
                                            edit_timestep=t, guidance_scale=w, seed=seed)
        return key, r["generated_01"]


@app.local_entrypoint()
def main(out: str, n: int = 50):
    import numpy as np
    import pandas as pd
    sys.path.insert(0, str(ROOT))
    from scripts.run_explainability_batch import build_target_condition, select_diverse_examples

    with vol.batch_upload(force=True) as b:
        b.put_file(str(ROOT / "checkpoints/diffusion/ax_t1_128/latest.pt"), "/diffusion_latest.pt")
    test = pd.read_csv(ROOT / "data/processed/ax_t1_128/splits/test.csv")
    ex = select_diverse_examples(test, 100, seed=42).iloc[:n]
    settings = [("field_strength", 200, 5.0), ("field_strength", 300, 5.0), ("field_strength", 999, 5.0),
                ("manufacturer", 500, 5.0), ("manufacturer", 999, 5.0)]
    jobs = []
    for control, t, w in settings:
        for idx, (_, row) in enumerate(ex.iterrows()):
            img = np.clip(np.load(ROOT / row["image_path"].replace("\\", "/")).astype(np.float32), 0, 1)
            _, tgt = build_target_condition(row, control)
            jobs.append(((control, t, w, idx), img, tgt["manufacturer_label"], tgt["field_strength_label"],
                         tgt["echo_time_norm"], t, w, 42 + idx))
    res = {}
    for key, cf in Editor().edit.map(jobs, order_outputs=False):
        res[key] = cf
    outd = Path(out)
    outd.mkdir(parents=True, exist_ok=True)
    for control, t, w in settings:
        arr = np.stack([res[(control, t, w, i)] for i in range(len(ex))])
        np.savez_compressed(outd / f"{control}_t{t}_w{int(w)}.npz", images=arr,
                            image_path=ex.image_path.to_numpy(str), patient_id=ex.patient_id.to_numpy(str))
        print("saved", control, t, w, arr.shape)
