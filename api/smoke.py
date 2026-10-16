"""Numerical compatibility check against an existing B3 image; never changes it."""
import json
from pathlib import Path

import numpy as np
from skimage.metrics import structural_similarity

from api import data
from api.main import EditRequest, run_edit
from api.models import editor, sample, INFERENCE_LOCK


def main():
    z = data.pack("main_field_strength_t100_w5_s0")
    index = 0
    source = data.slice_info(str(z["indices"][index]))
    np.testing.assert_array_equal(data.source_image(source["id"]), z["source_images"][index])
    with INFERENCE_LOCK:
        model = editor("converged")
        after = sample(model, z["source_images"][index], int(z["source_mfg"][index]),
                       1-int(z["source_fs"][index]), float(z["source_echo"][index]), 100, 5., 42)
    score = float(structural_similarity(z["cf"][index], after, data_range=1.))
    receipt = dict(reference=data.relative(data.B3 / "main_field_strength_t100_w5_s0/images.npz"),
                   index=index, slice_id=source["id"], model_metadata=model.checkpoint_metadata,
                   device=str(model.device), reference_ssim=score,
                   max_absolute_difference=float(np.max(np.abs(after-z["cf"][index]))),
                   passed=score > .999)
    # Also exercise the public result assembly and classifier models with a real edit.
    result = run_edit(EditRequest(slice_id=source["id"], manufacturer=source["manufacturer"],
                                 field_strength=1-source["field_strength"], seed=42))
    receipt.update(judges_loaded=len(result["judges"]), saved_cache_match=result["cache_kind"],
                   echo_fixed=result["target"]["echo"] == source["echo"])
    output = Path(__file__).parent / "runtime"
    output.mkdir(exist_ok=True)
    (output / "smoke.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps(receipt, indent=2))
    if not receipt["passed"]:
        raise SystemExit("B3 numerical compatibility check failed")


if __name__ == "__main__":
    main()
