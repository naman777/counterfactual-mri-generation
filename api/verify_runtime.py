"""Exercise the actual HTTP/SSE stack locally; writes only ignored QA artifacts."""
import io
import json
from pathlib import Path

import httpx
import numpy as np
from PIL import Image

from api import data


def main():
    out = Path(__file__).parent / "runtime"
    out.mkdir(exist_ok=True)
    receipt = {}
    with httpx.Client(base_url="http://127.0.0.1:3000/api", timeout=180.) as client:
        info = client.get("/slices").raise_for_status().json()
        source = next(r for r in info["items"] if r["id"] == info["recommended_id"])
        base = dict(slice_id=source["id"], manufacturer=source["manufacturer"], field_strength=1-source["field_strength"], timestep=100, guidance=5., seed=42, use_cache=False)
        for name, changes in [("epoch110", dict(model="converged")), ("null110", dict(model="converged", null_edit=True)), ("epoch10", dict(model="development", timestep=50))]:
            frames, result = 0, None
            with client.stream("POST", "/edit", json={**base, **changes}) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line.startswith("data: "):
                        continue
                    event = json.loads(line[6:])
                    if event["type"] == "error":
                        raise RuntimeError(event["message"])
                    frames += event["type"] == "frame"
                    if event["type"] == "result":
                        result = event["result"]
            assert result and frames > 0 and not result["cached"]
            assert result["target"]["echo"] == source["echo"]
            assert len(result["judges"]) == 4
            receipt[name] = dict(frames=frames, weights=result["weights"], ssim=result["metrics"]["ssim"], seconds=result["seconds"], echo_fixed=True)
            if name == "epoch110":
                for kind in ("png", "pdf", "json"):
                    report = client.get(f"/report/{result['id']}?format={kind}").raise_for_status()
                    (out / f"evidence-card.{kind}").write_bytes(report.content)
        for flip in (False, True):
            probe = client.post("/robustness", json=dict(slice_id=source["id"], sigma=0.03 if not flip else 0., vertical_flip=flip)).raise_for_status().json()
            assert len(probe["judges"]) == 5
            receipt["flip_probe" if flip else "noise_probe"] = dict(classifiers=len(probe["judges"]), note=probe["note"])
        row = data.csv_rows(data.TEST)[int(source["id"])]
        dicom = data.ROOT / row["slice_path"].replace("\\", "/")
        if dicom.exists():
            with dicom.open("rb") as f:
                uploaded = client.post("/upload", files={"file": ("source.dcm", f, "application/dicom")}).raise_for_status().json()
            assert uploaded["patient"] == "Local upload" and uploaded["warnings"]
            receipt["dicom_upload"] = dict(preprocessed=True, warnings=uploaded["warnings"], identifiers_returned=False)
        for route in ("/health", "/evidence", "/results/field_strength", "/results/manufacturer", "/tradeoff/field_strength", "/results-data/variance", "/results-data/high-frequency"):
            assert client.get(route).status_code == 200, route
        receipt["proxy_routes"] = "passed"
    (out / "integration.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
