"""Local data/API contract checks. Requires the research inputs on this laptop."""
import unittest
import io
import json
import zipfile
from PIL import Image

from fastapi.testclient import TestClient

from api.main import app, EditRequest, run_edit, make_batch, BATCHES, BatchRequest
from api import data


class WorkbenchContract(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_catalog_is_the_manifest_and_rejects_paths(self):
        rows = self.client.get("/slices").json()["items"]
        self.assertEqual(len(rows), len(data.csv_rows(data.TEST)))
        self.assertEqual(self.client.get("/slice/not-a-slice").status_code, 404)

    def test_evidence_maps_back_to_test_manifest(self):
        name = "main_field_strength_t100_w5_s0"
        rows = self.client.get(f"/evidence/{name}").json()["items"]
        self.assertEqual(len(rows), len(data.pack(name)["indices"]))
        self.assertTrue(all(len(r["judges"]) == 3 for r in rows))
        self.assertEqual(self.client.get(f"/evidence/{name}/-1/image").status_code, 404)
        self.assertEqual(self.client.get(f"/evidence/{name}/0/image").headers["content-type"], "image/png")

    def test_headlines_have_ordered_patient_intervals_and_sources(self):
        for axis in ("field_strength", "manufacturer"):
            rows = self.client.get(f"/results/{axis}").json()["items"]
            self.assertTrue(rows)
            self.assertTrue(any(r["model"] == "Epoch 10" for r in rows))
            for r in rows:
                self.assertLessEqual(r["lo"], r["mean"] + 1e-12)
                self.assertLessEqual(r["mean"], r["hi"] + 1e-12)
                self.assertTrue((data.ROOT / r["source"]).is_file())

    def test_edit_rejects_invalid_controls_and_echo_override(self):
        payload = dict(slice_id="0", manufacturer=0, field_strength=1, timestep=999)
        self.assertEqual(self.client.post("/edit", json=payload).status_code, 422)
        self.assertEqual(self.client.post("/edit", json=dict(slice_id="0", manufacturer=0, field_strength=1, echo=0.5)).status_code, 422)
        self.assertEqual(self.client.post("/edit", json=dict(slice_id="unknown", manufacturer=0, field_strength=1)).status_code, 404)

    def test_png_upload_requires_source_metadata_and_preserves_echo(self):
        content = data.png_bytes(data.source_image("0"))
        response = self.client.post("/upload", files={"file": ("source.png", content, "image/png")})
        self.assertEqual(response.status_code, 422)
        source = data.slice_info("0")
        response = self.client.post("/upload", files={"file": ("source.png", content, "image/png")},
                                    data=dict(manufacturer=source["manufacturer"], field_strength=source["field_strength"], echo_ms=source["echo_ms"]))
        self.assertEqual(response.status_code, 200)
        uploaded = response.json()
        self.assertTrue(uploaded["warnings"])
        self.assertAlmostEqual(uploaded["echo"], source["echo"])
        self.assertEqual(data.source_image(uploaded["id"]).shape, (128, 128))

    def test_patient_zip_has_original_float_arrays_and_manifest(self):
        name = "main_field_strength_t100_w5_s0"
        patient = data.evidence_rows(name)[0]["slice"]["patient"]
        BATCHES["contract"] = dict(status="queued")
        make_batch("contract", BatchRequest(set_name=name, patient=patient))
        self.assertEqual(BATCHES["contract"]["status"], "complete")
        with zipfile.ZipFile(io.BytesIO(BATCHES["contract"]["content"])) as archive:
            manifest = json.loads(archive.read("manifest.json"))
            self.assertTrue(manifest["research_only"])
            self.assertEqual(manifest["generation"]["t"], 100)
            for item in manifest["slices"]:
                self.assertEqual(item["slice"]["patient"], patient)
                import numpy as np
                array = np.load(io.BytesIO(archive.read(f"{patient}/example_{item['index']:03d}.npy")), allow_pickle=False)
                np.testing.assert_array_equal(array, data.pack(name)["cf"][item["index"]])

    def test_saved_edit_and_report_formats_are_real_artifacts(self):
        source = data.slice_info(str(data.pack("main_field_strength_t100_w5_s0")["indices"][0]))
        result = run_edit(EditRequest(slice_id=source["id"], manufacturer=source["manufacturer"], field_strength=1-source["field_strength"]))
        self.assertEqual(len(result["judges"]), 4)
        self.assertEqual(result["target"]["echo"], source["echo"])
        self.assertTrue(result["cached"])
        for kind in ("png", "pdf", "json"):
            response = self.client.get(f"/report/{result['id']}?format={kind}")
            self.assertEqual(response.status_code, 200)
            if kind == "png":
                self.assertEqual(Image.open(io.BytesIO(response.content)).size, (1400,1430))
            elif kind == "pdf":
                self.assertTrue(response.content.startswith(b"%PDF"))
            else:
                self.assertEqual(response.json()["id"], result["id"])

    def test_joint_evidence_requires_both_targets(self):
        name = "joint_both_t300_w5_s0"
        row = data.evidence_rows(name)[0]
        self.assertEqual(row["direction"], "joint")
        saved = next(r for r in data.csv_rows(data.B3/name/"metrics.csv") if r["judge"] == "original_predictor" and r["example_id"] == "example_000")
        self.assertEqual(row["original_hit"], bool(float(saved["both_hit"])))


if __name__ == "__main__":
    unittest.main()
