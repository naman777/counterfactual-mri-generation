"""Local data/API contract checks. Requires the research inputs on this laptop."""
import unittest

from fastapi.testclient import TestClient

from api.main import app, EditRequest
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


if __name__ == "__main__":
    unittest.main()
