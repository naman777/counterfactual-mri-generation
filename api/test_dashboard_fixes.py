import os
import unittest
from unittest.mock import patch

import numpy as np
from fastapi.testclient import TestClient

from api import data, demo_cache
from api.main import app, EditRequest, run_edit, saved_edit, CACHE
from api.metrics import analyse


class DashboardFixes(unittest.TestCase):
    def test_recommended_opener_is_a_saved_three_judge_case(self):
        catalogue = TestClient(app).get("/slices").json()
        source = next(s for s in catalogue["items"] if s["id"] == catalogue["recommended_id"])
        req = EditRequest(slice_id=source["id"], manufacturer=source["manufacturer"],
                          field_strength=1-source["field_strength"], seed=source["paper_seeds"]["fs"])
        saved = demo_cache.lookup(req)
        self.assertIsNotNone(saved)
        self.assertEqual(saved["verdict"]["confirmed"], 3)
        self.assertIn("44", catalogue["demo_ids"])

    def test_every_evaluation_slice_has_its_exact_paper_seed(self):
        for axis in ("field_strength", "manufacturer"):
            name = f"main_{axis}_t{100 if axis == 'field_strength' else 300}_w{5 if axis == 'field_strength' else 3}_s0"
            for position, index in enumerate(data.pack(name)["indices"]):
                source = data.slice_info(str(index))
                seed = source["paper_seeds"]["fs" if axis == "field_strength" else "mfg"]
                self.assertEqual(seed, 42 + position)
                req = EditRequest(slice_id=source["id"], manufacturer=1-source["manufacturer"] if axis == "manufacturer" else source["manufacturer"],
                                  field_strength=1-source["field_strength"] if axis == "field_strength" else source["field_strength"],
                                  timestep=100 if axis == "field_strength" else 300, guidance=5 if axis == "field_strength" else 3, seed=seed)
                self.assertIsNotNone(saved_edit(req, source))
                self.assertIsNotNone(saved_edit(req.model_copy(update={"null_edit": True}), source))
                self.assertIsNone(saved_edit(req.model_copy(update={"seed": seed + 999}), source))

    def test_histogram_uses_same_source_foreground_for_both_images(self):
        before = np.zeros((128, 128), dtype=np.float32)
        before[32:96, 32:96] = .4
        after = np.full_like(before, .1)
        after[32:96, 32:96] = .7
        result = analyse(before, after)
        self.assertEqual(sum(b["source"] for b in result["histogram"]), 64*64)
        self.assertEqual(sum(b["edited"] for b in result["histogram"]), 64*64)
        self.assertEqual(sum(b["source"] for b in analyse(np.zeros_like(before), after)["histogram"]), 0)

    def test_all_80_saved_requests_replay_without_any_inference(self):
        manifest = demo_cache.manifest()
        self.assertEqual(len(manifest["slices"]), 20)
        self.assertEqual(len({s["patient"] for s in manifest["slices"]}), 16)
        self.assertEqual(len(manifest["results"]), 80)
        with patch.dict(os.environ, MRONE_DEMO_ONLY="1"), patch("api.main.editor", side_effect=AssertionError("No GPU")), patch("api.main.probabilities", side_effect=AssertionError("No classifiers")):
            import json
            for filename in manifest["results"].values():
                saved = json.loads((demo_cache.CACHE_DIR / filename).read_text(encoding="utf-8"))
                req = EditRequest(**saved["request"])
                result = run_edit(req)
                self.assertEqual(result["metrics"], saved["metrics"])
                self.assertEqual(result["cache_kind"], "Saved demo evidence")
                self.assertEqual(result["target"]["echo"], result["source"]["echo"])
            client = TestClient(app)
            req.seed += 999
            self.assertEqual(client.post("/edit", json=req.model_dump()).status_code, 409)
            CACHE.clear()
            self.assertEqual(client.get(f"/report/{saved['id']}").status_code, 200)


if __name__ == "__main__":
    unittest.main()
