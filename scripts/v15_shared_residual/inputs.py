"""Load only the frozen source data and the recorded final adapters."""
import json

import pandas as pd
import torch

from scripts.v6_structure.model import BoundStructureModel
from scripts.v8_generator_dev.data import CanonicalDataset
from scripts.v8_generator_dev.detail_candidates import DetailConditionedUNet, multiscale_guide
from scripts.v8_generator_dev.full_train import deterministic_runtime, load_ema
from scripts.v8_generator_dev.io import ROOT, digest, project_path
from scripts.v13_guide_dropout.protocol import canonical_hash

PREPARED = ROOT / "reports/v13_guide_dropout/preparation_20261118_01"
OLD_TRAINING = ROOT / "reports/v10_detail/guide_replicates_20261117_01_training/complete.json"
NEW_TRAINING = ROOT / "reports/v13_guide_dropout/guide_dropout_20261118_01_training/complete.json"
CONTROLS = ("correct", "zero", "absent", "wrong", "coarse_only", "fine_only")


class FrozenInputs:
    def __init__(self):
        deterministic_runtime()
        self.paths = {}
        self.protocol = self.read_json(PREPARED / "protocol.json")
        self.reference = self.protocol["reference_protocol"]
        self.parent_path = self.require(self.reference["parent"], self.reference["parent_sha256"])
        self.conditioner, self.parent, self.scheduler, payload = load_ema(self.parent_path, "cuda")
        del payload
        self.conditioner.eval().requires_grad_(False)
        self.parent.eval().requires_grad_(False)
        self.val = CanonicalDataset(self.reference["view"], "val")
        selection = self.require(PREPARED / "validation_selection.csv")
        self.indices = pd.read_csv(selection).dataset_index.astype(int).tolist()
        if len(self.indices) != 15 or self.val.frame.iloc[self.indices].patient_id.nunique() != 15:
            raise ValueError("Frozen source selection changed")
        for split in ("train", "val"):
            self.require(ROOT / self.reference["view"] / "splits" / f"{split}.csv")
            self.require(ROOT / "data/processed/ax_t1_128/splits" / f"{split}.csv")
        self.require(ROOT / self.reference["view"] / "view.json")
        self.receipts = {"old_full_guide": self.read_json(OLD_TRAINING),
                         "guide_dropout": self.read_json(NEW_TRAINING)}

    def require(self, path, expected=None):
        path = project_path(path)
        actual = digest(path)
        if expected is not None and actual != expected:
            raise ValueError(f"Frozen input changed: {path.name}")
        self.paths[path.relative_to(ROOT).as_posix()] = actual
        return path

    def read_json(self, path):
        return json.loads(self.require(path).read_text(encoding="utf-8"))

    def load_copy(self, study, seed, mode):
        record = self.receipts[study]["copies"][f"seed{seed}__{mode}"]
        path = self.require(record["checkpoint"], record["sha256"])
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        protocol = self.protocol if study == "guide_dropout" else self.reference
        if (checkpoint["engineering"] or checkpoint["step"] != 2780 or
            checkpoint["training_seed"] != seed or checkpoint["mode"] != mode or
            checkpoint["parent_sha256"] != self.reference["parent_sha256"] or
            checkpoint["protocol_sha256"] != canonical_hash(protocol)):
            raise ValueError("Unexpected final adapter identity")
        model = DetailConditionedUNet(self.parent, self.reference["training"]["adapter_width"]).cuda()
        model.adapter.load_state_dict(checkpoint["adapter"], strict=True)
        return model.eval().requires_grad_(False)

    def source(self, index):
        row = self.val.frame.iloc[index]
        self.require(row.source_image_path, row.source_array_sha256)
        return self.val[index]

    def bound(self, model, index, mode, control):
        if control not in CONTROLS:
            raise ValueError("Unknown guide control")
        guide_index = index
        if control == "wrong":
            position = self.indices.index(index)
            guide_index = self.indices[(position + 1) % len(self.indices)]
        row = self.val.frame.iloc[guide_index]
        self.require(row.source_image_path, row.source_array_sha256)
        image = torch.from_numpy(self.val.image_01(guide_index).copy())[None, None].cuda()
        guide = multiscale_guide(image, mode)
        if control == "zero":
            guide.zero_()
        if control == "coarse_only":
            guide[:, 3:] = 0
        if control == "fine_only":
            guide[:, :3] = 0
        present = torch.tensor([float(control != "absent")], device="cuda")
        return BoundStructureModel(model, guide, present).eval()

    def arguments(self, item, label, generation, timestep, guidance):
        if label not in (item["manufacturer_label"], 1 - item["manufacturer_label"]):
            raise ValueError("Manufacturer request outside the binary axis")
        return (self.scheduler, item["image"][None].cuda(),
                torch.tensor([label], device="cuda"),
                torch.tensor([item["field_strength_label"]], device="cuda"),
                torch.tensor([item["echo_time_norm"]], device="cuda"),
                generation + item["index"], timestep, guidance)
