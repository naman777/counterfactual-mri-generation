"""Assemble a small Kaggle sampler script and a minimum inference-only input bundle."""
import argparse
import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from scripts.v8_generator_dev.data import CanonicalDataset
from scripts.v8_generator_dev.io import ROOT, digest, json_write, verify_inputs
from scripts.v13_guide_dropout.protocol import canonical_hash
from scripts.v15_shared_residual.inputs import PREPARED, NEW_TRAINING

OWNER = "namankundra18072005"
BASE = ROOT/"reports/v16_kaggle"
PLAN = ROOT/"reports/v15_shared_residual/pilot_plan_20261118_01"
DATASET = OWNER + "/capstone-primary-input-20261118-01"
DEFINITIONS = {
    "src/models/conditioning.py": ["AcquisitionConditionEmbedding"],
    "src/models/ddpm_scheduler.py": ["extract_values", "DDPMScheduler"],
    "src/models/diffusion_unet.py": ["get_group_count", "SinusoidalTimeEmbedding", "ResidualBlock",
        "Downsample", "Upsample", "MRIConditionalUNet"],
    "scripts/v8_generator_dev/model.py": ["ScaleShiftResidualBlock", "ScaleShiftMRIUNet", "build_models"],
    "scripts/v6_structure/model.py": ["StructureAdapter", "StructureConditionedUNet", "BoundStructureModel"],
    "scripts/v8_generator_dev/detail_candidates.py": ["local_order_native", "order_maps", "multiscale_guide",
        "MatchedDetailAdapter", "DetailConditionedUNet"],
    "scripts/v8_generator_dev/train_smoke.py": ["sample_wiring"],
    "scripts/v14_dropout_audit/shared_residual.py": ["parent_response_score"],
    "scripts/v15_shared_residual/sampler.py": ["sample"],
    "scripts/v8_generator_dev/adapter_pilot.py": ["state_digest"],
    "scripts/v8_generator_dev/full_train.py": ["deterministic_runtime"],
}


def create_kernel(output, settings, name, attachments):
    kernel = output/"kernel"
    kernel.mkdir()
    code = ["import hashlib, math, numpy as np, torch\nfrom torch import nn\nimport torch.nn.functional as F\n"
        "from typing import List, Optional", "SETTINGS = "+repr(settings)]
    for path, wanted in DEFINITIONS.items():
        source = (ROOT/path).read_text(encoding="utf-8")
        nodes = {node.name:node for node in ast.parse(source).body if isinstance(node,(ast.ClassDef, ast.FunctionDef))}
        for symbol in wanted:
            node = nodes[symbol]
            first = min([node.lineno]+[decorator.lineno for decorator in node.decorator_list])
            code.append("\n".join(source.splitlines()[first-1:node.end_lineno]))
    worker = ROOT/"scripts/v16_kaggle_worker.py"
    code.append(worker.read_text(encoding="utf-8"))
    combined = "\n\n".join(code)+"\n"
    compile(combined, "sampler_job.py", "exec")
    source_path=kernel/"sampler_job.py"
    source_path.write_text(combined, encoding="utf-8", newline="\n")
    json_write(kernel/"kernel-metadata.json", dict(id=OWNER+"/"+name, code_file=source_path.name,
        title=name.replace("-"," ").title(), language="python", kernel_type="script", is_private=True,
        enable_gpu=True, enable_internet=False, dataset_sources=attachments, competition_sources=[], kernel_sources=[]))
    evidence = {path:digest(ROOT/path) for path in DEFINITIONS}
    for path in (source_path, kernel/"kernel-metadata.json", worker, PLAN/"protocol.json"):
        evidence[path.relative_to(ROOT).as_posix()] = digest(path)
    json_write(output/"inputs.json", evidence)


def prepare(mode):
    verify_inputs(json.loads((PLAN/"inputs.json").read_text()))
    protocol = json.loads((PREPARED/"protocol.json").read_text())
    design = json.loads((PLAN/"protocol.json").read_text())
    reference = protocol["reference_protocol"]
    parent = ROOT/reference["parent"]
    if digest(parent) != reference["parent_sha256"]:
        raise ValueError("Unexpected parent bytes")
    weights = torch.load(parent, map_location="cpu", weights_only=False)
    if weights["variant"] != "additive":
        raise ValueError("Unexpected parent backbone")
    settings = dict(mode=mode, config={key:weights["config"][key] for key in ("condition","model","diffusion")},
        adapter_width=reference["training"]["adapter_width"], point=design["point"],
        study_plan_sha256=digest(PLAN/"protocol.json"))
    output = BASE/(mode+"_20261118_01")
    output.mkdir(parents=True, exist_ok=False)
    if mode == "benchmark":
        create_kernel(output, settings, "capstone-sampler-benchmark-20261118-01", [])
        readme = ("Runs the actual ordinary and shared residual sampler code at the frozen manufacturer settings "
                  "on synthetic inputs and newly initialised weights. Two concurrent T4 lanes measure paired case time.")
    else:
        data = output/"dataset"
        data.mkdir()
        canonical = CanonicalDataset(reference["view"], "val")
        selected = pd.read_csv(PREPARED/"validation_selection.csv")
        indices = selected.dataset_index.astype(int).tolist()
        if indices != design["dataset_indices"] or canonical.frame.iloc[indices].patient_id.nunique() != 15:
            raise ValueError("Validation selection changed")
        sources = np.stack([canonical.image_01(index) for index in indices])
        with (data/"validation.npy").open("xb") as handle:
            np.save(handle, sources, allow_pickle=False)
        conditions = []
        for index in indices:
            item = canonical[index]
            conditions.append(dict(dataset_index=index, manufacturer=item["manufacturer_label"],
                                   field_strength=item["field_strength_label"], echo_time_norm=float(item["echo_time_norm"])))
        torch.save(dict(parent=weights["unet_state_dict"], conditioner=weights["conditioning_state_dict"]), data/"parent.pt")
        trained = json.loads(NEW_TRAINING.read_text())["copies"]
        for seed in design["training_seeds"]:
            copy = trained[f"seed{seed}__coarse_fine"]
            path = ROOT/copy["checkpoint"]
            if digest(path) != copy["sha256"]:
                raise ValueError("Unexpected adapter bytes")
            state = torch.load(path, map_location="cpu", weights_only=False)
            if state["protocol_sha256"] != canonical_hash(protocol) or state["engineering"] or state["step"] != 2780:
                raise ValueError("Unexpected adapter identity")
            torch.save(state["adapter"], data/f"seed{seed}.pt")
        (data/"ATTRIBUTION.txt").write_text(
            "Duke-Breast-Cancer-MRI, Mazurowski et al., The Cancer Imaging Archive.\n"
            "https://www.cancerimagingarchive.net/collection/duke-breast-cancer-mri/\n"
            "Source images are licensed CC BY-NC 4.0. Used for noncommercial academic research.\n"
            "https://creativecommons.org/licenses/by-nc/4.0/\n"
            "Changes include slice selection, rescaling, intensity normalisation, resizing and canonical orientation.\n",
            encoding="utf-8")
        manifest = dict(files={path.name:digest(path) for path in data.iterdir()}, conditions=conditions,
                        training_seeds=design["training_seeds"], generation_seeds=design["generation_seeds"])
        json_write(data/"pilot-input.json", manifest)
        json_write(data/"dataset-metadata.json", dict(title="Capstone primary sampler inputs 20261118",
                   id=DATASET, licenses=[dict(name="CC-BY-NC-4.0")]))
        settings.update(dataset_path="/kaggle/input/capstone-primary-input-20261118-01",
                        input_manifest_sha256=digest(data/"pilot-input.json"))
        create_kernel(output, settings, "capstone-primary-sampler-20261118-01", [DATASET])
        readme = ("kernel/ runs the frozen three-profile matrix in two T4 lanes. dataset/ contains fifteen "
                  "preprocessed public TCIA validation slices, their minimal acquisition conditions, and four "
                  "inference-only weight files. No DICOMs, subject identifiers, judge weights or test inputs. "
                  "The images retain their source attribution and CC BY-NC 4.0 terms. The private dataset is "
                  "for this team's noncommercial computation. Treat the source licence as authoritative.")
    (output/"README.md").write_text("Producer is scripts.v16_kaggle_prepare. "+readme+
        " inputs.json binds the exact staged source.\n", encoding="utf-8")
    print(output, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("mode", choices=("benchmark","pilot"))
    prepare(parser.parse_args().mode)
