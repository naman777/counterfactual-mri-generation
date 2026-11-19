"""Freeze headline provenance and local handoff evidence without rerunning research."""
import argparse
import csv
import hashlib
import importlib.metadata as metadata
import json
import platform
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT.parent
B2 = ROOT / "reports/v4_heldout_judge/b2_local_20261016_01"
AN = ROOT / "reports/v4_diffusion/b1_20261016_01/remote/analysis/20261016T153744Z"
HEADER = ROOT / "reports/v4_header_confounds_20261017_01"
PAPER = BASE / "research-paper-v11-evaluation"


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_json(path, value):
    with path.open("x", encoding="utf-8") as f:
        json.dump(value, f, indent=2)


def main(out):
    out.mkdir(parents=True, exist_ok=False)
    inputs = {}
    ledger = []

    def remember(path):
        path = Path(path)
        inputs[path.relative_to(BASE).as_posix()] = sha(path)
        return path

    def select(path, **selector):
        frame = rows(remember(path))
        matched = [(i + 2, r) for i, r in enumerate(frame)
                   if all(r[k] == str(v) for k, v in selector.items())]
        assert len(matched) == 1, (path, selector, len(matched))
        return matched[0]

    def claim(name, path, selector, fields, expected):
        line, row = select(path, **selector)
        values = [float(row[k]) for k in fields]
        assert all(abs(a - b) < .000001 for a, b in zip(values, expected)), (name, values, expected)
        ledger.append(dict(claim=name, source=path.relative_to(ROOT).as_posix(), csv_line=line,
                           selector=selector, fields=dict(zip(fields, values))))

    claim("Epoch-10 original predictor FS", ROOT / "reports/v2_quickwins/bootstrap/bootstrap_ci.csv",
          dict(run="field_strength_t100_w5 (v2_quickwins)", subset="all", metric="hit"),
          ["mean", "patient_lo", "patient_hi"], [.94, .8514790115375174, 1.])
    claim("Epoch-10 training-cohort robust judge 0 FS", ROOT / "reports/v2_quickwins/regrade/regrade_summary.csv",
          dict(set="diffusion_fs_t100_w5", judge_seed=0, subset="all"),
          ["hit"], [.07])
    for stage, name, expected in [
        ("regrade_old", "old_fs_t100_w5", [.09, .030295454545454552, .16091954022988506]),
        ("regrade_b3", "main_field_strength_t100_w5_s0", [.5, .326080850024426, .6634637850467289]),
        ("regrade_b3", "main_manufacturer_t300_w3_s0", [.11, .047619047619047616, .19101123595505617]),
    ]:
        claim(name + " held-out judge 0", B2 / stage / "summary.csv",
              dict(family="orientation_robust", seed=0, set=name, variant="cf", subset="all"),
              ["hit_mean", "hit_patient_lo", "hit_patient_hi"], expected)
    claim("Real foreground HF difference", HEADER / "hf_fs_differences.csv", dict(stratum="all"),
          ["hf_fg_diff_3.0T_minus_1.5T", "hf_fg_diff_patient_lo", "hf_fg_diff_patient_hi"],
          [10.397984160860446, 7.521256944726557, 13.357848329847828])
    claim("Original whole-image HF difference", HEADER / "hf_fs_differences.csv", dict(stratum="original"),
          ["hf_diff_3.0T_minus_1.5T", "hf_diff_patient_lo", "hf_diff_patient_hi"],
          [.9951914602170255, -.24090383927115827, 2.285586617432053])
    claim("Epoch-110 foreground edit/null HF to 3T", HEADER / "edit_hf_regions_summary.csv",
          dict(direction="to 3.0T", metric="ratio_fg_cf_over_null"),
          ["mean", "patient_lo", "patient_hi"], [2.880574151685552, 2.4338288274824182, 3.2739538851459415])
    claim("Epoch-110 foreground edit/null HF to 1.5T", HEADER / "edit_hf_regions_summary.csv",
          dict(direction="to 1.5T", metric="ratio_fg_cf_over_null"),
          ["mean", "patient_lo", "patient_hi"], [.42806250602647294, .3849798774893335, .4620489174638439])
    for vendor in ("GE", "Siemens"):
        claim("Training exact EchoTime overlap " + vendor, HEADER / "echo_time_overlap.csv",
              dict(population="original train (diffusion training data)", manufacturer=vendor),
              ["shared_exact_TE_values"], [0])
    # Independently recompute frozen hit means from the saved per-slice grading.
    checks = []
    for stage in ("regrade_old", "regrade_b3"):
        summary = rows(remember(B2 / stage / "summary.csv"))
        per_slice = rows(remember(B2 / stage / "per_slice.csv"))
        for summary_row in summary:
            if summary_row["family"] != "orientation_robust" or summary_row["subset"] != "all":
                continue
            fields = ("family", "seed", "set", "axis", "variant")
            group = [r for r in per_slice if all(r[k] == summary_row[k] for k in fields)]
            assert len(group) == int(summary_row["n"])
            hits = [float(r["hit"]) for r in group]
            assert all(v in (0, 1) for v in hits)
            assert len({(r["test_index"], r["patient_id"]) for r in group}) == len(group)
            mean = sum(hits) / len(hits)
            assert abs(mean - float(summary_row["hit_mean"])) < 1e-12
            checks.append(dict(stage=stage, **{k: summary_row[k] for k in fields}, n=len(group), hit_mean=mean))
    # Every protocol separator quoted in the new prose must exist in the original-cohort audit.
    for field, stratum in [("acq_phase", "all"), ("ScanOptions", "all"),
                           ("ReceiveCoilName", "GE only"), ("ManufacturerModelName", "Siemens only")]:
        matches = [r for r in rows(remember(HEADER / "separability.csv"))
                   if r["population"] == "original" and r["target"] == "field_strength"
                   and r["field"] == field and r["stratum"] == stratum]
        assert len(matches) == 1 and float(matches[0]["lookup_lopo_balanced_acc"]) == 1., (field, matches)
    counts = {}
    patient_sets = {}
    for split in ("train", "val", "test"):
        table = rows(remember(ROOT / f"data/processed/ax_t1_128/splits/{split}.csv"))
        patients = {r["patient_id"] for r in table}
        patient_sets[split] = patients
        counts[split] = dict(slices=len(table), patients=len(patients))
    assert counts == {"train": {"slices": 4434, "patients": 72}, "val": {"slices": 987, "patients": 15},
                      "test": {"slices": 934, "patients": 16}}
    assert all(not patient_sets[a] & patient_sets[b] for a, b in (("train", "val"), ("train", "test"), ("val", "test")))
    # Historical numerical ledgers, result summaries and immutable qualitative provenance.
    for path in [AN / "summary_ci.csv", AN / "soundness_gaps_ci.csv", AN / "hf_energy_ci.csv",
                 ROOT / "reports/v4_stats_heldout_20261016_01/paired_tests_heldout.csv",
                 ROOT / "reports/v4_diffusion/b1_20261016_01/remote/full/research_complete.json",
                 B2 / "frozen_checkpoints.json", HEADER / "summary.json",
                 PAPER / "numbers.md", PAPER / "numbers_v4.md",
                 BASE / "research-paper-v10-judges/verification.json",
                 ROOT / "reports/v10_qualitative_judges_20261116/figure_values.json"]:
        remember(path)
    for path in [ROOT / "checkpoints/diffusion/ax_t1_128/latest.pt",
                 ROOT / "checkpoints/diffusion/ax_t1_128_v4/b1_20261016_01/selected_epoch_0110.pt",
                 ROOT / "checkpoints/predictor/ax_t1_128/best.pt",
                 *[B2 / f"training/orientation_robust_seed{s}/judge_heldout_resnet18_seed{s}.pt" for s in range(3)]]:
        remember(path)
    old_tex = (BASE / "research-paper-v10-judges/paper_v10_ieee.tex").read_text(encoding="utf-8")
    new_tex = (PAPER / "paper_v11_ieee.tex").read_text(encoding="utf-8")
    for environment in ("table", "table*", "algorithm"):
        pattern = r"\\begin\{" + re.escape(environment) + r"\}.*?\\end\{" + re.escape(environment) + r"\}"
        assert re.findall(pattern, old_tex, re.S) == re.findall(pattern, new_tex, re.S), environment
    for pattern in [r"\\author\{.*?\\maketitle", r"\\section\*\{Acknowledgment\}.*?\\balance",
                    r"\\begin\{thebibliography\}.*?\\end\{thebibliography\}"]:
        assert re.findall(pattern, old_tex, re.S) == re.findall(pattern, new_tex, re.S)
    for path in (PAPER / "figures").iterdir():
        assert sha(path) == sha(BASE / "research-paper-v10-judges/figures" / path.name)
        remember(path)
    environment = {"python": sys.version, "executable": sys.executable, "platform": platform.platform(), "packages": {}}
    names = ["numpy", "scipy", "pandas", "scikit-learn", "torch", "torchvision", "pydicom", "scikit-image",
             "opencv-python", "matplotlib", "Pillow", "PyYAML", "fastapi", "uvicorn", "httpx",
             "python-multipart", "PyMuPDF", "pytest", "tqdm", "pydantic", "modal"]
    for name in names:
        environment["packages"][name] = metadata.version(name)
    with (out / "environment-observed.txt").open("x", encoding="utf-8") as f:
        f.write("# Observed historical runtime; not a dependency-compatible clean-install lock.\n")
        f.write("\n".join(f"{n}=={v}" for n, v in environment["packages"].items()) + "\n")
    result = subprocess.run([sys.executable, "-m", "pip", "check"], text=True, capture_output=True)
    (out / "pip-check-observed.txt").write_text(result.stdout + result.stderr, encoding="utf-8")
    environment["pip_check_exit_code"] = result.returncode
    write_json(out / "environment.json", environment)
    write_json(out / "claims.json", ledger)
    write_json(out / "dataset_summary.json", counts)
    write_json(out / "saved_hit_mean_checks.json", checks)
    write_json(out / "inputs.json", inputs)
    write_json(out / "complete.json", dict(headline_checks=len(ledger), saved_hit_groups=len(checks),
               input_hashes=len(inputs), tables_algorithms_authors_acknowledgment_references_preserved=True,
               figures_byte_identical=True, research_retrained=False, clean_install_validated=False))
    (out / "README.md").write_text(
        "# v11 local submission handoff checks\n\nGenerated by `scripts/v11_submission/prepare_handoff.py`.\n\n"
        "- `claims.json`: exact row selectors and values for corrected headlines/header claims.\n"
        "- `dataset_summary.json`: split counts, checked for patient disjointness.\n"
        "- `saved_hit_mean_checks.json`: held-out hit means recomputed from frozen per-slice grading.\n"
        "- `inputs.json`: SHA-256 for models, splits, result sources and unchanged figure assets.\n"
        "- `environment.json`, `environment-observed.txt`: the actual local package versions. This is a historical snapshot, not a compatible installation lock.\n"
        "- `pip-check-observed.txt`: actual dependency conflicts, retained honestly.\n"
        "- `complete.json`: verification coverage and limits. No retraining, new efficacy result, clean-install claim, external review or submission.\n"
        "API/replay and PDF checks are written separately by `verify_release.py`.\n", encoding="utf-8")
    print(json.dumps(json.loads((out / "complete.json").read_text()), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args().output.resolve())
