"""Package only the read-only inputs needed for a CPU/replay rehearsal.

python -m api.package_demo --output demo/20261017/mrone-demo.zip
Dependencies must be installed and the frontend built on the destination before
going offline. Diffusion checkpoints and raw DICOMs are deliberately unnecessary.
"""
import argparse
import json
import zipfile
from pathlib import Path

from api import data
from api.publication import PAPER_PATH, PAPER_RELATIVE_PATH


def collect_files():
    """Return code-root inputs; publication is added at its sibling path."""
    files = set()
    for folder in ("api", "src", "web-app/src", "web-app/public"):
        files.update(p for p in (data.ROOT / folder).rglob("*") if p.is_file() and "__pycache__" not in p.parts and "runtime" not in p.parts)
    for pattern in ("web-app/*.json", "web-app/*.yaml", "web-app/*.ts", "web-app/*.mjs", "web-app/*.md", "configs/*.yaml", "data/processed/ax_t1_128/label_maps/*.json"):
        files.update(data.ROOT.glob(pattern))
    for filename in ("start_demo.bat", "start_demo.ps1", "src/__init__.py", "requirements.txt", "AGENTS.md", "README.md", "REPRODUCIBILITY.md"):
        if (data.ROOT / filename).is_file():
            files.add(data.ROOT / filename)
    files.add(data.TEST)
    files.update(data.ROOT / s["path"] for s in data.slices())
    for folder in (data.B3, data.ANALYSIS, data.B2 / "regrade_b3", data.B2 / "regrade_old", data.ROOT / "reports/v4_stats_heldout_20261016_01"):
        files.update(p for p in folder.rglob("*") if p.suffix in {".csv", ".json", ".npz", ".md"})
    # Optional CPU robustness probes still work with the frozen classifier weights.
    files.add(data.ROOT / "checkpoints/predictor/ax_t1_128/best.pt")
    for seed in range(3):
        files.add(data.B2 / f"training/orientation_robust_seed{seed}/judge_heldout_resnet18_seed{seed}.pt")
    files.add(data.B2 / "training/legacy_comparison_seed0/judge_heldout_resnet18_seed0.pt")
    demo_assets = data.ROOT / "demo/20261017"
    files.update(p for p in demo_assets.rglob("*") if p.is_file() and p.suffix in {".mp4", ".jpg", ".md", ".srt", ".concat", ".py"})
    files.update(p for p in (data.ROOT / "environment").glob("*.txt"))
    return files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest-only", action="store_true", help="Write the exact archive plan without copying large saved arrays")
    args = parser.parse_args()
    files = collect_files()
    missing = [str(p) for p in files if not p.is_file()]
    if missing:
        raise FileNotFoundError("Missing demo inputs: " + ", ".join(missing))
    if not PAPER_PATH.is_file():
        raise FileNotFoundError(f"Current publication missing: {PAPER_PATH}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.manifest_only:
        plan = {"counterfactual-mri-generation/" + data.relative(p): p.stat().st_size for p in sorted(files)}
        plan[PAPER_RELATIVE_PATH.as_posix()] = PAPER_PATH.stat().st_size
        with args.output.open("x", encoding="utf-8") as f:
            json.dump(dict(files=plan, total_bytes=sum(plan.values()), archive_written=False), f, indent=2)
        print(f"Checked {len(plan)} planned entries; archive not written")
        return
    with zipfile.ZipFile(args.output, "x", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(files):
            archive.write(path, "counterfactual-mri-generation/" + data.relative(path))
        archive.write(PAPER_PATH, PAPER_RELATIVE_PATH.as_posix())
    print(f"Packaged {len(files)} input/code files in {args.output}")


if __name__ == "__main__":
    main()
