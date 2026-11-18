"""Append-only downloads without importing Modal app decorators on Windows.

python -m scripts.v4_convergence.fetch --run-id b1_20261016_01
Safe to use during training: checkpoint files require a completed block marker.
"""
import argparse
import json
import shutil
from pathlib import Path
from uuid import uuid4

import modal

ROOT = Path(__file__).resolve().parents[2]


def fetch(run_id):
    if not run_id or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for c in run_id):
        raise ValueError("Invalid run id")
    volume = modal.Volume.from_name("capstone-judge-data", create_if_missing=False)
    base = ROOT / "reports/v4_diffusion" / run_id
    prefix = f"v4_convergence/{run_id}/"
    entries = list(volume.iterdir("/" + prefix, recursive=True))
    names = {e.path.lstrip("/") for e in entries}
    count = 0
    for entry in entries:
        name = entry.path.lstrip("/")
        if entry.type.name != "FILE" or "/inputs/" in name:
            continue
        relative = name.removeprefix(prefix)
        if relative == name or ".." in Path(relative).parts:
            raise ValueError("Unexpected remote path")
        if relative.endswith(".pt") and not any(p.startswith(name.rsplit("/", 1)[0] + "/block_") for p in names):
            continue
        target = base / "remote" / relative
        if target.exists():
            if target.stat().st_size != entry.size:
                raise ValueError(f"Immutable snapshot size differs: {target}. Inspect before a new download.")
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_name(target.name + ".download-" + uuid4().hex)
        with partial.open("xb") as f:
            for chunk in volume.read_file(entry.path):
                f.write(chunk)
        if partial.stat().st_size != entry.size:
            raise ValueError(f"Remote file changed while downloading; partial preserved: {partial}")
        partial.rename(target)
        count += 1
    ckptdir = ROOT / "checkpoints/diffusion/ax_t1_128_v4" / run_id
    for source in sorted((base / "remote/full/train").glob("to_*/epoch_*.pt")):
        dest = ckptdir / source.name
        ckptdir.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            with source.open("rb") as f, dest.open("xb") as g:
                shutil.copyfileobj(f, g, 1024 * 1024)
            print("Checkpoint:", dest)
    print("Fetched", count, "new artifacts to", base / "remote")
    return base / "remote"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--run-id", required=True)
    fetch(parser.parse_args().run_id)
