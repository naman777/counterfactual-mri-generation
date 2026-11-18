"""Small append-only provenance helpers without model/checkpoint imports."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def project_path(value, root=ROOT):
    root = Path(root).resolve()
    path = Path(str(value).replace('\\', '/'))
    path = (path if path.is_absolute() else root / path).resolve()
    if not path.is_relative_to(root):
        raise ValueError('Path must stay inside the project')
    return path


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def json_write(path, value):
    with Path(path).open('x', encoding='utf-8') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)


def csv_write(path, frame):
    with Path(path).open('x', encoding='utf-8', newline='') as handle:
        frame.to_csv(handle, index=False)


def verify_inputs(inputs, root=ROOT):
    for name, expected in inputs.items():
        if digest(project_path(name, root)) != expected:
            raise ValueError(f'Frozen input changed: {name}')
