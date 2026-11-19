"""Build a source-only, hash-checked Colab check; never uploads or launches it."""
import argparse
import ast
import hashlib
import json
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]


def cell(kind,text):
    result = dict(cell_type=kind,metadata={},source=text.splitlines(keepends=True))
    if kind == 'code':
        result.update(execution_count=None,outputs=[])
    return result


def build(report_id):
    if not report_id.replace('_','').isalnum():
        raise ValueError('Safe fresh report ID required')
    out = ROOT/'reports/v12_colab'/report_id
    out.mkdir(parents=True,exist_ok=False)
    p = json.loads((ROOT/'reports/v10_detail/preparation_20261117_02/protocol.json').read_text())
    parent = ROOT/p['parent']
    if hashlib.sha256(parent.read_bytes()).hexdigest() != p['parent_sha256']:
        raise ValueError('Parent identity changed')
    ck = torch.load(parent,map_location='cpu',weights_only=False)
    config = dict(format='colab_synthetic_compatibility_v1',model_config={k:ck['config'][k] for k in ('model','condition','diffusion')},
                  objective_settings={k:p['training'][k] for k in ('structure_proxy_max_timestep','structure_proxy_weight')},
                  config_source_parent_sha256=p['parent_sha256'],parent_weights_included=False)
    del ck
    old = ROOT/'scripts/v8_generator_dev/adapter_pilot.py'
    text = old.read_text(encoding='utf-8-sig')
    fn = next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name == 'objective')
    snapshot = ('"""Exact objective function snapshot for the synthetic compatibility check."""\n'
                'import torch\nimport torch.nn.functional as F\n'
                'from scripts.v8_generator_dev.soft_structure import soft_structure\n\n'+ast.get_source_segment(text,fn)+'\n')
    # objective_snapshot is generated in the bundle only, never injected into historical source.
    payload = {'compatibility_config.json':json.dumps(config,indent=2).encode(),
               'scripts/v12_colab/objective_snapshot.py':snapshot.encode()}
    pending = ['scripts/v12_colab/runtime_check.py']
    while pending:
        name = pending.pop()
        if name in payload:
            continue
        data = (ROOT/name).read_bytes();payload[name] = data
        tree = ast.parse(data.decode('utf-8-sig'))
        for node in ast.walk(tree):
            modules = ([node.module] if isinstance(node,ast.ImportFrom) and node.module else
                       [a.name for a in node.names] if isinstance(node,ast.Import) else [])
            for module in modules:
                if module.split('.')[0] not in ('scripts','src'):
                    continue
                dependency = module.replace('.','/')+'.py'
                if (ROOT/dependency).is_file():
                    pending.append(dependency)
    for name in list(payload):
        for parent_dir in Path(name).parents:
            if parent_dir.as_posix() != '.':
                init = parent_dir.as_posix()+'/__init__.py'
                # Empty namespace initialisers keep package imports self-contained.
                payload.setdefault(init,b'')
    manifest = {k:hashlib.sha256(v).hexdigest() for k,v in sorted(payload.items())}
    payload['bundle_manifest.json'] = json.dumps(manifest,indent=2).encode()
    archive = out/'colab_compatibility.zip'
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
        for name,data in sorted(payload.items()):
            z.writestr(name,data)
    archive_hash = hashlib.sha256(archive.read_bytes()).hexdigest()
    notebook = dict(nbformat=4,nbformat_minor=5,metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3'),
                       colab=dict(name='Generator T4 compatibility.ipynb'),accelerator='GPU'),cells=[
        cell('markdown','# Generator GPU compatibility check\n\nUpload the accompanying source-only ZIP below. This checks the existing model architecture using synthetic images and freshly initialised weights: two guide modes, two disposable updates each, batch 32, all five guide states. No MRI data or trained checkpoint is included. No packages are installed, no Drive access is requested, and no full training starts. A pass does not validate efficacy, interrupted resume, or equivalence to the old software stack. Download the generated report ZIP and return it for review.\n'),
        cell('code','import sys, torch\nprint("Python:", sys.version)\nprint("PyTorch:", torch.__version__, "CUDA:", torch.version.cuda)\nassert torch.cuda.is_available(), "Select a GPU runtime first"\nprint("GPU:", torch.cuda.get_device_name(0))\n'),
        cell('code',f'''from google.colab import files
from pathlib import Path, PurePosixPath
import hashlib, json, zipfile, os
uploaded = files.upload()
assert len(uploaded) == 1, "Upload only colab_compatibility.zip"
archive_bytes = next(iter(uploaded.values()))
assert hashlib.sha256(archive_bytes).hexdigest() == {archive_hash!r}, "Unexpected ZIP: do not execute"
import io, tempfile
project = Path(tempfile.mkdtemp(prefix="generator_compatibility_", dir="/content"))
with zipfile.ZipFile(io.BytesIO(archive_bytes)) as z:
    names = z.namelist()
    assert len(names) == len(set(names)), "Duplicate archive entries"
    manifest = json.loads(z.read("bundle_manifest.json"))
    assert set(names) == set(manifest) | {{"bundle_manifest.json"}}
    for name, expected in manifest.items():
        p = PurePosixPath(name)
        assert not p.is_absolute() and ".." not in p.parts and "\\\\" not in name and ":" not in name
        data = z.read(name)
        assert hashlib.sha256(data).hexdigest() == expected, name
        target = project / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    (project / "bundle_manifest.json").write_bytes(z.read("bundle_manifest.json"))
print("Verified source-only package:", project)
'''),
        cell('code','''import subprocess, shutil, datetime
run_id = "compatibility_" + datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
output = Path("/content") / run_id
log = Path("/content") / (run_id + ".log")
with log.open("x") as handle:
    try:
        completed = subprocess.run([sys.executable, "-u", "-m", "scripts.v12_colab.runtime_check",
            "--output", str(output)], cwd=project, stdout=handle, stderr=subprocess.STDOUT, timeout=600)
        exit_code = completed.returncode
    except subprocess.TimeoutExpired:
        exit_code = "timeout"
print(log.read_text()[-16000:])
output.mkdir(exist_ok=True)
shutil.copyfile(log, output / "process.log")
(output / "process_exit.json").write_text(json.dumps({"exit_code": exit_code}))
report_zip = shutil.make_archive(str(output), "zip", output)
files.download(report_zip)
print("Compatibility check finished. Exit:", exit_code)
print("Return the downloaded report ZIP. Full training has not started.")
''')])
    (out/'Generator_T4_compatibility.ipynb').write_text(json.dumps(notebook,indent=2),encoding='utf-8')
    (out/'manifest.json').write_text(json.dumps(dict(zip_sha256=archive_hash,zip_bytes=archive.stat().st_size,
        files=manifest,objective_source_sha256=hashlib.sha256(old.read_bytes()).hexdigest(),
        objective_function_exact_snapshot=True,parent_weights_included=False,medical_images_included=False,
        full_training_ready=False,cloud_started=False),indent=2))
    (out/'README.md').write_text('Producer: python -m scripts.v12_colab.build_compatibility --report-id '+report_id+
        '\n\nGenerator_T4_compatibility.ipynb: upload to Colab, then run its cells in order. '
        'colab_compatibility.zip: source-only dependency closure and architecture config (no data/weights). '
        'manifest.json: exact source hashes and ZIP identity. '
        'offline_import.txt/json: isolated extracted-package import and integrity checks. '
        'Local synthetic execution, if run, has its own new report folder. '
        'Colab-generated reports must be downloaded: this notebook uses ephemeral VM storage and has no training/resume workflow.\n',encoding='utf-8')
    with tempfile.TemporaryDirectory(prefix='colab_check_') as temporary:
        with zipfile.ZipFile(archive) as z:
            z.extractall(temporary)
        checked = subprocess.run([sys.executable,'-c',
            'from scripts.v12_colab.runtime_check import verify_bundle; print(verify_bundle()); print("ISOLATED IMPORT PASS")'],
            cwd=temporary,capture_output=True,text=True,timeout=90)
        (out/'offline_import.txt').write_text(checked.stdout+checked.stderr,encoding='utf-8')
        (out/'offline_import.json').write_text(json.dumps(dict(exit_code=checked.returncode,isolated_extracted_package=True)))
        if checked.returncode:
            raise RuntimeError('Isolated package import failed; see offline_import.txt')
    print(out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--report-id',required=True)
    build(parser.parse_args().report_id)
