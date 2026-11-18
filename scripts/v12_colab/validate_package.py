"""Validate notebook syntax, extracted integrity rejection and disposable local updates."""
import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def validate(package,report_id):
    package = Path(package).resolve()
    out = ROOT/'reports/v12_colab'/report_id
    out.mkdir(parents=True,exist_ok=False)
    (out/'README.md').write_text('Producer: scripts.v12_colab.validate_package. inputs.json: package hashes; '
        'checks.json: notebook syntax, isolated import and modified-source rejection; '
        'synthetic/: actual local batch-five disposable updates; execution.log: full subprocess output; '
        'complete.json: check outcome. This is not the Colab T4/batch-32 result or training readiness.\n')
    nb = json.loads((package/'Generator_T4_compatibility.ipynb').read_text())
    code_cells = [c for c in nb['cells'] if c['cell_type']=='code']
    for i,c in enumerate(code_cells):
        compile(''.join(c['source']),f'notebook_cell_{i}','exec')
    archive = package/'colab_compatibility.zip'
    bound = json.loads((package/'manifest.json').read_text())
    if hashlib.sha256(archive.read_bytes()).hexdigest()!=bound['zip_sha256']:
        raise ValueError('ZIP hash mismatch')
    (out/'inputs.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [archive,package/'Generator_T4_compatibility.ipynb',package/'manifest.json']},indent=2))
    with tempfile.TemporaryDirectory(prefix='validate_colab_') as temporary:
        with zipfile.ZipFile(archive) as z:
            z.extractall(temporary)
        command = [sys.executable,'-c','from scripts.v12_colab.runtime_check import verify_bundle; verify_bundle()']
        good = subprocess.run(command,cwd=temporary,capture_output=True,text=True,timeout=90)
        if good.returncode:
            raise RuntimeError(good.stderr)
        target = Path(temporary)/'compatibility_config.json'
        original = target.read_bytes()
        target.write_bytes(original+b' ')
        bad = subprocess.run(command,cwd=temporary,capture_output=True,text=True,timeout=90)
        if not bad.returncode or 'Bundle identity failed' not in bad.stderr:
            raise ValueError('Modified package was not explicitly rejected')
        target.write_bytes(original)
        (out/'checks.json').write_text(json.dumps(dict(notebook_code_cells_compiled=len(code_cells),
            exact_zip_hash=True,isolated_import=True,modified_source_rejected=True),indent=2))
        with (out/'execution.log').open('x') as log:
            done = subprocess.run([sys.executable,'-u','-m','scripts.v12_colab.runtime_check',
                '--output',str(out/'synthetic'),'--batch-size','5','--device','cuda'],
                cwd=temporary,stdout=log,stderr=subprocess.STDOUT,timeout=600)
        (out/'complete.json').write_text(json.dumps(dict(complete=done.returncode==0,
            local_batch_size=5,colab_runtime_validated=False,intended_batch_32_validated=False,
            full_training_ready=False,cloud_started=False,exit_code=done.returncode),indent=2))
        if done.returncode:
            raise RuntimeError('Synthetic check failed; inspect execution.log')
    print(out,flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--package',required=True)
    parser.add_argument('--report-id',required=True)
    args = parser.parse_args()
    validate(args.package,args.report_id)
