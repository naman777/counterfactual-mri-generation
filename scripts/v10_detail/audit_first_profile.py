"""Read-only early audit of one completed evaluation; never a study conclusion."""
import argparse
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import torch

from scripts.v2_quickwins.judge_model import load_judge
from scripts.v8_generator_dev.data import CanonicalDataset
from scripts.v8_generator_dev.evaluate_full import source_native, predict_judge
from scripts.v8_generator_dev.full_train import deterministic_runtime
from scripts.v8_generator_dev.grading_io import read_grading_csv
from scripts.v8_generator_dev.io import ROOT, digest, json_write, project_path, verify_inputs
from scripts.v10_detail.metrics import measure
from scripts.v10_detail.verify_saved_v2 import metric_agrees


@torch.inference_mode()
def audit(prepared, run_id, report_id):
    if not all(s.replace('_', '').isalnum() for s in (run_id, report_id)):
        raise ValueError('Safe identifiers required')
    prepared = project_path(prepared)
    protocol = json.loads((prepared / 'protocol.json').read_text())
    seed = protocol['training_seeds'][0]
    profile = f'seed{seed}__coarse_control__manufacturer__correct'
    relative = f'reports/v10_detail/{run_id}_eval_{profile}'
    base = f'v10_detail/{run_id}'
    os.environ['MODAL_PROFILE'] = 'namankundra7'
    import modal
    volume = modal.Volume.from_name('capstone-judge-data', create_if_missing=False)
    def read(name):
        return b''.join(volume.read_file(name))
    phase = json.loads(read(base + '/eval_' + profile + '.json'))
    receipt = json.loads(read(base + '/project/' + relative + '/complete.json'))
    if phase['exit_code'] or not receipt['complete'] or receipt['engineering'] or receipt['cases'] != 45:
        raise ValueError('Completed full prospective profile required')
    output = ROOT / 'reports/v10_detail' / report_id
    output.mkdir(parents=True, exist_ok=False)
    (output / 'README.md').write_text(
        'Producer: scripts.v10_detail.audit_first_profile. phase_receipt.json: original phase receipt; '
        'manifest.json: immutable retrieved completed first-profile files; inputs.json: audit dependencies; '
        'complete.json: source, image-metric and three named-judge checks for this profile only. '
        'No study completion, comparative efficacy, advancement or clinical preservation is asserted.\n', encoding='utf-8')
    json_write(output / 'phase_receipt.json', phase)
    entries = volume.listdir(base + '/project/' + relative, recursive=True)
    def retrieve(entry):
        name = entry.path.lstrip('/')
        if '.' not in name.rsplit('/', 1)[-1]:
            return None
        dest = project_path(name.removeprefix(base + '/project/'))
        data = read(name)
        sha = hashlib.sha256(data).hexdigest()
        if dest.exists():
            if digest(dest) != sha:
                raise ValueError('Immutable evaluation evidence collision')
        else:
            dest.parent.mkdir(parents=True, exist_ok=True)
            with dest.open('xb') as f:
                f.write(data)
        return dict(local=dest.relative_to(ROOT).as_posix(), sha256=sha, bytes=len(data))
    with ThreadPoolExecutor(max_workers=8) as pool:
        files = [r for r in pool.map(retrieve, entries) if r is not None]
    home = project_path(relative)
    meta = json.loads((home / 'protocol.json').read_text())
    if meta['scientific_protocol'] != protocol:
        raise ValueError('Scientific protocol drift')
    verify_inputs(json.loads((home / 'inputs.json').read_text()))
    requests = pd.read_csv(home / 'requests.csv')
    table = read_grading_csv(home / 'per_image.csv')
    selected = pd.read_csv(prepared / 'validation_selection.csv').dataset_index.astype(int)
    generations = [protocol['evaluation']['selection_generation_seed'],
                   *protocol['evaluation']['confirmation_generation_seeds']]
    if set(zip(requests.dataset_index, requests.generation_seed)) != {(i, s) for i in selected for s in generations}:
        raise ValueError('Missing source/generation cell')
    if len(requests) != 45 or len(table) != 405:
        raise ValueError('Incomplete profile')
    deterministic_runtime()
    val = CanonicalDataset(protocol['view'], 'val')
    judges = {f'heldout_orientation_robust_seed{j}': load_judge(project_path(path), 'cuda').eval()
              for j, path in enumerate(protocol['evaluation']['judge_paths'])}
    max_error = 0.
    grades = 0
    for r in requests.itertuples():
        source = val.image_01(r.dataset_index)
        path = home / r.array_file
        if digest(path) != r.array_sha256:
            raise ValueError('Saved-array hash mismatch')
        with np.load(path, allow_pickle=False) as z:
            np.testing.assert_array_equal(z['source'], source)
            if float(z['source_echo_time_norm']) != float(z['target_echo_time_norm']):
                raise ValueError('EchoTime edited')
            images = [z[k].copy() for k in ('source', 'null', 'target')]
        native = source_native(np.stack(images), val.plans[r.dataset_index])
        target = 1 - val[r.dataset_index]['manufacturer_label']
        metrics = [measure(source, image) for image in images]
        for name, judge in judges.items():
            prob = predict_judge(judge, native, 'cuda')
            null_hit = int(prob['mfg'][1].argmax() == target)
            for j, variant in enumerate(('identity', 'null', 'target')):
                rows = table[(table.case == r.case) & (table.judge == name) & (table.variant == variant)]
                if len(rows) != 1:
                    raise ValueError('Missing/duplicate named-judge cell')
                saved = rows.iloc[0]
                err = abs(float(prob['mfg'][j, target]) - saved.target_probability)
                max_error = max(max_error, err)
                hit = int(prob['mfg'][j].argmax() == target)
                if err > 2e-6 or hit != saved.target_hit or hit - null_hit != saved.paired_hit_delta:
                    raise ValueError('Named-judge probability/decision drift')
                if int(prob['fs'][j].argmax() == prob['fs'][0].argmax()) != saved.other_axis_stable:
                    raise ValueError('Other-axis decision drift')
                if not all(metric_agrees(v, saved[k]) for k, v in metrics[j].items()):
                    raise ValueError('Image-metric or missingness drift')
                grades += 1
    json_write(output / 'manifest.json', files)
    evidence = {r['local']: r['sha256'] for r in files}
    for path in (project_path(__file__), ROOT / 'scripts/v10_detail/verify_saved_v2.py', prepared / 'protocol.json'):
        evidence[path.relative_to(ROOT).as_posix()] = digest(path)
    json_write(output / 'inputs.json', evidence)
    json_write(output / 'complete.json', dict(complete=True, profile_verified=True, cases=45, grades=grades,
        max_probability_error=max_error, workflow_completion_not_asserted=True, comparative_efficacy_not_assessed=True,
        clinical_preservation='unknown', paid_gpu_calls=0))
    print('FIRST PROFILE VERIFIED', grades, 'grades; full study still pending', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(__doc__)
    for name in ('prepared', 'run-id', 'report-id'):
        ap.add_argument('--' + name, required=True)
    a = ap.parse_args()
    audit(a.prepared, a.run_id, a.report_id)
