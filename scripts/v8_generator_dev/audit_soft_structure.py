"""Compare a differentiable local-order guide with the previous coarse gradients."""
import argparse
import json
import re

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from scripts.v6_structure.model import structure_maps
from scripts.v8_generator_dev.data import CanonicalDataset
from scripts.v8_generator_dev.guide_audit import balanced_score, confidence
from scripts.v8_generator_dev.io import ROOT, csv_write, digest, json_write, project_path, verify_inputs
from scripts.v8_generator_dev.soft_structure import soft_structure


def audit(run_id):
    if not re.fullmatch('[A-Za-z0-9_-]+', run_id):
        raise ValueError('Safe new guide audit ID required')
    torch.set_num_threads(4)
    base = ROOT / 'reports/v8_generator_dev'
    selection_path = base / 'canonical_guide_audit_20261116_01/patient_selection.csv'
    frame = pd.read_csv(selection_path)
    view = 'data/processed/ax_t1_128_canonical/view_v1_20261116_01'
    datasets = {split: CanonicalDataset(view, split) for split in ('train', 'val')}
    output = base / run_id
    output.mkdir(exist_ok=False)
    json_write(output / 'protocol.json', dict(view=view, patient_selection=str(selection_path.relative_to(ROOT)),
        representations=['previous_coarse_gradients', 'soft_local_order'], feature_dimensions=192,
        probe='Train-standardised class-balanced ridge alpha=10, no tuning',
        guide_size=[3,32,32], source_raw_pixel_bypass=False, differentiable_candidate=True,
        interpretation='Development audit; neither weak probe nor contrast stability proves acquisition invariance or anatomical preservation',
        perturbations=['gamma .7', 'gamma 1.3', 'gain .8 plus .04', 'translation8px'],
        test_pixels_used=False, generator_weights_changed=False))
    inputs = {selection_path.relative_to(ROOT).as_posix(): digest(selection_path)}
    for name in ('audit_soft_structure.py', 'soft_structure.py', 'guide_audit.py'):
        path = ROOT / 'scripts/v8_generator_dev' / name
        inputs[path.relative_to(ROOT).as_posix()] = digest(path)
    inputs['scripts/v6_structure/model.py'] = digest(ROOT / 'scripts/v6_structure/model.py')
    features, maps, sensitivities = {}, {}, []
    for row in frame.itertuples():
        dataset = datasets[row.split]
        image = torch.from_numpy(dataset.image_01(row.dataset_index))[None,None]
        if dataset.frame.iloc[row.dataset_index].patient_id != row.patient_id:
            raise ValueError('Saved patient selection does not match canonical membership')
        path = project_path(row.source_image_path)
        inputs[path.relative_to(ROOT).as_posix()] = digest(path)
        for name, function in [('previous_coarse_gradients', structure_maps), ('soft_local_order', soft_structure)]:
            with torch.no_grad():
                values = function(image)
                photos = [function(image**gamma) for gamma in (.7,1.3)]
                affine = function(image*.8+.04)
                moved = function(torch.roll(image,(8,8),(-2,-1)))
            features.setdefault(name, []).append(F.adaptive_avg_pool2d(values,(8,8)).flatten().numpy())
            maps.setdefault(name, []).append(values[0].numpy())
            sensitivities.append(dict(patient_id=row.patient_id, split=row.split, representation=name,
                gamma07_l1=float((values-photos[0]).abs().mean()), gamma13_l1=float((values-photos[1]).abs().mean()),
                affine_l1=float((values-affine).abs().mean()), translation8_l1=float((values-moved).abs().mean()),
                anatomy_validated=False))
    train = frame.split.eq('train').to_numpy()
    val = ~train
    probes, predictions = [], []
    for name, records in features.items():
        values = np.stack(records)
        mean, std = values[train].mean(0), values[train].std(0).clip(1e-4)
        x = np.column_stack([np.ones(len(values)), (values-mean)/std]).astype(np.float64)
        for axis in ('manufacturer','field_strength'):
            y = frame[axis].to_numpy()
            counts = np.bincount(y[train],minlength=2)
            weights = 1/counts[y[train]]
            weights /= weights.mean()
            penalty = np.eye(x.shape[1])*10
            penalty[0,0] = 0
            coefficient = np.linalg.solve(x[train].T@(weights[:,None]*x[train])+penalty,
                                          x[train].T@(weights*y[train]))
            score = x[val]@coefficient
            prediction = (score>=.5).astype(int)
            lo, hi = confidence(y[val],prediction)
            probes.append(dict(representation=name,axis=axis,balanced_accuracy=balanced_score(y[val],prediction),
                ci95_lo=lo,ci95_hi=hi,train_patients=int(train.sum()),validation_patients=int(val.sum()),
                acquisition_invariance_established=False))
            predictions.extend(dict(representation=name,axis=axis,patient_id=patient,truth=int(t),prediction=int(p),score=float(s))
                for patient,t,p,s in zip(frame[val].patient_id,y[val],prediction,score))
    csv_write(output / 'leakage_probe.csv',pd.DataFrame(probes))
    csv_write(output / 'probe_predictions.csv',pd.DataFrame(predictions))
    csv_write(output / 'sensitivities.csv',pd.DataFrame(sensitivities))
    csv_write(output / 'patient_selection.csv',frame)
    with (output / 'guide_maps.npz').open('xb') as handle:
        np.savez_compressed(handle,**{name:np.stack(records) for name,records in maps.items()})
    for split in ('train','val'):
        for path in (ROOT / view / 'splits' / (split+'.csv'),ROOT / 'data/processed/ax_t1_128/splits' / (split+'.csv')):
            inputs[path.relative_to(ROOT).as_posix()] = digest(path)
    verify_inputs(inputs)
    json_write(output / 'inputs.json',inputs)
    json_write(output / 'complete.json',dict(complete=True,guide_audited=True,differentiable_candidate=True,
        acquisition_invariance_established=False,anatomy_validated=False,adapter_efficacy_established=False,
        test_pixels_used=False,backbone_trained=False))
    (output / 'README.md').write_text(
        '# Differentiable guide revision audit\n\nProducer: `python -m scripts.v8_generator_dev.audit_soft_structure`.\n'
        'protocol.json fixes sources/features/probes. sensitivities.csv reports photometric/spatial map response; '
        'leakage_probe.csv and probe_predictions.csv retain named fixed-probe CIs and every prediction. '
        'patient_selection.csv and guide_maps.npz retain sources/maps. inputs.json hashes original inputs/code; '
        'complete.json denies acquisition invariance, clinical preservation or adapter efficacy. '
        'A coarse local-order guide can retain patient/protocol/acquisition information even if photometric '
        'responses are small. No guide is promoted automatically.\n',encoding='utf-8')
    print('SOFT GUIDE AUDIT',pd.DataFrame(probes).to_string(index=False),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--run-id',required=True)
    audit(parser.parse_args().run_id)
