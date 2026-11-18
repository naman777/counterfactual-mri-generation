"""Use a new canonical-frame checkpoint on a known validation source, offline."""
import argparse
import json
import math
import re

import numpy as np
import torch

from scripts.v7_geometry.orientation import invert_orientation
from scripts.v7_geometry.support import request_support, require_research_support
from scripts.v8_generator_dev.data import CanonicalDataset
from scripts.v8_generator_dev.evaluate_full import predict_judge, source_native
from scripts.v8_generator_dev.full_train import deterministic_runtime, load_ema
from scripts.v8_generator_dev.io import ROOT, digest, json_write, project_path, verify_inputs
from scripts.v8_generator_dev.train_smoke import sample_wiring


def edit(checkpoint, index, axis, timestep, guidance, seed, run_id, device='cuda', allow_unverified=False):
    from scripts.v2_quickwins.judge_model import load_judge
    from scripts.v5_anatomy.metrics import diagnostics
    from skimage.metrics import structural_similarity

    if not re.fullmatch('[A-Za-z0-9_-]+', run_id) or axis not in {'manufacturer', 'field_strength'}:
        raise ValueError('Safe new output ID and one supported edit axis required')
    if not math.isfinite(guidance) or guidance < 0:
        raise ValueError('Guidance must be finite and nonnegative')
    deterministic_runtime()
    checkpoint = project_path(checkpoint)
    c, model, scheduler, ck = load_ema(checkpoint, device)
    if ck['engineering']:
        raise ValueError('A wiring-smoke checkpoint is not a trained editing checkpoint')
    if not 0 <= timestep < scheduler.num_train_timesteps:
        raise ValueError('Invalid editing timestep')
    train, val = [CanonicalDataset(ck['config']['view'], split) for split in ('train', 'val')]
    if not 0 <= index < len(val):
        raise ValueError('Select an existing validation index')
    item, row = val[index], val.frame.iloc[index]
    mfg, fs = item['manufacturer_label'], item['field_strength_label']
    target_m, target_f = (1 - mfg, fs) if axis == 'manufacturer' else (mfg, 1 - fs)
    echo = float(item['echo_time_norm'])
    support = request_support(train.frame, target_m, target_f, echo, ck['config']['data']['minimum_cell_patients'])
    require_research_support(support, allow_unsupported=allow_unverified)
    output = ROOT / 'reports/v8_generator_dev' / run_id
    if output.exists():
        raise FileExistsError('Use a new edit output ID')
    output.mkdir(parents=True)
    inputs = {checkpoint.relative_to(ROOT).as_posix(): digest(checkpoint)}
    for split in ('train', 'val'):
        path = project_path(ck['config']['view']) / 'splits' / f'{split}.csv'
        inputs[path.relative_to(ROOT).as_posix()] = digest(path)
    source_path = project_path(row.source_image_path)
    inputs[source_path.relative_to(ROOT).as_posix()] = row.source_array_sha256
    source = val.image_01(index)
    image = item['image'].unsqueeze(0).to(device)
    arrays = {'source': source}
    for name, m, f in [('null', mfg, fs), ('target', target_m, target_f)]:
        arrays[name] = sample_wiring(c, model, scheduler, image,
            torch.tensor([m], device=device), torch.tensor([f], device=device),
            torch.tensor([echo], device=device), seed, timestep, guidance)[0, 0]
    arrays['abs_target_source'] = np.abs(arrays['target'] - source)
    arrays['abs_target_null'] = np.abs(arrays['target'] - arrays['null'])
    _, arrays['local_ssim_source_target'] = structural_similarity(source, arrays['target'],
                                                                 data_range=1, win_size=7, full=True)
    for name in ('source', 'null', 'target'):
        arrays[f'{name}_native'] = invert_orientation(arrays[name], val.plans[index])
    if not np.array_equal(arrays['source_native'], np.load(source_path, allow_pickle=False)):
        raise AssertionError('Source-native round trip changed the source')
    with (output / 'edit_arrays.npz').open('xb') as handle:
        np.savez_compressed(handle, **arrays)
    histograms = {}
    for name in ('source', 'null', 'target'):
        counts, edges = np.histogram(arrays[name], bins=128, range=(0, 1))
        histograms[name] = counts.tolist()
    json_write(output / 'histograms.json', dict(bin_edges=edges.tolist(), counts=histograms))
    metrics = {name: diagnostics(source, arrays[name]) for name in ('null', 'target')}
    target_label = target_m if axis == 'manufacturer' else target_f
    key, other = ('mfg', 'fs') if axis == 'manufacturer' else ('fs', 'mfg')
    native = np.stack([arrays[f'{name}_native'] for name in ('source', 'null', 'target')])
    judges = {}
    for judge_seed, value in enumerate(ck['config']['evaluation']['judge_paths']):
        path = project_path(value)
        inputs[path.relative_to(ROOT).as_posix()] = digest(path)
        judge = load_judge(path, device).eval().requires_grad_(False)
        probability = predict_judge(judge, native, device)
        judges[f'heldout_orientation_robust_seed{judge_seed}'] = dict(
            target_probabilities={name: float(probability[key][i, target_label])
                                  for i, name in enumerate(('source', 'null', 'target'))},
            target_hit=bool(probability[key][2].argmax() == target_label),
            null_target_hit=bool(probability[key][1].argmax() == target_label),
            other_axis_stable=bool(probability[other][2].argmax() == probability[other][0].argmax()),
            mfg_probabilities=probability['mfg'].tolist(), fs_probabilities=probability['fs'].tolist(),
        )
        del judge
    json_write(output / 'source_record.json', dict(patient_id=item['patient_id'], sop_uid=row.sop_uid,
                                                   dataset_index=index, split='val',
                                                   source_path=row.source_image_path,
                                                   source_sha256=row.source_array_sha256,
                                                   orientation_plan=val.plans[index]))
    json_write(output / 'protocol.json', dict(
        checkpoint=str(checkpoint.relative_to(ROOT)), checkpoint_sha256=digest(checkpoint),
        model_variant=ck['variant'], training_progress=ck['progress'],
        full_training_completion_not_inferred_from_checkpoint=True,
        source_condition=dict(manufacturer=mfg, field_strength=fs, echo_time_norm=echo),
        target_condition=dict(manufacturer=target_m, field_strength=target_f, echo_time_norm=echo),
        axis=axis, timestep=timestep, guidance=guidance, seed=seed, device=device,
        target_null_noise_matched=True, support=support,
        research_override=allow_unverified, scope='Individual local development example, distinct from frozen cloud aggregate evaluation',
    ))
    json_write(output / 'metrics.json', dict(correspondence_and_proxies=metrics, named_judges=judges,
                                            anatomy_validated=False, clinical_preservation='unknown'))
    for value in ('scripts/v8_generator_dev/edit_full.py', 'scripts/v8_generator_dev/full_train.py',
                  'scripts/v8_generator_dev/model.py', 'scripts/v8_generator_dev/data.py',
                  'scripts/v8_generator_dev/train_smoke.py', 'scripts/v5_anatomy/metrics.py'):
        inputs[value] = digest(ROOT / value)
    verify_inputs(inputs)
    json_write(output / 'inputs.json', inputs)
    json_write(output / 'complete.json', dict(complete=True, source_native_roundtrip_exact=True,
                                              source_echo_time_fixed=True, named_judges=len(judges),
                                              test_pixels_used=False, anatomy_validated=False,
                                              aggregate_efficacy_established=False))
    (output / 'README.md').write_text(
        '# Individual offline canonical-generator edit\n\nProducer: `python -m scripts.v8_generator_dev.edit_full`.\n'
        'protocol.json fixes checkpoint/source/target/settings/support flags. source_record.json retains '
        'source provenance and exact inverse geometry. edit_arrays.npz contains canonical/native sources, '
        'matched null/target arrays, absolute differences and the local SSIM map. histograms.json gives '
        'all histogram bins/counts; metrics.json gives correspondence/proxies and each frozen judge. '
        'inputs.json verifies unchanged source/model/code; complete.json records software scope. '
        'A single example is not a new test cohort or independent anatomical/clinical validation.\n', encoding='utf-8')
    print('LOCAL EDIT COMPLETE', output, flush=True)
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--index', type=int, required=True)
    parser.add_argument('--axis', choices=['manufacturer', 'field_strength'], required=True)
    parser.add_argument('--timestep', type=int, required=True)
    parser.add_argument('--guidance', type=float, default=3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--allow-unverified-condition', action='store_true')
    args = parser.parse_args()
    edit(args.checkpoint, args.index, args.axis, args.timestep, args.guidance,
         args.seed, args.run_id, args.device, args.allow_unverified_condition)
