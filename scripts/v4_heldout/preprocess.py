"""Preprocess the verified held-out cohort, preserving its frozen patient split.

python -m scripts.v4_heldout.preprocess --plan-run preflight_20261016_01
Only reads completed download receipts. Never modifies the original cohort.
"""
import argparse
import json
import shutil

import numpy as np
import pandas as pd
import pydicom

from .audit import ROOT, sha
from .download import valid_id
from src.data.preprocess import (
    compute_foreground_fraction, percentile_clip_and_normalize,
    read_dicom_image, resize_image,
)


def main(plan_run, download_attempt=None):
    valid_id(plan_run)
    preflight = ROOT / 'reports/v4_heldout_judge' / plan_run
    suffix = '' if download_attempt is None else '_' + valid_id(download_attempt)
    download = ROOT / 'reports/v4_heldout_judge' / f'download_{plan_run}{suffix}'
    receipt = json.loads((download / 'result.json').read_text(encoding='utf-8'))
    if not receipt['complete']:
        raise ValueError('Download did not complete and verify every header')
    plan_path = preflight / 'download_plan.csv'
    if sha(plan_path) != receipt['plan_sha256']:
        raise ValueError('Frozen plan changed')
    plan = pd.read_csv(plan_path)
    headers_path = download / 'verified_dicom_headers.csv'
    headers = pd.read_csv(headers_path)
    excluded = pd.read_csv(preflight / 'excluded_original_patients.csv')
    assert set(headers.patient_id) == set(plan.PatientID)
    assert not (set(headers.patient_id) & set(excluded.patient_id))
    assert headers.sop_uid.is_unique
    assert len(headers) == int(plan.instanceCount.sum())
    original = ROOT / 'data/processed/ax_t1_128'
    map_paths = list((original / 'label_maps').glob('*.json'))
    protected = map_paths + list((original / 'splits').glob('*.csv'))
    protected_hashes = {str(p): sha(p) for p in protected}
    maps = {p.name: json.loads(p.read_text(encoding='utf-8')) for p in map_paths}
    mfgmap = maps['manufacturer_to_id.json']
    fsmap = maps['field_strength_to_id.json']
    echo = maps['echo_time_stats.json']
    assert echo['normalization'] == 'min_max' and echo['max'] > echo['min']
    out = ROOT / 'data/processed/ax_t1_128_heldout' / plan_run
    report = ROOT / 'reports/v4_heldout_judge' / f'preprocess_{plan_run}'
    out.mkdir(parents=True, exist_ok=False)
    report.mkdir(parents=True, exist_ok=False)
    (out / 'images').mkdir()
    (out / 'splits').mkdir()
    (out / 'label_maps').mkdir()
    for path in map_paths:
        shutil.copyfile(path, out / 'label_maps' / path.name)
    rows, rejected = [], []
    arrays = {'train': [], 'val': []}
    for i, row in headers.sort_values(['patient_id', 'instance_number', 'sop_uid']).iterrows():
        try:
            image = resize_image(percentile_clip_and_normalize(read_dicom_image(row.slice_path), 1, 99), 128)
            assert image.shape == (128, 128) and image.dtype == np.float32
            assert np.isfinite(image).all() and image.min() >= 0 and image.max() <= 1
            foreground = compute_foreground_fraction(image, threshold=0.05)
            if foreground < 0.01:
                rejected.append({'slice_path': row.slice_path, 'patient_id': row.patient_id, 'reason': 'low_foreground', 'foreground_fraction': foreground})
                continue
            if not np.isfinite(row.echo_time):
                raise ValueError('Missing/non-finite DICOM EchoTime')
            image_path = out / 'images' / f'{row.patient_id}_{row.sop_uid}.npy'
            with image_path.open('xb') as stream:
                np.save(stream, image, allow_pickle=False)
            mfg, fs = int(mfgmap[row.manufacturer]), int(fsmap[row.field_strength])
            ds = pydicom.dcmread(row.slice_path, stop_before_pixels=True)
            entry = dict(row)
            entry.update(image_path=str(image_path.relative_to(ROOT)), study_uid=str(ds.StudyInstanceUID), manufacturer_label=mfg, field_strength_label=fs, magnetic_field_strength=float(ds.MagneticFieldStrength), echo_time_norm=float((row.echo_time - echo['min']) / (echo['max'] - echo['min'])), foreground_fraction=foreground, series_description=str(ds.SeriesDescription))
            rows.append(entry)
            arrays[row.judge_split].append(image)
        except Exception as exc:
            rejected.append({'slice_path': row.slice_path, 'patient_id': row.patient_id, 'reason': 'processing_error', 'error': repr(exc)})
        if (i + 1) % 250 == 0:
            print(f'Processed {len(rows)} slices; rejected {len(rejected)}', flush=True)
    frame = pd.DataFrame(rows)
    rejected_frame = pd.DataFrame(rejected)
    rejected_frame.to_csv(report / 'rejected_slices.csv', index=False)
    frame.to_csv(out / 'processed_metadata.csv', index=False)
    frame.to_csv(report / 'processed_metadata.csv', index=False)
    if len(rejected_frame) and rejected_frame.reason.eq('processing_error').any():
        raise ValueError('Processing errors recorded; inspect before creating a training pack')
    assert set(frame.patient_id) == set(plan.PatientID), 'At least one patient lost all slices'
    assert frame.groupby('patient_id').judge_split.nunique().max() == 1
    pack = {'excluded_patient_ids': excluded.patient_id.to_numpy(dtype=str)}
    for split in ['train', 'val']:
        data = frame[frame.judge_split.eq(split)]
        data.to_csv(out / 'splits' / f'{split}.csv', index=False)
        assert data.patient_id.nunique() == (48 if split == 'train' else 16)
        x = np.stack(arrays[split])
        assert len(x) == len(data)
        pack[f'{split}_x'] = x
        for key, column in [('mfg', 'manufacturer_label'), ('fs', 'field_strength_label')]:
            pack[f'{split}_{key}'] = data[column].to_numpy(dtype=np.int64)
        pack[f'{split}_patient'] = data.patient_id.to_numpy(dtype=str)
        pack[f'{split}_sop'] = data.sop_uid.to_numpy(dtype=str)
        pack[f'{split}_echo'] = data.echo_time_norm.to_numpy(dtype=np.float32)
        # Check alignment against saved per-slice arrays in both classes.
        for j in np.linspace(0, len(data) - 1, 8, dtype=int):
            np.testing.assert_array_equal(x[j], np.load(ROOT / data.iloc[j].image_path, allow_pickle=False))
    counts = frame.groupby(['judge_split', 'manufacturer_label', 'field_strength_label']).agg(patients=('patient_id', 'nunique'), slices=('sop_uid', 'size')).reset_index()
    assert counts.patients.tolist() == [12] * 4 + [4] * 4
    counts.to_csv(report / 'cohort_counts.csv', index=False)
    assert set(pack['train_patient']).isdisjoint(set(pack['val_patient']))
    npz_path = report / 'heldout_train_val_float32.npz'
    with npz_path.open('xb') as stream:
        np.savez_compressed(stream, **pack)
    with np.load(npz_path, allow_pickle=False) as saved:
        assert not any(k.startswith('test_') for k in saved.files)
        for key in pack:
            np.testing.assert_array_equal(saved[key], pack[key])
    assert protected_hashes == {str(p): sha(p) for p in protected}
    # Reproducible central slices for visual QA, one patient per split/cell.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 4, figsize=(10, 5))
    qa = []
    for axis, (_, cell) in zip(axes.ravel(), frame.groupby(['judge_split', 'manufacturer_label', 'field_strength_label'], sort=True)):
        patient = sorted(cell.patient_id.unique())[0]
        group = cell[cell.patient_id.eq(patient)].sort_values('instance_number')
        selected = group.iloc[len(group) // 2]
        axis.imshow(np.load(ROOT / selected.image_path), cmap='gray', vmin=0, vmax=1)
        axis.set_title(f'{selected.judge_split} {patient}\nmfg={selected.manufacturer_label} fs={selected.field_strength_label}', fontsize=8)
        axis.axis('off')
        qa.append(selected.to_dict())
    fig.tight_layout()
    fig.savefig(report / 'preprocessing_qc.png', dpi=160)
    plt.close(fig)
    pd.DataFrame(qa).to_csv(report / 'preprocessing_qc_sources.csv', index=False)
    summary = {'complete': True, 'plan_run': plan_run, 'download_receipt': str(download / 'result.json'), 'processed_patients': int(frame.patient_id.nunique()), 'processed_slices': len(frame), 'rejected_slices': len(rejected), 'train_slices': len(pack['train_x']), 'val_slices': len(pack['val_x']), 'pack_path': str(npz_path), 'pack_sha256': sha(npz_path), 'pack_bytes': npz_path.stat().st_size, 'plan_sha256': sha(plan_path), 'verified_headers_sha256': sha(headers_path), 'preprocess_source_sha256': sha(__file__), 'shared_preprocess_source_sha256': sha(ROOT / 'src/data/preprocess.py'), 'original_files_unchanged': True, 'original_file_sha256': protected_hashes, 'echo_normalisation': 'original statistics; not refitted or clipped', 'echo_outside_original_range': int(((frame.echo_time_norm < 0) | (frame.echo_time_norm > 1)).sum()), 'cloud_training_launched': False, 'pack_contains_original_test': False}
    (report / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    (report / 'README.md').write_text(f'''# Held-out cohort preprocessing

Generated by `python -m scripts.v4_heldout.preprocess --plan-run {plan_run}{' --download-attempt ' + download_attempt if download_attempt else ''}`. `summary.json` records provenance, hashes, counts and completed checks. `processed_metadata.csv` and `cohort_counts.csv` describe retained slices; `rejected_slices.csv` records any exclusions. `heldout_train_val_float32.npz` contains only new-cohort training/validation images, manufacturer/field labels, source EchoTime, patient and SOP IDs. It contains no original test images. All arrays are checked against saved images and against the reloaded pack. Original split/map hashes are verified unchanged.

`preprocessing_qc.png` shows one central slice per split/condition cell; `preprocessing_qc_sources.csv` gives exact inputs. Review this image before training. Images, patient splits and copies of original label/echo maps are in `{out.relative_to(ROOT)}`. The existing `src/data/preprocess.py` functions perform identical rescale/clip/resize/foreground operations; new label maps and random splits are never fitted. EchoTime stays a source value and is unused by the acquisition judge.

No judge has been trained, no cloud allowance is implied, and B2 remains incomplete. A future versioned training runner must use this explicit pack path and a new checkpoint namespace, select on the new validation patients, then evaluate frozen checkpoints on the original test cohort.
''', encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--plan-run', required=True)
    parser.add_argument('--download-attempt')
    args = parser.parse_args()
    main(args.plan_run, args.download_attempt)
