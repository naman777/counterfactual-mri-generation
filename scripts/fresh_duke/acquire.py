"""Verify and lock the frozen Duke replication cohort without running any model."""

import argparse
import json
from pathlib import Path
from time import sleep

import cv2
import numpy as np
import pandas as pd
import pydicom

from scripts.fresh_duke.plan import CELLS, ROOT, read, sha, valid_id
from src.data.preprocess import (
    apply_rescale_if_present, compute_foreground_fraction, normalize_field_strength,
    normalize_manufacturer, percentile_clip_and_normalize, resize_image,
)


LABELS = ROOT / 'data/processed/ax_t1_128/label_maps'
PROFILE = {
    'max_axial_tilt_deg': 15, 'direction_tolerance': 1e-4, 'spacing_relative_tolerance': .10,
    'spacing_absolute_tolerance_mm': .2, 'image_size': 128, 'percentiles': [1, 99],
    'foreground_pixel_threshold': .05, 'min_foreground_fraction': .01,
    'slice_selection': 'Remove first and last physical slices. Choose the midpoints of up to five equal rank bins.',
    'resize_backend': f'OpenCV {cv2.__version__} INTER_AREA', 'numpy': np.__version__,
    'pydicom': pydicom.__version__,
}


class SeriesRejected(ValueError):
    """An observed data failure that permits an in-cell replacement."""


def require(condition, reason):
    if not condition:
        raise SeriesRejected(reason)


def check_header(ds, entry):
    require(
        (str(ds.PatientID), str(ds.StudyInstanceUID), str(ds.SeriesInstanceUID)) ==
        (entry.PatientID, entry.StudyInstanceUID, entry.SeriesInstanceUID),
        'patient_study_or_series_uid_mismatch',
    )
    require(ds.Modality == 'MR' and ds.SeriesDescription.strip().lower() == 'ax t1', 'sequence_mismatch')
    manufacturer = {'GE': 'GE MEDICAL SYSTEMS', 'Siemens': 'SIEMENS'}[entry.manufacturer]
    require(normalize_manufacturer(ds.Manufacturer) == manufacturer, 'manufacturer_mismatch')
    require(normalize_field_strength(ds.MagneticFieldStrength) == entry.field_strength, 'field_strength_mismatch')
    require(np.isfinite(float(ds.EchoTime)) and float(ds.EchoTime) > 0, 'invalid_echo_time')
    require(int(getattr(ds, 'NumberOfFrames', 1)) == 1 and int(ds.SamplesPerPixel) == 1, 'unsupported_frame_layout')


def inspect_series(entry, files, historical_sops):
    require(len(files) == int(entry.instanceCount) and len(files) >= 3, 'missing_or_extra_instances')
    records, sops = [], set()
    reference = None
    for path in sorted(files):
        try:
            ds = pydicom.dcmread(path)
            check_header(ds, entry)
            int(ds.InstanceNumber)
            sop = str(ds.SOPInstanceUID)
            require(sop not in sops and sop not in historical_sops, 'duplicate_or_historical_sop_uid')
            require(sop == str(ds.file_meta.MediaStorageSOPInstanceUID), 'file_meta_sop_uid_mismatch')
            directions = np.asarray(ds.ImageOrientationPatient, dtype=float).reshape(2, 3)
            position = np.asarray(ds.ImagePositionPatient, dtype=float).reshape(3)
            spacing = np.asarray(ds.PixelSpacing, dtype=float).reshape(2)
            require(np.isfinite(position).all() and np.isfinite(spacing).all() and (spacing > 0).all(),
                    'invalid_position_or_pixel_spacing')
            tol = PROFILE['direction_tolerance']
            require(np.allclose(directions @ directions.T, np.eye(2), rtol=0, atol=tol), 'invalid_orientation')
            normal = np.cross(*directions)
            require(abs(normal[2]) >= np.cos(np.deg2rad(PROFILE['max_axial_tilt_deg'])), 'non_axial_orientation')
            grid = np.r_[directions.ravel(), spacing, ds.Rows, ds.Columns, float(ds.EchoTime)]
            if reference is None:
                reference, first_normal = grid, normal * np.sign(normal[2])
            require(np.allclose(grid, reference, rtol=0, atol=tol), 'inconsistent_grid_or_echo_time')
            pixels = ds.pixel_array
            require(pixels.shape == (ds.Rows, ds.Columns), 'unexpected_decoded_shape')
            sops.add(sop)
            records.append((float(position @ first_normal), sop, path, ds))
        except (AttributeError, TypeError, ValueError, RuntimeError, OSError, NotImplementedError) as exc:
            raise SeriesRejected(f'{path.name}: {exc}') from exc
    records.sort(key=lambda row: row[0])
    gaps = np.diff([row[0] for row in records])
    require((gaps > .01).all(), 'duplicate_slice_positions')
    require(np.allclose(gaps, np.median(gaps), rtol=PROFILE['spacing_relative_tolerance'],
                        atol=PROFILE['spacing_absolute_tolerance_mm']), 'irregular_slice_spacing')
    interior = len(records) - 2
    count = min(5, interior)
    selected = {1 + ((2 * i + 1) * interior) // (2 * count) for i in range(count)}
    result = []
    for i, (position, sop, path, ds) in enumerate(records):
        image = None
        if i in selected:
            try:
                scaled = apply_rescale_if_present(ds, ds.pixel_array.astype(np.float32))
                clipped = percentile_clip_and_normalize(scaled, *PROFILE['percentiles'])
                image = resize_image(clipped, PROFILE['image_size'])
                require(image.shape == (128, 128) and image.dtype == np.float32 and np.isfinite(image).all()
                        and 0 <= image.min() <= image.max() <= 1, 'invalid_preprocessed_array')
                require(compute_foreground_fraction(image, PROFILE['foreground_pixel_threshold']) >=
                        PROFILE['min_foreground_fraction'], 'low_foreground_at_frozen_slice_index')
            except (ValueError, RuntimeError) as exc:
                raise SeriesRejected(f'{path.name}: {exc}') from exc
        result.append((i, position, sop, path, ds, image))
    return result


def load_plan(plan_run):
    folder = ROOT / 'reports/v21_fresh_duke' / valid_id(plan_run)
    meta = json.loads((folder / 'plan.json').read_text(encoding='utf-8'))
    order = folder / 'seeded_series_order.csv'
    checks = {order: meta['order_sha256'], folder / 'patient_manifest.csv': meta['patient_manifest_sha256']}
    checks.update({ROOT / path: digest for path, digest in meta['frozen_code_and_maps'].items()})
    if any(sha(path) != digest for path, digest in checks.items()):
        raise ValueError('Frozen selection or preprocessing inputs changed')
    return meta, pd.read_csv(order)


def fetch_series(client, entry, destination):
    """Retry a partial or failed transfer once, reusing files already on disk."""
    files = []
    error = None
    for attempt in range(2):
        try:
            client.download_dicom_series(
                entry.SeriesInstanceUID, str(destination), quiet=True, show_progress_bar=False,
                use_s5cmd_sync=True, dirTemplate=None,
            )
            error = None
        except (RuntimeError, OSError) as exc:
            error = exc
        files = list(destination.rglob('*.dcm'))
        if error is None and len(files) >= int(entry.instanceCount):
            return files, attempt + 1
        if attempt == 0:
            sleep(2)
    if error is not None or not files:
        raise RuntimeError(f'Download unavailable after two attempts for {entry.PatientID}') from error
    return files, 2


def dry_run(out):
    """Replay four original training patients, one per scanner cell, byte for byte."""
    train = read('data/processed/ax_t1_128/splits/train.csv')
    rows, failures = [], []
    for manufacturer, field in CELLS:
        raw_label = {'GE': 'GE MEDICAL SYSTEMS', 'Siemens': 'SIEMENS'}[manufacturer]
        subset = train[train.manufacturer.eq(raw_label) & train.field_strength.eq(field)]
        for (patient, study, series), group in subset.groupby(['patient_id', 'study_uid', 'series_uid']):
            folder = (ROOT / group.iloc[0].slice_path).parent
            files = list(folder.glob('*.dcm'))
            entry = pd.Series(dict(PatientID=patient, StudyInstanceUID=study, SeriesInstanceUID=series,
                                   manufacturer=manufacturer, field_strength=field, instanceCount=len(files)))
            try:
                checked = inspect_series(entry, files, set())
                by_sop = group.set_index('sop_uid')
                matched = []
                for i, position, sop, _, _, image in checked:
                    if image is not None:
                        require(sop in by_sop.index, 'no_original_array_for_frozen_slice')
                        ref = ROOT / by_sop.loc[sop, 'image_path']
                        np.testing.assert_array_equal(np.load(ref, allow_pickle=False), image)
                        matched.append(dict(patient_id=patient, series_uid=series, sop_uid=sop,
                                            manufacturer=manufacturer, field_strength=field,
                                            physical_index=i, position_mm=position, original_array_sha256=sha(ref)))
                rows.extend(matched)
                break
            except SeriesRejected as exc:
                failures.append(dict(patient_id=patient, series_uid=series, reason=str(exc)))
        else:
            raise ValueError(f'No historical replay completed in {manufacturer} / {field}')
    out.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(rows).to_csv(out / 'historical_replay.csv', index=False)
    pd.DataFrame(failures, columns=['patient_id', 'series_uid', 'reason']).to_csv(out / 'historical_skips.csv', index=False)
    (out / 'README.md').write_text(
        'Produced by `python -m scripts.fresh_duke.acquire --dry-run-id <id>`. '
        'historical_replay.csv records the exact array matches by SOP UID. '
        'historical_skips.csv records rejected historical series encountered before a match.\n', encoding='utf-8'
    )
    print(f'{len(rows)} exact array matches, four scanner cells, {len(failures)} historical skips')


def acquire(plan_run, acquisition_id, max_gb):
    from idc_index import IDCClient

    meta, order = load_plan(plan_run)
    if not np.isfinite(max_gb) or max_gb < meta['initial_catalogue_GB']:
        raise ValueError('An approved cap covering the initial series is required')
    series_order = order[order.role.ne('unreleased')]
    client = IDCClient()
    live = client.index.set_index('SeriesInstanceUID')
    raw = ROOT / 'data/raw_fresh_duke' / valid_id(acquisition_id)
    out = ROOT / 'reports/v21_fresh_duke' / valid_id(acquisition_id)
    out.mkdir(parents=True, exist_ok=False)
    raw.mkdir(parents=True, exist_ok=False)
    (out / 'README.md').write_text(
        'Produced by `python -m scripts.fresh_duke.acquire`. attempts.jsonl records retrieval, QC and exclusions. '
        'dicom_inventory.csv records verified instance identity, order and SHA-256. slice_manifest.csv and images/ '
        'record the frozen arrays and source EchoTime. lock.json records counts, profile, and content hashes. '
        'The lock exists only after data verification finishes.\n', encoding='utf-8',
    )
    images = out / 'images'
    images.mkdir()
    echo = json.loads((LABELS / 'echo_time_stats.json').read_text(encoding='utf-8'))
    historical_sops = set()
    for path, column in [
        ('data/metadata/dicom_metadata.csv', 'SOPInstanceUID'),
        ('data/metadata/dicom_metadata_sample.csv', 'SOPInstanceUID'),
        ('data/processed/ax_t1_128_heldout/preflight_20261016_01/processed_metadata.csv', 'sop_uid'),
        ('reports/v8_generator_dev/prospective_series_audit_20261117_01/headers.csv', 'sop_uid'),
    ]:
        historical_sops.update(read(path, usecols=[column])[column].dropna().astype(str))
    slices, all_instances, counts = [], [], {}
    actual = 0
    with (out / 'attempts.jsonl').open('x', encoding='utf-8') as log:
        for manufacturer, field in CELLS:
            count = 0
            cell = series_order[series_order.manufacturer.eq(manufacturer) & series_order.field_strength.eq(field)]
            for entry in cell.itertuples(index=False):
                if count == meta['target_per_cell']:
                    break
                pid, uid = entry.PatientID, entry.SeriesInstanceUID
                receipt = {'patient_id': pid, 'series_uid': uid, 'rank': entry.within_cell_rank}
                try:
                    row = live.loc[uid] if uid in live.index else None
                    if row is None:
                        raise SeriesRejected('series_absent_from_current_catalogue')
                    require(isinstance(row, pd.Series) and row.PatientID == pid and
                            row.StudyInstanceUID == entry.StudyInstanceUID and
                            row.collection_id == 'duke_breast_cancer_mri' and row.Modality == 'MR' and
                            row.SeriesDescription.strip().lower() == 'ax t1' and
                            int(row.instanceCount) == int(entry.instanceCount), 'live_catalogue_mismatch')
                    estimate = float(row.series_size_MB) * 1e6
                    if not np.isfinite(estimate) or actual + estimate * 1.10 > max_gb * 1e9:
                        raise RuntimeError('Remaining approved byte budget cannot cover the next frozen series')
                    destination = raw / pid
                    destination.mkdir()
                    try:
                        files, receipt['transfer_attempts'] = fetch_series(client, entry, destination)
                    finally:
                        actual += sum(f.stat().st_size for f in destination.rglob('*') if f.is_file())
                        receipt['stored_bytes_total'] = actual
                    if actual > max_gb * 1e9:
                        raise RuntimeError('Actual stored DICOM bytes exceeded the cap; further transfer stopped')
                    checked = inspect_series(entry, files, historical_sops)
                except SeriesRejected as exc:
                    receipt.update(status='excluded', reason=str(exc))
                except Exception as exc:
                    receipt.update(status='interrupted', reason=str(exc))
                    print(json.dumps(receipt), file=log, flush=True)
                    raise
                else:
                    for i, position, sop, path, ds, image in checked:
                        instance = dict(patient_id=pid, study_uid=entry.StudyInstanceUID, series_uid=uid,
                                        sop_uid=sop, manufacturer=manufacturer, field_strength=field,
                                        instance_number=int(ds.InstanceNumber), physical_index=i, position_mm=position,
                                        image_orientation_patient=list(map(float, ds.ImageOrientationPatient)),
                                        image_position_patient=list(map(float, ds.ImagePositionPatient)),
                                        dicom_path=path.relative_to(ROOT).as_posix(), dicom_sha256=sha(path),
                                        selected=image is not None)
                        all_instances.append(instance)
                        historical_sops.add(sop)
                        if image is not None:
                            dest = images / f'{pid}_{sop}.npy'
                            with dest.open('xb') as stream:
                                np.save(stream, image, allow_pickle=False)
                            source_echo = float(ds.EchoTime)
                            slices.append(dict(instance, image_path=dest.relative_to(ROOT).as_posix(),
                                               image_sha256=sha(dest), echo_time=source_echo,
                                               echo_time_norm=(source_echo - echo['min']) / (echo['max'] - echo['min'])))
                    count += 1
                    receipt.update(status='accepted')
                print(json.dumps(receipt), file=log, flush=True)
            counts[f'{manufacturer} {field}'] = count
    for name, records in [('dicom_inventory.csv', all_instances), ('slice_manifest.csv', slices)]:
        pd.DataFrame(records).to_csv(out / name, index=False)
    status = ('target_met' if all(n == meta['target_per_cell'] for n in counts.values()) else
              'minimum_met' if all(n >= meta['minimum_per_cell'] for n in counts.values()) else 'below_minimum')
    lock = dict(status=status, counts=counts, total_slices=len(slices), dicom_bytes=actual,
                preprocessing=PROFILE, source_echo_stats=echo, plan_run=plan_run,
                order_sha256=meta['order_sha256'], frozen_code_and_maps=meta['frozen_code_and_maps'],
                files={name: sha(out / name) for name in ('attempts.jsonl', 'dicom_inventory.csv', 'slice_manifest.csv')})
    (out / 'lock.json').write_text(json.dumps(lock, indent=2), encoding='utf-8')
    print(json.dumps(lock, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run-id')
    parser.add_argument('--plan-run')
    parser.add_argument('--acquisition-id')
    parser.add_argument('--max-gb', type=float)
    args = parser.parse_args()
    if args.dry_run_id:
        dry_run(ROOT / 'reports/v21_fresh_duke' / valid_id(args.dry_run_id))
    elif args.plan_run and args.acquisition_id and args.max_gb is not None:
        acquire(args.plan_run, args.acquisition_id, args.max_gb)
    else:
        parser.error('Specify --dry-run-id or all of --plan-run, --acquisition-id and approved --max-gb')
