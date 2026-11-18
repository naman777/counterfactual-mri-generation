"""Download only the frozen B2 series plan, after explicit user approval.

Default is a LOCAL dry run, with no network or image download:
python -m scripts.v4_heldout.download --plan-run preflight_20261016_01
Execution additionally requires --execute --max-gb <approved cap>.
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .audit import ROOT, sha


def valid_id(value):
    if not value or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789_-' for c in value):
        raise ValueError('Invalid plan ID')
    return value


def main(args):
    folder = ROOT / 'reports/v4_heldout_judge' / valid_id(args.plan_run)
    summary = json.loads((folder / 'summary.json').read_text(encoding='utf-8'))
    path = folder / 'download_plan.csv'
    if sha(path) != summary['download_plan_sha256']:
        raise ValueError('Download plan changed after audit')
    plan = pd.read_csv(path)
    excluded = pd.read_csv(folder / 'excluded_original_patients.csv')
    assert len(plan) == 64 and plan.PatientID.is_unique
    assert not (set(plan.PatientID) & set(excluded.patient_id))
    assert plan.Modality.eq('MR').all()
    assert plan.SeriesDescription.str.strip().str.lower().eq('ax t1').all()
    gb = float(plan.series_size_MB.sum() / 1000)
    print(json.dumps({'mode': 'execute' if args.execute else 'local_dry_run', 'patients': len(plan), 'decimal_GB': gb, 'instances': int(plan.instanceCount.sum()), 'plan_sha256': sha(path)}, indent=2))
    if not args.execute:
        return
    if args.max_gb is None or args.max_gb <= 0 or gb > args.max_gb:
        raise ValueError('Specify an approved --max-gb cap covering the catalogue estimate')
    # Intentionally imported only after the execution guard.
    from idc_index import IDCClient
    import pydicom
    from src.data.preprocess import normalize_field_strength, normalize_manufacturer

    client = IDCClient()
    live = client.index
    now = live[live.SeriesInstanceUID.isin(plan.SeriesInstanceUID)].copy()
    if len(now) != len(plan) or not now.SeriesInstanceUID.is_unique:
        raise ValueError('Catalogue no longer resolves the exact planned series')
    if float(pd.to_numeric(now.series_size_MB).sum() / 1000) > args.max_gb:
        raise ValueError('Current catalogue estimate exceeds approved cap')
    raw = ROOT / 'data/raw/duke_breast_cancer_mri_heldout' / args.plan_run
    raw.mkdir(parents=True, exist_ok=False)
    receipts = ROOT / 'reports/v4_heldout_judge' / f'download_{args.plan_run}'
    receipts.mkdir(parents=True, exist_ok=False)
    receipt = {'plan': str(path), 'plan_sha256': sha(path), 'max_decimal_GB': args.max_gb, 'started_utc': datetime.now(timezone.utc).isoformat(), 'raw_folder': str(raw), 'complete': False}
    (receipts / 'request.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    headers = []
    try:
        for _, row in plan.iterrows():
            # Scalar series UID avoids the API's unfiltered whole-collection selection.
            client.download_dicom_series(str(row.SeriesInstanceUID), str(raw), quiet=False, use_s5cmd_sync=False)
            files = list((raw / 'duke_breast_cancer_mri' / row.PatientID).rglob('*.dcm'))
            if len(files) != int(row.instanceCount):
                raise ValueError(f'{row.PatientID}: expected {row.instanceCount} DICOM instances, found {len(files)}')
            for file in files:
                ds = pydicom.dcmread(file, stop_before_pixels=True)
                field = normalize_field_strength(getattr(ds, 'MagneticFieldStrength', None))
                mfg = normalize_manufacturer(getattr(ds, 'Manufacturer', ''))
                checks = {
                    'patient': str(getattr(ds, 'PatientID', '')) == row.PatientID,
                    'series': str(getattr(ds, 'SeriesInstanceUID', '')) == row.SeriesInstanceUID,
                    'description': str(getattr(ds, 'SeriesDescription', '')).strip().lower() == 'ax t1',
                    'manufacturer': mfg == row.Manufacturer,
                    'field_strength': field == ('1.5T' if int(row.planning_field_strength_label) == 0 else '3.0T'),
                }
                if not all(checks.values()):
                    raise ValueError(f'Header mismatch at {file}: {checks}')
                headers.append({'patient_id': row.PatientID, 'series_uid': row.SeriesInstanceUID, 'sop_uid': str(ds.SOPInstanceUID), 'slice_path': str(file), 'judge_split': row.judge_split, 'manufacturer': mfg, 'field_strength': field, 'echo_time': float(getattr(ds, 'EchoTime', 'nan')), 'instance_number': int(getattr(ds, 'InstanceNumber', -1))})
            if sum(f.stat().st_size for f in raw.rglob('*') if f.is_file()) > args.max_gb * 1e9:
                raise ValueError('Actual downloaded bytes exceeded approved cap; stopping before next series')
        frame = pd.DataFrame(headers)
        assert frame.sop_uid.is_unique
        frame.to_csv(receipts / 'verified_dicom_headers.csv', index=False)
        receipt['complete'] = True
        receipt['instances_verified'] = len(frame)
    except Exception as exc:
        receipt['error'] = repr(exc)
        if headers:
            pd.DataFrame(headers).to_csv(receipts / 'partial_verified_headers.csv', index=False)
        raise
    finally:
        receipt['finished_utc'] = datetime.now(timezone.utc).isoformat()
        receipt['bytes_on_disk'] = sum(f.stat().st_size for f in raw.rglob('*') if f.is_file())
        (receipts / 'result.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
        (receipts / 'README.md').write_text('Generated by `python -m scripts.v4_heldout.download`. request.json freezes the authorised plan/cap. result.json records completion or the error. verified_dicom_headers.csv (or partial_verified_headers.csv on failure) records checked metadata. Original data and results are unchanged. An actual transfer can exceed the catalogue estimate by up to the current series before the between-series guard stops.\n', encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--plan-run', required=True)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--max-gb', type=float)
    main(parser.parse_args())
