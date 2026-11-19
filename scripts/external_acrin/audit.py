"""Audit ACRIN 6698 against the Duke training sequence using public metadata."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import itertools
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET

import idc_index_data
import numpy as np
import pandas as pd
import pydicom
import requests


ROOT = Path(__file__).resolve().parents[2]
PUBLIC = 'https://www.cancerimagingarchive.net/wp-content/uploads/'
REGISTRY = 'https://clinicaltrials.gov/api/v2/studies/NCT01564368'
DOCUMENTS = [
    'ACRIN-6698_protocol.pdf',
    'ACRIN-6698-MRI-Acquisition-Parameters_20210520.pdf',
    'ACRIN-6698-ISPY2-DWI-and-DCE-MRI-Data-Descriptions_20210520.pdf',
]
TAGS = '''PatientID SeriesInstanceUID StudyInstanceUID Manufacturer ManufacturerModelName
ClinicalTrialSiteID ClinicalTrialTimePointID InstitutionName ImageType ScanningSequence
SequenceVariant ScanOptions MRAcquisitionType SequenceName ContrastBolusAgent
ContrastBolusStartTime AcquisitionTime SeriesTime ImageOrientationPatient RepetitionTime
EchoTime FlipAngle MagneticFieldStrength ProtocolName SeriesDescription TemporalPositionIdentifier
NumberOfTemporalPositions EchoTrainLength'''.split()
VENDOR = {'GE MEDICAL SYSTEMS': 'GE', 'SIEMENS': 'Siemens',
          'Philips Medical Systems': 'Philips', 'Philips Healthcare': 'Philips'}


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def header_fields(dataset):
    record = {tag: str(dataset.get(tag, '')).strip() for tag in TAGS}
    orientation = dataset.get('ImageOrientationPatient', [])
    if len(orientation) == 6:
        normal = np.cross(np.array(orientation[:3], float), np.array(orientation[3:], float))
        norm = np.linalg.norm(normal)
        record['axial_within_15_deg'] = bool(norm and abs(normal[2]) / norm >= np.cos(np.deg2rad(15)))
    else:
        record['axial_within_15_deg'] = None
    options = dataset.get('ScanOptions', [])
    options = options.split('\\') if isinstance(options, str) else list(options)
    record['fat_suppression_evidence'] = (
        'explicit_in_header' if {'FS', 'SFS', 'SPAIR', 'SPIR', 'FATSAT'} & set(options)
        or re.search(r'SPAIR|\bFS\b|FAT[ -]?SAT', record['SeriesDescription'], re.I)
        else 'unknown_from_public_tags'
    )
    return record


def duke_reference():
    source = pd.read_csv(ROOT / 'data/metadata/ax_t1_metadata.csv')
    train = pd.read_csv(ROOT / 'data/processed/ax_t1_128/splits/train.csv')
    source = source[source.PatientID.isin(train.patient_id)].drop_duplicates('SeriesInstanceUID')
    if set(source.PatientID) != set(train.patient_id) or not source.PatientID.is_unique:
        raise ValueError('Expected exactly one source series per Duke training patient')
    return pd.DataFrame([header_fields(pydicom.dcmread(ROOT / path, stop_before_pixels=True))
                         for path in source.slice_path])


def family(value):
    label = value.lower().removeprefix('ispy2: ').strip()
    if label.startswith('acrin-6698:'):
        return 'diffusion_or_diffusion_derivative'
    if 'volser:' in label:
        return 'dce_analysis_export'
    if re.search(r'scout|loc(?:alizer)?\b|survey', label):
        return 'localiser'
    if re.search(r't2|stir', label):
        return 't2_or_fluid_sensitive'
    if re.search(r'dyn|vibrant|ethrive|spair|t1|^\(\d+/|^ph\d+/', label):
        return 't1_or_dce_named'
    return 'ambiguous'


def catalogue_samples(acrin):
    mr = acrin.loc[acrin.Modality.eq('MR')].copy()
    mr['SeriesDescription'] = mr.SeriesDescription.fillna('')
    mr['family'] = mr.SeriesDescription.map(family)
    mr['series_number'] = pd.to_numeric(mr.SeriesNumber, errors='coerce')
    order = ['series_number', 'SeriesInstanceUID']
    baseline = mr[mr.StudyDescription.eq('ACRIN-6698_ISPY2_MRI_T0')]
    likely_t1 = baseline[baseline.family.eq('t1_or_dce_named')]
    # Use T1/DCE-labelled series first; then a native T2 or scout if it is missing.
    # Sort within each class by series number, with a stable UID tie break.
    other = baseline[baseline.family.isin(['t2_or_fluid_sensitive', 'localiser', 'ambiguous'])]
    panel = pd.concat([likely_t1.sort_values(order), other.sort_values(order), baseline.sort_values(order)])
    panel = panel.drop_duplicates('PatientID').assign(selection='baseline_patient_anchor')
    t1 = mr[mr.family.eq('t1_or_dce_named')].sort_values(order)
    reps = t1.drop_duplicates(['SeriesDescription', 'Manufacturer', 'ManufacturerModelName', 'StudyDescription'])
    ambiguous = mr[mr.family.eq('ambiguous')]
    samples = pd.concat([panel, reps.assign(selection='t1_description_model_visit'),
                         ambiguous.assign(selection='all_ambiguous')]).drop_duplicates('SeriesInstanceUID')
    return mr, samples


def fetch_header(row):
    base = f'https://{row.aws_bucket}.s3.amazonaws.com/'
    result = {name: getattr(row, name) for name in
              ('PatientID', 'SeriesInstanceUID', 'StudyDescription', 'family', 'selection')}
    result.update(catalogue_description=row.SeriesDescription, catalogue_vendor=row.Manufacturer)
    for _ in range(3):
        try:
            listing = requests.get(base, params={'list-type': 2, 'max-keys': 1,
                                   'prefix': row.crdc_series_uuid + '/'}, timeout=12)
            listing.raise_for_status()
            key = ET.fromstring(listing.content).findtext('.//{*}Key')
            if not key or not key.endswith('.dcm'):
                raise ValueError('Missing DICOM instance key')
            for limit in (4096, 16384, 32768):
                with requests.get(base + key, headers={'Range': f'bytes=0-{limit - 1}'},
                                  stream=True, timeout=12) as response:
                    match = re.fullmatch(r'bytes 0-(\d+)/(\d+)', response.headers.get('Content-Range', ''))
                    if response.status_code != 206 or not match or not int(match[1]) < min(limit, int(match[2]) - 1):
                        raise ValueError('Range response cannot be verified as capped and partial')
                    data = response.content
                dataset = pydicom.dcmread(io.BytesIO(data), stop_before_pixels=True, specific_tags=TAGS)
                fields = header_fields(dataset)
                if all(fields[k] for k in ('ScanningSequence', 'MRAcquisitionType',
                                            'ImageOrientationPatient', 'MagneticFieldStrength', 'ClinicalTrialSiteID')):
                    break
            if (fields['PatientID'], fields['SeriesInstanceUID'], fields['Manufacturer']) != (
                row.PatientID, row.SeriesInstanceUID, row.Manufacturer
            ):
                raise ValueError('DICOM identity or manufacturer mismatch')
            return {**result, **fields, 'object_url': base + key, 'last_range_bytes': len(data), 'error': ''}
        except (requests.RequestException, ValueError, ET.ParseError, OSError) as exc:
            error = f'{type(exc).__name__}: {exc}'
    return {**result, 'error': error}


def count_cells(panel, out):
    axes = ['ClinicalTrialSiteID', 'vendor', 'field_t']
    panel = panel.copy()
    panel['vendor'] = panel.Manufacturer.map(VENDOR).fillna('unknown')
    panel['field_t'] = pd.to_numeric(panel.MagneticFieldStrength, errors='coerce')
    valid = panel[panel.error.eq('') & panel.ClinicalTrialSiteID.fillna('').ne('')
                  & panel.field_t.isin([1.5, 3.0]) & panel.vendor.ne('unknown')]
    cells = itertools.product(sorted(valid.ClinicalTrialSiteID.unique()), sorted(set(VENDOR.values())), [1.5, 3.0])
    grid = pd.MultiIndex.from_tuples(list(cells), names=axes)
    counts = valid.groupby(axes).PatientID.nunique().reindex(grid, fill_value=0).rename('baseline_anchor_patients')
    counts.to_frame().assign(verified_eligible_patients=0).to_csv(out / 'site_vendor_field_counts.csv')
    return valid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--workers', type=int, default=12)
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    index = Path(idc_index_data.IDC_INDEX_PARQUET_FILEPATH)
    acrin = pd.read_parquet(index, filters=[('collection_id', '==', 'acrin_6698')])
    duke = pd.read_parquet(index, filters=[('collection_id', '==', 'duke_breast_cancer_mri')])
    reference = duke_reference()
    reference.to_csv(out / 'duke_training_headers.csv', index=False)
    mr, samples = catalogue_samples(acrin)
    mr.groupby(['family', 'SeriesDescription'], dropna=False).agg(
        series=('SeriesInstanceUID', 'size'), patients=('PatientID', 'nunique')
    ).to_csv(out / 'catalogue_inventory.csv')
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        headers = pd.DataFrame(pool.map(fetch_header, samples.itertuples(index=False)))
    headers.to_csv(out / 'acrin_sampled_headers.csv', index=False)
    valid = count_cells(headers[headers.selection.eq('baseline_patient_anchor')], out)
    t1 = headers[headers.family.eq('t1_or_dce_named') & headers.error.eq('')]
    structural = headers[headers.family.isin(['t1_or_dce_named', 'ambiguous']) & headers.error.eq('')]
    structural_matches = structural[structural.MRAcquisitionType.eq('2D') & structural.ScanningSequence.eq('SE')]
    trial = requests.get(REGISTRY, timeout=20)
    trial.raise_for_status()
    locations = trial.json()['protocolSection']['contactsLocationsModule']['locations']
    documents = {}
    for name in DOCUMENTS:
        response = requests.get(PUBLIC + name, timeout=20)
        response.raise_for_status()
        if not response.content.startswith(b'%PDF-'):
            raise ValueError(f'Expected PDF at {response.url}')
        documents[response.url] = {'bytes': len(response.content), 'sha256': hashlib.sha256(response.content).hexdigest()}
    facts = {
        'index_version': idc_index_data.__version__, 'index_sha256': sha(index),
        'script_sha256': sha(__file__), 'documents': documents,
        'registry': REGISTRY, 'registry_listed_facilities': [row['facility'] for row in locations],
        'duke_reference_series': len(reference), 'acrin_patients': acrin.PatientID.nunique(),
        'acrin_series': len(acrin), 'acrin_mr_series': len(mr),
        'catalogue_shared_values_with_duke': {col: len(set(acrin[col].dropna()) & set(duke[col].dropna()))
                                             for col in ('PatientID', 'StudyInstanceUID', 'SeriesInstanceUID')},
        'probes': len(headers), 'probe_failures': int(headers.error.ne('').sum()),
        'baseline_patients': mr.loc[mr.StudyDescription.str.endswith('_T0'), 'PatientID'].nunique(),
        'baseline_patients_with_scanner_and_site': len(valid),
        'sampled_t1_or_dce': len(t1), 'sampled_t1_2d_se': int((t1.MRAcquisitionType.eq('2D') &
                                                             t1.ScanningSequence.eq('SE')).sum()),
        'ambiguous_series_total': int(mr.family.eq('ambiguous').sum()),
        'possible_2d_se_series_for_review': structural_matches.SeriesInstanceUID.tolist(),
        'sampled_t1_dce_signature_counts': t1.groupby(['MRAcquisitionType', 'ScanningSequence'], dropna=False)
            .size().reset_index(name='probes').to_dict('records'),
        'baseline_overlap_by_strength': [
            {'tesla': field, 'sites': group.ClinicalTrialSiteID.nunique(), 'vendors': group.vendor.nunique(),
             'patients': len(group)} for field, group in valid.groupby('field_t')],
        'verified_eligible_patients': 0,
    }
    (out / 'audit.json').write_text(json.dumps(facts, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(facts, indent=2))


if __name__ == '__main__':
    main()
