"""Small bounded prospective-series retrieval and full header audit; never changes splits."""
import argparse
import csv
import hashlib
import io
import json
import re
import stat
import zipfile
from pathlib import PurePosixPath

import numpy as np
import pandas as pd
import pydicom
import requests

from scripts.v7_geometry.orientation import orientation_plan
from scripts.v8_generator_dev.io import ROOT,csv_write,digest,json_write,project_path,verify_inputs


def prepare(run_id):
    if not re.fullmatch('[A-Za-z0-9_-]+',run_id):
        raise ValueError('Safe new proposal ID required')
    base=ROOT/'reports/v8_generator_dev'
    audit=base/'public_metadata_candidates_20261117_01'
    verify_inputs(json.loads((audit/'inputs.json').read_text()))
    source=audit/'candidate_headers.csv'
    frame=pd.read_csv(source,keep_default_na=False)
    frame=frame[(frame.field_strength_label==0)&frame.sample_2d_spin_echo_family&frame.sample_geometry_compatible&
        frame.field_strength_target_support.isin(['observed_condition','interpolation_only'])]
    if len(frame)!=2:
        raise ValueError('This bounded proposal is for the two identified candidates only')
    output=base/run_id
    output.mkdir(exist_ok=False)
    targets=frame[['patient_id','series_uid','image_count','declared_series_bytes','echo_time_ms','manufacturer_label',
        'field_strength_label','field_strength_target_support']].to_dict('records')
    expected=sum(int(row['declared_series_bytes']) for row in targets)
    if expected>25*1024**2:
        raise ValueError('Candidate scope exceeds the small-download proposal')
    json_write(output/'proposal.json',dict(candidates=targets,expected_uncompressed_bytes=expected,
        maximum_network_bytes=30*1024**2,maximum_uncompressed_bytes=30*1024**2,
        scope='Small prospective raw-series/header audit under next-step data preparation; no large cohort retrieval',
        source='https://www.cancerimagingarchive.net/collection/duke-breast-cancer-mri/',
        download_endpoint='https://services.cancerimagingarchive.net/nbia-api/services/v1/getImage',
        no_dataset_split_change=True,no_current_model_evaluation=True,contrast_status_unresolved=True))
    json_write(output/'inputs.json',{source.relative_to(ROOT).as_posix():digest(source),
        'scripts/v8_generator_dev/retrieve_candidate_series.py':digest(__file__)})
    (output/'PROPOSAL.md').write_text(
        '# Small prospective series audit\n\nRetrieve only the two metadata-selected unseen 1.5T series, '
        f'with {expected/1024**2:.2f} MiB total declared DICOM payload, capped at 30 MiB network '
        'and 30 MiB extracted bytes. Audit all instance headers and preserve original raw bytes. '
        'This is a small data-preparation retrieval, not a large Duke cohort download. '
        'No split, training dataset, running experiment or final assessment is changed. '
        'These candidates have representative-header support only; full protocol/contrast and '
        'image quality still require assessment before any separately versioned use.\n',encoding='utf-8')
    (output/'README.md').write_text(
        'Producer: `python -m scripts.v8_generator_dev.retrieve_candidate_series prepare`.\n'
        'proposal.json/PROPOSAL.md freeze the two series, bytes and no-ingestion scope; '
        'inputs.json hashes candidate evidence/code. No retrieval occurs during preparation.\n',encoding='utf-8')
    print('SMALL DATA PROPOSAL',output,expected,'bytes',flush=True)


def retrieve(proposal,run_id):
    if not re.fullmatch('[A-Za-z0-9_-]+',run_id):
        raise ValueError('Safe new retrieval ID required')
    proposal=project_path(proposal)
    verify_inputs(json.loads((proposal/'inputs.json').read_text()))
    config=json.loads((proposal/'proposal.json').read_text())
    if len(config['candidates'])!=2 or config['maximum_network_bytes']!=30*1024**2:
        raise ValueError('Refuse a widened download')
    raw_root=(ROOT/'data/raw/duke_breast_cancer_mri').resolve()
    for row in config['candidates']:
        if not re.fullmatch('Breast_MRI_[0-9]{3}',row['patient_id']) or not re.fullmatch('[0-9.]+',row['series_uid']):
            raise ValueError('Unsafe candidate identity')
        target=(raw_root/row['patient_id']).resolve()
        if not target.is_relative_to(raw_root) or target.exists():
            raise ValueError('Refuse existing/out-of-root candidate target')
    membership_hashes={}
    for split in ('train','val','test'):
        path=ROOT/'data/processed/ax_t1_128/splits'/(split+'.csv')
        membership_hashes[path.relative_to(ROOT).as_posix()]=digest(path)
        members=set(pd.read_csv(path,usecols=['patient_id']).patient_id)
        if any(row['patient_id'] in members for row in config['candidates']):
            raise ValueError('Candidate overlaps an existing split')
    output=ROOT/'reports/v8_generator_dev'/run_id
    output.mkdir(exist_ok=False)
    (output/'README.md').write_text(
        'Producer: `python -m scripts.v8_generator_dev.retrieve_candidate_series retrieve`.\n'
        'protocol.json fixes the bounded proposal; *.zip preserves original responses; headers.csv '
        'records every retrieved instance and new raw path/hash. series_checks.json audits full header '
        'identity/grid/condition consistency without decoding pixel arrays. inputs.json hashes evidence '
        'and unchanged memberships; complete.json never grants ingestion, contrast validity or anatomy truth.\n',encoding='utf-8')
    json_write(output/'protocol.json',config)
    downloaded,uncompressed=0,0
    records,checks=[],[]
    session=requests.Session()
    # Basic public image API; no account credentials or private bearer tokens.
    try:
        for candidate in config['candidates']:
            patient,series=candidate['patient_id'],candidate['series_uid']
            response=session.get(config['download_endpoint'],params=dict(SeriesInstanceUID=series,NewFileNames='Yes'),
                timeout=(15,60),stream=True)
            response.raise_for_status()
            archive_path=output/(patient+'.zip')
            with archive_path.open('xb') as handle:
                for chunk in response.iter_content(65536):
                    downloaded+=len(chunk)
                    if downloaded>config['maximum_network_bytes']:
                        response.close()
                        raise ValueError('Network byte cap reached; partial evidence retained')
                    handle.write(chunk)
            instance_rows=[]
            with zipfile.ZipFile(archive_path) as archive:
                infos=[info for info in archive.infolist() if not info.is_dir()]
                if sum(info.file_size for info in infos)+uncompressed>config['maximum_uncompressed_bytes']:
                    raise ValueError('Uncompressed byte cap reached')
                seen=set()
                for info in infos:
                    name=PurePosixPath(info.filename)
                    if name.is_absolute() or '..' in name.parts or '\\' in info.filename or stat.S_ISLNK(info.external_attr>>16):
                        raise ValueError('Unsafe archive member')
                    if name.suffix.lower()!='.dcm':
                        continue
                    content=archive.read(info)
                    uncompressed+=len(content)
                    header=pydicom.dcmread(io.BytesIO(content),stop_before_pixels=True)
                    if str(header.PatientID)!=patient or str(header.SeriesInstanceUID)!=series:
                        raise ValueError('Retrieved header identity differs from proposal')
                    sop=str(header.SOPInstanceUID)
                    study=str(header.StudyInstanceUID)
                    if not re.fullmatch('[0-9.]+',sop) or not re.fullmatch('[0-9.]+',study) or sop in seen:
                        raise ValueError('Duplicate/invalid source instance identity')
                    seen.add(sop)
                    orientation_plan(list(header.ImageOrientationPatient),(int(header.Rows),int(header.Columns)))
                    filename=hashlib.sha256(sop.encode()).hexdigest()[:32]+'.dcm'
                    destination=(raw_root/patient/study/series/filename).resolve()
                    if not destination.is_relative_to(raw_root/patient):
                        raise ValueError('Raw destination escapes patient folder')
                    destination.parent.mkdir(parents=True,exist_ok=True)
                    with destination.open('xb') as handle:
                        handle.write(content)
                    instance_rows.append(dict(patient_id=patient,series_uid=series,sop_uid=sop,
                        source_path=str(destination.relative_to(ROOT)),sha256=hashlib.sha256(content).hexdigest(),
                        rows=int(header.Rows),columns=int(header.Columns),
                        manufacturer=str(header.Manufacturer),field_strength_t=float(header.MagneticFieldStrength),
                        echo_time_ms=float(header.EchoTime),repetition_time_ms=float(header.RepetitionTime),
                        slice_thickness_mm=float(header.SliceThickness),
                        pixel_spacing=json.dumps([float(x) for x in header.PixelSpacing]),
                        iop=json.dumps([float(x) for x in header.ImageOrientationPatient]),
                        ipp=json.dumps([float(x) for x in header.ImagePositionPatient]),
                        acquisition_type=str(header.MRAcquisitionType),scanning_sequence=str(header.ScanningSequence),
                        contrast_bolus_agent=str(getattr(header,'ContrastBolusAgent','')),
                        series_description=str(header.SeriesDescription)))
            if len(instance_rows)!=int(candidate['image_count']):
                raise ValueError('Retrieved instance count differs from metadata')
            frame=pd.DataFrame(instance_rows)
            consistency={key:bool(frame[key].nunique()==1) for key in (
                'rows','columns','manufacturer','field_strength_t','echo_time_ms','repetition_time_ms',
                'pixel_spacing','iop','acquisition_type','scanning_sequence')}
            orientation=np.array(json.loads(frame.iop.iloc[0]))
            normal=np.cross(orientation[:3],orientation[3:])
            positions=np.sort(np.array([json.loads(value) for value in frame.ipp])@normal)
            spacing=np.diff(positions)
            checks.append(dict(patient_id=patient,instances=len(frame),header_consistency=consistency,
                all_geometry_plans_valid=True,unique_projected_positions=bool(np.all(spacing>1e-4)),
                projected_spacing_min_mm=float(spacing.min()),projected_spacing_max_mm=float(spacing.max()),
                contrast_status_confirmed=False,full_protocol_compatibility_established=False,
                admitted_to_training_or_validation=False,archive_sha256=digest(archive_path)))
            records.extend(instance_rows)
            print('RETRIEVED HEADER AUDIT',patient,len(frame),'instances',flush=True)
        csv_write(output/'headers.csv',pd.DataFrame(records))
        json_write(output/'series_checks.json',checks)
        verify_inputs(membership_hashes)
        inputs={proposal.joinpath('proposal.json').relative_to(ROOT).as_posix():digest(proposal/'proposal.json'),
            **membership_hashes,'scripts/v8_generator_dev/retrieve_candidate_series.py':digest(__file__)}
        for row in records:
            inputs[row['source_path']]=row['sha256']
        json_write(output/'inputs.json',inputs)
        json_write(output/'complete.json',dict(complete=True,prospective_raw_patients=2,instances=len(records),
            actual_network_bytes=downloaded,actual_uncompressed_dicom_bytes=uncompressed,
            existing_splits_unchanged=True,pixel_arrays_decoded=False,model_outputs_used=False,
            current_training_changed=False,contrast_status_confirmed=False,
            full_protocol_compatibility_established=False,clinical_anatomy_validated=False))
    except Exception as error:
        json_write(output/'failed.json',dict(complete=False,error=str(error),downloaded_bytes=downloaded,
            extracted_dicom_bytes=uncompressed,existing_splits_changed=False))
        raise
    print('SMALL PROSPECTIVE RETRIEVAL COMPLETE',output,downloaded,'bytes',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('action',choices=['prepare','retrieve'])
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--proposal')
    args=parser.parse_args()
    if args.action=='prepare':
        prepare(args.run_id)
    else:
        retrieve(args.proposal,args.run_id)
