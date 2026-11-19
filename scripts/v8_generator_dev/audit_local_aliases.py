"""Header-only follow-up of nonexact T1 names; no automatic pooling or pixel loading."""
import argparse
import csv
import json
import re

import numpy as np
import pandas as pd
import pydicom

from scripts.v7_geometry.orientation import orientation_plan
from scripts.v8_generator_dev.data import CanonicalDataset
from scripts.v8_generator_dev.io import ROOT, csv_write, digest, json_write, project_path

FIELDS=('ScanningSequence','SequenceVariant','MRAcquisitionType','Manufacturer',
        'MagneticFieldStrength','EchoTime','RepetitionTime','FlipAngle','SliceThickness',
        'PixelSpacing','Rows','Columns','ContrastBolusAgent','ImageType')


def read_header(path):
    ds=pydicom.dcmread(path,stop_before_pixels=True)
    result={name:str(getattr(ds,name,'')) for name in FIELDS}
    result['geometry_losslessly_canonicalisable']=False
    try:
        orientation_plan(list(ds.ImageOrientationPatient),(int(ds.Rows),int(ds.Columns)))
        result['geometry_losslessly_canonicalisable']=True
    except (AttributeError,ValueError):
        pass
    return result


def audit(run_id):
    if not re.fullmatch('[A-Za-z0-9_-]+',run_id):
        raise ValueError('Safe new audit ID required')
    base=ROOT/'reports/v8_generator_dev'
    inventory=base/'unused_local_inventory_20261117_01/unused_series.csv'
    table=pd.read_csv(inventory,keep_default_na=False)
    candidates=table[table.series_description.str.lower().str.contains('t1') &
        ~table.series_description.str.lower().str.contains('+c',regex=False)].copy()
    candidate_keys=set(zip(candidates.patient_id,candidates.series_uid))
    paths={key:[] for key in candidate_keys}
    metadata=ROOT/'data/metadata/dicom_metadata.csv'
    with metadata.open(newline='',encoding='utf-8-sig') as handle:
        for row in csv.DictReader(handle):
            key=(row['PatientID'],row['SeriesInstanceUID'])
            if key in paths:
                paths[key].append(row['slice_path'])
    inputs={p.relative_to(ROOT).as_posix():digest(p) for p in (inventory,metadata)}
    dataset=CanonicalDataset('data/processed/ax_t1_128_canonical/view_v1_20261116_01','train')
    references=[]
    for row in dataset.frame.drop_duplicates('patient_id').itertuples():
        path=project_path(row.source_dicom_path)
        inputs[path.relative_to(ROOT).as_posix()]=digest(path)
        references.append(dict(patient_id=row.patient_id,manufacturer_label=row.manufacturer_label,
            field_strength_label=row.field_strength_label,**read_header(path)))
    reference=pd.DataFrame(references)
    samples=[]
    for row in candidates.itertuples():
        series_paths=sorted(set(paths[(row.patient_id,row.series_uid)]))
        if not series_paths:
            series_paths=[row.representative_path]
        for index in sorted({0,len(series_paths)//2,len(series_paths)-1}):
            path=project_path(series_paths[index])
            inputs[path.relative_to(ROOT).as_posix()]=digest(path)
            samples.append(dict(patient_id=row.patient_id,series_uid=row.series_uid,
                series_description=row.series_description,source_path=str(path.relative_to(ROOT)),
                sample_ordinal=index,total_metadata_instances=len(series_paths),**read_header(path)))
    samples=pd.DataFrame(samples)
    summaries=[]
    for (patient,series),group in samples.groupby(['patient_id','series_uid']):
        first=group.iloc[0]
        manufacturer=0 if 'GE' in first.Manufacturer.upper() else 1 if 'SIEMENS' in first.Manufacturer.upper() else None
        fields=pd.to_numeric(group.MagneticFieldStrength,errors='coerce')
        strength=0 if fields.iloc[0]==1.5 else 1 if fields.iloc[0]==3 else None
        ref=reference[(reference.manufacturer_label==manufacturer)&(reference.field_strength_label==strength)]
        ref_te=pd.to_numeric(ref.EchoTime,errors='coerce')
        ref_tr=pd.to_numeric(ref.RepetitionTime,errors='coerce')
        candidate_te=float(first.EchoTime)
        candidate_tr=float(first.RepetitionTime)
        pulse=bool(first.MRAcquisitionType=='2D' and 'SE' in first.ScanningSequence)
        summaries.append(dict(patient_id=patient,series_uid=series,series_description=first.series_description,
            sampled_instances=len(group),manufacturer_label=manufacturer,field_strength_label=strength,
            echo_time_ms=candidate_te,repetition_time_ms=candidate_tr,
            same_2d_spin_echo_family=pulse,
            sampled_geometry_canonicalisable=bool(group.geometry_losslessly_canonicalisable.all()),
            sampled_TE_consistent=group.EchoTime.nunique()==1,
            explicit_contrast_agent_in_sample=bool(group.ContrastBolusAgent.ne('').any()),
            absent_agent_field_proves_precontrast=False,
            TE_within_current_joint_cell=bool(len(ref) and ref_te.min()<=candidate_te<=ref_te.max()),
            TR_within_current_joint_cell=bool(len(ref) and ref_tr.min()<=candidate_tr<=ref_tr.max()),
            directly_compatible_or_added=False,
            status='requires_full_series_contrast_and_protocol_audit' if pulse else 'different_3d_or_pulse_profile_requires_separate_protocol'))
    output=base/run_id
    output.mkdir(exist_ok=False)
    csv_write(output/'current_training_headers.csv',reference)
    csv_write(output/'candidate_sample_headers.csv',samples)
    csv_write(output/'candidate_compatibility.csv',pd.DataFrame(summaries))
    inputs['scripts/v8_generator_dev/audit_local_aliases.py']=digest(__file__)
    json_write(output/'inputs.json',inputs)
    json_write(output/'complete.json',dict(complete=True,header_only=True,alias_candidate_series=len(candidates),
        sample_headers=len(samples),current_training_patient_headers=len(reference),
        same_2d_spin_echo_family_series=sum(row['same_2d_spin_echo_family'] for row in summaries),
        full_series_compatibility_established=False,contrast_status_established=False,
        candidates_added_to_dataset=False,test_acquisition_fields_used=False,image_pixels_loaded=False))
    (output/'README.md').write_text(
        '# Nonexact local T1 aliases: header-only follow-up\n\nProducer: `python -m scripts.v8_generator_dev.audit_local_aliases`.\n'
        'current_training_headers.csv gives one existing training header per patient; '
        'candidate_sample_headers.csv samples first/middle/last metadata instances of each unused noncontrast-named T1 alias; '
        'candidate_compatibility.csv distinguishes pulse/geometry/TE/TR checks from full compatibility. '
        'inputs.json hashes files/code; complete.json retains unresolved contrast/full-series status. '
        'No pixel array is loaded and no patient is added. A missing contrast-agent field or a familiar '
        'T1 name does not prove a compatible precontrast sequence. 3D gradient-echo cannot silently '
        'replace the current 2D spin-echo profile. Current test patients are not part of this audit.\n',encoding='utf-8')
    print('LOCAL ALIAS HEADER AUDIT',output,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--run-id',required=True)
    audit(parser.parse_args().run_id)
