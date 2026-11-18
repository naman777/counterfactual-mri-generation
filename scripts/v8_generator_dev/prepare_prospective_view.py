"""Versioned unsplit candidate pixel QC, using existing preprocessing functions."""
import argparse
import json
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydicom

from src.data.preprocess import read_dicom_image,percentile_clip_and_normalize,resize_image,compute_foreground_fraction
from scripts.v7_geometry.orientation import orientation_plan,apply_orientation,invert_orientation,resized_pixel_map
from scripts.v8_generator_dev.data import CanonicalDataset
from scripts.v8_generator_dev.io import ROOT,csv_write,digest,json_write,project_path,verify_inputs


def prepare(run_id):
    if not re.fullmatch('[A-Za-z0-9_-]+',run_id):
        raise ValueError('Safe new candidate profile ID required')
    base=ROOT/'reports/v8_generator_dev'
    raw=base/'prospective_series_audit_20261117_01'
    verify_inputs(json.loads((raw/'inputs.json').read_text()))
    headers=pd.read_csv(raw/'headers.csv',keep_default_na=False)
    train=CanonicalDataset('data/processed/ax_t1_128_canonical/view_v1_20261116_01','train').frame
    slope,offset=np.polyfit(train.echo_time,train.echo_time_norm,1)
    used_patients,used_sops=set(),set()
    for split in ('train','val','test'):
        membership=pd.read_csv(ROOT/'data/processed/ax_t1_128/splits'/(split+'.csv'),usecols=['patient_id','sop_uid'])
        used_patients.update(membership.patient_id)
        used_sops.update(membership.sop_uid)
    if set(headers.patient_id)&used_patients or set(headers.sop_uid)&used_sops:
        raise ValueError('Prospective candidates overlap historical splits')
    output=base/run_id
    data=ROOT/'data/processed'/('ax_t1_128_'+run_id)
    output.mkdir(exist_ok=False)
    data.mkdir(exist_ok=False)
    (data/'source_images').mkdir()
    (data/'canonical_images').mkdir()
    json_write(data/'profile.json',dict(format='prospective_candidate_view_v1_not_training_dataset',
        assigned_to_split=False,image_size=128,percentiles=[1,99],source_echo_time_fixed=True,
        echo_transform_slope=float(slope),echo_transform_offset=float(offset),
        source_preprocessing='Existing src.data.preprocess functions; raw patient/series bytes retained',
        contrast_status_confirmed=False,full_protocol_compatibility_established=False,
        historical_train_val_test_modified=False,model_outputs_used=False,clinical_labels_present=False))
    inputs={raw.joinpath('headers.csv').relative_to(ROOT).as_posix():digest(raw/'headers.csv')}
    rows=[]
    for patient,group in headers.groupby('patient_id'):
        # Fixed physical ordering; all source slices retained, no outcome-based selection.
        group=group.assign(z=group.ipp.map(lambda value:json.loads(value)[2])).sort_values(['z','sop_uid'])
        fig,axes=plt.subplots(8,8,figsize=(10,10),constrained_layout=True)
        for ax in axes.flat:
            ax.axis('off')
        for ordinal,row in enumerate(group.itertuples()):
            path=project_path(row.source_path)
            inputs[path.relative_to(ROOT).as_posix()]=row.sha256
            header=pydicom.dcmread(path,stop_before_pixels=True)
            native=read_dicom_image(path)
            image=resize_image(percentile_clip_and_normalize(native,1,99),128)
            if not np.isfinite(image).all() or image.min()<0 or image.max()>1:
                raise ValueError('Invalid prospective image after existing preprocessing')
            transform=orientation_plan(list(header.ImageOrientationPatient),(128,128))
            canonical=apply_orientation(image,transform)
            np.testing.assert_array_equal(image,invert_orientation(canonical,transform))
            stem=patient+'_'+path.stem
            source_path=data/'source_images'/(stem+'.npy')
            canonical_path=data/'canonical_images'/(stem+'.npy')
            for destination,values in ((source_path,image),(canonical_path,canonical)):
                with destination.open('xb') as handle:
                    np.save(handle,values)
            manufacturer=0 if 'GE' in row.manufacturer.upper() else 1 if 'SIEMENS' in row.manufacturer.upper() else None
            strength=0 if row.field_strength_t==1.5 else 1 if row.field_strength_t==3 else None
            if manufacturer is None or strength is None:
                raise ValueError('Unsupported prospective acquisition label')
            norm=float(row.echo_time_ms*slope+offset)
            if not 0<=norm<=1:
                raise ValueError('Candidate exceeds frozen global EchoTime normalisation; no clipping permitted')
            pixel_map=resized_pixel_map(list(header.ImageOrientationPatient),list(header.ImagePositionPatient),
                list(header.PixelSpacing),(int(header.Rows),int(header.Columns)),(128,128),transform)
            rows.append(dict(patient_id=patient,study_uid=str(header.StudyInstanceUID),series_uid=row.series_uid,
                sop_uid=row.sop_uid,physical_order=ordinal,split='unassigned_prospective_candidate',
                manufacturer_label=manufacturer,field_strength_label=strength,echo_time=row.echo_time_ms,
                echo_time_norm=norm,source_image_path=str(source_path.relative_to(ROOT)),
                source_array_sha256=digest(source_path),canonical_image_path=str(canonical_path.relative_to(ROOT)),
                canonical_array_sha256=digest(canonical_path),source_dicom_path=row.source_path,
                source_dicom_sha256=row.sha256,orientation_plan=json.dumps(transform),
                physical_pixel_map=json.dumps(pixel_map),source_iop=row.iop,source_ipp=row.ipp,
                foreground_fraction=compute_foreground_fraction(image),
                contrast_status_confirmed=False,anatomy_validated=False))
            axes.flat[ordinal].imshow(canonical,cmap='gray',vmin=0,vmax=1,interpolation='none')
            axes.flat[ordinal].set_title(str(ordinal),fontsize=7)
        fig.suptitle(patient+' | all prospective source slices in physical order\nPixel QC only; no split assignment or generated/model-selected images',fontsize=11)
        fig.savefig(output/(patient+'_contact.png'),dpi=120)
        plt.close(fig)
    manifest=pd.DataFrame(rows)
    csv_write(data/'manifest.csv',manifest)
    csv_write(output/'pixel_qc.csv',manifest[['patient_id','sop_uid','physical_order','foreground_fraction',
        'source_array_sha256','canonical_array_sha256']])
    for split in ('train','val','test'):
        path=ROOT/'data/processed/ax_t1_128/splits'/(split+'.csv')
        inputs[path.relative_to(ROOT).as_posix()]=digest(path)
    for name in ('scripts/v8_generator_dev/prepare_prospective_view.py','src/data/preprocess.py',
                 'scripts/v7_geometry/orientation.py'):
        inputs[name]=digest(ROOT/name)
    verify_inputs(inputs)
    json_write(output/'inputs.json',inputs)
    json_write(output/'complete.json',dict(complete=True,prospective_patients=manifest.patient_id.nunique(),
        source_images=len(manifest),profile=str(data.relative_to(ROOT)),all_source_roundtrips_exact=True,
        all_images_retained=True,clinical_anatomy_validated=False,contrast_status_confirmed=False,
        admitted_to_current_training_or_validation=False,historical_splits_modified=False,model_outputs_used=False))
    (output/'README.md').write_text(
        'Producer: `python -m scripts.v8_generator_dev.prepare_prospective_view`.\n'
        'The new unassigned profile contains profile.json, manifest.csv and separate source/canonical arrays. '
        'pixel_qc.csv records every source instance/hash/foreground diagnostic; *_contact.png renders every '
        'source in physical order. inputs.json hashes raw bytes, preprocessing and old memberships. '
        'complete.json records pixel/round-trip checks only. Constant/edge slices are not silently dropped; '
        'contrast/protocol/clinical compatibility is unresolved and no current data/model/evaluation changes.\n',encoding='utf-8')
    print('UNASSIGNED PROSPECTIVE PIXEL QC',output,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--run-id',required=True)
    prepare(parser.parse_args().run_id)
