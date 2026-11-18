"""Create an append-only, lazy canonical train/validation view from DICOM headers."""
import argparse
import json
import re
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
import pydicom
from scripts.v4_convergence.core import digest,json_write
from scripts.v5_anatomy.run_pilot import project_path,read_splits
from scripts.v7_geometry.orientation import orientation_plan,resized_pixel_map,apply_orientation,invert_orientation
from scripts.v7_geometry.support import request_support

ROOT=Path(__file__).resolve().parents[2]


def load_canonical(row,verify=True):
    path=project_path(row['source_image_path'])
    if verify and digest(path)!=row['source_array_sha256']:raise ValueError('Source array changed')
    source=np.load(path,allow_pickle=False)
    plan=json.loads(row['orientation_plan'])
    return apply_orientation(source,plan),plan


def prepare(config,run_id):
    if not re.fullmatch('[A-Za-z0-9_-]+',run_id):raise ValueError('Invalid version ID')
    config=project_path(config);cfg=json.loads(config.read_text())
    if cfg['splits']!=['train','val'] or cfg['test_images_permitted']:
        raise ValueError('This development view is restricted to train/val')
    splits=read_splits()
    data=ROOT/'data/processed/ax_t1_128_canonical'/run_id
    report=ROOT/'reports/v7_geometry'/run_id
    if data.exists() or report.exists():raise FileExistsError('Use a new version ID')
    data.mkdir(parents=True);(data/'splits').mkdir();report.mkdir(parents=True);(report/'code').mkdir()
    inputs={str(config.relative_to(ROOT)):digest(config)}
    for path in sorted(Path(__file__).parent.glob('*.py')):
        inputs[str(path.relative_to(ROOT))]=digest(path);shutil.copyfile(path,report/'code'/path.name)
    json_write(report/'protocol.json',dict(**cfg,run_id=run_id,
        standard='https://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_C.7.6.2.html',
        grid_mapping='DICOM first IOP triplet follows increasing column index; second follows increasing row index',
        no_causal_or_anatomy_validation_claim=True))
    (report/'README.md').write_text('# Reversible canonical source view\n\n'
        'Producer: `python -m scripts.v7_geometry.prepare_view`.\n'
        'protocol.json declares geometry policy; inputs.json fingerprints config, splits, code and immutable sources.\n'
        'geometry_manifest.csv records DICOM-derived transforms/physical maps and source hashes. rejected.csv retains failures.\n'
        'support_requests.csv diagnoses fixed-source-EchoTime opposite-label requests using training metadata only.\n'
        'summary.json records counts and limitations; code/ snapshots implementation.\n'
        'The new lazy dataset index is under data/processed/ax_t1_128_canonical/<version>/splits/.\n'
        'No historical image, report or checkpoint is changed; no test image or annotation is loaded.\n'
        'Canonicalisation is not registration, scanner invariance or validation of old model weights on a new image frame.\n',encoding='utf-8')
    records=[];rejected=[];support=[]
    for split in cfg['splits']:
        index_path=ROOT/f'data/processed/ax_t1_128/splits/{split}.csv';inputs[str(index_path.relative_to(ROOT))]=digest(index_path)
        for i,(_,row) in enumerate(splits[split].iterrows()):
            source=project_path(row.image_path);header_path=project_path(row.slice_path)
            try:
                ds=pydicom.dcmread(header_path,stop_before_pixels=True)
                if str(getattr(ds,'AnatomicalOrientationType','BIPED'))!='BIPED':raise ValueError('Only biped patient coordinates supported')
                if str(ds.SOPInstanceUID)!=str(row.sop_uid) or str(ds.SeriesInstanceUID)!=str(row.series_uid):
                    raise ValueError('DICOM/reference UID mismatch')
                image=np.load(source,allow_pickle=False)
                if image.shape!=(cfg['image_size'],cfg['image_size']) or not np.isfinite(image).all():
                    raise ValueError('Invalid preprocessed source grid')
                plan=orientation_plan(ds.ImageOrientationPatient,image.shape,cfg['orientation_tolerance'])
                physical=resized_pixel_map(ds.ImageOrientationPatient,ds.ImagePositionPatient,ds.PixelSpacing,
                                           (int(ds.Rows),int(ds.Columns)),image.shape,plan)
                np.testing.assert_array_equal(image,invert_orientation(apply_orientation(image,plan),plan))
                array_hash,header_hash=digest(source),digest(header_path)
                record=row.to_dict();record.pop('image_path');record.pop('slice_path')
                record.update(split=split,source_image_path=str(source.relative_to(ROOT)),source_dicom_path=str(header_path.relative_to(ROOT)),
                    source_array_sha256=array_hash,source_dicom_sha256=header_hash,
                    orientation_plan=json.dumps(plan,separators=(',',':')),
                    physical_pixel_map=json.dumps(physical,separators=(',',':')),
                    source_iop=json.dumps(list(map(float,ds.ImageOrientationPatient))),
                    source_ipp=json.dumps(list(map(float,ds.ImagePositionPatient))),
                    native_rows=int(ds.Rows),native_columns=int(ds.Columns),
                    native_row_spacing_mm=float(ds.PixelSpacing[0]),native_column_spacing_mm=float(ds.PixelSpacing[1]),
                    transform='transpose' if plan['transpose'] else 'identity_axes',
                    flip_rows=plan['flip_rows'],flip_columns=plan['flip_columns'],
                    canonical_row_spacing_mm=physical['canonical_row_spacing_mm'],canonical_column_spacing_mm=physical['canonical_column_spacing_mm'],
                    frame_of_reference_uid=str(getattr(ds,'FrameOfReferenceUID','')),
                    acquisition_matrix=str(getattr(ds,'AcquisitionMatrix','')),pixel_bandwidth=getattr(ds,'PixelBandwidth',None),
                    scanner_model=str(getattr(ds,'ManufacturerModelName','')),anatomy_validated=False)
                records.append(record)
                inputs[str(source.relative_to(ROOT))]=array_hash;inputs[str(header_path.relative_to(ROOT))]=header_hash
            except (ValueError,AttributeError,AssertionError) as error:
                rejected.append(dict(split=split,patient_id=row.patient_id,source_image_path=row.image_path,
                                     reason=str(error),included_in_view=False))
            if (i+1)%500==0:print(split,i+1,'headers indexed',flush=True)
    manifest=pd.DataFrame(records)
    if manifest.empty:raise ValueError('No geometry-compatible images')
    manifest.to_csv(report/'geometry_manifest.csv',index=False)
    pd.DataFrame(rejected,columns=['split','patient_id','source_image_path','reason','included_in_view']).to_csv(report/'rejected.csv',index=False)
    for split in cfg['splits']:
        manifest[manifest.split==split].to_csv(data/'splits'/f'{split}.csv',index=False)
    # All supported validation sources are screened, not selected by output quality.
    for row in manifest[manifest.split=='val'].itertuples():
        for axis in ['manufacturer','field_strength']:
            mfg=int(row.manufacturer_label);fs=int(row.field_strength_label)
            if axis=='manufacturer':mfg=1-mfg
            else:fs=1-fs
            support.append(dict(patient_id=row.patient_id,sop_uid=row.sop_uid,axis=axis,
                target_manufacturer=mfg,target_field_strength=fs,source_echo_time_norm=float(row.echo_time_norm),
                target_echo_time_norm=float(row.echo_time_norm),
                **request_support(splits['train'],mfg,fs,float(row.echo_time_norm),cfg['support_min_patients'])))
    pd.DataFrame(support).to_csv(report/'support_requests.csv',index=False)
    json_write(report/'inputs.json',inputs)
    json_write(data/'view.json',dict(format='canonical_axial_view_v1',report=str(report.relative_to(ROOT)),
        image_column='source_image_path',loader='scripts.v7_geometry.prepare_view.load_canonical',
        warning='Never feed source_image_path into a legacy image loader and call it canonical; apply the recorded transform',
        historical_arrays_unchanged=True))
    (data/'README.md').write_text('Versioned lazy geometry view; generated by scripts/v7_geometry/prepare_view.py.\n'
        'splits/*.csv references immutable original arrays plus a reversible header-derived transform.\n'
        'Use load_canonical; old loaders intentionally have no image_path column to consume.\n'
        'view.json records format and report. No new scanner labels, cropping, interpolation or model compatibility claim.\n',encoding='utf-8')
    summary=dict(complete=True,run_id=run_id,view=str(data.relative_to(ROOT)),included=len(records),rejected=len(rejected),
        rows_per_split=manifest.groupby('split').size().to_dict(),patients_per_split=manifest.groupby('split').patient_id.nunique().to_dict(),
        transform_counts=manifest.groupby(['flip_rows','flip_columns','transform']).size().reset_index(name='images').to_dict('records'),
        exact_pixel_roundtrip_checked_every_included_image=True,test_images_used=False,
        preprocessing_reproduction_audit_pending=True,old_model_validated_on_view=False,scanner_invariance_established=False)
    json_write(report/'summary.json',summary);print(summary,flush=True)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--config',default='configs/geometry_view_v1.json');p.add_argument('--run-id',required=True)
    a=p.parse_args();print(prepare(a.config,a.run_id))
