"""Assemble local official MANUAL DCE SEG frames on their exact native MRI grids.

Only original train/validation patients; no new image download, resampling,
registration, whole-breast inference, current-ax-t1 label transfer or test data.
"""
import argparse
import json
import re
from pathlib import Path
import numpy as np
import pandas as pd
import pydicom
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts.v4_convergence.core import digest,json_write
from scripts.v5_anatomy.run_pilot import project_path,read_splits
from scripts.v7_geometry.annotation_compatibility import seg_references

ROOT=Path(__file__).resolve().parents[2]
CATALOG=ROOT/'reports/v4_downstream/mask_feasibility_20261016_01/official_2d_segmentation_catalog_matches.csv'


def frame_reference(frame):
    refs=[]
    for derivation in getattr(frame,'DerivationImageSequence',[]):
        refs.extend(str(item.ReferencedSOPInstanceUID) for item in getattr(derivation,'SourceImageSequence',[]))
    refs=sorted(set(refs))
    if len(refs)!=1:raise ValueError('A SEG frame must reference exactly one source SOP instance')
    return refs[0]


def frame_geometry(seg,frame,source):
    shared=seg.SharedFunctionalGroupsSequence[0]
    orient=getattr(frame,'PlaneOrientationSequence',getattr(shared,'PlaneOrientationSequence',None))
    measures=getattr(frame,'PixelMeasuresSequence',getattr(shared,'PixelMeasuresSequence',None))
    if orient is None or measures is None or not hasattr(frame,'PlanePositionSequence'):
        raise ValueError('Missing per-frame/shared source geometry')
    iop=np.asarray(orient[0].ImageOrientationPatient,float)
    spacing=np.asarray(measures[0].PixelSpacing,float)
    position=np.asarray(frame.PlanePositionSequence[0].ImagePositionPatient,float)
    if int(seg.Rows)!=int(source.Rows) or int(seg.Columns)!=int(source.Columns):raise ValueError('Native image grid mismatch')
    errors=dict(orientation_error=float(np.max(np.abs(iop-np.asarray(source.ImageOrientationPatient,float)))),
                spacing_error_mm=float(np.max(np.abs(spacing-np.asarray(source.PixelSpacing,float)))),
                position_error_mm=float(np.max(np.abs(position-np.asarray(source.ImagePositionPatient,float)))))
    if errors['orientation_error']>1e-5 or errors['spacing_error_mm']>1e-4 or errors['position_error_mm']>1e-3:
        raise ValueError('Native geometry mismatch: '+str(errors))
    return errors


def assemble(run_id):
    if not re.fullmatch('[A-Za-z0-9_-]+',run_id):raise ValueError('Invalid version ID')
    report=ROOT/'reports/v7_geometry'/run_id;data=ROOT/'data/processed/duke_dce_manual_native'/run_id
    if report.exists() or data.exists():raise FileExistsError('Use a new version ID')
    splits=read_splits();allowed=pd.concat([splits[s].assign(original_split=s) for s in ['train','val']])
    patients=allowed.drop_duplicates('patient_id').set_index('patient_id')
    catalog=pd.read_csv(CATALOG);catalog=catalog[catalog.PatientID.isin(patients.index)].sort_values('PatientID')
    report.mkdir(parents=True);data.mkdir(parents=True);(data/'frames').mkdir();(report/'qa').mkdir()
    inputs={str(CATALOG.relative_to(ROOT)):digest(CATALOG)};records=[];rejected=[]
    json_write(report/'protocol.json',dict(selection='all locally available official SEG entries overlapping original train/val patients',
        require_algorithm_type='MANUAL',require_single_per_frame_source_sop=True,
        orientation_tolerance=1e-5,pixel_spacing_tolerance_mm=1e-4,position_tolerance_mm=1e-3,
        resize=False,registration=False,test_images_allowed=False,current_ax_t1_label_transfer=False,
        masks_for_generated_outputs=False,expert_review_of_our_outputs_completed=False))
    for _,entry in catalog.iterrows():
        patient=str(entry.PatientID);root=ROOT/'data/raw/duke_breast_cancer_mri'/patient
        folders=list(root.glob('*/SEG_'+str(entry.SeriesInstanceUID)))
        if len(folders)!=1:
            rejected.append(dict(patient_id=patient,reason='Official SEG not uniquely available locally',frame=None));continue
        segfiles=list(folders[0].glob('*.dcm'))
        if len(segfiles)!=1:
            rejected.append(dict(patient_id=patient,reason='Expected one local SEG instance',frame=None));continue
        segpath=segfiles[0];seg=pydicom.dcmread(segpath)
        if str(seg.SeriesInstanceUID)!=str(entry.SeriesInstanceUID):raise ValueError('Official SEG series mismatch')
        inputs[str(segpath.relative_to(ROOT))]=digest(segpath)
        references=seg_references(seg);source_index={}
        for uid in references['referenced_series_uids']:
            for folder in root.glob('*/MR_'+uid):
                for path in folder.glob('*.dcm'):
                    header=pydicom.dcmread(path,stop_before_pixels=True)
                    source_index[str(header.SOPInstanceUID)]=(path,header)
        masks=np.asarray(seg.pixel_array)
        if masks.ndim==2:masks=masks[None]
        if len(masks)!=len(seg.PerFrameFunctionalGroupsSequence):raise ValueError('SEG frame count mismatch')
        definitions={int(s.SegmentNumber):s for s in seg.SegmentSequence}
        patient_records=[]
        for i,frame in enumerate(seg.PerFrameFunctionalGroupsSequence):
            try:
                sop=frame_reference(frame)
                if sop not in source_index:raise ValueError('Referenced source SOP not available in local referenced series')
                sourcepath,header=source_index[sop]
                if str(header.PatientID)!=patient:raise ValueError('Patient ID mismatch')
                number=int(frame.SegmentIdentificationSequence[0].ReferencedSegmentNumber)
                definition=definitions[number]
                if str(definition.SegmentAlgorithmType)!='MANUAL':raise ValueError('Annotation is not declared MANUAL')
                errors=frame_geometry(seg,frame,header)
                mask=masks[i]
                if not np.isin(mask,[0,1]).all() or not mask.any():raise ValueError('Mask is empty or not binary')
                source=pydicom.dcmread(sourcepath)
                image=source.pixel_array.astype(np.float32)
                image=image*float(getattr(source,'RescaleSlope',1.))+float(getattr(source,'RescaleIntercept',0.))
                if image.shape!=mask.shape or not np.isfinite(image).all():raise ValueError('Invalid native source pixels')
                name=f'{patient}_segment{number:02d}_frame{i:03d}.npz';path=data/'frames'/name
                np.savez_compressed(path,image=image,mask=mask.astype(np.uint8))
                inputs[str(sourcepath.relative_to(ROOT))]=digest(sourcepath)
                record=dict(patient_id=patient,split=patients.loc[patient].original_split,
                    source_sop_uid=sop,source_series_uid=str(header.SeriesInstanceUID),
                    source_series_description=str(getattr(header,'SeriesDescription','')),
                    source_image_path=str(sourcepath.relative_to(ROOT)),source_seg_path=str(segpath.relative_to(ROOT)),
                    array_path=str(path.relative_to(ROOT)),array_sha256=digest(path),
                    segment_number=number,segment_label=str(definition.SegmentLabel),algorithm_type='MANUAL',
                    native_rows=int(header.Rows),native_columns=int(header.Columns),foreground_pixels=int(mask.sum()),
                    pixel_spacing=json.dumps(list(map(float,header.PixelSpacing))),
                    image_orientation_patient=json.dumps(list(map(float,header.ImageOrientationPatient))),
                    image_position_patient=json.dumps(list(map(float,header.ImagePositionPatient))),
                    echo_time_ms=float(getattr(header,'EchoTime',np.nan)),
                    source_reference_verified=True,native_geometry_verified=True,
                    current_ax_t1_sop_match=sop in set(allowed.sop_uid.astype(str)),
                    usable_as_current_ax_t1_ground_truth=False,
                    expert_annotation_of_generated_output=False,**errors)
                records.append(record);patient_records.append((image,mask,record))
            except (ValueError,AttributeError,KeyError) as error:
                rejected.append(dict(patient_id=patient,frame=i,reason=str(error)))
        if patient_records:
            fig,axes=plt.subplots(1,len(patient_records),figsize=(3*len(patient_records),3.8),squeeze=False)
            for column,(image,mask,record) in enumerate(patient_records):
                lo,hi=np.percentile(image,[1,99]);shown=np.clip((image-lo)/max(hi-lo,1e-8),0,1)
                axes[0,column].imshow(shown,cmap='gray',vmin=0,vmax=1,interpolation='none')
                axes[0,column].contour(mask,levels=[.5],colors=['#ffd34e'],linewidths=.7)
                axes[0,column].set_title(f'Segment {record["segment_number"]}\n{record["segment_label"]}',fontsize=8)
                axes[0,column].axis('off')
            fig.suptitle(f'{patient}: official MANUAL labels on exact native referenced DCE images; no ax-t1 transfer',fontsize=10)
            fig.tight_layout(rect=[0,.02,1,.88]);fig.savefig(report/'qa'/f'{patient}.png',dpi=120);plt.close(fig)
        print(patient,'verified frames',len(patient_records),flush=True)
    frame=pd.DataFrame(records)
    frame.to_csv(report/'annotation_manifest.csv',index=False)
    pd.DataFrame(rejected,columns=['patient_id','frame','reason']).to_csv(report/'rejected.csv',index=False)
    if not frame.empty:
        for split in ['train','val']:frame[frame.split==split].to_csv(data/f'{split}.csv',index=False)
    inputs[str(Path(__file__).relative_to(ROOT))]=digest(__file__);json_write(report/'inputs.json',inputs)
    json_write(report/'summary.json',dict(complete=True,verified_annotation_frames=len(records),rejected_frames=len(rejected),
        patients=int(frame.patient_id.nunique()) if not frame.empty else 0,
        patients_per_split=frame.groupby('split').patient_id.nunique().to_dict() if not frame.empty else {},
        source_instances=int(frame.source_sop_uid.nunique()) if not frame.empty else 0,
        existing_ax_t1_matches=int(frame.current_ax_t1_sop_match.sum()) if not frame.empty else 0,
        dataset=str(data.relative_to(ROOT)),native_reference_geometry_verified=True,
        test_images_used=False,extra_mri_downloads=False,generated_output_expert_review=False,
        current_editor_preservation_validated=False))
    (data/'README.md').write_text('Native-grid official MANUAL DCE annotation subset; generated by scripts/v7_geometry/native_annotations.py.\n'
        'frames/*.npz contains rescaled native source image and the exact referenced binary SEG frame, without resize or registration.\n'
        'train.csv/val.csv preserve original patient split. See the reports/v7_geometry version for provenance and limitations.\n'
        'These are selected tissue labels, not automatically whole-breast or tumour masks, and not labels for current ax-t1/generated outputs.\n',encoding='utf-8')
    (report/'README.md').write_text('# Verified local native DCE annotation subset\n\n'
        'Producer: `python -m scripts.v7_geometry.native_annotations`.\n'
        'protocol.json declares fail-closed references/geometry/algorithm checks. annotation_manifest.csv maps source/SEG UIDs, labels, grids and output hashes.\n'
        'rejected.csv retains failures; inputs.json hashes source data; summary.json reports verified coverage and limitations.\n'
        'qa/*.png overlays each accepted mask on its own native referenced image; display-only percentile normalisation does not change saved pixels.\n'
        'Data files are under data/processed/duke_dce_manual_native/<version>/. No test images, extra MRI downloads or existing-data changes.\n'
        'MANUAL status and official collection provenance do not mean our generated images have been reviewed. Tissue semantics remain distinct.\n',encoding='utf-8')
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--run-id',required=True);print(assemble(p.parse_args().run_id))
