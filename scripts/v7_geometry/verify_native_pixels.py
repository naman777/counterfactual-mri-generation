"""Verify exported images/masks bitwise against their original DICOM pixel arrays."""
import argparse
import json
import re
from pathlib import Path
import numpy as np
import pandas as pd
import pydicom
from scripts.v4_convergence.core import digest,json_write
from scripts.v5_anatomy.run_pilot import project_path
from scripts.v7_geometry.native_annotations import frame_reference,frame_geometry

ROOT=Path(__file__).resolve().parents[2]


def verify(native_report,public_probe,output):
    native_report,public_probe,output=map(project_path,[native_report,public_probe,output])
    if output.exists():raise FileExistsError(output)
    manifest=pd.read_csv(native_report/'annotation_manifest.csv');segs={};rows=[]
    for entry in manifest.itertuples():
        path=project_path(entry.source_seg_path)
        if entry.source_seg_path not in segs:
            ds=pydicom.dcmread(path);pixels=np.asarray(ds.pixel_array)
            if pixels.ndim==2:pixels=pixels[None]
            segs[entry.source_seg_path]=(ds,pixels)
        seg,masks=segs[entry.source_seg_path]
        match=re.search(r'_frame(\d+)\.npz$',entry.array_path)
        if not match:raise ValueError('Missing explicit frame index in artifact filename')
        frame_index=int(match.group(1));frame=seg.PerFrameFunctionalGroupsSequence[frame_index]
        assert frame_reference(frame)==entry.source_sop_uid
        source=pydicom.dcmread(project_path(entry.source_image_path))
        frame_geometry(seg,frame,source)
        original=source.pixel_array.astype(np.float32)
        original=original*float(getattr(source,'RescaleSlope',1.))+float(getattr(source,'RescaleIntercept',0.))
        with np.load(project_path(entry.array_path)) as arrays:
            np.testing.assert_array_equal(arrays['mask'],masks[frame_index])
            np.testing.assert_array_equal(arrays['image'],original)
        rows.append(dict(patient_id=entry.patient_id,source_sop_uid=entry.source_sop_uid,segment_number=entry.segment_number,
            original_seg_frame_index=frame_index,mask_pixels_equal_original_seg=True,
            image_pixels_equal_rescaled_original_mri=True,array_sha256=digest(project_path(entry.array_path))))
    probe=json.loads((public_probe/'complete.json').read_text())
    assert digest(public_probe/'public_seg.dcm')==probe['seg_sha256']
    output.mkdir(parents=True,exist_ok=False);pd.DataFrame(rows).to_csv(output/'pixel_fidelity.csv',index=False)
    json_write(output/'complete.json',dict(complete=True,annotation_frames_checked=len(rows),
        all_masks_bitwise_equal_source_seg=True,all_images_equal_declared_source_mri=True,
        public_seg_source_hash_verified=True,generated_output_clinical_validation=False,
        script_sha256=digest(__file__)))
    (output/'README.md').write_text('# Native annotation pixel fidelity\n\n'
        'Producer: `python -m scripts.v7_geometry.verify_native_pixels`.\n'
        'pixel_fidelity.csv maps each exported frame back to its original SEG index and source MRI, with exact pixel comparisons.\n'
        'complete.json records count, public SEG hash verification and limits. No resampling or geometry change is performed.\n'
        'These checks establish export fidelity, not independent review of generated anatomy.\n',encoding='utf-8')
    return output


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__)
    for name in ['native-report','public-probe','output']:p.add_argument('--'+name,required=True)
    a=p.parse_args();print(verify(a.native_report,a.public_probe,a.output))
