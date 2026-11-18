"""Stage-3 data integrity, source/geometry checks and next-data readiness profile."""
import argparse
import json
import shutil
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
import pydicom
from scripts.v4_convergence.core import digest,json_write
from scripts.v5_anatomy.csv_io import read_table
from scripts.v5_anatomy.run_pilot import project_path,read_splits
from scripts.v7_geometry.orientation import invert_orientation
from scripts.v7_geometry.prepare_view import load_canonical

ROOT=Path(__file__).resolve().parents[2]


def verify(view_report,audit,native,seg_probe,output):
    view_report,audit,native,seg_probe,output=map(project_path,[view_report,audit,native,seg_probe,output])
    if output.exists():raise FileExistsError(output)
    checked=0
    for folder in [view_report,audit,native,seg_probe]:
        for path,expected in json.loads((folder/'inputs.json').read_text()).items():
            assert digest(ROOT/path)==expected,path;checked+=1
    summary=json.loads((view_report/'summary.json').read_text())
    view=project_path(summary['view'])
    source_splits=read_splits()
    manifests={s:read_table(view/'splits'/f'{s}.csv') for s in ['train','val']}
    for split,manifest in manifests.items():
        assert 'image_path' not in manifest.columns
        assert set(manifest.patient_id)<=set(source_splits[split].patient_id)
        assert set(manifest.patient_id).isdisjoint(source_splits['test'].patient_id)
        for _,row in manifest.iterrows():
            canonical,plan=load_canonical(row)
            np.testing.assert_array_equal(invert_orientation(canonical,plan),np.load(project_path(row.source_image_path)))
    audited=json.loads((audit/'complete.json').read_text())
    assert audited['source_preprocessing_reproduced_on_audit_samples']
    support=read_table(view_report/'support_requests.csv')
    np.testing.assert_array_equal(support.source_echo_time_norm,support.target_echo_time_norm)
    assert not support.clinical_approval.any() and not support.causal_edit_identified.any()
    native_summary=json.loads((native/'summary.json').read_text());annotations=read_table(native/'annotation_manifest.csv')
    profiles=[]
    for row in annotations.itertuples():
        assert row.algorithm_type=='MANUAL' and row.native_geometry_verified and row.source_reference_verified
        assert row.patient_id in set(source_splits[row.split].patient_id)
        assert row.patient_id not in set(source_splits['test'].patient_id)
        assert digest(project_path(row.array_path))==row.array_sha256
        with np.load(project_path(row.array_path)) as arrays:
            assert arrays['image'].shape==arrays['mask'].shape==(row.native_rows,row.native_columns)
            assert np.isin(arrays['mask'],[0,1]).all() and arrays['mask'].any()
        header=pydicom.dcmread(project_path(row.source_image_path),stop_before_pixels=True)
        assert str(header.SOPInstanceUID)==row.source_sop_uid
        profiles.append(dict(patient_id=row.patient_id,split=row.split,source_sop_uid=row.source_sop_uid,
            source_series_uid=row.source_series_uid,source_series_description=row.source_series_description,
            segment_label=row.segment_label,manufacturer=str(getattr(header,'Manufacturer','unknown')),
            magnetic_field_strength=float(getattr(header,'MagneticFieldStrength',np.nan)),
            echo_time_ms=float(getattr(header,'EchoTime',np.nan)),
            temporal_position_identifier=str(getattr(header,'TemporalPositionIdentifier','unknown')),
            contrast_phase_confirmed=False))
    output.mkdir(parents=True);(output/'code').mkdir()
    profile=pd.DataFrame(profiles);profile.to_csv(output/'native_acquisition_profile.csv',index=False)
    patients=profile.drop_duplicates('patient_id')
    patients.groupby(['split','manufacturer','magnetic_field_strength']).size().reset_index(name='patients').to_csv(
        output/'native_split_coverage.csv',index=False)
    support.groupby(['axis','condition_support_status']).agg(images=('patient_id','size'),patients=('patient_id','nunique')).reset_index().to_csv(
        output/'support_coverage.csv',index=False)
    # Reserve candidates by metadata/directory presence only: do not inspect
    # their SEG frames or MRI pixels and silently consume a new test cohort.
    catalog=read_table(ROOT/'reports/v4_downstream/mask_feasibility_20261016_01/official_2d_segmentation_catalog_matches.csv')
    excluded=read_table(ROOT/'reports/v6_structure/annotation_sources_20261116_01/excluded_patients.csv')
    candidates=catalog[~catalog.PatientID.isin(excluded.patient_id)]
    reserves=[]
    for row in candidates.itertuples():
        folders=list((ROOT/'data/raw/duke_breast_cancer_mri'/row.PatientID).glob('*/SEG_'+row.SeriesInstanceUID))
        reserves.append(dict(patient_id=row.PatientID,seg_series_uid=row.SeriesInstanceUID,
            local_seg_directory_exists=bool(folders),not_in_recorded_original_or_judge_splits=True,
            pixels_or_annotations_inspected=False,final_test_eligibility_verified=False,
            status='reserve_candidate_only_not_a_frozen_or_verified_final_cohort'))
    pd.DataFrame(reserves).to_csv(output/'annotation_reserve_candidates.csv',index=False)
    from scripts.v7_geometry import test_geometry,test_annotations
    suite=unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromModule(m) for m in [test_geometry,test_annotations]])
    with (output/'tests.log').open('x',encoding='utf-8') as stream:
        tests=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    if not tests.wasSuccessful():raise AssertionError('Software checks failed')
    for path in Path(__file__).parent.glob('*.py'):shutil.copyfile(path,output/'code'/path.name)
    hashes={str(path.relative_to(ROOT)):digest(path) for folder in [view_report,audit,native,seg_probe]
            for path in folder.glob('*.json')}
    json_write(output/'artifact_hashes.json',hashes)
    json_write(output/'verification.json',dict(software_tests=tests.testsRun,all_software_tests_pass=True,
        frozen_input_hash_checks=checked,source_image_roundtrips=sum(len(f) for f in manifests.values()),
        source_preprocessing_audit_samples=audited['preprocessing_samples_checked'],
        native_manual_frames=len(annotations),native_annotation_patients=annotations.patient_id.nunique(),
        native_annotation_source_instances=annotations.source_sop_uid.nunique(),
        exact_current_ax_t1_annotation_matches=int(annotations.current_ax_t1_sop_match.sum()),
        local_uninspected_reserve_candidates=sum(r['local_seg_directory_exists'] for r in reserves),
        test_pixels_used=False,old_data_or_models_rewritten=False,scanner_invariance_established=False,
        clinical_preservation_established=False))
    (output/'README.md').write_text('# Geometry/annotation readiness verification\n\n'
        'Producer: `python -m scripts.v7_geometry.verify_stage`.\n'
        'verification.json and tests.log check exact source transforms, frozen input hashes, split separation, native MANUAL labels and source references.\n'
        'artifact_hashes.json fingerprints reports; code/ snapshots current implementation.\n'
        'native_acquisition_profile.csv/native_split_coverage.csv expose sequence and scanner coverage of the separate DCE annotation subset.\n'
        'support_coverage.csv retains target/EchoTime support limitations. annotation_reserve_candidates.csv identifies metadata-only reserve candidates,\n'
        'not a verified or frozen final test cohort. No reserve pixels/annotations were inspected.\n'
        'Native DCE annotations are not labels for ax-t1 or generated images. This is data/software verification, not clinical validation.\n',encoding='utf-8')
    return output


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__)
    for name in ['view-report','audit','native','seg-probe','output']:p.add_argument('--'+name,required=True)
    a=p.parse_args();print(verify(a.view_report,a.audit,a.native,a.seg_probe,a.output))
