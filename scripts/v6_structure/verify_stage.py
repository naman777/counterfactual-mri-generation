"""Verify stage-2 artifacts without treating engineering checks as efficacy."""
import argparse
import json
import re
import shutil
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from scripts.v4_convergence.core import digest,json_write
from scripts.v5_anatomy.csv_io import read_table
from scripts.v5_anatomy.run_pilot import project_path

ROOT=Path(__file__).resolve().parents[2]


def verify(training,inference,audit,resources,output):
    training,inference,audit,resources,output=map(project_path,[training,inference,audit,resources,output])
    if output.exists():raise FileExistsError(output)
    for folder in [training,inference,audit,resources]:
        for path,expected in json.loads((folder/'inputs.json').read_text()).items():
            assert digest(ROOT/path)==expected,path
    trained=json.loads((training/'complete.json').read_text());tested=json.loads((inference/'complete.json').read_text())
    assert trained['base_weights_unchanged'] and trained['conditioner_unchanged'] and trained['engineering_smoke']
    assert not trained['efficacy_established'] and not tested['expert_anatomy_validated']
    checkpoint=project_path(trained['checkpoint']);assert digest(checkpoint)==trained['checkpoint_sha256']==tested['checkpoint_sha256']
    selection=read_table(inference/'selection.csv')
    val=read_table(ROOT/'data/processed/ax_t1_128/splits/val.csv')
    assert set(selection.patient_id)<=set(val.patient_id)
    generation=read_table(inference/'generation.csv');predictions=read_table(inference/'predictions.csv')
    assert len(generation)==tested['output_images'] and len(predictions)==3*len(generation)
    assert generation.human_preservation_pass.isna().all()
    assert predictions.target_hit_and_human_preservation.isna().all()
    assert not generation.anatomy_validated.any()
    assert np.array_equal(generation.source_echo,generation.target_echo)
    for case,group in generation.groupby('case_id'):
        assert group.seed.nunique()==group.timestep.nunique()==group.guidance.nunique()==1
        with np.load(inference/'arrays'/f'{case}.npz') as arrays:
            np.testing.assert_array_equal(arrays['reference'],arrays['guide_absent'])
            src=np.load(project_path(group.iloc[0].source_path)).astype(np.float32)
            np.testing.assert_array_equal(src,arrays['source'])
            for key in arrays.files:
                assert arrays[key].shape==(128,128) and np.isfinite(arrays[key]).all()
                assert arrays[key].min()>=0 and arrays[key].max()<=1
        wrong=group[group.variant=='wrong_source_guide'].iloc[0]
        assert wrong.wrong_source_patient!=wrong.patient_id
    catalog=json.loads((resources/'source_catalog.json').read_text())
    assert not catalog['reviewer_recruited'] and not catalog['direct_labels_for_current_ax_t1_verified']
    excluded=read_table(resources/'excluded_patients.csv')
    assert not excluded.patient_id.duplicated().any()
    output.mkdir(parents=True);(output/'code').mkdir()
    from scripts.v6_structure import test_model
    suite=unittest.defaultTestLoader.loadTestsFromModule(test_model)
    with (output/'tests.log').open('x',encoding='utf-8') as stream:
        result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    if not result.wasSuccessful():raise AssertionError('Model tests failed; see log')
    for path in Path(__file__).parent.glob('*.py'):shutil.copyfile(path,output/'code'/path.name)
    hashes={str(path.relative_to(ROOT)):digest(path) for folder in [training,inference,audit,resources]
            for path in folder.glob('*.json')}
    for path in (inference/'arrays').glob('*.npz'):hashes[str(path.relative_to(ROOT))]=digest(path)
    json_write(output/'artifact_hashes.json',hashes)
    json_write(output/'verification.json',dict(software_tests_run=result.testsRun,software_tests_pass=True,
        frozen_inputs_verified=True,base_and_conditioner_unchanged=True,adapter_checkpoint_verified=True,
        validation_patients=len(selection),validation_images=len(generation),judge_rows=len(predictions),
        absent_guide_bitwise_equivalent=True,wrong_guides_from_distinct_patients=True,echo_fixed=True,
        expert_review_not_invented=True,untouched_evaluation_exclusion_patients=len(excluded),
        representation_scanner_invariance_established=False,clinical_or_efficacy_validation=False))
    (output/'README.md').write_text('# Stage-2 software and provenance checks\n\n'
        'Producer: `python -m scripts.v6_structure.verify_stage`.\n'
        'tests.log records model/gradient/copy-control tests. verification.json checks actual frozen input hashes,\n'
        'adapter checkpoint, split membership, matched seeds, missing human labels and bitwise absent-guide recovery.\n'
        'artifact_hashes.json fingerprints output artifacts; code/ preserves final scripts.\n'
        'No success in this check establishes efficacy, scanner-invariant anatomy or qualified reviewer validation.\n',encoding='utf-8')
    return output


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__)
    for name in ['training','inference','audit','resources','output']:p.add_argument('--'+name,required=True)
    a=p.parse_args();print(verify(a.training,a.inference,a.audit,a.resources,a.output))
