"""Declare the local diagnostic plan after the frozen primary outcome is known."""
import argparse
import json

from scripts.v8_generator_dev.io import ROOT,digest,json_write,verify_inputs
from scripts.v14_dropout_audit.common import METRICS


def prepare(run_id):
    out=ROOT/'reports/v14_dropout_audit'/run_id;out.mkdir(parents=True,exist_ok=False)
    (out/'README.md').write_text('Producer: scripts.v14_dropout_audit.prepare. protocol.json: '
        'post-hoc diagnostic scope, every contrast/control/stratum and fixed uncertainty plan; '
        'inputs.json: completed receipts, raw grading provenance and source identities; '
        'complete.json: planning only. No cloud call, training, test pixels or changed advancement rule.\n',encoding='utf-8')
    prior=ROOT/'reports/v13_guide_dropout'
    verify_inputs(json.loads((prior/'final_review_20261118_01/inputs.json').read_text()))
    verify_inputs(json.loads((prior/'analysis_20261118_01/inputs.json').read_text()))
    p=dict(format='dropout_failure_diagnosis_v1',post_hoc=True,primary_outcome_already_known=True,
        old_receipt='reports/v10_detail/results_20261117_01/complete.json',
        new_receipt='reports/v13_guide_dropout/results_20261118_01/complete.json',
        scientific_protocol='reports/v13_guide_dropout/preparation_20261118_01/protocol.json',
        frozen_advancement='reports/v13_guide_dropout/analysis_20261118_01/advancement.json',
        training='reports/v13_guide_dropout/guide_dropout_20261118_01_training',
        training_verified='reports/v13_guide_dropout/training_verified_20261118_01/complete.json',
        selection='reports/v13_guide_dropout/preparation_20261118_01/validation_selection.csv',
        training_seeds=[583301,583302,583303],generation_seeds=[482941,482942,482943],
        bootstrap_seed=583481,bootstrap_resamples=10000,
        interactions_family=dict(size=18,alpha=.05,quantiles=[.05/36,1-.05/36],
            members='Six manufacturer guide-effect interactions x three named judges, editing only'),
        metrics=METRICS,all_original_cases_retained=True,
        primary_case_contrasts=['guide_dropout__fine_minus_coarse__correct','old_full_guide__fine_minus_coarse__correct',
            'coarse_fine__dropout_minus_old__correct','coarse_control__dropout_minus_old__correct','guide_effect_interaction__correct'],
        added_intervals='Descriptive crossed 95% intervals, except the explicitly declared 18-member interaction family; not a new primary advancement test',
        training_audit='Actual sample exposure by gate/label-drop/timestep/source-cell/patient/epoch, all copies; no per-state causal gradient attribution',
        prospective_forward_probe=dict(local_only=True,max_seconds=600,training_seeds=[583301,583302,583303],
            generation_seeds=[482941],timesteps=[100,300],studies=['old_full_guide','guide_dropout'],
            modes=['coarse_control','coarse_fine'],controls=['correct','zero','absent','wrong','coarse_only','fine_only'],
            expected_observations=2160,raw_epsilon_arrays_saved=True,
            semantics='At fixed q(source,t,noise), source vs target manufacturer and null labels; field strength/EchoTime fixed. Not reverse-trajectory states, new images or classifier efficacy.'),
        paid_gpu_calls=0,test_pixels_used=False,echo_time_edited=False,advancement_retested=False,
        clinical_preservation='unknown',no_new_cloud_authority=True)
    json_write(out/'protocol.json',p)
    paths=[ROOT/p[k] for k in ('old_receipt','new_receipt','scientific_protocol','frozen_advancement','training_verified','selection')]
    paths += [prior/'final_review_20261118_01/inputs.json',prior/'analysis_20261118_01/inputs.json']
    paths += list((ROOT/'scripts/v14_dropout_audit').glob('*.py'))
    json_write(out/'inputs.json',{path.relative_to(ROOT).as_posix():digest(path) for path in paths})
    json_write(out/'complete.json',dict(complete=True,diagnosis_executed=False,paid_gpu_calls=0))
    print('LOCAL DIAGNOSTIC PLAN FROZEN',out,flush=True)


if __name__=='__main__':
    a=argparse.ArgumentParser(__doc__);a.add_argument('--run-id',required=True);prepare(a.parse_args().run_id)
