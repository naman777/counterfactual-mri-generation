"""Read-only proof that the recovered failing draw was applied in real continuation."""
import argparse
import hashlib
import json
import os
import re

os.environ['MODAL_PROFILE']='namankundra7'
import modal

from scripts.v8_generator_dev.io import ROOT,digest,json_write


def audit(run_id,report_id):
    if not all(re.fullmatch('[A-Za-z0-9_-]+',value) for value in (run_id,report_id)):
        raise ValueError('Safe workflow/report IDs required')
    output=ROOT/'reports/v8_generator_dev'/report_id
    output.mkdir(exist_ok=False)
    source=ROOT/'reports/v8_generator_dev/diagnostic_results_20261117_01/remote/numerical_diagnostic_20261117_01/results/failure.json'
    failure=json.loads(source.read_text())
    volume=modal.Volume.from_name('capstone-judge-data',create_if_missing=False)
    folder=f'v8_generator_dev/{run_id}/project/reports/v8_generator_dev/{run_id}_training/block_0001'
    data=b''.join(volume.read_file(folder+'/additive_steps.json'))
    marker=b''.join(volume.read_file(folder+'/complete.json'))
    with (output/'additive_steps.json').open('xb') as handle:
        handle.write(data)
    with (output/'pair_complete.json').open('xb') as handle:
        handle.write(marker)
    pair=json.loads(marker)
    if pair['progress']['additive']!=pair['progress']['scale_shift'] or not pair['matched_sources_timesteps_noise']:
        raise ValueError('Published continuation block is not matched')
    rows=json.loads(data)
    expected=failure['progress_before']['global_step']+1
    matches=[row for row in rows if row['global_step']==expected]
    if len(matches)!=1:
        raise ValueError('Recovered draw missing/duplicated in real continuation')
    row=matches[0]
    for key in ('sample_indices','patient_ids','timesteps','noise_sha256'):
        if row[key]!=failure['draw'][key]:
            raise ValueError('Real continuation differs from recovered failure draw: '+key)
    if not row['applied'] or row['amp_retries']<1:
        raise ValueError('Recovered draw was not applied via bounded retry')
    json_write(output/'known_draw.json',row)
    json_write(output/'remote_sources.json',dict(steps=folder+'/additive_steps.json',
        steps_sha256=hashlib.sha256(data).hexdigest(),pair_marker=folder+'/complete.json',
        marker_sha256=hashlib.sha256(marker).hexdigest()))
    json_write(output/'inputs.json',{source.relative_to(ROOT).as_posix():digest(source),
        'scripts/v8_generator_dev/audit_live_retry.py':digest(__file__)})
    json_write(output/'complete.json',dict(complete=True,known_recovered_draw_matched=True,
        applied_in_real_continuation=True,amp_retries=row['amp_retries'],global_step=expected,
        published_common_epoch=pair['progress']['additive']['epoch'],
        all_recorded_updates_applied=all(value['applied'] for value in rows),
        complete_full_training=False,editing_efficacy_established=False,anatomy_validated=False,
        new_gpu_app_created=False))
    (output/'README.md').write_text(
        'Producer: `python -m scripts.v8_generator_dev.audit_live_retry`.\n'
        'additive_steps.json/pair_complete.json preserve exact published continuation bytes. '
        'known_draw.json matches source/patient/timestep/noise identities against the recovered failure '
        'and records the actually applied retry; remote_sources.json/inputs.json bind provenance. '
        'complete.json closes this draw-level audit, never full training, counterfactual efficacy or anatomy validation.\n',encoding='utf-8')
    print('REAL RETRY AUDIT PASSED',expected,'retries',row['amp_retries'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--report-id',required=True)
    args=parser.parse_args()
    audit(args.run_id,args.report_id)
