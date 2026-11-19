"""Read completed backed-up pair records with the SDK; no GPU app is created."""
import argparse
import hashlib
import json
import os
import re

os.environ['MODAL_PROFILE']='namankundra7'
import modal
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.v8_generator_dev.io import ROOT, csv_write, digest, json_write


def snapshot(run_id,report_id):
    if not all(re.fullmatch('[A-Za-z0-9_-]+',value) for value in (run_id,report_id)):
        raise ValueError('Safe run/report IDs required')
    output=ROOT/'reports/v8_generator_dev'/report_id
    output.mkdir(exist_ok=False)
    rows,records,inputs=[],[],{}
    original=ROOT/'reports/v8_generator_dev/cloud_resume_fetch_20261116_01/training'
    for path in sorted(original.glob('block_*/*_validation.json')):
        variant=path.name.removesuffix('_validation.json')
        value=json.loads(path.read_text())
        rows.append(dict(variant=variant,phase='original',**value))
        inputs[path.relative_to(ROOT).as_posix()]=digest(path)
    volume=modal.Volume.from_name('capstone-judge-data',create_if_missing=False)
    remote=f'v8_generator_dev/{run_id}/project/reports/v8_generator_dev/{run_id}_training'
    entries=volume.listdir(remote,recursive=True)
    complete_paths=sorted(entry.path.lstrip('/') for entry in entries if re.search(r'block_\d+/complete.json$',entry.path))
    for complete_path in complete_paths:
        marker=b''.join(volume.read_file(complete_path))
        pair=json.loads(marker)
        if pair['progress']['additive']!=pair['progress']['scale_shift']:
            raise ValueError('Snapshot refuses unmatched progress')
        block=complete_path.split('/')[-2]
        for variant in ('additive','scale_shift'):
            path=complete_path.rsplit('/',1)[0]+f'/{variant}_validation.json'
            content=b''.join(volume.read_file(path))
            value=json.loads(content)
            if value['epoch']!=pair['progress'][variant]['epoch']:
                raise ValueError('Marker/validation epoch mismatch')
            local=output/'raw'/block/f'{variant}_validation.json'
            local.parent.mkdir(parents=True,exist_ok=True)
            with local.open('xb') as handle:
                handle.write(content)
            records.append(dict(remote=path,local=str(local.relative_to(ROOT)),sha256=hashlib.sha256(content).hexdigest()))
            rows.append(dict(variant=variant,phase='continuation',**value))
        marker_path=output/'raw'/block/'complete.json'
        with marker_path.open('xb') as handle:
            handle.write(marker)
        records.append(dict(remote=complete_path,local=str(marker_path.relative_to(ROOT)),sha256=digest(marker_path)))
    frame=pd.DataFrame(rows)
    common=set.intersection(*[set(group.epoch) for _,group in frame.groupby('variant')])
    frame=frame[frame.epoch.isin(common)].sort_values(['variant','epoch'])
    if frame.duplicated(['variant','epoch']).any() or not np.isfinite(frame.patient_macro_noise_mse).all():
        raise ValueError('Invalid curve records')
    csv_write(output/'validation_curve.csv',frame)
    fig,axes=plt.subplots(2,1,figsize=(9,7),sharex=True,constrained_layout=True)
    for ax,metric,label in zip(axes,['patient_macro_noise_mse','natural_slice_noise_mse'],
                              ['Equal-patient validation noise MSE','Natural-slice validation noise MSE']):
        for variant,color in [('additive','#245e87'),('scale_shift','#ad642f')]:
            group=frame[frame.variant==variant]
            ax.plot(group.epoch,group[metric],marker='o',markersize=3,color=color,label=variant)
        ax.axvline(70,color='gray',linestyle='--',linewidth=1)
        ax.set_ylabel(label)
        ax.set_yscale('log')
        ax.grid(alpha=.2)
    axes[0].legend(frameon=False)
    axes[0].set_title('Actual completed matched validation records\nContinuation enables bounded AMP retries after epoch 70; editing efficacy is not measured here')
    axes[1].set_xlabel('Completed common epoch')
    fig.savefig(output/'validation_curve.png',dpi=150)
    plt.close(fig)
    json_write(output/'remote_records.json',records)
    inputs['scripts/v8_generator_dev/snapshot_continuation_curve.py']=digest(__file__)
    json_write(output/'inputs.json',inputs)
    json_write(output/'complete.json',dict(complete=True,latest_matched_epoch=int(frame.epoch.max()),
        full_training_completed=False,read_only=True,new_modal_app_created=False,editing_efficacy_established=False))
    (output/'README.md').write_text(
        '# Matched continuation curve snapshot\n\nProducer: `python -m scripts.v8_generator_dev.snapshot_continuation_curve`.\n'
        'raw/ and remote_records.json retain exact completed-pair validation/marker bytes from the existing volume. '
        'validation_curve.csv/png combines original and continued actual records only. inputs.json hashes '
        'original evidence/code; complete.json records snapshot completion, never full training or editing gains. '
        'No GPU app, altered checkpoint or extrapolated curve is created.\n',encoding='utf-8')
    print('CURVE SNAPSHOT',output,'epoch',int(frame.epoch.max()),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--report-id',required=True)
    args=parser.parse_args()
    snapshot(args.run_id,args.report_id)
