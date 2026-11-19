"""Decompose automated flags into geometry/local-detail changes and paired null controls."""
import argparse
import json
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.v8_generator_dev.full_protocol import bootstrap_patients
from scripts.v8_generator_dev.io import ROOT, csv_write, digest, json_write, project_path, verify_inputs
from scripts.v8_generator_dev.preservation_screen_v2 import METRICS


def explain(benchmark, run_id):
    if not re.fullmatch('[A-Za-z0-9_-]+', run_id):
        raise ValueError('Safe new report ID required')
    benchmark = project_path(benchmark)
    verify_inputs(json.loads((benchmark / 'inputs.json').read_text()))
    frame = pd.read_csv(benchmark / 'edit_metrics.csv', keep_default_na=False)
    thresholds = json.loads((benchmark / 'thresholds.json').read_text())['thresholds']
    output = ROOT / 'reports/v8_generator_dev' / run_id
    output.mkdir(exist_ok=False)
    rows=[]
    for (model,axis,control),group in frame[frame.control!='identity'].groupby(['model','axis','control']):
        for metric in METRICS:
            values=group[['patient_id']].copy()
            values['ratio']=group[metric]/thresholds[metric]
            values['exceeded']=(group[metric]>thresholds[metric]).astype(float)
            for name in ('ratio','exceeded'):
                rows.append(dict(model=model,axis=axis,control=control,metric=metric,quantity=name,
                    **bootstrap_patients(values,name,482944)))
    table=pd.DataFrame(rows)
    csv_write(output/'metric_contributions.csv',table)
    keys=['model','patient_id','case','axis','direction','generation_seed','condition_support_status']
    paired=frame[frame.control=='target'].merge(frame[frame.control=='null'],on=keys,
        how='outer',suffixes=('_target','_null'),indicator=True,validate='one_to_one')
    if not paired._merge.eq('both').all():
        raise ValueError('Unmatched null/target cases')
    differences=[]
    for (model,axis),group in paired.groupby(['model','axis']):
        for metric in (*METRICS,'ssim','mad'):
            values=group[['patient_id']].copy()
            values['difference']=group[metric+'_target']-group[metric+'_null']
            differences.append(dict(model=model,axis=axis,metric=metric,contrast='target_minus_matched_null',
                **bootstrap_patients(values,'difference',482944)))
    csv_write(output/'paired_null_differences.csv',pd.DataFrame(differences))
    fig,axes=plt.subplots(2,2,figsize=(13,8),constrained_layout=True)
    for ax,metric in zip(axes.flat,METRICS):
        points=table[(table.metric==metric)&(table.quantity=='ratio')]
        y=np.arange(len(points))
        ax.errorbar(points['mean'],y,xerr=np.vstack([points['mean']-points.ci95_lo,points.ci95_hi-points['mean']]),
                    fmt='o',capsize=3,color='#235d87')
        ax.set_yticks(y,[f'{r.model}, {r.axis}, {r.control}' for r in points.itertuples()],fontsize=8)
        ax.axvline(1,color='#a85532',linestyle='--',linewidth=1)
        ax.invert_yaxis()
        ax.set_title(metric)
        ax.set_xlabel('Mean ratio to training-control cutoff [95% patient CI]')
    fig.suptitle('Structural-screen decomposition: flags are engineering warnings, not anatomy diagnoses')
    fig.savefig(output/'metric_contributions.png',dpi=150)
    plt.close(fig)
    lines=['# What the automated preservation screen establishes','',
        'No qualified reader is available. This report provides a reproducible automated workaround '
        'while retaining unknown breast/tumour preservation. Source and edited MRI images are not '
        'paired true scans under different acquisitions.','',
        '- The first benchmark detected spatial displacement but missed small erased patches. '
        'Its complete negative results remain unchanged.',
        '- The separately versioned local-detail extension detects those artificial erasures. '
        'It was developed after inspecting the first diagnostic, so it is development evidence, '
        'not independently confirmed lesion sensitivity.',
        '- Both null and target edits exceed the mild-transform envelope. This points to a common '
        'denoising/source-detail effect as well as any target-conditioned changes. It does not '
        'mean every edit has damaged anatomy: legitimate acquisition-related texture/contrast '
        'changes can also trigger these measurements.',
        '- Rank/local-variance measures examine texture/order; edge displacement examines a coarse '
        'geometry proxy. A failure on texture alone must not be called a displaced breast boundary '
        'or an erased lesion. Inspect metric_contributions.csv separately.',
        '- paired_null_differences.csv isolates target-minus-null proxy changes on identical '
        'source/noise/settings, with generation seeds averaged within patient. Its intervals are '
        'descriptive development uncertainty, not a scanner-effect or clinical-validity estimate.','',
        'Immediate generator work: finish the matched stable reference run, compare its outputs '
        'using these unchanged screens, then evaluate a source-guided adapter with correct/absent/wrong '
        'guides and matched nulls. Do not lower cutoffs or select only favourable patients. '
        'The guide remains acquisition-leaky and its software integration is not efficacy training.','',
        '![Metric decomposition](metric_contributions.png)','',
        'Without compatible exact-source labels or expert review, a proxy screen cannot certify '
        'lesion preservation. All control false flags, missed damage, support flags and small-sample '
        'interval limitations remain part of the result.','']
    (output/'RESULTS.md').write_text('\n'.join(lines),encoding='utf-8')
    inputs={path.relative_to(ROOT).as_posix():digest(path) for path in (
        benchmark/'edit_metrics.csv',benchmark/'thresholds.json',benchmark/'complete.json',
        benchmark/'control_intervals.csv',ROOT/'scripts/v8_generator_dev/explain_preservation.py')}
    json_write(output/'inputs.json',inputs)
    json_write(output/'complete.json',dict(complete=True,paired_nulls_verified=True,
        clinical_anatomy_validated=False,lesion_preservation='unknown',generator_winner_selected=False))
    (output/'README.md').write_text(
        'Producer: `python -m scripts.v8_generator_dev.explain_preservation`.\n'
        'RESULTS.md interprets automated flags; metric_contributions.csv/png separates proxy components; '
        'paired_null_differences.csv gives every matched target-minus-null contrast with patient CIs. '
        'inputs.json hashes the benchmark/code; complete.json retains unknown clinical validity.\n',encoding='utf-8')
    print('PRESERVATION INTERPRETATION',output,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--benchmark',required=True)
    parser.add_argument('--run-id',required=True)
    args=parser.parse_args()
    explain(args.benchmark,args.run_id)
