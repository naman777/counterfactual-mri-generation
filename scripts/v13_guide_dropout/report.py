"""Human-readable complete report and fixed all-patient panels after verification."""
import argparse
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.v8_generator_dev.io import ROOT,digest,json_write,project_path,verify_inputs


def report(prepared,results,analysis,run_id):
    prepared,results,analysis=map(project_path,(prepared,results,analysis))
    p=json.loads((prepared/'protocol.json').read_text())
    verify_inputs(json.loads((analysis/'inputs.json').read_text()))
    outcome=json.loads((analysis/'advancement.json').read_text())
    out=ROOT/'reports/v13_guide_dropout'/run_id;out.mkdir(parents=True,exist_ok=False)
    (out/'README.md').write_text('Producer: scripts.v13_guide_dropout.report. RESULTS.md: '
        'all ten gates, safeguards, scope and limits; gates.png: every simultaneous gate; '
        'patients_*.png: every frozen source patient, first declared training/generation seed, '
        'source/cached-old-fine/new-coarse/new-fine and fixed-scale difference images; '
        'inputs.json: source and array hashes; complete.json: report production only. '
        'Inspect every figure visually before closing the reporting task.\n',encoding='utf-8')
    lines=['# Guide-state dropout development experiment','',
        f"Frozen advancement requirements met: **{outcome['research_advancement']}**. No default is promoted.",'',
        'Six fresh adapters complete the declared exposure on the frozen epoch-130 parent. '
        'The evaluation uses 15 reused development patients, three training seeds and three generation seeds; '
        'these repeated cases are not independent new patients. EchoTime stays source. '
        'Interpret effects by the named independent judge. These are source-correspondence proxies, '
        'not clinical anatomy/lesion preservation or causal scanner effects.','',
        '| Comparison | Control / metric / judge | Difference [simultaneous CI] | Interval / replica pass |',
        '|---|---|---|---|']
    for g in outcome['gates']:
        rule=g['rule'];scale=100 if rule['metric']=='paired_hit_delta' else 1
        value=(f"{scale*g['estimate']:.5f} [{scale*g['ci_low']:.5f}, {scale*g['ci_high']:.5f}]"+
               (' percentage points' if scale==100 else '')) if g['available'] else 'unavailable: '+g['error']
        lines.append(f"| {rule['comparison']} | {rule['control']} / {rule['metric']} / {g['judge']} | {value} | {g['interval_passed']} / {g['replica_direction_passed']} |")
    lines += ['', 'Intervals use the prospectively declared ten-member family. Research margins '
        'are not clinical equivalence or safety thresholds. Failed noninferiority does not prove harm. '
        'Passing identifies a development candidate only; untouched confirmation remains separate.','',
        'All named judges, guide controls, source warnings, unsupported requests and adverse directions '
        'remain in the accompanying analysis tables. Field strength remains exploratory with unchanged '
        'limited direction support. No patient is removed for a warning.','',
        'The patient panels show all sources using the first declared training seed and generation seed; '
        'the displayed examples are not selected by improvement. All other seeds remain in the CSVs.','',
        f'Analysis: `{analysis.relative_to(ROOT).as_posix()}`.']
    (out/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    fig,axes=plt.subplots(1,2,figsize=(17,7),layout='constrained')
    for ax,editing in zip(axes,(False,True)):
        selected=[g for g in outcome['gates'] if (g['rule']['metric']=='paired_hit_delta')==editing]
        for i,g in enumerate(selected):
            if not g['available']:continue
            scale=100 if editing else 1
            ax.errorbar(scale*g['estimate'],i,xerr=[[scale*(g['estimate']-g['ci_low'])],[scale*(g['ci_high']-g['estimate'])]],
                fmt='o',color='#206653' if g['interval_passed'] and g['replica_direction_passed'] else '#a33e3e',capsize=4)
            threshold=g['rule'].get('upper_less_than',g['rule'].get('lower_greater_than'))*scale
            ax.plot(threshold,i,'|',color='black',markersize=15)
        ax.set_yticks(range(len(selected)),[g['rule']['comparison'].replace('new_fine_minus_','fine vs ')+'\n'+
            (g['judge'] if editing else g['rule']['control']+' '+g['rule']['metric']) for g in selected],fontsize=8)
        ax.axvline(0,color='#bbbbbb',linewidth=.8);ax.grid(axis='x',alpha=.15);ax.invert_yaxis()
        ax.set_xlabel('Editing difference (percentage points)' if editing else 'Normalised image-distance difference')
        ax.set_title('Named-judge editing safeguards' if editing else 'Source-correspondence requirements')
    fig.savefig(out/'gates.png',dpi=160);plt.close(fig)
    new=json.loads((results/'complete.json').read_text())
    old=json.loads((ROOT/'reports/v10_detail/results_20261117_01/complete.json').read_text())
    seed=p['reference_protocol']['training_seeds'][0];generation=p['reference_protocol']['evaluation']['selection_generation_seed']
    sources=pd.read_csv(prepared/'validation_selection.csv').dataset_index.astype(int).tolist()
    homes=[];evidence={}
    for receipt,mode in ((old,'coarse_fine'),(new,'coarse_control'),(new,'coarse_fine')):
        key=f'seed{seed}__{mode}__manufacturer__correct';home=project_path(receipt['evaluations'][key])
        requests=pd.read_csv(home/'requests.csv');homes.append((home,requests[requests.generation_seed==generation].set_index('dataset_index')))
    for page in range(3):
        fig,axes=plt.subplots(5,5,figsize=(11,11),layout='constrained')
        for row,index in enumerate(sources[page*5:(page+1)*5]):
            images=[];pid=None
            for home,requests in homes:
                request=requests.loc[index];pid=request.patient_id;path=home/request.array_file
                evidence[path.relative_to(ROOT).as_posix()]=digest(path)
                with np.load(path,allow_pickle=False) as z:
                    if not images:source=z['source'].copy()
                    images.append(z['target'].copy())
            displayed=[source,*images,np.abs(images[-1]-source)]
            for col,(ax,value) in enumerate(zip(axes[row],displayed)):
                ax.imshow(value,cmap='magma' if col==4 else 'gray',vmin=0,vmax=.3 if col==4 else 1)
                ax.set_xticks([]);ax.set_yticks([])
                if col==0:ax.set_ylabel(str(pid),fontsize=8)
                if row==0:ax.set_title(['Source','Cached old fine','New coarse','New fine','|New fine − source|'][col],fontsize=9)
        fig.savefig(out/f'patients_{page+1:02d}.png',dpi=150);plt.close(fig)
    for path in (analysis/'advancement.json',prepared/'protocol.json',project_path(__file__)):
        evidence[path.relative_to(ROOT).as_posix()]=digest(path)
    json_write(out/'inputs.json',evidence)
    json_write(out/'complete.json',dict(complete=True,report_produced=True,figures=4,visual_inspection_pending=True,
        default_promoted=False,clinical_preservation='unknown'))
    print('DROPOUT REPORT PRODUCED',out,flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(__doc__)
    for name in ('prepared','results','analysis','run-id'):ap.add_argument('--'+name,required=True)
    report(**vars(ap.parse_args()))
