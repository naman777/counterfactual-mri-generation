"""Plot every matched judge-specific epoch-70-to-130 conditioning contrast."""
import argparse
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.v8_generator_dev.io import ROOT, digest, json_write, project_path


def plot(source,run_id):
    source=project_path(source)
    frame=pd.read_csv(source,keep_default_na=False)
    selected=frame[(frame.stratum=='all_flagged_and_in_range')&(frame.control=='target')&
        (frame.metric=='paired_hit_delta')].copy()
    if len(selected)!=12:
        raise ValueError('All two backbones / two axes / three named judges required')
    output=ROOT/'reports/v8_generator_dev'/run_id
    output.mkdir(exist_ok=False)
    fig,axes=plt.subplots(1,2,figsize=(11,5.8),sharey=True)
    ordered=[(model,f'heldout_orientation_robust_seed{seed}') for model in ('additive','scale_shift') for seed in range(3)]
    for ax,axis,title in zip(axes,('manufacturer','field_strength'),('Manufacturer','Field strength')):
        part=selected[selected.axis==axis].set_index(['model','judge'])
        for y,key in enumerate(ordered):
            row=part.loc[key]
            center,lo,hi=[float(row[name])*100 for name in ('mean','ci95_lo','ci95_hi')]
            color='#245e87' if key[0]=='additive' else '#ad642f'
            ax.errorbar(center,y,xerr=[[center-lo],[hi-center]],fmt='o',color=color,capsize=4,markersize=6)
            ax.text(50,y,f'{center:.1f} [{lo:.1f}, {hi:.1f}]',va='center',ha='right',fontsize=8.5,color=color)
        ax.axvline(0,color='#777777',linestyle='--',linewidth=1)
        ax.set_title(title,fontweight='bold')
        ax.set_xlim(-22,53)
        ax.set_xlabel('Change in target-minus-null hit rate (percentage points)')
        ax.grid(axis='x',alpha=.15)
        ax.set_axisbelow(True)
        ax.spines[['top','right']].set_visible(False)
    axes[0].set_yticks(np.arange(6),[f'{model.replace("_"," ")} / judge {judge[-1]}' for model,judge in ordered])
    axes[0].invert_yaxis()
    fig.suptitle('Matched conditioning change: common epoch 130 minus epoch 70',fontsize=13)
    fig.text(.5,.025,'All heldout_orientation_robust_seed{0,1,2} judges; 15 reused development patients; three generation seeds.\n'
        '95% patient-bootstrap intervals, not multiplicity adjusted. Flagged requests retained; clinical preservation unknown.',
        ha='center',fontsize=9)
    fig.subplots_adjust(left=.20,right=.98,top=.87,bottom=.21,wspace=.16)
    fig.savefig(output/'matched_conditioning_change.png',dpi=180)
    fig.savefig(output/'matched_conditioning_change.pdf')
    plt.close(fig)
    json_write(output/'inputs.json',{source.relative_to(ROOT).as_posix():digest(source),
        'scripts/v8_generator_dev/plot_continuation.py':digest(__file__)})
    json_write(output/'complete.json',dict(complete=True,all_twelve_contrasts_shown=True,
        rates_and_intervals_from_source_only=True,anatomy_validated=False,fresh_cohort=False))
    (output/'README.md').write_text(
        'Producer: scripts.v8_generator_dev.plot_continuation. matched_conditioning_change.png/pdf '
        'shows all twelve actual paired epoch-130-minus-70 target-minus-null contrasts and patient '
        'CIs; no model/judge is omitted. inputs.json hashes the source CSV/code. The source table '
        'retains direction/support strata; the figure shows the declared overall development stratum. '
        'complete.json is plotting completion, not independent clinical/causal confirmation.\n',encoding='utf-8')
    print('CONTINUATION PLOT COMPLETE',output,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--source',required=True)
    p.add_argument('--run-id',required=True)
    a=p.parse_args()
    plot(a.source,a.run_id)
