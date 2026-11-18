"""Render every pilot request and judge-specific trade-offs without selecting wins."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts.v4_convergence.core import digest, json_write
from scripts.v5_anatomy.csv_io import read_table


def render(pilot, output_name='figures'):
    pilot = Path(pilot).resolve()
    if not json.loads((pilot/'complete.json').read_text())['complete']:
        raise ValueError('Pilot has not completed')
    if Path(output_name).name != output_name:
        raise ValueError('Output name must be a simple new subfolder')
    output = pilot/output_name
    output.mkdir(exist_ok=False)
    protocol = json.loads((pilot/'protocol.json').read_text())
    generation = read_table(pilot/'generation.csv')
    summary = read_table(pilot/'summary.csv')
    plt.rcParams.update({'font.family':'Arial','font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    variants = ['identity','null','sdedit']+[f'anchor_{s:g}' for s in protocol['anchor_strengths']]
    panels = []
    for case, group in generation.groupby('case_id',sort=True):
        with np.load(pilot/'arrays'/f'{case}.npz') as arrays:
            fig, axes = plt.subplots(2,len(variants)+1,figsize=(14,6.1),gridspec_kw={'height_ratios':[1,1]})
            axes[0,0].imshow(arrays['source'],cmap='gray',vmin=0,vmax=1,interpolation='none')
            axes[0,0].set_title('Source');axes[0,0].axis('off');axes[1,0].axis('off')
            indexed=group.set_index('variant')
            for column,variant in enumerate(variants,1):
                row=indexed.loc[variant]
                axes[0,column].imshow(arrays[variant],cmap='gray',vmin=0,vmax=1,interpolation='none')
                axes[0,column].set_title(variant);axes[0,column].axis('off')
                im=axes[1,column].imshow(np.abs(arrays[variant]-arrays['source']),cmap='magma',vmin=0,vmax=1,interpolation='none')
                axes[1,column].axis('off')
                axes[1,column].set_title(f'SSIM {row.ssim:.3f}\nProxy Dice {row.proxy_dice_min:.3f}; HD95 {row.proxy_boundary_hd95_px_max:.1f} px',fontsize=8)
            first=group.iloc[0]
            fig.suptitle(f'{case} | {first.patient_id} | t={first.timestep}, w={first.guidance:g}\nValidation development; threshold-mask proxies, human anatomy review pending',fontsize=11)
            fig.subplots_adjust(top=.80,bottom=.17,wspace=.12,hspace=.32)
            ca=fig.add_axes([.38,.07,.28,.018]);fig.colorbar(im,cax=ca,orientation='horizontal',ticks=[0,.5,1])
            ca.tick_params(labelsize=8,pad=2)
            ca.set_xlabel('Absolute difference from the common source',fontsize=8,labelpad=2)
            name=f'{case}.png';fig.savefig(output/name,dpi=140);plt.close(fig);panels.append(name)
    fig, axes = plt.subplots(2,3,figsize=(13,7.3),sharey=True)
    colors=['#888888','#9a6237','#235d87','#167166','#a13b73']
    markers=['x','D','o','s','^']
    for row,axis in enumerate(['field_strength','manufacturer']):
        for seed in range(3):
            ax=axes[row,seed]
            group=summary[(summary.axis==axis)&(summary.judge==f'heldout_orientation_robust{seed}')]
            for variant,color,marker in zip(variants,colors,markers):
                g=group[group.variant==variant].set_index('metric')
                x=g.loc['proxy_boundary_hd95_px_max'];y=g.loc['target_hit']
                ax.errorbar(x['mean'],y['mean']*100,
                    xerr=[[max(0,x['mean']-x.ci95_lo)],[max(0,x.ci95_hi-x['mean'])]],
                    yerr=[[max(0,y['mean']-y.ci95_lo)*100],[max(0,y.ci95_hi-y['mean'])*100]],
                    marker=marker,color=color,capsize=2,ls='none',label=variant,alpha=.85)
            ax.set_title(f'{axis.replace("_"," ").title()} | held-out seed {seed}')
            ax.grid(alpha=.2);ax.set_ylim(-5,105)
            ax.set_xlabel('Mean proxy boundary HD95 (pixels; lower is closer)')
            if seed==0:ax.set_ylabel('Judge-specific target-hit rate (%)')
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=5,frameon=False,bbox_to_anchor=(.5,.02))
    fig.suptitle('Exploratory validation trade-off: target hits versus a geometry proxy\nPatient-bootstrap 95% intervals; no clinical preservation claim or winner selection',fontsize=12)
    fig.tight_layout(rect=[0,.08,1,.91]);fig.savefig(output/'tradeoff.png',dpi=160);plt.close(fig)
    # Summarise first held-out judge only with its name and intervals; do not
    # collapse different judges or label the proxy as anatomical validation.
    selected=summary[(summary.judge=='heldout_orientation_robust0')&
                     summary.metric.isin(['target_hit','proxy_screen_pass','target_hit_and_proxy_screen'])]
    lines=['# Exploratory validation findings','',
           'This is a development pilot, not independent final testing. All candidates are reported.',
           'Automated foreground geometry is contrast-sensitive. Human anatomy/lesion preservation remains unmeasured.',
           'All intervals below resample patients. Degenerate intervals in this small pilot are not population certainty.','',
           '| Axis | Variant | Metric | Held-out seed 0: rate [95% CI] |',
           '| --- | --- | --- | --- |']
    for r in selected.itertuples():
        lines.append(f'| {r.axis} | {r.variant} | {r.metric} | {100*r.mean:.1f}% [{100*r.ci95_lo:.1f}, {100*r.ci95_hi:.1f}] |')
    lines+=['','Other judge seeds, original predictor, continuous diagnostics and paired differences are in the CSVs.',
            'A target hit plus proxy-screen pass is not a target hit plus verified preservation.',
            'No algorithm was selected from these outcomes. Read the blinded review instructions before collecting human labels.']
    (output/'findings.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    json_write(output/'manifest.json',dict(producer='scripts/v5_anatomy/render_pilot.py',
               inputs={name:digest(pilot/name) for name in ['complete.json','generation.csv','summary.csv','protocol.json']},
               script_sha256=digest(__file__),request_panels=panels,tradeoff='tradeoff.png'))
    return output


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--pilot',required=True)
    p.add_argument('--output-name',default='figures')
    args=p.parse_args();print(render(args.pilot,args.output_name))
