"""Scientific visual QA for the engineering adapter; every case is retained."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts.v4_convergence.core import digest,json_write
from scripts.v5_anatomy.csv_io import read_table


def render(pilot):
    pilot=Path(pilot).resolve()
    if not json.loads((pilot/'complete.json').read_text())['complete']:raise ValueError('Incomplete evaluation')
    output=pilot/'figures';output.mkdir(exist_ok=False)
    generation=read_table(pilot/'generation.csv');summary=read_table(pilot/'summary.csv')
    variants=['reference','source_guide','guide_absent','wrong_source_guide'];controls=[]
    plt.rcParams.update({'font.family':'Arial','font.size':9})
    for case,group in generation.groupby('case_id'):
        with np.load(pilot/'arrays'/f'{case}.npz') as arrays:
            fig,axes=plt.subplots(2,5,figsize=(13,6.1))
            axes[0,0].imshow(arrays['source'],cmap='gray',vmin=0,vmax=1,interpolation='none')
            axes[0,0].set_title('Source');axes[0,0].axis('off');axes[1,0].axis('off')
            table=group.set_index('variant')
            for column,variant in enumerate(variants,1):
                axes[0,column].imshow(arrays[variant],cmap='gray',vmin=0,vmax=1,interpolation='none')
                axes[0,column].set_title(variant.replace('_',' '));axes[0,column].axis('off')
                im=axes[1,column].imshow(np.abs(arrays[variant]-arrays['source']),cmap='magma',vmin=0,vmax=1,interpolation='none')
                axes[1,column].set_title(f'SSIM {table.loc[variant,"ssim"]:.3f}\nProxy Dice {table.loc[variant,"proxy_dice_min"]:.3f}')
                axes[1,column].axis('off')
            controls.append(dict(case_id=case,source_guide_vs_absent_mad=float(np.abs(arrays['source_guide']-arrays['guide_absent']).mean()),
                correct_vs_wrong_guide_mad=float(np.abs(arrays['source_guide']-arrays['wrong_source_guide']).mean()),
                absent_equals_base=bool(np.array_equal(arrays['guide_absent'],arrays['reference']))))
        fig.suptitle(f'{case}: short engineering-trained adapter\nWiring check only; expert anatomy preservation and efficacy remain unvalidated',fontsize=12)
        fig.subplots_adjust(top=.82,bottom=.16,hspace=.28)
        ca=fig.add_axes([.36,.07,.3,.018]);fig.colorbar(im,cax=ca,orientation='horizontal',ticks=[0,.5,1])
        ca.tick_params(labelsize=8,pad=2);ca.set_xlabel('Absolute difference from common source',fontsize=8,labelpad=2)
        fig.savefig(output/f'{case}.png',dpi=140);plt.close(fig)
    pd.DataFrame(controls).to_csv(output/'guide_ablation_response.csv',index=False)
    lines=['# Engineering adapter inference check','','These are technical smoke outputs, not trained-model efficacy results.',
           'Correct, absent and wrong-source guide controls share the source/target, timestep, CFG and RNG seed.',
           'The absent-guide output reproduces the frozen reference exactly. Nonzero guide response checks wiring only.',
           'The representation audit found residual scanner-label leakage; no scanner-invariance claim is supported.',
           'Expert anatomy/lesion review and qualified reader participation remain pending.','',
           '| Axis | Variant | Judge | Target-hit rate [patient-bootstrap 95% CI] |',
           '| --- | --- | --- | --- |']
    for row in summary[summary.metric=='target_hit'].itertuples():
        lines.append(f'| {row.axis} | {row.variant} | {row.judge} | {100*row.mean:.1f}% [{100*row.ci95_lo:.1f}, {100*row.ci95_hi:.1f}] |')
    lines+=['','Very small sample; degenerate bootstrap intervals are not certainty. No winner or clinical pass is declared.']
    (output/'findings.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    json_write(output/'manifest.json',dict(script_sha256=digest(__file__),
        inputs={p:digest(pilot/p) for p in ['complete.json','generation.csv','summary.csv']},
        request_panels=len(controls),clinical_validation=False))
    return output


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--pilot',required=True);print(render(p.parse_args().pilot))
