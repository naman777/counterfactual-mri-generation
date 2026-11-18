"""Package saved-array diagnosis and explicitly gated engineering proposals."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.v5_anatomy.metrics import foreground_proxy
from scripts.v8_generator_dev.io import ROOT, csv_write, digest, json_write, project_path, verify_inputs

BASE=ROOT/'reports/v8_generator_dev'


def report(diagnosis,preparation,run_id):
    if not run_id.replace('_','').isalnum():
        raise ValueError('Safe fresh ID required')
    diagnosis,preparation=map(project_path,(diagnosis,preparation))
    for home in (diagnosis,preparation):
        if not json.loads((home/'complete.json').read_text())['complete']:
            raise ValueError('Completed local inputs required')
        verify_inputs(json.loads((home/'inputs.json').read_text()))
    output=BASE/run_id
    output.mkdir(exist_ok=False)
    summary=pd.read_csv(diagnosis/'patient_intervals.csv')
    cases=pd.read_csv(diagnosis/'case_index.csv')
    metrics=pd.read_csv(diagnosis/'case_metrics.csv')
    selected=cases[cases.profile.eq('additive_alpha050')].drop_duplicates('dataset_index')
    masks=[]
    for row in selected.itertuples():
        p=project_path(json.loads((diagnosis/'protocol.json').read_text())['profiles'][row.profile])/row.array_file
        with np.load(p,allow_pickle=False) as data:
            source=data['source']
            for threshold in (.03,.05,.1):
                mask=foreground_proxy(source,threshold)
                fraction=float(mask.mean())
                masks.append(dict(patient_id=row.patient_id,dataset_index=row.dataset_index,threshold=threshold,
                    foreground_fraction=fraction,
                    near_full_frame_diagnostic_flag=fraction>=.90,
                    flag_role='Post-view engineering warning only; no case exclusion/clinical threshold'))
    mask_frame=pd.DataFrame(masks)
    csv_write(output/'mask_qc.csv',mask_frame)
    # Independently recompute one table entry and its bootstrap from saved case rows.
    checked=[]
    for region in ('whole','interior','boundary','background'):
        for comparison in ('source_null','source_target','target_minus_null_source_error'):
            group=metrics[(metrics.profile=='additive_alpha050')&(metrics.axis=='manufacturer')&
                (metrics.threshold==.05)&(metrics.region==region)&(metrics.comparison==comparison)]
            for metric in ('mad','ssim','rank_mad','hf_residual_mad','hf_energy_ratio'):
                patients=group.groupby('patient_id')[metric].mean().dropna().to_numpy()
                protocol=json.loads((diagnosis/'protocol.json').read_text())
                rng=np.random.default_rng(protocol['bootstrap_seed'])
                draws=rng.integers(len(patients),size=(protocol['bootstrap_resamples'],len(patients)))
                lo,hi=np.quantile(patients[draws].mean(1),[.025,.975])
                original=summary[(summary.profile=='additive_alpha050')&(summary.axis=='manufacturer')&
                    (summary.threshold==.05)&(summary.region==region)&(summary.comparison==comparison)&
                    (summary.metric==metric)&(summary.stratum=='all')].iloc[0]
                error=float(np.max(np.abs([patients.mean()-original.estimate,lo-original.ci_low,hi-original.ci_high])))
                if error>1e-12:
                    raise AssertionError('Independent summary recomputation mismatch')
                checked.append(dict(region=region,comparison=comparison,metric=metric,max_error=error))
    csv_write(output/'summary_recheck.csv',pd.DataFrame(checked))
    def entry(region,comparison,metric,threshold=.05):
        row=summary[(summary.profile=='additive_alpha050')&(summary.axis=='manufacturer')&
            (summary.threshold==threshold)&(summary.region==region)&(summary.comparison==comparison)&
            (summary.metric==metric)&(summary.stratum=='all')].iloc[0]
        return f'{row.estimate:.4f} [{row.ci_low:.4f}, {row.ci_high:.4f}]'
    lines=['# Matched source-detail diagnosis and next experiments','',
        'The saved development outputs show substantial source changes under original-label denoising, '
        'with further changes when the acquisition label is switched. This motivates testing source-detail '
        'conditioning in both null and target branches. It does not establish anatomical damage or a clinical preservation rate.','',
        '## Evidence and scope','',
        f'- Diagnosis: `{diagnosis.relative_to(ROOT).as_posix()}`; 16 profiles, 1,440 fixed-point cases, '
        '15 reused validation patients and three generation seeds. Both axes, every guide/control, all directions and support flags remain. '
        'Case count includes repeated matched sources and exactly duplicated absent-parent controls; it is not independent sample size.',
        '- Manufacturer t=300, w=3; field strength t=100, w=5. EchoTime stays fixed. '
        'These development settings/cohort are distinct from the older publication protocol.',
        '- Exact consumed arrays and request/protocol files match immutable retrieval hashes; source pixels, sampling seeds, '
        'settings and support agree between profiles. Prior strict judge grading is reused; no new grade is claimed.',
        '- All intervals below are 95% equal-patient percentile bootstrap intervals; generation seeds are averaged within patient. '
        'They are exploratory, with no multiplicity adjustment or independent-cohort confirmation.','',
        '## Additive alpha .50 manufacturer reference','',
        'The following use source-derived proxy regions at threshold .05. Intensities are normalised to [0,1]. '
        'MAD measures appearance change; SSIM and high-frequency energy are not anatomy labels.','',
        '| Region / metric | Source vs null | Source vs target | Paired target-minus-null source discrepancy |',
        '|---|---|---|---|']
    for region,metric in [('whole','mad'),('interior','mad'),('boundary','mad'),('background','mad'),
                          ('whole','ssim'),('interior','ssim'),('interior','rank_mad'),
                          ('interior','hf_residual_mad'),('interior','hf_energy_ratio')]:
        lines.append('| '+region+' / '+metric+' | '+' | '.join(entry(region,c,metric) for c in
            ('source_null','source_target','target_minus_null_source_error'))+' |')
    lines += ['', 'Most importantly, the interior local-rank discrepancy and high-frequency residual difference '
        'between target and null have intervals spanning zero, although intensity MAD rises and SSIM falls. '
        'Thus these proxies disagree about the additional target-associated fine-detail change. '
        'High-frequency energy exceeds source energy on average; a simple global-blur explanation is insufficient. '
        'Energy may be added or redistributed while source correspondence worsens. '
        'Null and target paths share noise, but nonlinear metric differences are not a causal or additive decomposition of damage.','',
        'Whole-image SSIM includes easy background and differs from the lower proxy-interior score. '
        'Regional SSIM averages a 7x7 map at region centres; those windows can cross boundaries. '
        'It also differs slightly from historical global SSIM, which discards the image border; '
        'do not replace the historical SSIM table with this diagnostic.','',
        '## Mask and sensitivity failure retained','',
        '`mask_qc.csv` reports every selected source at all three fixed thresholds. '
        'The all-patient panel was visually inspected. The source mask for Breast_MRI_445 is nearly full-frame at .05; '
        'background signal and hole filling contaminate the proposed interior. These regions are source-intensity proxies, '
        'not breast, tumour or verified tissue masks. The >=.90 warning was added after viewing as a diagnostic flag only; '
        'no case is excluded and it is not a validated admission threshold.','',
        '| Source proxy threshold | Interior source-null MAD | Interior source-target MAD | Paired change |',
        '|---|---|---|---|']
    for threshold in (.03,.05,.1):
        lines.append('| '+str(threshold)+' | '+' | '.join(entry('interior',c,'mad',threshold) for c in
            ('source_null','source_target','target_minus_null_source_error'))+' |')
    lines += ['', 'Direction and target-support strata, all other profiles, null-target correspondence and every threshold '
        'remain in `patient_intervals.csv`; aggregate results must not conceal these differences. '
        'A fresh mask-QC/revision step is required before the foreground efficacy ablation. '
        'Copying a proxy-defined exterior cannot demonstrate preservation inside edited tissue.','',
        '## Candidate preparation','',
        f'Local evidence: `{preparation.relative_to(ROOT).as_posix()}`. Eight synthetic engineering tests pass: '
        'coarse-guide numerical identity, differentiability, zero-initialised parent equality, zero-gate/absent equality, '
        'discarded adapter updates and shared CFG guides on both tiny backbones; masked sampling has exact full-mask '
        'historical-sampler equality, empty-mask source equality in sampler coordinates and partial-mask exterior retention. '
        'No final-parent, full-batch or efficacy claim follows.','',
        'The fixed pooled ridge probe still predicts manufacturer from the proposed fine/coarse guide: '
        'balanced accuracy 0.8750 [0.6250, 1.0000]; field strength 0.7411 [0.5357, 0.9375]. '
        'Intervals use a class-stratified patient bootstrap. Confounding and anatomical correlations can contribute, '
        'and a weak probe cannot prove invariance. Both matched input variants have the same feature/adapter capacity. '
        'The historical three-channel descriptor has a different feature budget and is retained as context only.','',
        'The NumPy/SciPy compatibility warning is preserved in the local runtime record. No environment was changed. '
        'Read `EXPERIMENT_PLAN.md` for the separate guide and masking controls, next checks and execution gates. '
        'No paid job, new efficacy checkpoint, split admission, default promotion or publication was performed.']
    (output/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    for ax,metric,title in zip(axes,('mad','ssim'),('Source discrepancy (MAD)','Source correspondence (regional SSIM)')):
        for i,comparison in enumerate(('source_null','source_target')):
            part=summary[(summary.profile=='additive_alpha050')&(summary.axis=='manufacturer')&
                (summary.threshold==.05)&(summary.comparison==comparison)&(summary.metric==metric)&
                (summary.stratum=='all')].set_index('region').loc[['whole','interior','boundary','background']]
            ax.errorbar(np.arange(4)+(i-.5)*.12,part.estimate,
                yerr=[part.estimate-part.ci_low,part.ci_high-part.estimate],fmt='o',capsize=3,
                label='Original-label null' if i==0 else 'Target')
        ax.set_xticks(range(4),['Whole','Proxy interior','Proxy boundary','Proxy background'],rotation=15)
        ax.set_title(title);ax.legend();ax.grid(axis='y',alpha=.2)
    fig.suptitle('Additive alpha .50 manufacturer; 95% patient intervals\nReused development patients; proxy masks; clinical preservation unknown')
    fig.savefig(output/'regional_diagnosis.png',dpi=150)
    plt.close(fig)
    plan = '''# Controlled next experiments: development preparation only

Status: local prototypes and weak-probe audit complete. No trained finer-guide
model, paid allowance, final cloud package or efficacy result is implied.

## Reference and fixed protocol

Retain additive alpha .50 as a post-hoc manufacturer research reference, with
its immutable parent/full-guide/wrong/absent/learned-zero outputs. Historical
defaults and publication results stay unchanged. Freeze both axes' original
development points, all source/patient memberships, three generation seeds,
fixed source EchoTime, source-native judge frame, three named robust judges,
direction/support rules and patient-bootstrap intervals before new generation.
Field-strength support is still infeasible in one direction; no guide or mask
revision can repair that patient deficit. No test-patient access is needed for
these development checks.

## A. Finer multiscale guide: matched training comparison

Hypothesis: adding finer source local-order structure reduces self-source
detail discrepancy while retaining target-minus-null editing. Native generator
resolution stays 128x128; this is not higher-resolution generator training.

New guide is coarse 3x32 local-order/gradients upsampled to 64, concatenated
with fine 3x64 local-order/gradients. Both derive from the same 5x5 local order;
no raw intensity channel or target label is passed through the guide. This
does not make it acquisition-invariant: the new leakage probe remains strong.

Compare three NEWLY trained copies of the same six-input adapter: coarse/coarse,
coarse/fine, and learned-zero. Same parent, architecture, initial tensors,
parameter count, patient draws, noise/timesteps, photometric augmentation,
condition drops, optimiser, precision retry, exposure and final-epoch rule.
Retain the existing historical three-channel parent/guide as context, not the
matched capacity control. The frozen backbone and conditioner never receive
gradients. Keep the existing coarse structure-consistency loss identical in all
three arms for the first comparison, so guide input and training loss resolution
are not changed together. A finer loss requires a subsequent separate ablation.

Start with the additive reference; retain scale-shift diagnostic evidence. A
separate matched scale-shift replication is conditional on the first comparison,
not an architecture rejection. Use multiple independent adapter training seeds
before a reproducibility claim; generation seeds are not training replicas.

Evaluate correct, zero, absent and deterministic wrong-patient guides, plus
coarse-only/fine-only input erasure as labelled distribution-shift diagnostics.
Use the same guide in both conditional/unconditional CFG arms. Absent-guide
outputs must equal the frozen parent exactly. Keep identity, same-label null,
and target outputs for every case with matched stochastic draws.

Predeclare paired fine-minus-coarse source-MAD/SSIM, rank and HF residuals in
both null and target branches, whole and proxy-interior regions; report every
metric rather than choosing a favourable proxy. Acquisition co-primary is each
named judge's target-minus-null change; raw target hits and other-axis stability
are secondary, with direction/support strata. Directional improvement in detail
alone is insufficient if editing declines. No non-inferiority margin or efficacy
winner is declared in this preparation: a numerically specified margin, training
seed schedule and multiplicity policy must be frozen in the final run protocol
before any paid training/output viewing. Do not choose them from future results.

Source-copy audit before promotion: fixed train-only probes for manufacturer,
field strength and source-intensity reconstruction, with patient-separated
validation and same feature/model capacity; explicit uncertainty and baseline
reconstruction. Check guide amplitude/input erasure/wrong-guide behaviour,
near-identity source outputs alongside null-adjusted editing, photometric and
spatial responses, and injected fine-detail controls. No weak negative probe
certifies absence of leakage or copy shortcuts. Never optimise to frozen judges.

## B. Foreground restriction: separate inference ablation

Do not combine a guide revision with a mask revision in one comparison.
First audit/revise masks on training sources, with source-only sensitivity and
saved validation failure disclosure. The existing .05 foreground proxy can
be almost full-frame; it is currently blocked as an efficacy-ready mask.
No current image-level proxy is verified breast/tumour truth. A future mask
specification, QC/invalid-mask policy and erosion/dilation widths must be frozen
before generated-output evaluation; retain and report every failure.

Prototype uses a binary source mask: 1 edits, 0 projects to a source-consistent
noised exterior after each reverse step and to the clean source at termination.
The exterior path is sampled from q(x_(t-1)|x_t,source) using a separate RNG,
starting from the exact historical noisy source x_t. Thus it is a coherent
forward-DDPM bridge with the right q(x_t|source) marginals. The interior starts
with exactly the old seed/noise; reverse RNG draws remain identical to the old
sampler. Masking may affect interior predictions through convolutions; it is
not equivalent to a final output composite. Exact exterior retention refers
to [-1,1] sampler coordinates converted to [0,1]; floating conversion need not
be bitwise equal to an independently stored original [0,1] array.

Keep a 2x2 comparison: parent unmasked, parent masked (masking only), fixed
existing alpha-.50 adapter unmasked (adapter only), same adapter masked. Keep
identity/null/target in every cell. Add final-compositing-only controls to
separate cosmetic copied-background gains from iterative mask effects; no
compositing output is called a masked-sampler result. No new guide weights are
trained for this ablation.

Check exterior error, boundary-band MAD/gradient jumps, interior rank/HF/SSIM,
whole-image results, mask coverage and copied-pixel fraction. Regional metrics
with windows must disclose cross-boundary support; use strict interior margins
for the actual efficacy protocol. Grade all named frozen judges in the original
source-native frame and retain direction/support differences. Add empty/full
mask and wrong/translated mask failure controls. Do not credit copied background
as preserved internal tissue, or lower thresholds to pass all images.

## Required next local checks and execution boundary

1. Complete train-calibrated mask QC/revision without choosing from edit scores.
2. Add source-reconstruction probes and fine-erasure/texture/boundary controls,
   measured against geometry-preserving transforms with explicit false flags.
3. Wire a NEW versioned trainer/evaluator to these prototypes. Existing trained
   three-input adapters cannot consume the six-input guide; no shape conversion
   or silent checkpoint migration is permitted. Run real frozen-parent full-batch
   CPU/CUDA engineering smoke, memory/timing and exact noise/absent/full-mask
   checks; discard diagnostic updates. No training or sampler defaults change.
4. Freeze exact source/config/data/parent/judge hashes, training seeds, clinical
   limitations, numerical decision criteria, resource/time/cost caps, partial-run
   handling, and one-app/one-GPU execution before a concrete paid proposal.
5. Seek the required separate paid allowance only after that reviewable package
   exists. Old allowances have ended and are not transferred or inferred here.
   No large download, data admission, contact or publication follows from this plan.

Untouched-patient confirmation and protocol-compatible prospective data admission
remain separate later work. Automated proxies cannot certify clinical lesions.
'''
    (output/'EXPERIMENT_PLAN.md').write_text(plan,encoding='utf-8')
    inputs={str(Path(__file__).relative_to(ROOT).as_posix()):digest(__file__)}
    for home in (diagnosis,preparation):
        for name in ('inputs.json','complete.json','protocol.json'):
            p=home/name;inputs[p.relative_to(ROOT).as_posix()]=digest(p)
    for p in (diagnosis/'case_metrics.csv',diagnosis/'case_index.csv',diagnosis/'patient_intervals.csv',
              preparation/'leakage_probe.csv',preparation/'engineering_tests.txt'):
        inputs[p.relative_to(ROOT).as_posix()]=digest(p)
    verify_inputs(inputs)
    json_write(output/'inputs.json',inputs)
    json_write(output/'complete.json',dict(complete=True,summary_entries_recomputed=len(checked),
        max_summary_error=max(c['max_error'] for c in checked),mask_failure_retained=True,
        trained_candidate=False,paid_compute=False,clinical_preservation='unknown',cloud_ready=False))
    (output/'README.md').write_text(
        '# Detail diagnosis interpretation and proposed controls\n\n'
        'Producer: `python -m scripts.v8_generator_dev.report_detail_diagnosis`. '
        '`RESULTS.md`: evidence, patient CIs and limitations. `mask_qc.csv`: all source-proxy coverage/thresholds. '
        '`summary_recheck.csv`: independent recomputation of the reference summary. '
        '`regional_diagnosis.png`: source-null/target regional results. '
        '`EXPERIMENT_PLAN.md`: explicitly gated guide and mask proposals. '
        '`inputs.json`/`complete.json`: provenance and scope. No source report or checkpoint is modified.\n',encoding='utf-8')
    print('DETAIL REPORT COMPLETE',output,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--diagnosis',required=True)
    p.add_argument('--preparation',required=True)
    p.add_argument('--run-id',required=True)
    a=p.parse_args()
    report(a.diagnosis,a.preparation,a.run_id)
