"""Paired local diagnosis with all controls, replicas, judges and unavailable cells."""
import argparse
import hashlib
import json

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

from scripts.v8_generator_dev.io import ROOT, csv_write, digest, json_write, project_path, verify_inputs
from scripts.v8_generator_dev.grading_io import read_grading_csv
from scripts.v9_detail.masks import noise_aware_mask

KEYS = ['patient_id','dataset_index','training_seed','generation_seed','judge','axis']
META = ['direction','condition_support_status','qc_warning']


def terms(axis):
    guided = ['coarse_control','coarse_fine']
    controls = ['correct','zero','absent','wrong','coarse_only','fine_only'] if axis == 'manufacturer' else ['correct']
    output = []
    for control in controls:
        output.append((f'fine_minus_coarse__{control}', [('coarse_fine',control,1),('coarse_control',control,-1)], False))
    if axis == 'manufacturer':
        for mode in guided:
            for control in controls[1:]:
                output.append((f'{mode}__correct_minus_{control}', [(mode,'correct',1),(mode,control,-1)], False))
        for control in controls[1:]:
            output.append((f'interaction__correct_minus_{control}', [('coarse_fine','correct',1),('coarse_control','correct',-1),
                ('coarse_fine',control,-1),('coarse_control',control,1)], True))
        output.append(('learned_zero__correct_minus_absent',[('learned_zero','correct',1),('learned_zero','absent',-1)],False))
    for mode in guided:
        output.append((f'{mode}__correct_minus_learned_zero',[(mode,'correct',1),('learned_zero','correct',-1)],False))
    return output


def linear_pair(frame, specifications, metrics):
    parts = []
    reference = None
    for mode, control, coefficient in specifications:
        part = frame[(frame['mode'] == mode) & (frame.guide_control == control)].copy()
        if part.empty or part.duplicated(KEYS).any():
            raise ValueError('Missing/duplicate paired route')
        part = part.set_index(KEYS).sort_index()
        if reference is None:
            reference = part[META].copy()
        elif not part.index.equals(reference.index) or not part[META].equals(reference):
            raise ValueError('Unpaired design or source metadata drift')
        parts.append(part[metrics] * coefficient)
    result = parts[0]
    for part in parts[1:]:
        result = result + part
    return reference.join(result).reset_index()


class CrossedBootstrap:
    def __init__(self, protocol):
        self.p = protocol
        self.weights = {}

    def design(self, patients):
        if patients not in self.weights:
            n = self.p['bootstrap_resamples']; seeds = len(self.p['training_seeds'])
            rng = np.random.default_rng(self.p['bootstrap_seed'])
            pi = rng.integers(patients, size=(n,patients)); si = rng.integers(seeds,size=(n,seeds))
            wp = np.zeros((n,patients)); ws = np.zeros((n,seeds))
            np.add.at(wp,(np.repeat(np.arange(n),patients),pi.ravel()),1)
            np.add.at(ws,(np.repeat(np.arange(n),seeds),si.ravel()),1)
            self.weights[patients] = (wp,ws)
        return self.weights[patients]

    def intervals(self, frame, metrics, family=False):
        patients = sorted(frame.patient_id.unique()); seeds = self.p['training_seeds']; generations = self.p['generation_seeds']
        keys = ['patient_id','training_seed','generation_seed']
        if frame.duplicated(keys).any():
            raise ValueError('Duplicate crossed cell')
        index = pd.MultiIndex.from_product([patients,seeds,generations],names=keys)
        observed = pd.MultiIndex.from_frame(frame[keys])
        if len(observed) != len(index) or set(observed) != set(index):
            raise ValueError('Missing patient/training/generation cell')
        raw = frame.set_index(keys)[metrics].reindex(index).to_numpy(float).reshape(len(patients),len(seeds),len(generations),len(metrics))
        available = np.isfinite(raw).all(axis=(0,1,2))
        values = raw.mean(axis=2)
        wp, ws = self.design(len(patients))
        draws = np.einsum('np,psm,ns->nm',wp,values[:,:,available],ws,optimize=False) / (len(patients)*len(seeds))
        quantiles = np.quantile(draws,[.025,.975,*self.p['interactions_family']['quantiles']],axis=0)
        summaries = []; replicas = []; column = 0
        for j, metric in enumerate(metrics):
            common = dict(metric=metric,patients=len(patients),training_replicas=len(seeds),generation_seeds=len(generations),
                          invalid_cells=int((~np.isfinite(raw[:,:,:,j])).sum()),available=bool(available[j]))
            if available[j]:
                summaries.append(dict(common,estimate=float(values[:,:,j].mean()),ci_low=float(quantiles[0,column]),
                    ci_high=float(quantiles[1,column]),family_ci_low=float(quantiles[2,column]) if family and metric=='paired_hit_delta' else None,
                    family_ci_high=float(quantiles[3,column]) if family and metric=='paired_hit_delta' else None,
                    family_adjusted=family and metric=='paired_hit_delta'))
                column += 1
            else:
                summaries.append(dict(common,estimate=None,ci_low=None,ci_high=None,family_adjusted=False,
                    family_ci_low=None,family_ci_high=None))
            for s, seed in enumerate(seeds):
                per = values[:,s,j]
                if np.isfinite(per).all():
                    boot = np.einsum('np,p->n',wp,per) / len(patients)
                    lo,hi = np.quantile(boot,[.025,.975])
                    replicas.append(dict(common,training_seed=seed,estimate=float(per.mean()),ci_low=float(lo),ci_high=float(hi)))
                else:
                    replicas.append(dict(common,training_seed=seed,estimate=None,ci_low=None,ci_high=None))
        return summaries, replicas


def change_metrics(source, null, target, interior):
    arrays = [source,null,target]
    if any(a.shape != (128,128) or not np.isfinite(a).all() for a in arrays):
        raise ValueError('Invalid saved image')
    if interior.sum() < 64:
        return dict(null_to_target_interior_mad=np.nan,null_to_target_hf_mad=np.nan)
    change = target.astype(np.float64) - null.astype(np.float64)
    high = change - ndi.gaussian_filter(change,1,mode='reflect',truncate=3.)
    return dict(null_to_target_interior_mad=float(np.abs(change[interior]).mean()),
                null_to_target_hf_mad=float(np.abs(high[interior]).mean()))


def load_cases(receipt, metrics):
    frames = []; evidence = {}; masks = {}; sources = {}
    for profile, relative in receipt['evaluations'].items():
        home = project_path(relative)
        table = read_grading_csv(home/'per_image.csv')
        target = table[table.variant.eq('target')].sort_values(['case','judge']).copy()
        null = table[table.variant.eq('null')].sort_values(['case','judge'])
        if not target[['case','judge']].reset_index(drop=True).equals(null[['case','judge']].reset_index(drop=True)):
            raise ValueError('Null/target pairing mismatch')
        target['null_hit'] = null.target_hit.to_numpy()
        target['null_probability'] = null.target_probability.to_numpy()
        target['probability_shift'] = target.target_probability-target.null_probability
        for name in ('interior_mad','interior_hf_residual_mad','ssim'):
            target['null_'+name] = null[name].to_numpy()
        if not np.array_equal(target.paired_hit_delta,target.target_hit-target.null_hit):
            raise ValueError('Editing decomposition mismatch')
        requests = pd.read_csv(home/'requests.csv',keep_default_na=False,na_values=[''])
        additions = []
        for r in requests.itertuples():
            path = home/r.array_file
            if digest(path) != r.array_sha256:
                raise ValueError('Saved array identity changed')
            evidence[path.relative_to(ROOT).as_posix()] = r.array_sha256
            with np.load(path,allow_pickle=False) as z:
                source = z['source']; index = r.dataset_index
                sha = hashlib.sha256(source.tobytes()).hexdigest()
                if index in sources and sources[index] != sha:
                    raise ValueError('Source differs across paired cases')
                if index not in masks:
                    masks[index] = ndi.binary_erosion(noise_aware_mask(source)[0],iterations=8,border_value=0)
                    sources[index] = sha
                additions.append(dict(case=r.case,**change_metrics(source,z['null'],z['target'],masks[index])))
        target = target.merge(pd.DataFrame(additions),on='case',validate='many_to_one')
        target['profile'] = profile
        frames.append(target[['profile','case',*KEYS,*META,'mode','guide_control',*metrics]])
        for filename in ('per_image.csv','requests.csv','protocol.json'):
            path=home/filename; evidence[path.relative_to(ROOT).as_posix()]=digest(path)
        print('LOADED',profile,flush=True)
    frame = pd.concat(frames,ignore_index=True)
    if len(frame) != 2295*3:
        raise ValueError('Incomplete case/judge matrix')
    return frame,evidence


def strata(frame):
    return [('all',frame),*[(f'direction:{k}',v) for k,v in frame.groupby('direction')],
        *[(f'support:{k}',v) for k,v in frame.groupby('condition_support_status')],
        *[(f'warning:{k}',v) for k,v in frame.groupby('qc_warning')]]


def primary_attribution(frame, metrics):
    output = []
    for axis in ('manufacturer','field_strength'):
        part = frame[(frame.axis==axis)&frame.guide_control.eq('correct')]
        fine = part[part['mode'].eq('coarse_fine')].set_index(KEYS).sort_index()
        coarse = part[part['mode'].eq('coarse_control')].set_index(KEYS).sort_index()
        if not fine.index.equals(coarse.index):
            raise ValueError('Primary attribution pairing failed')
        joined = fine[META].copy()
        for metric in metrics:
            joined[metric+'_difference'] = fine[metric]-coarse[metric]
        for metric in ('target_hit','null_hit','paired_hit_delta'):
            joined[metric+'_fine'] = fine[metric]; joined[metric+'_coarse'] = coarse[metric]
        delta = joined.paired_hit_delta_difference
        joined['editing_outcome'] = np.select([delta.lt(0),delta.gt(0)],['loss','gain'],default='unchanged')
        output.append(joined.reset_index())
    return pd.concat(output,ignore_index=True)


def analyse(prepared, run_id):
    prepared = project_path(prepared); p = json.loads((prepared/'protocol.json').read_text())
    verify_inputs(json.loads((prepared/'inputs.json').read_text()))
    receipt = json.loads(project_path(p['previous_results']).read_text())
    output = ROOT/'reports/v11_guide_audit'/run_id
    output.mkdir(parents=True,exist_ok=False)
    (output/'source.py').write_text(project_path(__file__).read_text(),encoding='utf-8')
    json_write(output/'protocol.json',p)
    (output/'README.md').write_text('Producer: scripts.v11_guide_audit.analyse (source.py is the exact snapshot). '
        'protocol.json: prior exploratory plan; case_metrics.csv: every source/generation/training/judge/control case; '
        'contrast_intervals.csv: all paired contrasts/strata with descriptive crossed CIs and the 15-member interaction '
        'family where declared; per_training_seed.csv: paired patient CIs within every replica; '
        'primary_cases.csv and primary_patients.csv: all primary gains/losses and target/null decomposition; '
        'leave_one_patient_out.csv: full influence sensitivity, not case selection; inputs.json: exact immutable inputs; '
        'complete.json: scope. No clinical, fresh-test, causal-mechanism or advancement claim.\n',encoding='utf-8')
    metrics=p['metrics']; frame,evidence = load_cases(receipt,metrics)
    csv_write(output/'case_metrics.csv',frame)
    boot = CrossedBootstrap(p); summaries=[]; replicas=[]; contrast_count=0
    for axis, part in frame.groupby('axis'):
        for name, specifications, interaction in terms(axis):
            paired=linear_pair(part,specifications,metrics); contrast_count+=1
            for judge, group in paired.groupby('judge'):
                for stratum, rows in strata(group):
                    result, per_seed = boot.intervals(rows,metrics,family=interaction and stratum=='all')
                    meta=dict(axis=axis,contrast=name,judge=judge,stratum=stratum,interaction=interaction)
                    summaries.extend(dict(meta,**r) for r in result)
                    replicas.extend(dict(meta,**r) for r in per_seed)
            print('PAIRED',axis,name,flush=True)
    csv_write(output/'contrast_intervals.csv',pd.DataFrame(summaries))
    csv_write(output/'per_training_seed.csv',pd.DataFrame(replicas))
    cases = primary_attribution(frame,metrics)
    csv_write(output/'primary_cases.csv',cases)
    patient_keys=['axis','judge','patient_id','dataset_index','training_seed',*META]
    means=cases.groupby(patient_keys,dropna=False)[[c for c in cases if c.endswith('_difference')]].mean().reset_index()
    counts=cases.groupby(patient_keys,dropna=False).editing_outcome.agg(lambda x:';'.join(x)).reset_index(name='generation_outcomes')
    means=means.merge(counts,on=patient_keys,validate='one_to_one')
    csv_write(output/'primary_patients.csv',means)
    influence=[]
    for (axis,judge), part in cases.groupby(['axis','judge']):
        for patient in sorted(part.patient_id.unique()):
            remaining=part[part.patient_id.ne(patient)]
            influence.append(dict(axis=axis,judge=judge,omitted_patient=patient,
                editing_difference=float(remaining.paired_hit_delta_difference.mean()),
                all_patient_difference=float(part.paired_hit_delta_difference.mean())))
    csv_write(output/'leave_one_patient_out.csv',pd.DataFrame(influence))
    for path in (prepared/'protocol.json',prepared/'inputs.json',project_path(__file__)):
        evidence[path.relative_to(ROOT).as_posix()]=digest(path)
    json_write(output/'inputs.json',evidence)
    json_write(output/'complete.json',dict(complete=True,exploratory=True,cases=2295,case_judge_rows=len(frame),
        contrasts=contrast_count,contrast_interval_rows=len(summaries),all_judges_controls_patients_retained=True,
        new_sampling=False,paid_gpu_calls=0,clinical_preservation='unknown',advancement_retested=False))
    return output


if __name__=='__main__':
    a=argparse.ArgumentParser(__doc__); a.add_argument('--prepared',required=True); a.add_argument('--run-id',required=True)
    q=a.parse_args(); analyse(q.prepared,q.run_id)
