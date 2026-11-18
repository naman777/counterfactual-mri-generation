"""Strict paired decomposition; literal null controls and warning flags stay distinct."""
import numpy as np
import pandas as pd

KEYS=['patient_id','dataset_index','training_seed','generation_seed','judge','axis']
META=['direction','condition_support_status','qc_warning','timestep','guidance',
      'source_echo_time_norm','target_echo_time_norm']
METRICS=['target_hit','null_hit','paired_hit_delta','target_probability','null_probability','probability_shift',
         'target_interior_mad','null_interior_mad','target_interior_hf_residual_mad',
         'null_interior_hf_residual_mad','target_ssim','target_other_axis_stable']
CONTROLS=('correct','zero','absent','wrong','coarse_only','fine_only')
OLD='old_full_guide';NEW='guide_dropout'


def warnings(values):
    mapping={0:False,1:True,'False':False,'True':True,'0':False,'1':True}
    result=values.map(mapping)
    if result.isna().any():raise ValueError('Unknown warning value')
    return result.astype(bool)


def decompose(frame):
    frame=frame.copy();frame['qc_warning']=warnings(frame.qc_warning)
    route=['study','mode','guide_control',*KEYS]
    parts={variant:frame[frame.variant==variant].set_index(route).sort_index() for variant in ('target','null','identity')}
    target=parts['target'];null=parts['null'];identity=parts['identity']
    if target.empty or target.index.has_duplicates or not target.index.equals(null.index) or not target.index.equals(identity.index):
        raise ValueError('Missing/duplicate original/null/target case')
    if not target[META].equals(null[META]) or not target[META].equals(identity[META]):
        raise ValueError('Control/source metadata changed')
    result=target[META].copy()
    for prefix,part in (('target',target),('null',null)):
        for metric in ('target_hit','target_probability','interior_mad','interior_hf_residual_mad','ssim','other_axis_stable'):
            name=prefix+'_'+metric.removeprefix('target_')
            result[name]=part[metric]
    result['paired_hit_delta']=result.target_hit-result.null_hit
    if not np.array_equal(result.paired_hit_delta,target.paired_hit_delta):raise ValueError('Target/null editing decomposition drift')
    result['probability_shift']=result.target_probability-result.null_probability
    return result.reset_index()


def paired(frame,terms,metrics=METRICS):
    result=None;metadata=None
    for study,mode,control,coefficient in terms:
        part=frame[(frame.study==study)&(frame['mode']==mode)&(frame.guide_control==control)].set_index(KEYS).sort_index()
        if part.empty or part.index.has_duplicates:raise ValueError('Missing/duplicate paired route')
        if metadata is None:metadata=part[META]
        elif not metadata.index.equals(part.index) or not metadata.equals(part[META]):
            raise ValueError('Unpaired source/design metadata')
        value=part[metrics]*coefficient
        result=value if result is None else result+value
    return metadata.join(result).reset_index()


def specifications(axis):
    controls=CONTROLS if axis=='manufacturer' else ('correct',)
    result=[]
    for control in controls:
        for study in (OLD,NEW):
            result.append((f'{study}__fine_minus_coarse__{control}',
                [(study,'coarse_fine',control,1),(study,'coarse_control',control,-1)],False))
        for mode in ('coarse_control','coarse_fine'):
            result.append((f'{mode}__dropout_minus_old__{control}',
                [(NEW,mode,control,1),(OLD,mode,control,-1)],False))
        result.append((f'guide_effect_interaction__{control}',
            [(NEW,'coarse_fine',control,1),(NEW,'coarse_control',control,-1),
             (OLD,'coarse_fine',control,-1),(OLD,'coarse_control',control,1)],axis=='manufacturer'))
    if axis=='manufacturer':
        for study in (OLD,NEW):
            for mode in ('coarse_control','coarse_fine'):
                for control in controls[1:]:
                    result.append((f'{study}__{mode}__correct_minus_{control}',
                        [(study,mode,'correct',1),(study,mode,control,-1)],False))
        for mode in ('coarse_control','coarse_fine'):
            for control in controls[1:]:
                result.append((f'{mode}__dropout_change_in_correct_minus_{control}',
                    [(NEW,mode,'correct',1),(NEW,mode,control,-1),
                     (OLD,mode,'correct',-1),(OLD,mode,control,1)],False))
    for mode in ('coarse_control','coarse_fine'):
        result.append((f'{mode}__dropout_minus_cached_learned_zero',
                       [(NEW,mode,'correct',1),(OLD,'learned_zero','correct',-1)],False))
    return result


def strata(frame):
    return [('all',frame),*[(f'direction:{k}',v) for k,v in frame.groupby('direction')],
            *[(f'support:{k}',v) for k,v in frame.groupby('condition_support_status')],
            *[(f'warning:{k}',v) for k,v in frame.groupby('qc_warning')]]
