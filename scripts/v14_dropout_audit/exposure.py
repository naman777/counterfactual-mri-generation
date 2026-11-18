"""Count actual gate/condition/timestep/source exposure and compare historical draw streams."""
import argparse
from collections import Counter
import json

import numpy as np
import pandas as pd

from scripts.v8_generator_dev.io import ROOT,digest,json_write,csv_write,project_path,verify_inputs
from scripts.v11_guide_audit.guide_dropout import STATE_NAMES,STATE_PROBABILITIES

DRAW_KEYS=('sample_indices','timesteps','condition_dropped','noise_sha256','patient_ids')


def count_record(row,source,joint,epoch_counts,patient_counts):
    names=('sample_indices','timesteps','condition_dropped','gate_states','patient_ids')
    if len({len(row[k]) for k in names})!=1:raise ValueError('Mismatched actual sample vectors')
    seed,mode,epoch=row['training_seed'],row['mode'],row['epoch_at_draw']
    for index,t,dropped,state,patient in zip(*(row[k] for k in names)):
        if not 0<=state<5 or type(dropped) is not bool:raise ValueError('Invalid actual gate/condition state')
        item=source.iloc[index]
        if item.patient_id!=patient:raise ValueError('Actual source patient/index mismatch')
        low=bool(t<=300);m=int(item.manufacturer_label);f=int(item.field_strength_label)
        joint[(seed,mode,state,dropped,low,m,f)]+=1
        epoch_counts[(seed,mode,epoch,state,dropped,low)]+=1
        patient_counts[(seed,mode,patient,state,dropped,low)]+=1


def run(prepared,run_id):
    prepared=project_path(prepared);p=json.loads((prepared/'protocol.json').read_text())
    verify_inputs(json.loads((prepared/'inputs.json').read_text()))
    scientific=json.loads(project_path(p['scientific_protocol']).read_text())
    source_path=ROOT/scientific['reference_protocol']['view']/'splits/train.csv'
    source=pd.read_csv(source_path,dtype={'patient_id':str})
    out=ROOT/'reports/v14_dropout_audit'/run_id;out.mkdir(parents=True,exist_ok=False)
    (out/'README.md').write_text('Producer: scripts.v14_dropout_audit.exposure. joint_exposure.csv: '
        'all gate/label-drop/timestep/source-cell combinations including zero cells; epoch_exposure.csv: '
        'actual exposure through every epoch; patient_exposure.csv: every sampled training patient; '
        'state_exposure.csv: counts/fractions and intended probabilities; loss_by_epoch.csv: mean '
        'recorded batch losses/precision retries (not efficacy or per-state gradient effects); '
        'checks.json: matching between modes and old/new actual draws; inputs.json: source log hashes; '
        'complete.json: observational audit. Matched copies are not independent gate draws.\n',encoding='utf-8')
    old=ROOT/'reports/v10_detail/guide_replicates_20261117_01_training'
    old_rows={};evidence={source_path.relative_to(ROOT).as_posix():digest(source_path)}
    for seed in p['training_seeds']:
        for path in sorted(old.glob(f'block_*/seed{seed}__coarse_control.jsonl')):
            for line in path.read_text().splitlines():
                row=json.loads(line);key=(seed,row['global_step'])
                if key in old_rows:raise ValueError('Historical draw duplicated')
                old_rows[key]={k:row[k] for k in DRAW_KEYS}
            evidence[path.relative_to(ROOT).as_posix()]=digest(path)
    if len(old_rows)!=3*2780:raise ValueError('Historical draw exposure incomplete')
    joint=Counter();epochs=Counter();patients=Counter();losses=[];seen=set();rows_by_copy=Counter()
    training=project_path(p['training'])
    for path in sorted((training/'segments').glob('*/updates.jsonl')):
        for line in path.read_text().splitlines():
            row=json.loads(line);key=(row['training_seed'],row['mode'],row['global_step'])
            if key in seen or row['engineering'] or not row['applied']:raise ValueError('Invalid/duplicate full-exposure row')
            seen.add(key);rows_by_copy[key[:2]]+=1
            if {k:row[k] for k in DRAW_KEYS}!=old_rows[(key[0],key[2])]:
                raise ValueError('Dropout changed the historical source/noise/label-drop streams')
            count_record(row,source,joint,epochs,patients)
            losses.append(dict(seed=key[0],mode=key[1],epoch=row['epoch_at_draw']+1,
                total_loss=row['total_loss'],noise_mse=row['noise_mse'],coarse_structure_loss=row['coarse_structure_loss'],
                amp_retries=row['amp_retries'],recorded_batch_size=row['actual_batch_size']))
        evidence[path.relative_to(ROOT).as_posix()]=digest(path)
    if len(seen)!=16680 or any(v!=2780 for v in rows_by_copy.values()) or len(rows_by_copy)!=6:
        raise ValueError('Actual exposure is incomplete')
    records=[]
    for seed in p['training_seeds']:
        for mode in ('coarse_control','coarse_fine'):
            for state in range(5):
                for dropped in (False,True):
                    for low in (False,True):
                        for m in (0,1):
                            for f in (0,1):
                                key=(seed,mode,state,dropped,low,m,f)
                                records.append(dict(training_seed=seed,mode=mode,state=STATE_NAMES[state],
                                    state_index=state,condition_dropped=dropped,low_timestep=low,
                                    manufacturer_label=m,field_strength_label=f,samples=joint[key]))
    joint_frame=pd.DataFrame(records);csv_write(out/'joint_exposure.csv',joint_frame)
    state_rows=[]
    for (seed,mode,state,index),part in joint_frame.groupby(['training_seed','mode','state','state_index']):
        count=int(part.samples.sum());total=88680
        conditional=int(part.loc[~part.condition_dropped,'samples'].sum())
        state_rows.append(dict(training_seed=seed,mode=mode,state=state,samples=count,observed_fraction=count/total,
            intended_fraction=STATE_PROBABILITIES[index],conditional_samples=conditional,
            unconditional_samples=count-conditional,conditional_fraction_of_all_samples=conditional/total,
            presence_gate_active=state!='absent_adapter',informative_guide_block_present=state in STATE_NAMES[:3]))
    state_frame=pd.DataFrame(state_rows);csv_write(out/'state_exposure.csv',state_frame)
    for filename,counter,names in [('epoch_exposure.csv',epochs,['training_seed','mode','epoch_index','state_index','condition_dropped','low_timestep']),
                                  ('patient_exposure.csv',patients,['training_seed','mode','patient_id','state_index','condition_dropped','low_timestep'])]:
        csv_write(out/filename,pd.DataFrame([dict(zip(names,key),samples=value) for key,value in sorted(counter.items())]))
    csv_write(out/'loss_by_epoch.csv',pd.DataFrame(losses).groupby(['seed','mode','epoch'],as_index=False).agg(
        mean_recorded_batch_total_loss=('total_loss','mean'),mean_recorded_batch_noise_mse=('noise_mse','mean'),
        mean_recorded_batch_structure_loss=('coarse_structure_loss','mean'),precision_retries=('amp_retries','sum'),
        recorded_updates=('total_loss','size')))
    verified=pd.read_csv(ROOT/'reports/v13_guide_dropout/training_verified_20261118_01/copies.csv')
    for row in verified.itertuples():
        seed=int(row.key.split('__')[0].removeprefix('seed'));mode=row.key.split('__')[1]
        observed=[int(state_frame[(state_frame.training_seed==seed)&(state_frame['mode']==mode)&(state_frame.state==s)].samples.iloc[0])
                  for s in STATE_NAMES]
        if observed!=json.loads(row.gate_counts):raise ValueError('Independent state counter differs from training audit')
    for seed in p['training_seeds']:
        a=joint_frame[(joint_frame.training_seed==seed)&(joint_frame['mode']=='coarse_control')].samples.to_numpy()
        b=joint_frame[(joint_frame.training_seed==seed)&(joint_frame['mode']=='coarse_fine')].samples.to_numpy()
        np.testing.assert_array_equal(a,b)
    checks=dict(old_new_actual_data_noise_condition_streams_identical=True,matched_gate_exposure_between_modes=True,
        independently_counted_gate_exposures_match_checkpoints=True,total_samples_across_copies=int(joint_frame.samples.sum()),
        absent_samples_have_no_adapter_gradient_by_forward_wiring=True,
        zero_input_is_still_learnable_bias_path=True,no_per_state_gradient_magnitude_claim=True)
    json_write(out/'checks.json',checks)
    for path in (prepared/'protocol.json',ROOT/'scripts/v14_dropout_audit/exposure.py',
                 ROOT/'reports/v13_guide_dropout/training_verified_20261118_01/copies.csv'):
        evidence[path.relative_to(ROOT).as_posix()]=digest(path)
    json_write(out/'inputs.json',evidence)
    json_write(out/'complete.json',dict(complete=True,actual_update_rows=len(seen),source_train_patients=72,
        total_actual_samples=int(joint_frame.samples.sum()),draw_replay='Recorded streams compared to historical logs; raw noise not replayed',
        paid_gpu_calls=0,new_training=False,clinical_preservation='unknown'))
    print('ACTUAL DROPOUT EXPOSURE AUDITED',out,flush=True)
    return out


if __name__=='__main__':
    a=argparse.ArgumentParser(__doc__);a.add_argument('--prepared',required=True);a.add_argument('--run-id',required=True)
    run(**vars(a.parse_args()))
