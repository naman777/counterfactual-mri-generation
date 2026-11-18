import unittest

import pandas as pd

from scripts.v13_guide_dropout.analyse import gates
from scripts.v13_guide_dropout.test_training import protocol


def fixture(p):
    reference=p['reference_protocol'];e=reference['evaluation']
    rows=[]
    for patient in ('a','b'):
        for seed in reference['training_seeds']:
            for generation in (e['selection_generation_seed'],*e['confirmation_generation_seeds']):
                for j in range(3):
                    for variant in ('null','target'):
                        for mode in p['modes']:
                            rows.append(dict(patient_id=patient,dataset_index=0 if patient=='a' else 1,
                                training_seed=seed,generation_seed=generation,judge=f'heldout_orientation_robust_seed{j}',
                                variant=variant,mode=mode,axis='manufacturer',direction='0_to_1',
                                condition_support_status='observed_condition',timestep=300,guidance=3.,
                                source_echo_time_norm=.3,target_echo_time_norm=.3,qc_warning=False,
                                guide_control='correct',paired_hit_delta=.2,
                                interior_mad=.09 if mode=='coarse_fine' else .1,
                                interior_hf_residual_mad=.09 if mode=='coarse_fine' else .1))
    return pd.DataFrame(rows)


class AnalysisChecks(unittest.TestCase):
    def test_all_ten_gates_and_named_adverse_old_baseline(self):
        p=protocol();new=fixture(p);old=new.copy()
        result,_=gates(new,old,p)
        self.assertEqual(len(result),10)
        self.assertTrue(all(r['available'] and r['interval_passed'] and r['replica_direction_passed'] for r in result))
        # One adverse named judge relative to the old fine guide must prevent advancement.
        old.loc[(old['mode']=='coarse_fine')&(old.judge=='heldout_orientation_robust_seed1'),'paired_hit_delta']=.4
        result,_=gates(new,old,p)
        failed=[r for r in result if not r['interval_passed']]
        self.assertEqual(len(failed),1)
        self.assertEqual(failed[0]['rule']['comparison'],'new_fine_minus_old_fine')
        self.assertEqual(failed[0]['judge'],'heldout_orientation_robust_seed1')

    def test_invalid_primary_metric_remains_unavailable(self):
        p=protocol();new=fixture(p);old=new.copy()
        new.loc[0,'interior_mad']=float('nan')
        result,_=gates(new,old,p)
        self.assertTrue(any(not r['available'] for r in result))


if __name__=='__main__':unittest.main()
