import unittest
import pandas as pd

from scripts.v14_dropout_audit.common import decompose,paired,warnings


class PairChecks(unittest.TestCase):
    def base(self):
        return dict(patient_id='p',dataset_index=1,training_seed=2,generation_seed=3,judge='j',axis='manufacturer',
            direction='0_to_1',condition_support_status='observed_condition',qc_warning=1,timestep=300,guidance=3.,
            source_echo_time_norm=.3,target_echo_time_norm=.3)

    def test_target_null_decomposition_and_missing_control_rejection(self):
        rows=[]
        for variant,hit,prob in [('identity',0,.1),('null',1,.6),('target',1,.9)]:
            rows.append(dict(self.base(),study='new',mode='fine',guide_control='correct',variant=variant,
                target_hit=hit,target_probability=prob,paired_hit_delta=hit-1,interior_mad=.1,
                interior_hf_residual_mad=.01,ssim=.8,other_axis_stable=1))
        result=decompose(pd.DataFrame(rows))
        self.assertEqual(result.paired_hit_delta.iloc[0],0)
        self.assertAlmostEqual(result.probability_shift.iloc[0],.3)
        with self.assertRaises(ValueError):decompose(pd.DataFrame(rows[:-1]))

    def test_four_cell_interaction_and_metadata_rejection(self):
        frame=pd.DataFrame([dict(self.base(),study=s,mode=m,guide_control='correct',x=v) for s,m,v in
            [('new','fine',7),('new','coarse',4),('old','fine',4),('old','coarse',2)]])
        terms=[('new','fine','correct',1),('new','coarse','correct',-1),('old','fine','correct',-1),('old','coarse','correct',1)]
        self.assertEqual(paired(frame,terms,['x']).x.iloc[0],1)
        frame.loc[0,'qc_warning']=0
        with self.assertRaises(ValueError):paired(frame,terms,['x'])

    def test_integer_warning_is_a_mask_not_a_row_label(self):
        frame=pd.DataFrame(dict(patient=['a','b','c'],warning=[0,1,0]))
        self.assertEqual(frame.loc[warnings(frame.warning),'patient'].tolist(),['b'])
        with self.assertRaises(ValueError):warnings(pd.Series(['unknown']))


if __name__=='__main__':unittest.main()
