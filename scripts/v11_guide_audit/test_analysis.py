import unittest
import numpy as np
import pandas as pd
from scripts.v11_guide_audit.analyse import CrossedBootstrap,linear_pair,change_metrics


class DiagnosisChecks(unittest.TestCase):
    def setUp(self):
        self.p=dict(training_seeds=[11,12,13],generation_seeds=[1,2,3],bootstrap_seed=9,bootstrap_resamples=300,
                    interactions_family=dict(quantiles=[.05/30,1-.05/30]))
        self.frame=pd.DataFrame([dict(patient_id=p,training_seed=s,generation_seed=g,x=i+2*j+.1*k)
            for i,p in enumerate(['a','b']) for j,s in enumerate(self.p['training_seeds']) for k,g in enumerate([1,2,3])])

    def test_crossed_matches_independent_index_resampling(self):
        summary,_=CrossedBootstrap(self.p).intervals(self.frame,['x'])
        cube=self.frame.x.to_numpy().reshape(2,3,3).mean(2)
        rng=np.random.default_rng(9); pp=rng.integers(2,size=(300,2)); ss=rng.integers(3,size=(300,3))
        direct=cube[pp[:,:,None],ss[:,None,:]].mean((1,2))
        np.testing.assert_allclose([summary[0]['estimate'],summary[0]['ci_low'],summary[0]['ci_high']],
            [cube.mean(),*np.quantile(direct,[.025,.975])],atol=1e-14)

    def test_missing_design_rejected_missing_metric_retained(self):
        boot=CrossedBootstrap(self.p)
        with self.assertRaises(ValueError):boot.intervals(self.frame.iloc[:-1],['x'])
        f=self.frame.copy(); f.loc[0,'x']=np.nan
        rows,_=boot.intervals(f,['x'])
        self.assertFalse(rows[0]['available']);self.assertEqual(rows[0]['invalid_cells'],1)

    def test_difference_interaction_and_mismatched_patient(self):
        base=dict(patient_id='a',dataset_index=3,training_seed=11,generation_seed=1,judge='j',axis='manufacturer',
                  direction='0_to_1',condition_support_status='ok',qc_warning=0)
        f=pd.DataFrame([dict(base,mode=m,guide_control=c,x=v) for m,c,v in
            [('f','correct',7),('c','correct',4),('f','zero',3),('c','zero',2)]])
        specifications=[('f','correct',1),('c','correct',-1),('f','zero',-1),('c','zero',1)]
        self.assertEqual(linear_pair(f,specifications,['x']).x.iloc[0],2)
        f.loc[0,'patient_id']='wrong'
        with self.assertRaises(ValueError):linear_pair(f,specifications,['x'])

    def test_null_target_change_ignores_source_copy_and_background(self):
        source=np.zeros((128,128)); mask=np.zeros_like(source,dtype=bool);mask[30:90,30:90]=True
        null=source.copy(); target=source.copy();target[mask]=.2
        a=change_metrics(source,null,target,mask)
        self.assertAlmostEqual(a['null_to_target_interior_mad'],.2)
        target[~mask]=.9
        self.assertAlmostEqual(change_metrics(source,null,target,mask)['null_to_target_interior_mad'],.2)
        self.assertEqual(change_metrics(source,null,null,mask)['null_to_target_interior_mad'],0.)
        self.assertTrue(np.isnan(change_metrics(source,null,target,np.zeros_like(mask))['null_to_target_interior_mad']))


if __name__=='__main__':unittest.main()
