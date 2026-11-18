import unittest
import numpy as np
from scipy import ndimage as ndi

from scripts.v5_anatomy.metrics import foreground_proxy
from scripts.v9_detail.masks import noise_aware_mask,mask_iou
from scripts.v9_detail.reconstruction import fit_global,predict_global,fit_local,predict_local
from scripts.v9_detail.controls import measure,damage_controls,domains
from scripts.v9_detail.common import interval
import pandas as pd


class LocalPreparationTests(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(91)
        self.source=np.full((128,128),.07,np.float32)
        self.source[25:103,25:103]=rng.uniform(.3,.8,(78,78))

    def test_noisy_background_and_photometric_stability(self):
        old=foreground_proxy(self.source,.05)
        candidate,_=noise_aware_mask(self.source)
        self.assertGreater(old.mean(),.9)
        self.assertLess(candidate.mean(),.6)
        self.assertTrue(candidate[64,64])
        affine,_=noise_aware_mask(self.source*.8+.04)
        self.assertGreater(mask_iou(candidate,affine),.9)

    def test_invalid_constant_and_empty_domain_are_explicit(self):
        source=np.zeros((128,128),np.float32)
        self.assertFalse(measure(source,source)['valid'])
        self.assertEqual(damage_controls(source,1),[])
        with self.assertRaises(ValueError):noise_aware_mask(np.full((128,128),2.))

    def test_identity_and_small_erasure_change_the_measurements(self):
        identity=measure(self.source,self.source)
        self.assertTrue(identity['valid'])
        for key in ('rank_patch_max','texture_deficit','hf_patch_max','boundary_change'):
            self.assertEqual(identity[key],0)
        edit=self.source.copy();edit[64,64]=0
        damaged=measure(self.source,edit)
        self.assertGreater(damaged['hf_patch_max'],0)
        self.assertGreater(damaged['rank_patch_max'],0)

    def test_controls_are_inside_strict_source_domain_and_reproducible(self):
        _,interior,_=domains(self.source)
        a=damage_controls(self.source,32);b=damage_controls(self.source,32)
        self.assertEqual(len(a),20)
        for left,right in zip(a,b):
            name,location,edit,info=left
            np.testing.assert_array_equal(edit,right[2])
            if info['y']>=0:
                self.assertTrue(interior[info['y'],info['x']])
            self.assertTrue(np.isfinite(edit).all())

    def test_global_decoder_train_centring_and_wrong_guides(self):
        rng=np.random.default_rng(17)
        x=rng.normal(size=(20,6))
        y=(.5+.1*x[:,:2]).clip(0,1)
        model=fit_global(x[:15],y[:15],alpha=.001)
        np.testing.assert_allclose(model['mean'],x[:15].mean(0))
        predicted=predict_global(model,x[15:])
        wrong=predict_global(model,np.roll(x[15:],1,axis=0))
        self.assertLess(np.abs(predicted-y[15:]).mean(),np.abs(wrong-y[15:]).mean())
        before=model['coefficient'].copy()
        predict_global(model,x[15:]+100)
        np.testing.assert_array_equal(before,model['coefficient'])

    def test_local_zero_guide_is_training_mean(self):
        y=np.arange(2*8*8).reshape(2,8,8)/128.
        maps=np.zeros((2,6,8,8))
        model=fit_local(maps,y)
        np.testing.assert_allclose(predict_local(model,np.zeros((3,6,8,8))),y.mean())

    def test_patient_intervals_average_locations_before_resampling(self):
        frame=pd.DataFrame(dict(patient_id=['a','a','a','b'],value=[0,0,0,1]))
        self.assertEqual(interval(frame,'value',n=100)['estimate'],.5)


if __name__=='__main__':unittest.main()
