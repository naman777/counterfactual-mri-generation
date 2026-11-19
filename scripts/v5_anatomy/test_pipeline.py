"""Numerical, geometry and leakage checks for the anatomy development pilot."""
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from scripts.v4_convergence.core import sample, target
from src.models.ddpm_scheduler import DDPMScheduler
from scripts.v5_anatomy.anchored_sampler import sample_anchored, lowpass
from scripts.v5_anatomy.metrics import diagnostics, foreground_proxy, mask_metrics, landmark_metrics, mask_array
from scripts.v5_anatomy.run_pilot import select_validation, support_flags, read_splits
from scripts.v5_anatomy.score_review import label


class TinyCondition(nn.Module):
    def forward(self,m,f,e,force_unconditional=False):
        return (m.float()*.01+f.float()*.02+e*.01)[:,None]*int(not force_unconditional)


class TinyDenoiser(nn.Module):
    def __init__(self):
        super().__init__(); self.scale=nn.Parameter(torch.tensor(.05))
    def forward(self,x,t,c):
        return x*self.scale+c[:,:,None,None]


class AnatomyTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.image=np.zeros((128,128),np.float32)
        self.image[32:96,36:92]=.7
        self.cond=dict(mfg=np.array([0]),fs=np.array([1]),echo=np.array([.4],np.float32))
        self.c,self.u,self.s=TinyCondition(),TinyDenoiser(),DDPMScheduler(num_train_timesteps=20)

    def test_identity_and_translation_detected(self):
        a=self.image>.1; b=np.roll(a,12,axis=1)
        same=mask_metrics(a,a); moved=mask_metrics(a,b)
        self.assertEqual(same['dice'],1.)
        self.assertEqual(same['boundary_hd95_px'],0.)
        self.assertLess(moved['dice'],.9)
        self.assertAlmostEqual(moved['centroid_shift_px'],12.)
        self.assertGreater(moved['boundary_hd95_px'],8)

    def test_empty_masks_do_not_pass(self):
        result=mask_metrics(np.zeros((20,20)),np.zeros((20,20)))
        self.assertFalse(result['valid']); self.assertIsNone(result['dice'])

    def test_small_component_not_anatomy_ground_truth(self):
        image=self.image.copy(); image[0,0]=1
        mask=foreground_proxy(image)
        self.assertFalse(mask[0,0])
        self.assertTrue(mask[64,64])
        self.assertFalse(diagnostics(image,image)['anatomy_validated'])

    def test_metrics_reject_invalid_images(self):
        for value in [np.full((128,128),np.nan),np.ones((128,128))*1.1]:
            with self.assertRaises(ValueError): diagnostics(self.image,value)
        with self.assertRaises(ValueError): diagnostics(self.image,np.zeros((64,64)))

    def test_verified_masks_and_landmarks_require_same_native_grid(self):
        with self.assertRaises(ValueError): mask_array(np.ones((64,64)),(128,128))
        with self.assertRaises(ValueError): mask_array(np.ones((128,128))*.3,(128,128))
        result=landmark_metrics({'chest':[20,30]},{'chest':[23,34]},(128,128))
        self.assertEqual(result['landmark_mean_shift_px'],5.)
        with self.assertRaises(ValueError): landmark_metrics({'a':[0,0]},{'b':[0,0]},(128,128))

    def test_zero_anchor_bitwise_reference_for_cfg_branches(self):
        for guidance in [0.,1.,5.]:
            expected=sample(self.c,self.u,self.s,self.cond,[42],5,guidance,self.image[None])
            actual=sample_anchored(self.c,self.u,self.s,self.cond,[42],5,guidance,self.image[None],0.)
            np.testing.assert_array_equal(expected,actual)

    def test_anchor_is_deterministic_and_does_not_change_inputs(self):
        before=self.image.copy()
        a=sample_anchored(self.c,self.u,self.s,self.cond,[42],5,3.,self.image[None],.5)
        b=sample_anchored(self.c,self.u,self.s,self.cond,[42],5,3.,self.image[None],.5)
        np.testing.assert_array_equal(a,b); np.testing.assert_array_equal(self.image,before)
        self.assertTrue(np.isfinite(a).all()); self.assertGreaterEqual(a.min(),0); self.assertLessEqual(a.max(),1)

    def test_strong_anchor_reduces_coarse_error_in_controlled_fixture(self):
        a=sample_anchored(self.c,self.u,self.s,self.cond,[42],10,3.,self.image[None],0.)
        b=sample_anchored(self.c,self.u,self.s,self.cond,[42],10,3.,self.image[None],1.)
        source=torch.tensor(self.image)[None,None]
        error=lambda x: float((lowpass(torch.tensor(x)[:,None])-lowpass(source)).square().mean())
        self.assertLess(error(b),error(a))

    def test_sampler_rejects_invalid_controls(self):
        for strength in [-.1,1.1,float('nan')]:
            with self.assertRaises(ValueError):
                sample_anchored(self.c,self.u,self.s,self.cond,[42],5,3.,self.image[None],strength)

    def test_patient_selection_does_not_read_outcomes(self):
        frame=pd.DataFrame({'patient_id':['a','a','b','b','c'],'image_path':range(5)})
        selected=select_validation(frame,3,731)
        self.assertEqual(set(selected.patient_id),{'a','b','c'})
        self.assertFalse(selected.patient_id.duplicated().any())
        pd.testing.assert_frame_equal(selected,select_validation(frame,3,731))
        with self.assertRaises(ValueError): select_validation(frame,4,731)

    def test_target_keeps_echo_and_other_axis(self):
        for axis,key,other in [('manufacturer','mfg','fs'),('field_strength','fs','mfg')]:
            result=target(self.cond,axis)
            np.testing.assert_array_equal(result['echo'],self.cond['echo'])
            np.testing.assert_array_equal(result[other],self.cond[other])
            np.testing.assert_array_equal(result[key],1-self.cond[key])

    def test_unsupported_echo_is_reported(self):
        train=pd.DataFrame(dict(manufacturer_label=[0,0],field_strength_label=[0,1],
                                echo_time_norm=[.4,.8],patient_id=['a','b']))
        src=dict(mfg=np.array([0]),fs=np.array([0]),echo=np.array([.4]))
        flags=support_flags(train,src,target(src,'field_strength'))
        self.assertFalse(flags['target_echo_exact_seen']); self.assertFalse(flags['target_echo_in_train_range'])

    def test_missing_review_is_never_a_pass(self):
        self.assertIsNone(label('')); self.assertIsNone(label('nan'))
        self.assertTrue(label('yes')); self.assertFalse(label('no'))
        with self.assertRaises(ValueError): label('probably')

    def test_actual_split_patient_separation(self):
        splits=read_splits()
        selected=select_validation(splits['val'],15,731)
        self.assertEqual(len(selected),15)
        self.assertFalse(set(selected.patient_id)&set(splits['test'].patient_id))


if __name__=='__main__':
    unittest.main(verbosity=2)
