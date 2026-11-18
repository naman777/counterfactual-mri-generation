"""Source adapter numerical checks, copy controls and train/validation separation."""
import unittest
import numpy as np
import pandas as pd
import torch
from src.models.diffusion_unet import MRIConditionalUNet
from scripts.v6_structure.model import StructureConditionedUNet,BoundStructureModel,structure_maps,photometric_view
from scripts.v6_structure.train_smoke import state_digest,training_rows
from scripts.v5_anatomy.run_pilot import read_splits


class StructureTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2);torch.manual_seed(7)
        self.base=MRIConditionalUNet(base_channels=4,condition_dim=8,time_embedding_dim=8,dropout=.1).eval()
        self.model=StructureConditionedUNet(self.base,width=2)
        self.x=torch.randn(2,1,128,128);self.t=torch.tensor([20,100]);self.c=torch.randn(2,8)
        self.image=torch.zeros(2,1,128,128);self.image[0,:,20:80,30:95]=.7;self.image[1,:,40:100,10:80]=.6
        self.guide=structure_maps(self.image)

    def test_maps_bottleneck_and_constant_are_finite(self):
        self.assertEqual(tuple(self.guide.shape),(2,3,32,32))
        self.assertTrue(torch.isfinite(self.guide).all());self.assertLessEqual(float(self.guide.abs().max()),1.)
        self.assertTrue(torch.equal(structure_maps(torch.zeros_like(self.image)),torch.zeros_like(self.guide)))

    def test_positive_affine_intensity_change_is_reduced(self):
        other=structure_maps(self.image*.4+.2)
        self.assertLess(float((self.guide-other).abs().max()),.002)

    def test_spatial_changes_remain_visible_in_maps(self):
        shifted=structure_maps(torch.roll(self.image,20,dims=-1))
        self.assertGreater(float((shifted-self.guide).abs().mean()),.01)

    def test_zero_initial_adapter_matches_base_exactly(self):
        self.model.eval()
        with torch.no_grad():
            expected=self.base(self.x,self.t,self.c)
            actual=self.model(self.x,self.t,self.c,self.guide)
        torch.testing.assert_close(expected,actual,rtol=0,atol=0)

    def test_base_stays_frozen_and_eval_in_training(self):
        self.model.train()
        self.assertFalse(self.base.training)
        self.assertTrue(self.model.adapter.training)
        self.assertFalse(any(p.requires_grad for p in self.base.parameters()))

    def test_optimizer_updates_only_adapter_and_reaches_encoder(self):
        before=state_digest(self.base);optim=torch.optim.Adam(self.model.adapter.parameters(),lr=.01)
        self.model.train()
        target=torch.randn_like(self.x)
        for _ in range(2):
            optim.zero_grad();loss=(self.model(self.x,self.t,self.c,self.guide)-target).square().mean()
            loss.backward();optim.step()
        self.assertEqual(before,state_digest(self.base))
        self.assertTrue(all(p.grad is None for p in self.base.parameters()))
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in self.model.adapter.blocks.parameters()),0.)

    def test_presence_zero_disables_trained_branch(self):
        with torch.no_grad():
            for layer in self.model.adapter.outputs:layer.weight.fill_(.01);layer.bias.fill_(.01)
            self.model.eval()
            actual=self.model(self.x,self.t,self.c,self.guide,torch.zeros(2))
            expected=self.base(self.x,self.t,self.c)
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)

    def test_wrong_source_and_no_source_controls_are_distinct(self):
        with torch.no_grad():
            for layer in self.model.adapter.outputs:layer.weight.normal_(0,.1)
            self.model.eval()
            a=self.model(self.x,self.t,self.c,self.guide)
            b=self.model(self.x,self.t,self.c,self.guide.flip(0))
        self.assertFalse(torch.equal(a,b))

    def test_cfg_uses_same_source_guide_in_both_arms(self):
        seen=[]
        handle=self.model.adapter.register_forward_pre_hook(lambda module,args:seen.append(args[0].detach().clone()))
        BoundStructureModel(self.model,self.guide)(self.x.repeat(2,1,1,1),self.t.repeat(2),self.c.repeat(2,1))
        handle.remove()
        torch.testing.assert_close(seen[0][:2],seen[0][2:],rtol=0,atol=0)

    def test_reject_raw_clean_source_and_invalid_gate(self):
        with self.assertRaises(ValueError):self.model(self.x,self.t,self.c,self.image)
        with self.assertRaises(ValueError):self.model(self.x,self.t,self.c,self.guide,torch.tensor([2.,0.]))
        with self.assertRaises(ValueError):structure_maps(self.image*2)

    def test_photometric_view_deterministic_and_no_spatial_warp(self):
        g=torch.Generator().manual_seed(4);a=photometric_view(self.image,g)
        g=torch.Generator().manual_seed(4);b=photometric_view(self.image,g)
        torch.testing.assert_close(a,b,rtol=0,atol=0)
        # Uniform source regions remain uniform: no random spatial deformation.
        region=a[0,0,30:60,40:70]
        self.assertEqual(float(region.max()-region.min()),0.)

    def test_training_selection_only_original_train_patients(self):
        splits=read_splits();chosen=training_rows(splits['train'],16,4,194731)
        self.assertEqual(chosen.patient_id.nunique(),16)
        self.assertTrue(set(chosen.patient_id).isdisjoint(splits['val'].patient_id))
        self.assertTrue(set(chosen.patient_id).isdisjoint(splits['test'].patient_id))
        self.assertTrue((chosen.groupby('patient_id').size()<=4).all())


if __name__=='__main__':unittest.main(verbosity=2)
