import unittest
import numpy as np
import torch
import torch.nn.functional as F

from scripts.v6_structure.model import BoundStructureModel
from scripts.v8_generator_dev.detail_candidates import (
    local_order_native, order_maps, multiscale_guide, DetailConditionedUNet, masked_sample)
from scripts.v8_generator_dev.model import build_models
from scripts.v8_generator_dev.soft_structure import soft_structure
from scripts.v8_generator_dev.test_pipeline import tiny_config
from scripts.v8_generator_dev.train_smoke import sample_wiring


class DetailCandidatesTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.image=torch.rand(1,1,128,128,generator=torch.Generator().manual_seed(31))

    def test_coarse_implementation_matches_and_candidate_has_gradients(self):
        native=local_order_native(self.image)
        self.assertTrue(torch.equal(order_maps(native,4),soft_structure(self.image)))
        image=self.image.clone().requires_grad_()
        guide=multiscale_guide(image)
        self.assertEqual(guide.shape,(1,6,64,64))
        self.assertLessEqual(float(guide.detach().abs().max()),1.)
        guide.square().mean().backward()
        self.assertTrue(torch.isfinite(image.grad).all())
        self.assertGreater(float(image.grad.abs().sum()),0.)
        self.assertTrue(torch.equal(multiscale_guide(image,'learned_zero'),torch.zeros_like(guide)))
        control=multiscale_guide(image,'coarse_control')
        self.assertTrue(torch.equal(control[:,:3],control[:,3:]))

    def test_zero_initialisation_absence_and_actual_adapter_update_both_backbones(self):
        for variant in ('additive','scale_shift'):
            c,base,_=build_models(tiny_config(),variant)
            base.eval()
            model=DetailConditionedUNet(base,width=2).train()
            guide=multiscale_guide(self.image)
            x=self.image*2-1
            t=torch.ones(1,dtype=torch.long)
            cond=c(torch.zeros(1,dtype=torch.long),torch.zeros(1,dtype=torch.long),torch.zeros(1)).detach()
            parent=base(x,t,cond).detach()
            self.assertTrue(torch.equal(parent,model(x,t,cond,guide)))
            self.assertFalse(base.training)
            opt=torch.optim.AdamW(model.adapter.parameters(),lr=.01)
            # Arbitrary synthetic epsilon objective; update discarded, no efficacy checkpoint.
            for _ in range(2):
                opt.zero_grad(set_to_none=True)
                F.mse_loss(model(x,t,cond,guide),torch.ones_like(x)).backward()
                opt.step()
            with torch.no_grad():
                self.assertFalse(torch.equal(parent,model(x,t,cond,guide)))
                self.assertTrue(torch.equal(parent,model(x,t,cond,guide,torch.zeros(1))))
                bound=BoundStructureModel(model,guide)
                doubled=bound(x.repeat(2,1,1,1),t.repeat(2),cond.repeat(2,1))
                self.assertTrue(torch.equal(doubled[0],doubled[1]))
            self.assertTrue(all(p.grad is None for p in base.parameters()))

    def test_mask_endpoints_noise_and_exterior_for_both_backbones(self):
        for variant in ('additive','scale_shift'):
            c,base,scheduler=build_models(tiny_config(),variant)
            x=self.image*2-1
            args=(c,base,scheduler,x,torch.zeros(1,dtype=torch.long),
                  torch.zeros(1,dtype=torch.long),torch.zeros(1),71,3,3.)
            parent=sample_wiring(*args)
            full=masked_sample(*args,torch.ones_like(x))
            np.testing.assert_array_equal(parent,full)
            empty=masked_sample(*args,torch.zeros_like(x))
            expected=((x+1)/2).numpy()
            np.testing.assert_array_equal(empty,expected)
            mask=torch.zeros_like(x);mask[:,:,20:108,20:108]=1
            result=masked_sample(*args,mask)
            np.testing.assert_array_equal(result[~mask.bool().numpy()],expected[~mask.bool().numpy()])
            self.assertTrue(np.isfinite(result).all())
            np.testing.assert_array_equal(result,masked_sample(*args,mask))
            with self.assertRaises(ValueError):
                masked_sample(*args,mask+.1)


if __name__=='__main__':
    unittest.main()
