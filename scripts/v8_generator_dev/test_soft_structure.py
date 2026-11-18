import unittest

import torch

from scripts.v6_structure.model import StructureConditionedUNet, BoundStructureModel
from scripts.v8_generator_dev.model import build_models
from scripts.v8_generator_dev.soft_structure import soft_structure
from scripts.v8_generator_dev.test_pipeline import tiny_config


class SoftStructureTests(unittest.TestCase):
    def test_shape_finite_gradient_and_affine_response(self):
        torch.set_num_threads(2)
        image = (.1+.7*torch.rand(1,1,128,128, generator=torch.Generator().manual_seed(4))).requires_grad_()
        maps = soft_structure(image)
        self.assertEqual(maps.shape, (1,3,32,32))
        self.assertTrue(torch.isfinite(maps).all())
        maps.square().mean().backward()
        self.assertTrue(torch.isfinite(image.grad).all())
        self.assertGreater(float(image.grad.abs().sum()), 0)
        self.assertLess(float((maps.detach()-soft_structure(image.detach()*.8+.04)).abs().max()), .001)

    def test_zero_adapter_absent_gate_and_shared_cfg_guide_both_backbones(self):
        torch.set_num_threads(2)
        for variant in ('additive', 'scale_shift'):
            config = tiny_config()
            c, base, scheduler = build_models(config, variant)
            base.eval()
            model = StructureConditionedUNet(base, width=2).eval()
            source = torch.rand(1,1,128,128)
            guide = soft_structure(source)
            condition = c(torch.zeros(1,dtype=torch.long), torch.zeros(1,dtype=torch.long), torch.zeros(1))
            x, t = torch.randn(1,1,128,128), torch.ones(1,dtype=torch.long)
            with torch.no_grad():
                reference = base(x,t,condition)
                self.assertTrue(torch.equal(reference, model(x,t,condition,guide)))
                for layer in model.adapter.outputs:
                    layer.bias.fill_(.01)
                self.assertTrue(torch.equal(reference, model(x,t,condition,guide,present=torch.zeros(1))))
                bound = BoundStructureModel(model,guide)
                doubled = bound(x.repeat(2,1,1,1),t.repeat(2),condition.repeat(2,1))
                self.assertTrue(torch.equal(doubled[0],doubled[1]))


if __name__ == '__main__':
    unittest.main()
