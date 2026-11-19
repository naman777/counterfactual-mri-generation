import unittest
import torch

from scripts.v14_dropout_audit.shared_residual import parent_response_score


class AlgebraChecks(unittest.TestCase):
    def test_parent_and_zero_guidance_endpoints(self):
        parent_u=torch.tensor([1.,2.],dtype=torch.float64);parent_c=torch.tensor([3.,4.],dtype=torch.float64)
        guided_u=torch.tensor([9.,7.],dtype=torch.float64)
        torch.testing.assert_close(parent_response_score(parent_u,parent_c,parent_u,3.),parent_u+3*(parent_c-parent_u),rtol=0,atol=0)
        torch.testing.assert_close(parent_response_score(guided_u,parent_c,parent_u,0.),guided_u,rtol=0,atol=0)

    def test_target_source_contrast_is_parent_contrast(self):
        generator=torch.Generator().manual_seed(7)
        u,s,t,g=torch.randn(4,32,generator=generator,dtype=torch.float64)
        difference=parent_response_score(g,t,u,3.)-parent_response_score(g,s,u,3.)
        torch.testing.assert_close(difference,3*(t-s),atol=1e-14,rtol=0)
        with self.assertRaises(ValueError):parent_response_score(g,t,u,-1.)
        with self.assertRaises(ValueError):parent_response_score(g,t[:1],u,1.)


if __name__=='__main__':unittest.main()
