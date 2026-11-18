import unittest
import numpy as np
import torch
from scripts.v11_guide_audit.guide_dropout import apply_states,draw_states
from scripts.v11_guide_audit.forward_probe import response_metrics


class PrototypeChecks(unittest.TestCase):
    def test_gate_semantics_and_bias_gradient(self):
        guide=torch.ones(5,6,64,64);states=torch.arange(5)
        gated,present=apply_states(guide,states)
        torch.testing.assert_close(gated[:,:3].sum((1,2,3)),torch.tensor([12288.,12288.,0.,0.,12288.]))
        torch.testing.assert_close(gated[:,3:].sum((1,2,3)),torch.tensor([12288.,0.,12288.,0.,12288.]))
        torch.testing.assert_close(present,torch.tensor([1.,1.,1.,1.,0.]))
        bias=torch.tensor(2.,requires_grad=True)
        result=(gated.sum((1,2,3))+bias)*present
        self.assertEqual(result[3].item(),2.);self.assertEqual(result[4].item(),0.)
        result[4].backward(retain_graph=True);self.assertEqual(bias.grad.item(),0.)
        bias.grad.zero_();result[3].backward();self.assertEqual(bias.grad.item(),1.)

    def test_rng_independence_reproducibility_and_validation(self):
        before=torch.random.get_rng_state().clone()
        a=draw_states(4096,583301,1)
        torch.testing.assert_close(before,torch.random.get_rng_state())
        torch.testing.assert_close(a,draw_states(4096,583301,1))
        self.assertEqual(set(a.tolist()),set(range(5)))
        self.assertFalse(torch.equal(a,draw_states(4096,583301,2)))
        with self.assertRaises(ValueError):apply_states(torch.ones(1,6,64,64),torch.tensor([5]))

    def test_analytical_epsilon_response_and_undefined_denominator(self):
        parent=np.zeros((3,1,128,128));parent[1]=.5
        prediction=parent.copy();prediction[1]=1.;prediction[2]=2.
        mask=np.ones((128,128),dtype=bool)
        result=response_metrics(prediction,parent,mask)
        self.assertEqual(result['label_response_rms'],1.)
        self.assertEqual(result['response_ratio_to_parent'],2.)
        self.assertEqual(result['response_cosine_to_parent'],1.)
        self.assertEqual(result['unconditional_adapter_shift_rms'],2.)
        self.assertTrue(np.isnan(response_metrics(prediction,np.zeros_like(parent),mask)['response_ratio_to_parent']))
        self.assertTrue(all(np.isnan(v) for v in response_metrics(prediction,parent,np.zeros_like(mask)).values()))


if __name__=='__main__':unittest.main()
