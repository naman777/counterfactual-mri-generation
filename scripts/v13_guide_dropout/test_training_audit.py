import copy
import hashlib
import unittest

from scripts.v13_guide_dropout.protocol import canonical_hash
from scripts.v13_guide_dropout.verify_training import audit_rows,DRAW_FIELDS


class AuditChecks(unittest.TestCase):
    def test_segment_cursor_chain_and_corrupt_actual_row(self):
        initial=hashlib.sha256(b'guide_dropout_actual_draw_chain_v1').hexdigest()
        row=dict(global_step=1,training_seed=583301,mode='coarse_fine',engineering=False,
                 applied=True,fixed_echo_time=True,actual_batch_size=32,epoch_at_draw=0,
                 sample_indices=list(range(32)),timesteps=[100]*32,condition_dropped=[False]*32,
                 gate_states=[i%5 for i in range(32)],patient_ids=['p']*32,noise_sha256='a'*64,
                 total_loss=1.,gradient_norm=1.,noise_mse=1.,coarse_structure_loss=0.,seconds=.1,amp_retries=0)
        row['actual_draw_chain_sha256']=canonical_hash(dict(previous=initial,draw={k:row[k] for k in DRAW_FIELDS}))
        step,chain,counts=audit_rows([row],583301,'coarse_fine',0,initial,[0]*5)
        self.assertEqual(step,1);self.assertEqual(sum(counts),32);self.assertEqual(chain,row['actual_draw_chain_sha256'])
        for field,value in [('global_step',2),('applied',False),('noise_sha256','b'*64),('engineering',True),
                            ('actual_batch_size',5),('gradient_norm',float('nan'))]:
            changed=copy.deepcopy(row);changed[field]=value
            with self.assertRaises(ValueError):audit_rows([changed],583301,'coarse_fine',0,initial,[0]*5)


if __name__=='__main__':unittest.main()
