import copy
import unittest
import pandas as pd
from scripts.v10_detail.verify_training_logs import check_records


class TrainingJournalTests(unittest.TestCase):
    def test_source_exposure_and_skipped_updates_are_not_silently_accepted(self):
        frame=pd.DataFrame({'patient_id':['a','b']})
        row=dict(global_step=1,training_seed=1,mode='coarse_fine',applied=True,engineering=False,
            fixed_echo_time=True,amp_retries=0,total_loss=.1,noise_mse=.1,coarse_structure_loss=0.,gradient_norm=1.,
            epoch_at_draw=0,actual_batch_size=2,sample_indices=[0,1],patient_ids=['a','b'],timesteps=[1,2],
            condition_dropped=[False,True],noise_sha256='0'*64)
        self.assertEqual(len(check_records([row],0,1,frame,1,'coarse_fine')),1)
        for key,value in [('applied',False),('actual_batch_size',32),('patient_ids',['b','a']),('global_step',2)]:
            bad=copy.deepcopy(row);bad[key]=value
            with self.assertRaises(ValueError):check_records([bad],0,1,frame,1,'coarse_fine')
        with self.assertRaises(ValueError):check_records([],0,1,frame,1,'coarse_fine')


if __name__=='__main__':unittest.main()
