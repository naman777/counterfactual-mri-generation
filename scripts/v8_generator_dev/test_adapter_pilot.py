"""Pilot-specific controls: zero-guide leakage, independent draws, exact block resume."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from scripts.v8_generator_dev.adapter_pilot import draw, guide_for, pilot_settings, train_copy
from scripts.v8_generator_dev.io import ROOT
from scripts.v8_generator_dev.model import build_models
from scripts.v8_generator_dev.test_full import equal_nested
from scripts.v8_generator_dev.test_pipeline import tiny_config


class Data:
    split='train'
    def __init__(self):
        self.frame=pd.DataFrame([dict(patient_id=f'p{i}',manufacturer_label=i%2,field_strength_label=i//2)
            for i in range(4)])
        self.images=torch.rand(4,1,128,128,generator=torch.Generator().manual_seed(13))*2-1
    def __len__(self):
        return 4
    def __getitem__(self,index):
        row=self.frame.iloc[index]
        return dict(image=self.images[index],patient_id=row.patient_id,
            manufacturer_label=int(row.manufacturer_label),field_strength_label=int(row.field_strength_label),
            echo_time_norm=np.float32(.3))


class PilotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_zero_guide_hides_source_but_keeps_noise_and_dropout_streams(self):
        data=Data()
        weights=torch.full((4,),.25,dtype=torch.float64)
        for step in (0,1,9):
            a=draw(data,weights,step,2)
            guide_for((a[1]+1)/2,'soft_source_guide',step)
            zero=guide_for((a[1]+1)/2,'learned_zero_guide',step)
            self.assertEqual(int(torch.count_nonzero(zero)),0)
            b=draw(data,weights,step,2)
            self.assertEqual(a[-1],b[-1])
            self.assertTrue(torch.equal(a[3],b[3]))
            self.assertTrue(torch.equal(a[4],b[4]))

    def test_exact_adapter_block_resume_and_both_learned_controls(self):
        settings=pilot_settings()
        settings.update(batch_size=2,adapter_width=2)
        data=Data()
        for backbone in ('additive','scale_shift'):
            with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
                folder=Path(temporary)
                config=tiny_config()
                config['data']={'minimum_cell_patients':1}
                torch.manual_seed(9)
                c,u,_=build_models(config,backbone)
                parent=folder/'parent.pt'
                torch.save(dict(format='canonical_generator_full_v1',coordinate_frame='canonical_axial_view_v1',
                    config=config,variant=backbone,conditioning_state_dict=c.state_dict(),unet_state_dict=u.state_dict()),parent)
                draws=[]
                for mode in ('soft_source_guide','learned_zero_guide'):
                    direct=folder/(mode+'_direct.pt')
                    result=train_copy(parent,mode,data,settings,direct,folder/(mode+'_direct.jsonl'),
                        stop_step=2,device='cpu',engineering=True)
                    self.assertTrue(result['checks']['zero_init_equal'])
                    self.assertTrue(result['checks']['base_weights_unchanged'])
                    self.assertTrue(result['checks']['adapter_updated'])
                    first=folder/(mode+'_first.pt')
                    train_copy(parent,mode,data,settings,first,folder/(mode+'_first.jsonl'),
                        stop_step=1,device='cpu',engineering=True)
                    resumed=folder/(mode+'_resumed.pt')
                    train_copy(parent,mode,data,settings,resumed,folder/(mode+'_resumed.jsonl'),
                        start_step=1,stop_step=2,resume=first,device='cpu',engineering=True)
                    a,b=[torch.load(path,weights_only=False) for path in (direct,resumed)]
                    equal_nested(self,a['adapter_state_dict'],b['adapter_state_dict'])
                    equal_nested(self,a['optimizer_state_dict'],b['optimizer_state_dict'])
                    rows=[json.loads(line) for line in (folder/(mode+'_direct.jsonl')).read_text().splitlines()]
                    draws.append([{k:row[k] for k in ('sample_indices','timesteps','noise_sha256','condition_dropped')} for row in rows])
                self.assertEqual(*draws)


if __name__=='__main__':
    unittest.main()
