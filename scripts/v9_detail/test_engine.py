import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from scripts.v8_generator_dev.io import ROOT
from scripts.v8_generator_dev.model import build_models
from scripts.v8_generator_dev.test_adapter_pilot import Data
from scripts.v8_generator_dev.test_full import equal_nested
from scripts.v8_generator_dev.test_pipeline import tiny_config
from scripts.v9_detail.engine import EngineeringTrainer,MODES
from scripts.v9_detail.evaluation import evaluate_case,GUIDE_CONTROLS,MASK_CONTROLS


class EngineeringIntegrationTests(unittest.TestCase):
    def test_resume_caps_and_matched_draws_both_backbones(self):
        torch.set_num_threads(2)
        for backbone in ('additive','scale_shift'):
            with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
                home=Path(temporary);config=tiny_config();config['data']={'minimum_cell_patients':1}
                torch.manual_seed(17);c,u,_=build_models(config,backbone)
                parent=home/'parent.pt'
                torch.save(dict(format='canonical_generator_full_v1',coordinate_frame='canonical_axial_view_v1',
                    config=config,variant=backbone,conditioning_state_dict=c.state_dict(),unet_state_dict=u.state_dict()),parent)
                dataset=Data();draws=[];initial=None
                for mode in MODES:
                    engine=EngineeringTrainer(parent,dataset,mode)
                    state=engine.model.adapter.state_dict()
                    if initial is None:initial={k:v.clone() for k,v in state.items()}
                    else:equal_nested(self,initial,state)
                    first=engine.batch();snapshot=home/(mode+'_first.pt');engine.save(snapshot)
                    second=engine.batch();direct=engine.model.adapter.state_dict()
                    resumed=EngineeringTrainer(parent,dataset,mode);resumed.restore(snapshot);replayed=resumed.batch()
                    equal_nested(self,direct,resumed.model.adapter.state_dict())
                    equal_nested(self,engine.optimizer.state_dict(),resumed.optimizer.state_dict())
                    self.assertEqual(second['noise_sha256'],replayed['noise_sha256'])
                    with self.assertRaises(ValueError):engine.batch()
                    wrong=EngineeringTrainer(parent,dataset,'learned_zero' if mode!='learned_zero' else 'coarse_fine')
                    with self.assertRaises(ValueError):wrong.restore(snapshot)
                    draws.append([{k:r[k] for k in ('sample_indices','timesteps','noise_sha256','condition_dropped')} for r in (first,second)])
                self.assertEqual(draws[0],draws[1]);self.assertEqual(draws[1],draws[2])

    def test_guide_and_mask_routes_preserve_endpoint_controls(self):
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            home=Path(temporary);config=tiny_config();config['data']={'minimum_cell_patients':1}
            c,u,_=build_models(config,'additive');parent=home/'parent.pt'
            torch.save(dict(format='canonical_generator_full_v1',coordinate_frame='canonical_axial_view_v1',
                config=config,variant='additive',conditioning_state_dict=c.state_dict(),unet_state_dict=u.state_dict()),parent)
            data=Data();engine=EngineeringTrainer(parent,data,'coarse_fine');engine.batch()
            item=data[0];source=((item['image']+1)/2)[0].numpy();wrong=((data[1]['image']+1)/2)[0].numpy()
            baseline,_=evaluate_case(engine,item,source,'manufacturer',19)
            for control in GUIDE_CONTROLS:
                arrays,meta=evaluate_case(engine,item,source,'manufacturer',19,control,wrong_source=wrong)
                self.assertEqual(meta['source_echo_time_norm'],meta['target_echo_time_norm'])
                self.assertTrue(np.isfinite(arrays['target']).all())
            for control in MASK_CONTROLS:
                arrays,_=evaluate_case(engine,item,source,'manufacturer',19,mask_control=control,wrong_source=wrong)
                if control=='full':np.testing.assert_array_equal(arrays['target'],baseline['target'])
                if control=='empty':np.testing.assert_array_equal(arrays['target'],source)
            with self.assertRaises(ValueError):evaluate_case(engine,item,source,'manufacturer',19,timestep=100)


if __name__=='__main__':unittest.main()
