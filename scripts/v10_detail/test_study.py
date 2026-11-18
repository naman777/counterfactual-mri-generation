import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from scripts.v8_generator_dev.io import ROOT,digest
from scripts.v8_generator_dev.model import build_models
from scripts.v8_generator_dev.test_adapter_pilot import Data
from scripts.v8_generator_dev.test_full import equal_nested
from scripts.v8_generator_dev.test_pipeline import tiny_config
from scripts.v10_detail.protocol import make_protocol,validate,canonical_hash
from scripts.v10_detail.engine import Trainer
from scripts.v10_detail.analyse import crossed_interval,gate_rows
from scripts.v10_detail.evaluate import json_safe
from scripts.v10_detail.metrics import measure


def protocol():
    source=json.loads((ROOT/'reports/v8_generator_dev/adapter_preparation_20261117_02/protocol.json').read_text())
    return make_protocol(source)


class ProspectiveStudyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_frozen_design_and_independent_approval(self):
        p=protocol();validate(p)
        self.assertEqual(len(p['training_seeds'])*len(p['modes']),9)
        self.assertEqual(sum(len(v)*45*3 for axis in p['controls'].values() for v in axis.values()),2295)
        p['scope']['paid_approval']=True
        with self.assertRaises(ValueError):validate(p)

    def test_full_engine_resume_and_capacity_matched_draws(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            home=Path(temporary);config=tiny_config();config['data']={'minimum_cell_patients':1}
            torch.manual_seed(18);c,u,_=build_models(config,'additive');parent=home/'parent.pt'
            torch.save(dict(format='canonical_generator_full_v1',coordinate_frame='canonical_axial_view_v1',
                config=config,variant='additive',conditioning_state_dict=c.state_dict(),unet_state_dict=u.state_dict()),parent)
            p=protocol();p['parent']=parent.relative_to(ROOT).as_posix();p['parent_sha256']=digest(parent)
            data=Data();draws=[];initial=None
            for mode in p['modes']:
                engine=Trainer(p,data,mode,p['training_seeds'][0])
                state=engine.model.adapter.state_dict()
                if initial is None:initial={k:v.clone() for k,v in state.items()}
                else:equal_nested(self,initial,state)
                rows=[engine.batch() for _ in range(2)]
                snapshot=home/(mode+'.pt');engine.save(snapshot)
                rows.extend(engine.batch() for _ in range(2));engine.frozen_check()
                resumed=Trainer(p,data,mode,p['training_seeds'][0]);resumed.restore(snapshot)
                resumed.batch();resumed.batch()
                equal_nested(self,engine.model.adapter.state_dict(),resumed.model.adapter.state_dict())
                equal_nested(self,engine.optimizer.state_dict(),resumed.optimizer.state_dict())
                self.assertEqual(engine.step,4) # Proves this is the full-exposure engine, not v9's two-step cap.
                bad=Trainer(p,data,mode,p['training_seeds'][1])
                with self.assertRaises(ValueError):bad.restore(snapshot)
                bad=Trainer(p,data,mode,p['training_seeds'][0],engineering=True,batch_override=2)
                with self.assertRaises(ValueError):bad.restore(snapshot)
                bad.batch();bad.batch()
                with self.assertRaises(ValueError):bad.batch()
                draws.append([{k:r[k] for k in ('sample_indices','timesteps','noise_sha256','condition_dropped')} for r in rows])
            self.assertEqual(draws[0],draws[1]);self.assertEqual(draws[1],draws[2])

    def test_crossed_bootstrap_and_complete_cell_requirement(self):
        p=protocol();p['analysis']['bootstrap_resamples']=100
        generations=[p['evaluation']['selection_generation_seed'],*p['evaluation']['confirmation_generation_seeds']]
        rows=[dict(patient_id=pid,training_seed=s,generation_seed=g,value=v) for pid,v in [('a',0.),('b',1.)]
              for s in p['training_seeds'] for g in generations]
        frame=pd.DataFrame(rows)
        self.assertEqual(crossed_interval(frame,'value',p)['estimate'],.5)
        with self.assertRaises(ValueError):crossed_interval(frame.iloc[:-1],'value',p)
        with self.assertRaises(ValueError):crossed_interval(pd.concat([frame,frame.iloc[:1]]),'value',p)
        frame.loc[0,'value']=np.nan
        with self.assertRaises(ValueError):crossed_interval(frame,'value',p)

    def test_simultaneous_gate_cannot_choose_a_favourable_judge(self):
        p=protocol();p['analysis']['bootstrap_resamples']=100
        generations=[p['evaluation']['selection_generation_seed'],*p['evaluation']['confirmation_generation_seeds']]
        rows=[]
        for pid in ('a','b'):
            for seed in p['training_seeds']:
                for g in generations:
                    for j in range(3):
                        for variant in ('null','target'):
                            rows.append(dict(patient_id=pid,training_seed=seed,generation_seed=g,
                                judge=f'heldout_orientation_robust_seed{j}',variant=variant,
                                interior_mad_fine=.09,interior_mad_coarse=.1,
                                interior_hf_residual_mad_fine=.09,interior_hf_residual_mad_coarse=.1,
                                paired_hit_delta_fine=-.2 if j==2 else 0.,paired_hit_delta_coarse=0.))
        results=gate_rows(pd.DataFrame(rows),p)
        self.assertEqual(sum(not r['interval_passed'] for r in results),1)
        self.assertFalse(all(r['interval_passed'] for r in results))

    def test_mask_background_cannot_improve_strict_interior_metrics(self):
        rng=np.random.default_rng(17);source=np.zeros((128,128),np.float32)
        source[20:108,20:108]=rng.uniform(.2,.8,(88,88))
        edit=source.copy();edit[:10]=.1
        a=measure(source,edit);self.assertTrue(a['interior_valid'])
        self.assertEqual(a['interior_mad'],0.)
        self.assertEqual(a['interior_hf_residual_mad'],0.)
        self.assertGreater(a['whole_mad'],0.)

    def test_missing_measurements_are_explicit_json_nulls(self):
        self.assertEqual(json.dumps(json_safe({'x':np.nan,'y':np.float32(1)}),allow_nan=False),'{"x": null, "y": 1.0}')


if __name__=='__main__':unittest.main()
