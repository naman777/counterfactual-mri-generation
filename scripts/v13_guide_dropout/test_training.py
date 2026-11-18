"""Checks that exercise resumed optimisation and reject incompatible evidence."""
import copy
import json
import tempfile
import unittest
from pathlib import Path

import torch

from scripts.v8_generator_dev.io import ROOT,digest
from scripts.v8_generator_dev.model import build_models
from scripts.v8_generator_dev.test_adapter_pilot import Data
from scripts.v8_generator_dev.test_full import equal_nested
from scripts.v8_generator_dev.test_pipeline import tiny_config
from scripts.v11_guide_audit.guide_dropout import draw_states
from scripts.v13_guide_dropout.engine import Trainer
from scripts.v13_guide_dropout.protocol import make_protocol,validate,canonical_hash


def protocol():
    reference = json.loads((ROOT/'reports/v10_detail/preparation_20261117_02/protocol.json').read_text())
    return make_protocol(reference)


def tiny_parent(folder):
    config=tiny_config();config['data']={'minimum_cell_patients':1}
    torch.manual_seed(29)
    c,u,_=build_models(config,'additive')
    parent=folder/'parent.pt'
    torch.save(dict(format='canonical_generator_full_v1',coordinate_frame='canonical_axial_view_v1',
                    config=config,variant='additive',conditioning_state_dict=c.state_dict(),unet_state_dict=u.state_dict()),parent)
    p=protocol();p['reference_protocol']['parent']=parent.relative_to(ROOT).as_posix()
    p['reference_protocol']['parent_sha256']=digest(parent)
    return make_protocol(p['reference_protocol'])


class TrainingChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_policy_and_ten_gates_cannot_drift(self):
        p=protocol();validate(p)
        self.assertEqual(len(p['analysis']['gates']),10)
        self.assertEqual(p['analysis']['interval_quantiles'],[.0025,.9975])
        for field,value in [('probabilities',[1.,0.,0.,0.,0.]),('seed_offset',0)]:
            bad=copy.deepcopy(p);bad['guide_dropout'][field]=value
            with self.assertRaises(ValueError):validate(bad)

    def test_full_exposure_resume_matches_parameters_optimizer_and_actual_gates(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            home=Path(temporary);p=tiny_parent(home);data=Data();matched=[]
            for mode in p['modes']:
                engine=Trainer(p,data,mode,583301)
                rows=[engine.batch(),engine.batch()]
                snapshot=home/(mode+'.pt');engine.save(snapshot)
                rows.extend(engine.batch() for _ in range(4))
                restored=Trainer(p,data,mode,583301);restored.restore(snapshot)
                replay=[restored.batch() for _ in range(4)]
                equal_nested(self,engine.model.adapter.state_dict(),restored.model.adapter.state_dict())
                equal_nested(self,engine.optimizer.state_dict(),restored.optimizer.state_dict())
                self.assertEqual(engine.gate_counts,restored.gate_counts)
                self.assertEqual(engine.draw_chain,restored.draw_chain)
                engine.frozen_check();restored.frozen_check()
                self.assertEqual(engine.step,6)
                keys=('sample_indices','timesteps','condition_dropped','noise_sha256','gate_states','actual_draw_chain_sha256')
                self.assertEqual([{k:r[k] for k in keys} for r in rows[2:]],
                                 [{k:r[k] for k in keys} for r in replay])
                for step,row in enumerate(rows):
                    self.assertEqual(row['gate_states'],draw_states(len(data),583301,step).tolist())
                matched.append([{k:r[k] for k in keys} for r in rows])
            self.assertEqual(*matched)

    def test_missing_corrupt_changed_runtime_and_engineering_checkpoints_rejected(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            home=Path(temporary);p=tiny_parent(home);data=Data()
            engine=Trainer(p,data,'coarse_fine',583301);engine.batch()
            snapshot=home/'state.pt';engine.save(snapshot)
            with self.assertRaises(FileNotFoundError):engine.restore(home/'missing.pt')
            broken=home/'broken.pt';broken.write_bytes(b'not a torch checkpoint')
            with self.assertRaises(Exception):engine.restore(broken)
            original=torch.load(snapshot,weights_only=False)
            variants=[]
            for key,value in [('step',True),('gate_counts',[0]*5),('engineering',True),
                              ('mode','coarse_control'),('training_seed',583302)]:
                ck=copy.deepcopy(original);ck[key]=value;variants.append(ck)
            ck=copy.deepcopy(original);ck['runtime']['gpu']='changed GPU';variants.append(ck)
            ck=copy.deepcopy(original);del ck['rng_cpu'];variants.append(ck)
            ck=copy.deepcopy(original);next(iter(ck['adapter'].values())).fill_(float('nan'));variants.append(ck)
            for number,ck in enumerate(variants):
                path=home/f'bad{number}.pt';torch.save(ck,path)
                with self.assertRaises(ValueError):Trainer(p,data,'coarse_fine',583301).restore(path)
            engineering=Trainer(p,data,'coarse_fine',583301,engineering=True,batch_override=2)
            engineering.batch();engineering.batch()
            with self.assertRaises(ValueError):engineering.batch()
            with self.assertRaises(ValueError):engineering.restore(snapshot)

    def test_optimizer_references_and_full_batch_override_rejection(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            p=tiny_parent(Path(temporary));data=Data()
            engine=Trainer(p,data,'coarse_fine',583301)
            self.assertEqual({id(v) for v in engine.model.adapter.parameters()},
                             {id(v) for g in engine.optimizer.param_groups for v in g['params']})
            with self.assertRaises(ValueError):Trainer(p,data,'coarse_fine',583301,batch_override=2)
            with self.assertRaises(ValueError):Trainer(p,data,'learned_zero',583301)


if __name__=='__main__':unittest.main()
