"""Patient-separated scanner leakage probe; never assume proxy maps are invariant."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scripts.v4_convergence.core import digest,json_write
from scripts.v5_anatomy.run_pilot import read_splits,project_path,select_validation
from scripts.v6_structure.model import structure_maps,photometric_view

ROOT=Path(__file__).resolve().parents[2]


def balanced_accuracy(truth,pred):
    return float(np.mean([(truth[truth==label]==pred[truth==label]).mean() for label in [0,1]]))


def balanced_ci(truth,pred):
    rng=np.random.default_rng(731);groups=[np.where(truth==label)[0] for label in [0,1]]
    if any(not len(group) for group in groups):return None,None
    values=[]
    for _ in range(10000):
        idx=np.concatenate([rng.choice(group,len(group),replace=True) for group in groups])
        values.append(balanced_accuracy(truth[idx],pred[idx]))
    return float(np.quantile(values,.025)),float(np.quantile(values,.975))


def audit(output):
    output=project_path(output)
    if not output.is_relative_to(ROOT/'reports'):raise ValueError('Use a new reports directory')
    splits=read_splits();inputs={};frames=[];descriptors=[];raw=[];changes=[]
    generator=torch.Generator().manual_seed(731);torch.set_num_threads(4)
    for split in ['train','val']:
        selection=select_validation(splits[split],splits[split].patient_id.nunique(),731)
        for _,row in selection.iterrows():
            path=project_path(row.image_path);image=torch.tensor(np.load(path).astype(np.float32))[None,None]
            inputs[str(path.relative_to(ROOT))]=digest(path)
            with torch.no_grad():
                maps=structure_maps(image);other=structure_maps(photometric_view(image,generator))
                feature=F.adaptive_avg_pool2d(maps,(8,8)).flatten().numpy()
                raw_feature=F.adaptive_avg_pool2d(image,(8,8)).flatten().numpy()
            descriptors.append(feature);raw.append(raw_feature)
            changes.append(float((maps-other).abs().mean()))
            frames.append(dict(split=split,patient_id=row.patient_id,manufacturer=int(row.manufacturer_label),
                               field_strength=int(row.field_strength_label),photometric_map_l1=changes[-1]))
    frame=pd.DataFrame(frames);features={'structure_proxy':np.stack(descriptors),'raw_coarse_intensity':np.stack(raw)}
    train=(frame.split=='train').to_numpy();val=~train;results=[];predictions=[]
    for name,values in features.items():
        mean,std=values[train].mean(0),values[train].std(0).clip(1e-4)
        x=(values-mean)/std;x=np.column_stack([np.ones(len(x)),x]).astype(np.float64)
        for axis in ['manufacturer','field_strength']:
            y=frame[axis].to_numpy();counts=np.bincount(y[train],minlength=2)
            weights=1/counts[y[train]];weights=weights/weights.mean()
            penalty=np.eye(x.shape[1])*10.;penalty[0,0]=0
            coefficient=np.linalg.solve(x[train].T@(weights[:,None]*x[train])+penalty,
                                        x[train].T@(weights*y[train]))
            score=x[val]@coefficient;pred=(score>=.5).astype(int);truth=y[val]
            lo,hi=balanced_ci(truth,pred)
            results.append(dict(representation=name,axis=axis,train_patients=int(train.sum()),
                                validation_patients=int(val.sum()),balanced_accuracy=balanced_accuracy(truth,pred),
                                ci95_lo=lo,ci95_hi=hi,probe='fixed balanced ridge classifier alpha=10; no tuning',
                                interval='10,000 class-stratified patient bootstrap',
                                scanner_invariance_established=False))
            for patient,t,p,s in zip(frame[val].patient_id,truth,pred,score):
                predictions.append(dict(representation=name,axis=axis,patient_id=patient,truth=int(t),prediction=int(p),score=float(s)))
    output.mkdir(parents=True,exist_ok=False)
    frame.to_csv(output/'patient_features.csv',index=False)
    pd.DataFrame(results).to_csv(output/'leakage_probe.csv',index=False)
    pd.DataFrame(predictions).to_csv(output/'probe_predictions.csv',index=False)
    np.savez_compressed(output/'features.npz',**features)
    inputs[str(Path(__file__).relative_to(ROOT))]=digest(__file__)
    model=Path(__file__).parent/'model.py';inputs[str(model.relative_to(ROOT))]=digest(model)
    json_write(output/'inputs.json',inputs)
    json_write(output/'complete.json',dict(complete=True,train_patients=int(train.sum()),validation_patients=int(val.sum()),
        test_images_used=False,expert_annotations_used=False,representation_invariance_established=False,
        mean_photometric_map_l1=float(np.mean(changes)),full_training_approved_by_this_audit=False))
    (output/'README.md').write_text('# Structure-representation audit\n\n'
        'Producer: `python -m scripts.v6_structure.audit_representation`.\n'
        'patient_features.csv uses one seeded image per original train/validation patient.\n'
        'features.npz stores pooled proxy/raw features; leakage_probe.csv gives fixed ridge probe results and patient CIs;\n'
        'probe_predictions.csv retains each prediction; inputs.json fingerprints inputs; complete.json records limitations.\n'
        'No test images or expert labels. A weak probe cannot prove invariance; a successful probe exposes residual label information.\n'
        'Original orientation and protocol confounding remain. Full efficacy training is not approved or justified by this audit.\n',encoding='utf-8')
    print(pd.DataFrame(results).to_string(index=False),flush=True);return output


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--output',required=True);print(audit(p.parse_args().output))
