"""Compare native/canonical leakage on identical patients; verify source resize provenance."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts.v4_convergence.core import digest,json_write
from scripts.v5_anatomy.run_pilot import project_path,select_validation
from scripts.v6_structure.model import structure_maps
from scripts.v6_structure.audit_representation import balanced_accuracy,balanced_ci
from scripts.v7_geometry.prepare_view import load_canonical
from src.data.preprocess import read_dicom_image,percentile_clip_and_normalize,resize_image

ROOT=Path(__file__).resolve().parents[2]


def audit(view,output):
    view,output=project_path(view),project_path(output)
    if output.exists():raise FileExistsError(output)
    config=json.loads((view/'view.json').read_text())
    manifests={s:pd.read_csv(view/'splits'/f'{s}.csv') for s in ['train','val']}
    torch.set_num_threads(4);records=[];feature_rows={k:[] for k in ['native_proxy','canonical_proxy','native_intensity','canonical_intensity']}
    inputs={str((view/'view.json').relative_to(ROOT)):digest(view/'view.json')};preprocessing=[];examples={}
    for split,frame in manifests.items():
        index=view/'splits'/f'{split}.csv';inputs[str(index.relative_to(ROOT))]=digest(index)
        chosen=select_validation(frame,frame.patient_id.nunique(),731)
        for _,row in chosen.iterrows():
            canonical,plan=load_canonical(row)
            source_path=project_path(row.source_image_path);source=np.load(source_path).astype(np.float32)
            header=project_path(row.source_dicom_path)
            if digest(header)!=row.source_dicom_sha256:raise ValueError('Source DICOM changed')
            # Confirm no undocumented preprocessing rotation/crop underlies the
            # pixel-to-patient map. Only audit-selected train/val images decoded.
            raw=read_dicom_image(header);recreated=resize_image(percentile_clip_and_normalize(raw),128)
            difference=float(np.max(np.abs(recreated-source)))
            preprocessing.append(dict(split=split,patient_id=row.patient_id,sop_uid=row.sop_uid,
                max_abs_preprocessing_error=difference,declared_preprocessing_matches=difference<=1e-6))
            inputs[str(source_path.relative_to(ROOT))]=digest(source_path);inputs[str(header.relative_to(ROOT))]=digest(header)
            with torch.no_grad():
                for name,array in [('native',source),('canonical',canonical)]:
                    tensor=torch.tensor(array.copy())[None,None]
                    maps=structure_maps(tensor)
                    feature_rows[name+'_proxy'].append(F.adaptive_avg_pool2d(maps,(8,8)).flatten().numpy())
                    feature_rows[name+'_intensity'].append(F.adaptive_avg_pool2d(tensor,(8,8)).flatten().numpy())
            records.append(dict(split=split,patient_id=row.patient_id,manufacturer=int(row.manufacturer_label),field_strength=int(row.field_strength_label)))
            if split=='val' and int(row.manufacturer_label) not in examples:
                examples[int(row.manufacturer_label)]=(source,canonical,row,plan)
    frame=pd.DataFrame(records);train=(frame.split=='train').to_numpy();val=~train
    results=[];predictions=[];features={k:np.stack(v) for k,v in feature_rows.items()}
    for representation,values in features.items():
        mean,std=values[train].mean(0),values[train].std(0).clip(1e-4)
        x=np.column_stack([np.ones(len(values)),(values-mean)/std]).astype(np.float64)
        for axis in ['manufacturer','field_strength']:
            y=frame[axis].to_numpy();counts=np.bincount(y[train],minlength=2);w=1/counts[y[train]];w=w/w.mean()
            penalty=np.eye(x.shape[1])*10;penalty[0,0]=0
            coefficients=np.linalg.solve(x[train].T@(w[:,None]*x[train])+penalty,x[train].T@(w*y[train]))
            scores=x[val]@coefficients;pred=(scores>=.5).astype(int);lo,hi=balanced_ci(y[val],pred)
            results.append(dict(representation=representation,axis=axis,balanced_accuracy=balanced_accuracy(y[val],pred),
                ci95_lo=lo,ci95_hi=hi,train_patients=int(train.sum()),validation_patients=int(val.sum()),
                probe='fixed balanced ridge alpha=10; identical patient/source selection; no tuning',
                interval='10,000 class-stratified patient bootstrap; perfect-sample CIs may be degenerate',
                scanner_invariance_established=False))
            predictions.extend(dict(representation=representation,axis=axis,patient_id=patient,truth=int(t),prediction=int(p),score=float(s))
                for patient,t,p,s in zip(frame[val].patient_id,y[val],pred,scores))
    output.mkdir(parents=True)
    frame.to_csv(output/'patient_selection.csv',index=False);pd.DataFrame(results).to_csv(output/'leakage_probe.csv',index=False)
    pd.DataFrame(predictions).to_csv(output/'probe_predictions.csv',index=False)
    pd.DataFrame(preprocessing).to_csv(output/'preprocessing_reproduction.csv',index=False)
    np.savez_compressed(output/'features.npz',**features)
    fig,axes=plt.subplots(len(examples),2,figsize=(7.2,3.6*len(examples)),squeeze=False)
    for i,(label,(source,canonical,row,plan)) in enumerate(sorted(examples.items())):
        for column,(array,title) in enumerate([(source,'Historical stored frame'),(canonical,'Header-derived patient frame')]):
            axes[i,column].imshow(array,cmap='gray',vmin=0,vmax=1,interpolation='none');axes[i,column].axis('off')
            axes[i,column].set_title(title,fontsize=10)
        axes[i,0].text(.5,-.05,f'{row.patient_id}; vendor label {label}',ha='center',transform=axes[i,0].transAxes,fontsize=9)
    fig.suptitle('Lossless orientation normalisation; no registration or scanner-effect validation',fontsize=11)
    fig.tight_layout(rect=[0,.02,1,.95]);fig.savefig(output/'orientation_examples.png',dpi=160);plt.close(fig)
    inputs[str(Path(__file__).relative_to(ROOT))]=digest(__file__)
    json_write(output/'inputs.json',inputs)
    matched=all(p['declared_preprocessing_matches'] for p in preprocessing)
    json_write(output/'complete.json',dict(complete=True,train_patients=int(train.sum()),validation_patients=int(val.sum()),
        source_preprocessing_reproduced_on_audit_samples=matched,preprocessing_samples_checked=len(preprocessing),
        test_images_used=False,scanner_invariance_established=False,anatomy_validation=False,old_model_validated_on_view=False))
    lines=['# Canonical-view representation audit','','The same seeded source image per train/validation patient is used in every representation.',
           'Only array transpose/flips are applied. This is not anatomical registration or physical scanner-effect identification.',
           'Native DICOM preprocessing is reproduced on audit samples to check the pixel-map assumption.','',
           '| Representation | Axis | Fixed probe balanced accuracy [patient-bootstrap 95% CI] |',
           '| --- | --- | --- |']
    for row in results:lines.append(f'| {row["representation"]} | {row["axis"]} | {row["balanced_accuracy"]:.3f} [{row["ci95_lo"]:.3f}, {row["ci95_hi"]:.3f}] |')
    lines+=['','A lower probe score shows less information for this particular probe, not proven scanner invariance.',
            'A perfect sample score can produce a degenerate bootstrap CI; small validation classes remain limited.',
            'Protocol/cohort confounding can remain after orientation correction. No efficacy training or winner selection occurs.']
    (output/'findings.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (output/'README.md').write_text('# Canonical representation and preprocessing audit\n\n'
        'Producer: `python -m scripts.v7_geometry.audit_view`.\n'
        'patient_selection.csv fixes one source per train/validation patient; features.npz stores native/canonical descriptors.\n'
        'leakage_probe.csv and probe_predictions.csv retain results and individual predictions; findings.md explains limits.\n'
        'preprocessing_reproduction.csv checks native DICOM rescale/normalise/resize against frozen arrays.\n'
        'orientation_examples.png shows first-selected validation examples per vendor. inputs.json fingerprints sources and code.\n'
        'complete.json records coverage and validity limits. No test image, expert annotation or new scanner-editing result.\n',encoding='utf-8')
    if not matched:raise ValueError('Some source preprocessing does not match; do not trust pixel-to-patient maps without investigation')
    print(pd.DataFrame(results).to_string(index=False),flush=True);return output


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--view',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();print(audit(a.view,a.output))
