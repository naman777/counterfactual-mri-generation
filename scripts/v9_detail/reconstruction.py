"""Fixed train-only ridge reconstruction probes, never generator objectives."""
import numpy as np


def fit_global(features,targets,alpha=10.):
    x=np.asarray(features,np.float64)
    y=np.asarray(targets,np.float64)
    mean,std=x.mean(0),x.std(0).clip(1e-4)
    z=(x-mean)/std
    target_mean=y.mean(0)
    # Centred dual ridge, with an unpenalised intercept; each patient one row.
    dual=np.linalg.solve(z@z.T+alpha*np.eye(len(z)),y-target_mean)
    return dict(mean=mean,std=std,target_mean=target_mean,coefficient=z.T@dual)


def predict_global(model,features):
    values=(np.asarray(features,np.float64)-model['mean'])/model['std']
    return (values@model['coefficient']+model['target_mean']).clip(0,1)


def fit_local(maps,targets,alpha=10.):
    maps=np.asarray(maps,np.float64)
    # Equal spatial sample count per patient; no val pixels used in fitting.
    x=maps.transpose(0,2,3,1).reshape(-1,maps.shape[1])
    y=np.asarray(targets,np.float64).reshape(-1)
    mean,std=x.mean(0),x.std(0).clip(1e-4)
    z=(x-mean)/std
    target_mean=y.mean()
    coef=np.linalg.solve(z.T@z+alpha*np.eye(z.shape[1]),z.T@(y-target_mean))
    return dict(mean=mean,std=std,target_mean=np.array(target_mean),coefficient=coef)


def predict_local(model,maps):
    values=np.asarray(maps,np.float64).transpose(0,2,3,1)
    return (((values-model['mean'])/model['std'])@model['coefficient']+model['target_mean']).clip(0,1)
