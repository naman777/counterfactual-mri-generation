"""Local copy of the ResNet-18 judge architecture (must match modal_train_judge.build_model)."""
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class Judge(nn.Module):
    def __init__(self):
        super().__init__()
        m = torchvision.models.resnet18(weights=None)
        m.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        m.fc = nn.Identity()
        self.backbone = m
        self.drop = nn.Dropout(0.3)
        self.mfg = nn.Linear(512, 2)
        self.fs = nn.Linear(512, 2)

    def features(self, x):
        return self.backbone((x - 0.5) / 0.25)

    def forward(self, x):
        f = self.drop(self.features(x))
        return {"mfg": self.mfg(f), "fs": self.fs(f)}


def load_judge(path, device):
    j = Judge()
    j.load_state_dict(torch.load(path, map_location="cpu"))
    return j.to(device).eval()


@torch.no_grad()
def judge_probs(model, imgs, device, bs=128):
    """imgs: [N,H,W] float in [0,1] -> (mfg_probs [N,2], fs_probs [N,2])"""
    pm, pf = [], []
    for i in range(0, len(imgs), bs):
        x = torch.from_numpy(np.asarray(imgs[i:i + bs], dtype=np.float32)).unsqueeze(1).to(device)
        o = model(x)
        pm.append(F.softmax(o["mfg"], 1).cpu().numpy())
        pf.append(F.softmax(o["fs"], 1).cpu().numpy())
    return np.concatenate(pm), np.concatenate(pf)


@torch.no_grad()
def judge_features(model, imgs, device, bs=128):
    out = []
    for i in range(0, len(imgs), bs):
        x = torch.from_numpy(np.asarray(imgs[i:i + bs], dtype=np.float32)).unsqueeze(1).to(device)
        out.append(model.features(x).cpu().numpy())
    return np.concatenate(out)
