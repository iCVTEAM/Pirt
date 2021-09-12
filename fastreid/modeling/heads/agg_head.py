# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch.nn.functional as F
from torch import nn
import torch

from fastreid.utils.comm import get_local_rank
from fastreid.layers import *
from fastreid.utils.weight_init import weights_init_kaiming, weights_init_classifier
from .build import REID_HEADS_REGISTRY

class BNC(nn.Module):
    def __init__(self, inplane, num_classes, num_heads):
        super().__init__()
        self.bnnecks = nn.ModuleList([
            get_norm('BN', inplane, bias_freeze=True) for _ in range(num_heads)
        ])
        self.classifiers = nn.ModuleList([
            nn.Linear(inplane, num_classes, bias=False) for _ in range(num_heads)
        ])
        self.bnnecks.apply(weights_init_kaiming)
        self.classifiers.apply(weights_init_classifier)

    def forward(self, x):
        # x [p, b, c]
        b, c, p = x.shape
        out_list = []
        for i in range(p):
            out = x[:, :, i].reshape(b, c, 1, 1)
            out = self.bnnecks[i](out).reshape(b, c)
            out = self.classifiers[i](out)
            out_list.append(out)
        out = torch.stack(out_list, dim=-1).contiguous()
        return out

# back up code for basic piguhead
@REID_HEADS_REGISTRY.register()
class AGGHead(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # fmt: off
        num_classes   = cfg.MODEL.HEADS.NUM_CLASSES

        self.pool_layer =  nn.AdaptiveAvgPool2d(1)

        self.count = 0
        self.bnc = BNC(2048, num_classes, 3)

        self.slassifier = nn.Linear(2048, num_classes, bias=False)
        self.sbn = get_norm('BN', 2048, bias_freeze=True)
        self.slassifier.apply(weights_init_classifier)
        self.sbn.apply(weights_init_kaiming)

        self.plassifier = nn.Linear(2048, num_classes, bias=False)
        self.pbn = get_norm('BN', 2048, bias_freeze=True)
        self.plassifier.apply(weights_init_classifier)
        self.pbn.apply(weights_init_kaiming)

    def forward(self, seatures, peatures, cls_feats, targets=None, confs= None):
        """
        See :class:`ReIDHeads.forward`.
        """
        b, c, h, w = seatures.shape
        self.count = (self.count + 1) % 100
        score = confs

        seatures = self.pool_layer(seatures)
        bn_seatures = self.sbn(seatures.reshape(b, c, 1, 1)).squeeze()
        peatures = self.pool_layer(peatures)
        bn_peatures = self.pbn(peatures.reshape(b, c, 1, 1)).squeeze()

        # Evaluation
        # fmt: off
        if not self.training: return bn_seatures, bn_peatures, cls_feats, score,
        # fmt: on

        # Training
        stp_outputs = self.slassifier(bn_seatures)
        pth_outputs = self.plassifier(bn_peatures)
        key_outputs = self.bnc(cls_feats)

        return {
            "stp_outputs": stp_outputs,
            "pth_outputs": pth_outputs,
            "key_outputs": key_outputs,
            "global_features": seatures.squeeze(),
            "key_feats": cls_feats,
        }
