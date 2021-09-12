# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch
from torch import nn, sigmoid, einsum
from einops import rearrange
from fastreid.layers.batch_norm import IBN

class StripeAttention(nn.Module):

    def __init__(self, dim, heads=4, dimqk=128, dimv=128):
        super(StripeAttention, self).__init__()
        self.scale = dimqk ** -0.5
        self.heads = heads
        out_dimqk = heads * dimqk
        out_dimv = heads * dimv

        self.q = nn.Conv2d(dim, out_dimqk, 1, bias=False)
        self.k = nn.Conv2d(dim, out_dimqk, 1, bias=False)
        self.v = nn.Conv2d(dim, out_dimv, 1, bias=False)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, query_features, key_features, value_features, prob=False):
        heads = self.heads
        B, C, H, W = query_features.shape
        q = self.q(query_features)
        k = self.k(key_features)
        v = self.v(value_features)
        q, k, v = map(lambda x: rearrange(x, 'B (h d) H W -> B h (H W) d', h=heads), (q, k, v))

        q *= self.scale

        logits = einsum('bhxd,bhyd->bhxy',q, k)

        weights = self.softmax(logits)
        out = einsum('bhxy,bhyd->bhxd', weights, v)
        out = rearrange(out, 'B h (H W) d -> B (h d) H W', H=H)
        if prob is True:
            return out, weights
        else:
            return out

class StripeLayer(nn.Module):

    def __init__(self, inplanes, planes, num_stripe=8, p=0.3):
        super(StripeLayer, self).__init__()
        self.num_stripe = num_stripe
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=True)
        self.bn1 = IBN(planes, 'BN')
        self.dropout = nn.Dropout(p=p)

        self.sp = StripeAttention(planes, 4, planes // 4, planes // 4)
        self.bn2 = nn.BatchNorm2d(planes)

        self.conv3 = nn.Conv2d(planes, inplanes, kernel_size=1, bias=True)
        self.bn3 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, mask = None):
        residual = x if mask is None else x * mask

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # stripe attention
        num = self.num_stripe
        res_out = out
        stripes = out.chunk(num, dim=2)
        outs = [self.sp(stripes[i], stripes[i], stripes[i]) for i in range(len(stripes))]
        out = torch.cat(outs, dim=2).contiguous()

        out = out + self.dropout(res_out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += residual
        out = self.relu(out)

        return out