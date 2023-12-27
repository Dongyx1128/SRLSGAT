from thop import profile
from thop import clever_format
import math
import copy
import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Dropout, Softmax, Linear, Conv2d, LayerNorm
from common import *
import scipy.sparse as sp
from scipy.spatial.distance import cdist
from SRLSGAT import *

# device = torch.device("cpu")
cuda = 1
device = torch.device("cuda" if cuda else "cpu")

class SRLSGAT(nn.Module):
    def __init__(self, n_subs=31, n_ovls=0, in_feats=31, out_feats=64, n_blocks=6, n_scale=8, use_share=True, conv=default_conv):
        super(SRLSGAT, self).__init__()
        kernel_size = 3
        self.shared = use_share

        # calculate the group number (the number of branch networks)
        self.G = math.ceil((in_feats - n_ovls) / (n_subs - n_ovls))
        # calculate group indices
        self.start_idx = []
        self.end_idx = []

        for g in range(self.G):
            sta_ind = (n_subs - n_ovls) * g
            end_ind = sta_ind + n_subs
            if end_ind > in_feats:
                end_ind = in_feats
                sta_ind = in_feats - n_subs
            self.start_idx.append(
                sta_ind)
            self.end_idx.append(
                end_ind)

        if self.shared:
            self.branch = global_net(n_subs, out_feats, n_blocks, n_scale)
        else:
            self.branch = nn.ModuleList
            for i in range(self.G):
                self.branch.append(global_net(n_subs, out_feats, n_blocks, n_scale))
        self.skip_conv = conv(in_feats, out_feats, kernel_size)
        self.final = conv(out_feats, in_feats, kernel_size)

    def forward(self, x, lms):
        for g in range(self.G):
            sta_ind = self.start_idx[g]
            end_ind = self.end_idx[g]

            xi = x[:, sta_ind:end_ind, :, :]
            if self.shared:
                xi = self.branch(xi)
            else:
                xi = self.branch[g](xi)

        y = xi + self.skip_conv(lms)
        y = self.final(y)
        return y

if __name__ == "__main__":
    model = SRLSGAT().to(device)
    x = torch.randn(1, 31, 16, 16).to(device)
    lms = torch.randn(1, 31, 128, 128).to(device)
    flops, params = profile(model, inputs=(x, lms,))
    print(f"result_1: FLOPs {flops} Params {params}")
    flops, params = clever_format([flops, params], "%.3f")
    print(f"result_2: FLOPs {flops} Params {params}")
    # print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
    # print('Params = ' + str(params / 1000 ** 2) + 'M')