import math
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.nn import LayerNorm
from common import *
from scipy.spatial.distance import cdist
import pdb

# Encoder
class PatchEmbed(nn.Module):
    def __init__(self, in_feats, out_feats, kernel_size, stride, bias = True):
        super(PatchEmbed, self).__init__()
        self.head = prosessing_conv(in_feats, out_feats, kernel_size, stride, bias=bias)

    def forward(self, x):
        x = self.head(x)
        # print("conv.shape:", x.shape)
        y = x.flatten(2).transpose(1,2)     # [B, C, H, W] -> [B, N, C]
        # print("patchembed shape y:", y.shape)
        return y

# SpeAM: Spectral Attention Mechanism
class BiLSTM2D(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layer: int, dropout: float):
        super(BiLSTM2D, self).__init__()
        self.LSTM_v = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layer,
                            batch_first=True,
                            dropout=dropout,
                            bidirectional=True)
        self.LSTM_h = nn.LSTM(input_size, hidden_size, num_layer, batch_first=True, dropout=dropout, bidirectional=True)
        self.fc = nn.Linear(in_features=4 * input_size, out_features=hidden_size)
        # weight init
        for name, param in self.LSTM_v.named_parameters():
            if name.startswith("weight"):
                nn.init.xavier_normal_(param)
            else:
                nn.init.zeros_(param)
        for name, param in self.LSTM_h.named_parameters():
            if name.startswith("weight"):
                nn.init.xavier_normal_(param)
            else:
                nn.init.zeros_(param)
        nn.init.xavier_normal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, x):
        B, N, C = x.shape
        H, W = int(np.sqrt(N)), int(np.sqrt(N))
        # x = x.permute(0, 2, 1)  # [B, N, C] ---> [B, C, N]
        x = x.contiguous().view(B, H, W, C)

        # print("lstm_x.shape:", x.shape)
        # x.shape：(batch_size, len_sen(patches nums), input_size(dimension per patches))
        # output: (batch_size, len_sen, num_directions * hidden_size),
        # h_n: (num_layers * num_directions, batch_size, hidden_size),
        # c_n: (num_layers * num_directions, batch_size, hidden_size),
        # x: input tensor shape: (B, H, W, C)
        v, (v_n, v_n) = self.LSTM_v(x.permute(0, 2, 1, 3).reshape(-1, H, C))
        v = v.reshape(B, H, W, -1).permute(0, 2, 1, 3)
        h, (h_n, c_n) = self.LSTM_h(x.reshape(-1, W, C))
        h = h.reshape(B, H, W, -1)
        x = torch.cat([v, h], dim=-1)
        # print("x.shape:", x.shape)
        res = self.fc(x)
        # print("res1.shape:", res.shape)
        res = res.permute(0, 3, 1, 2).flatten(2).transpose(1,2)  # [B, H, W, C] -> [B, C, H, W] -> [B, C, N] -> [B, N, C]
        # print("res2.shape:", res.shape)

        # print("lstm_output.shape:", output.shape)
        # print("lstm_h_n.shape:", h_n.shape)
        # print("lstm_c_n.shape:", c_n.shape)
        # backward_out = h_n[-1, :, :]
        # forward_out = h_n[-2, :, :]
        # features = torch.cat((forward_out, backward_out), 1)
        # features = torch.cat((output, output), 2)
        return res


# MLP layer
class Mlp(nn.Module):
    """
    MLP as used in Vision Transformer, MLP-Mixer and related networks
    """

    def __init__(self, in_features, hidden_features, act_layer=nn.GELU, drop=0.1):
        super(Mlp, self).__init__()
        self.fc1 = nn.Linear(in_features, hidden_features * 4)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features * 4, hidden_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        # print("x1.shape:", x.shape)
        x = self.fc1(x)
        # print("x2.shape:", x.shape)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        # print("x3.shape:", x.shape)
        x = self.drop(x)
        return x


# VM-BiLSTM: vertical horizontal BiLSTM Spectral Sub-Network
class spectral_Unit(nn.Module):  
    def __init__(self, in_feats, out_feats):
        super(spectral_Unit, self).__init__()
        self.BiLstm = BiLSTM2D(in_feats, out_feats, num_layer=1, dropout=0.5)
        self.Mlp = Mlp(out_feats, out_feats)
        self.Layer_norm = LayerNorm(out_feats)
        self.layer_num = nn.ModuleList()

    def forward(self, x):
        h = x
        x = self.Layer_norm(x)
        x = self.BiLstm(x)
        x = self.Mlp(x)
        x = x + h

        h = x
        x = self.Layer_norm(x)
        x = self.Mlp(x)
        x = x + h
        # print("x_spe.shape:", x.shape)
        return x    # [B, N, C]


# Multi-Adjacent Weight Matrix
class Weighted_adj(nn.Module):
    def __init__(self,
                 dim,  # input token dimension
                 num_heads=1,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop_ratio=0.,
                 proj_drop_ratio=0.):
        super(Weighted_adj, self).__init__()
        self.num_heads = num_heads  # 12
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5  # 0.125
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)

    def Norm_dynamic_adj(self, x):
        # Distance：It is used to measure the spatial distance between individuals, with greater distance indicating greater differences between individuals.
        # Similarity：The degree of similarity between individuals is calculated. In contrast to the distance measure, the smaller the value of the similarity measure, the smaller the similarity and the larger the difference between individuals.
        B, N, C = x.shape
        # y = torch.reshape(x, [B, C, H * W])
        # N = H * W
        # adj:[N, N], 1 if both nodes have edges, 0 otherwise.
        adj_euclidean = torch.zeros(B, N, N).cuda()
        adj_chebyshev = torch.zeros(B, N, N).cuda()
        adj_corrcoef = torch.zeros(B, N, N).cuda()
        
        # k = 5
        for b in range(B):
            dist1 = cdist(x[b, :, :].cpu().detach().numpy(), x[b, :, :].cpu().detach().numpy(), metric='euclidean')
            dist1 = torch.from_numpy(dist1).type(torch.FloatTensor).cuda()
            dist1 = torch.unsqueeze(dist1, 0)
            adj_euclidean[b, :, :] = dist1

            dist2 = cdist(x[b, :, :].cpu().detach().numpy(), x[b, :, :].cpu().detach().numpy(), metric='chebyshev')
            dist2 = torch.from_numpy(dist2).type(torch.FloatTensor).cuda()
            dist2 = torch.unsqueeze(dist2, 0)
            adj_chebyshev[b, :, :] = dist2

            dist3 = np.corrcoef(x[b, :, :].cpu().detach().numpy())
            dist3 = torch.from_numpy(dist3).type(torch.FloatTensor).cuda()
            dist3 = torch.unsqueeze(dist3, 0)
            adj_corrcoef[b, :, :] = dist3
        return adj_euclidean, adj_chebyshev, adj_corrcoef

    def forward(self, x):
        # [batch_size, num_patches, total_embed_dim]
        B, N, C = x.shape
        # print("x_norm?", x)
        adj_euclidean, adj_chebyshev, adj_corrcoef = self.Norm_dynamic_adj(x)
        a, b, c = adj_euclidean.shape
        sums = b * b

        eucl_qkv = adj_euclidean.reshape(B, N, 1, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        cheb_qkv = adj_chebyshev.reshape(B, N, 1, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        corr_qkv = adj_corrcoef.reshape(B, N, 1, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)

        q, k, v = eucl_qkv[0], cheb_qkv[0], corr_qkv[0]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        weighted_adj = (attn @ v).transpose(1, 2).reshape(B, N, c)
        weight_adj = weighted_adj.reshape(B, 1, sums)
        weight_adj_cpu = weight_adj.cpu().numpy()
        sums_elements = np.floor_divide(sums, 6)
        weighted_adj = np.where(weight_adj_cpu.argsort(2).argsort(2) >= (sums - sums_elements), 1, 0)
        weighted_adj = weighted_adj.reshape(B, b, c)
        weighted_adj = torch.from_numpy(weighted_adj).type(torch.FloatTensor).cuda()

        return weighted_adj


# MAW-GAT: Multi-Adjacent Weight Matrix GAT Spatial Sub-Network
class spatial_Unit(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(spatial_Unit, self).__init__()
        n_heads = 2
        dropout = 0.6
        alpha = 0.2
        weighted_adj_dim = 64
        # weighted_adj_dim = 256
        self.body = GAT(in_feats, out_feats, dropout, alpha, n_heads)
        self.weighted_adj = Weighted_adj(weighted_adj_dim)

    def forward(self, x):
        weighted_adj = self.weighted_adj(x)
        y = self.body(x, weighted_adj)
        # print("y.shape:", y.shape)
        # Sorted by the index of each dimension, [B,N,C]->[B,C,N]
        # y = y.permute(0, 2, 1).contiguous()
        # [B, C, N] = y.shape
        # H = int(math.sqrt(N))
        # W = int(math.sqrt(N))
        # y = torch.reshape(y, [B, C, H, W])
        # y = self.last(y)
        return y

# SSAT:Spatial-Spectral Attention Transposition Module
class SSB(nn.Module):
    def __init__(self, in_feats, conv=default_conv):
        super(SSB, self).__init__()
        act = nn.ReLU(True)
        res_scale = 1
        kernel_spa = 3
        kernel_spc = 1
        self.spa = SpatialResBlock(conv, in_feats, kernel_spa, act, res_scale)
        self.spc = SpectralAttentionResBlock(conv, in_feats, kernel_spc, act, res_scale)

    def forward(self, x):
        # print("xx.shape:", x.shape)
        return self.spc(self.spa(x))

# SSAT sequence
class SSPN(nn.Module):
    def __init__(self, in_feats, n_blocks):
        super(SSPN, self).__init__()
        m = []
        for i in range(n_blocks):
            m.append(SSB(in_feats))
        self.net = nn.Sequential(*m)

    def forward(self, x):
        res = self.net(x)
        # print("res.shape:", res.shape)
        res += x
        return res

# Decoder
class Decoder(nn.Module):
    def __init__(self, in_feats, n_blocks, kernel_size, stride):
        super(Decoder, self).__init__()
        self.SSPN = SSPN(in_feats, n_blocks)
        self.transpose_conv = transpose_conv(in_feats, in_feats, kernel_size, stride)

    def forward(self, x):
        B, N, C = x.shape
        H, W = int(np.sqrt(N)), int(np.sqrt(N))
        x = x.permute(0, 2, 1)  # [B, N, C] ---> [B, C, N]
        x = x.contiguous().view(B, C, H, W)
        # print("xxx.shape:", x.shape)
        x = self.transpose_conv(x)
        # print("x_trans.shape:", x.shape)
        x = self.SSPN(x)
        # print("x.shape:", x.shape, x)
        return x

# SRLSGAT Network Structure
class global_net(nn.Module):
    def __init__(self, in_feats, out_feats, n_blocks, n_scale):
        super(global_net, self).__init__()
        if n_scale == 2:
            kernel_size = 5       # n_scale = 2
            stride = 4
        else:
            kernel_size = 3     # n_scale = 4 8
            stride = 2
        self.pre = PatchEmbed(in_feats, out_feats, kernel_size, stride)
        self.spatial = spatial_Unit(out_feats, out_feats)       # MAW-GAT
        # self.spectral = spectral_Unit(out_feats, out_feats)
        self.Layer_norm = LayerNorm(out_feats)
        self.layer = nn.ModuleList()
        layer_num = 2
        for _ in range(layer_num):
            layer = spectral_Unit(out_feats, out_feats)     # VH-BiLSTM
            self.layer.append(copy.deepcopy(layer))
        self.last = Decoder(out_feats, n_blocks, kernel_size, stride)
        self.upsample = Upsampler(default_conv, n_scale, out_feats)
        self.tail = default_conv(out_feats, out_feats, kernel_size=3)

    def forward(self, x):
        # print("global x.shape:", x.shape)
        # print("x1", x)
        x = self.pre(x)
        # print("x2.shape", x.shape)
        spatial_result = self.spatial(x)        # MAW-GAT
        # print("x3", spatial_result)
        for layer_spectral in self.layer:
            spectral_result = layer_spectral(x)  # VH-BiLSTM
        # spectral_result = self.spectral(x)
        # print("x4", spectral_result)
        x = spatial_result + spectral_result        # MAW-GAT + VH-BiLSTM
        x = self.Layer_norm(x)
        x = self.last(x)
        y = self.upsample(x)        # reconstruction
        # print("upsample:", y.shape)
        y = self.tail(y)
        # print("con.shape:", y.shape)
        return y

# The SRLSGAT Network
class SRLSGAT(nn.Module):
    def __init__(self, n_subs, n_ovls, in_feats, out_feats, n_blocks, n_scale, use_share=True, conv=default_conv):
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
            self.start_idx.append(sta_ind)
            self.end_idx.append(end_ind)

        if self.shared:
            self.branch = global_net(n_subs, out_feats, n_blocks, n_scale)  # input is n_subs, hidden layer is out_feats, output is n_subs.
        else:
            self.branch = nn.ModuleList
            for i in range(self.G):
                self.branch.append(global_net(n_subs, out_feats, n_blocks, n_scale))
        self.skip_conv = conv(in_feats, out_feats, kernel_size)
        self.final = conv(out_feats, in_feats, kernel_size)  # reconstruction

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
