
from turtle import forward
from matplotlib.pyplot import sca
from numpy import std
import torch
import torch.nn as nn
from torch.nn import init
import functools
import torch.nn.functional as F
import math

PI = math.pi
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ResBlock(nn.Module):
    def __init__(self, std_channel, inner_channel, w_dim, top=False):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2)
        self.conv1 = nn.Conv2d(std_channel, inner_channel, 3, 1, 1)
        self.A1 = AffineNet(w_dim, inner_channel)
        self.conv2 = nn.Conv2d(inner_channel, inner_channel, 3, 1, 1)
        self.A2 = AffineNet(w_dim, inner_channel)
        self.conv3 = nn.Conv2d(inner_channel, std_channel, 3, 1, 1)
        if top:
            self.forward = self.top_forward
        else:
            self.conv4 = nn.Conv2d(std_channel, std_channel, 3, 1, 1)
    def adain_merge(self, x, w):
        m1,s1 = self.A1(w)
        m2,s2 = self.A2(w)
        shape_1 = m1.shape + (1,1)
        shape_2 = m2.shape + (1,1)
        x = self.conv1(x)
        x = ins_nor(x) * s1.view(shape_1) + m1.view(shape_1)
        x = self.conv2(x)
        x = ins_nor(x) * s2.view(shape_2) + m2.view(shape_2)
        x = self.conv3(x)
        return x

    def top_forward(self, x, w):
        x = self.up(x)
        x = self.adain_merge(x, w)
        return x

    def forward(self, x, w):
        y = self.up(x)
        x = self.adain_merge(y, w)
        return x+self.conv4(y), x

class AffineNet(nn.Module):
    def __init__(self, w_dim, out_size):
        super().__init__()
        self.mean_path = nn.Sequential(
            nn.Linear(w_dim, out_size),
            nn.LeakyReLU()
        )
        self.std_path = nn.Sequential(
            nn.Linear(w_dim, out_size),
            nn.LeakyReLU()
        )
    def forward(self, w):
        return self.mean_path(w), self.std_path(w)

class MappingNet(nn.Module):
    def __init__(self, action_size, w_dim=512):
        super().__init__()
        self.map = nn.Sequential(
            nn.Linear(action_size, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, w_dim),
            nn.LeakyReLU()
        )
    def forward(self, x):
        return self.map(x)

class SRRender(nn.Module):
    def __init__(self, action_size, noise_dim=4, dim=32, w_dim=512):
        super().__init__()
        self.noise_dim = noise_dim
        self.w_dim = w_dim
        self.dim = dim 
        self.const = nn.Parameter(
            torch.ones((1,dim*2,4,4))
        )
        self.MapNet = MappingNet(action_size=action_size, w_dim=w_dim)
        self.Main = nn.ModuleDict({
            "L1":ResBlock(dim*2, dim*4, w_dim=w_dim), #8*8
            "L2":ResBlock(dim*2, dim*4, w_dim=w_dim), #16*16
            "L3":ResBlock(dim*2, dim*8, w_dim=w_dim), #32*32
            "L4":ResBlock(dim*2, dim*4, w_dim=w_dim), #64*64
            "L5":ResBlock(dim*2, dim*2, w_dim=w_dim) #128*128
        })
        self.Up = nn.ModuleDict({
            "L1":nn.Sequential(
                nn.Upsample(scale_factor=16),
                nn.Conv2d(dim*2, dim*2, 9, 1, 4)
                ),
            "L2":nn.Sequential(
                nn.Upsample(scale_factor=8),
                nn.Conv2d(dim*2, dim*2, 7, 1, 3)
                ),
            "L3":nn.Sequential(
                nn.Upsample(scale_factor=4),
                nn.Conv2d(dim*2, dim*2, 5, 1, 2)
                ),
            "L4":nn.Sequential(
                nn.Upsample(scale_factor=2),
                nn.Conv2d(dim*2, dim*2, 3, 1, 1)
                )
        })
        self.Merge = nn.ModuleDict({
            "L1":nn.Conv2d(dim*2, dim*2, 3, 1, 1),
            "L2":nn.Conv2d(dim*2, dim*2, 3, 1, 1),
            "L3":nn.Conv2d(dim*2, dim*2, 3, 1, 1),
            "L4":nn.Conv2d(dim*2, dim*2, 3, 1, 1),
        })
        self.to_rgb = nn.Sequential(
            nn.Conv2d(dim*4, 3, 1, 1, 0),
        )
        self.to_alpha = nn.Sequential(
            nn.Conv2d(dim*4, 1, 1, 1, 0),
        )
        self.list = ["L1", "L2", "L3", "L4", "L5"]

    def forward(self, actions):

        w = self.MapNet(actions)

        x = self.const
        res = []
        for layer_name, STYLayer in self.Main.items():
            if layer_name != "L5":
                x,y = STYLayer(x, w)
                res.append(self.Up[layer_name](y))
            else:
                x = STYLayer(x, w)
                res.append(x)
        
        for l in [3, 2, 1, 0]:
            res[l] = self.Merge[self.list[l]](res[l+1]+res[l])

        x = torch.cat(res, dim=1)
        return self.to_rgb(x), self.to_alpha(x)

def ins_nor(feat):
    size = feat.size()
    content_mean, content_std = calc_mean_std(feat)
    normalized_feat = (feat - content_mean.expand(
        size)) / content_std.expand(size)
    return normalized_feat

def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std