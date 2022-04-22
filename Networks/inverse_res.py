
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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class InvResBlock(nn.Module):
    def __init__(self, std_channel, w_dim):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(std_channel, std_channel, 3, 1, 1)
        )
        self.conv1 = nn.Conv2d(std_channel, std_channel, 3, 1, 1)
        self.A1 = AffineNet(w_dim, std_channel)
        self.conv2 = nn.Conv2d(std_channel, std_channel, 3, 1, 1)
        self.A2 = AffineNet(w_dim, std_channel)
        self.F_inv = nn.Sequential(
            nn.Conv2d(std_channel, std_channel, 3, 1, 1),
            nn.Conv2d(std_channel, std_channel, 3, 1, 1)
        )

    def R_inv(self, x, w):
        m1,s1 = self.A1(w)
        m2,s2 = self.A2(w)
        shape_1 = m1.shape + (1,1)
        shape_2 = m2.shape + (1,1)
        x = self.conv1(x)
        x = ins_nor(x) * s1.view(shape_1) + m1.view(shape_1)
        x = self.conv2(x)
        x = ins_nor(x) * s2.view(shape_2) + m2.view(shape_2)
        return x

    def forward(self, x, w):
        y = self.up(x)
        x_hat = self.R_inv(y, w)
        x = self.F_inv(y-x_hat)
        return x, ((x-x_hat)**2).mean()

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

class IRRender(nn.Module):
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
            "L1":InvResBlock(dim*2, w_dim=w_dim), #8*8
            "L2":InvResBlock(dim*2, w_dim=w_dim), #16*16
            "L3":InvResBlock(dim*2, w_dim=w_dim), #32*32
            "L4":InvResBlock(dim*2, w_dim=w_dim), #64*64
            "L5":InvResBlock(dim*2, w_dim=w_dim) #128*128
        })
        self.to_rgb = nn.Sequential(
            nn.Conv2d(dim*2*5, 3, 1, 1, 0),
        )
        self.to_alpha = nn.Sequential(
            nn.Conv2d(dim*2*5, 1, 1, 1, 0),
        )

    def forward(self, actions):

        w = self.MapNet(actions)

        x = self.const
        reg = 0.0
        for layer_name, STYLayer in self.Main.items():
            x, r = STYLayer(x, w)
            reg = reg + r

        return self.to_rgb(x), self.to_alpha(x), reg

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