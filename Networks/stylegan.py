
from turtle import forward
import torch
import torch.nn as nn
from torch.nn import init
import functools
import torch.nn.functional as F
import math

PI = math.pi
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class StyleBlock(nn.Module):
    def __init__(self, input_channel, output_channel, w_dim):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2)
        self.conv1 = nn.Conv2d(input_channel, input_channel, 3, 1, 1)
        self.A1 = AffineNet(w_dim, input_channel)
        self.conv2 = nn.Conv2d(input_channel, output_channel, 3, 1, 1)
        self.A2 = AffineNet(w_dim, output_channel)
        self.RGB = nn.Sequential(
            nn.Conv2d(output_channel, 8, 1, 1, 0),
            nn.Conv2d(8, 3, 1, 1, 0),
        ) 
        self.ALPHA = nn.Sequential(
            nn.Conv2d(output_channel, 8, 1, 1, 0),
            nn.Conv2d(8, 1, 1, 1, 0),
        ) 

    def to_res(self, x):
        return torch.sigmoid(self.RGB(x)), torch.sigmoid(self.ALPHA(x))

    def forward(self, x, w):
        m1,s1 = self.A1(w)
        m2,s2 = self.A2(w)
        shape_1 = m1.shape + (1,1)
        shape_2 = m2.shape + (1,1)
        x = self.up(x)
        x = self.conv1(x)
        x = ins_nor(x) * s1.view(shape_1) + m1.view(shape_1)
        x = self.conv2(x)
        x = ins_nor(x) * s2.view(shape_2) + m2.view(shape_2)
        return x

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
    def __init__(self, action_size, w_dim=256):
        super().__init__()
        self.map = nn.Sequential(
            nn.Linear(action_size, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, w_dim),
            nn.LeakyReLU()
        )
    def forward(self, x):
        return self.map(x)

class StyleRender(nn.Module):
    def __init__(self, action_size, noise_dim=4, dim=32, w_dim=256):
        super().__init__()
        self.noise_dim = noise_dim
        self.w_dim = w_dim
        self.dim = dim 
        self.const = nn.Parameter(
            torch.ones((1,dim*16,4,4))
        )
        self.MapNet = MappingNet(action_size=action_size, w_dim=w_dim)
        self.main = nn.ModuleDict({
            "L1":StyleBlock(dim*16, dim*16, w_dim=w_dim), #8*8
            "L2":StyleBlock(dim*16, dim*8, w_dim=w_dim), #16*16
            "L3":StyleBlock(dim*8, dim*4, w_dim=w_dim), #32*32
            "L4":StyleBlock(dim*4, dim*2, w_dim=w_dim), #64*64
            "L5":StyleBlock(dim*2, dim, w_dim=w_dim) #128*128
        })

    def forward(self, actions, layer=None, alpha=None):

        w = self.MapNet(actions)

        x = self.const

        if layer is None:
            for layer_name, STYLayer in self.main.items():
                x = STYLayer(x, w)
            return self.main["L5"].to_res(x)
        else:
            for layer_name, STYLayer in self.main.items():
                if layer_name == layer:
                    break
                x = STYLayer(x, w)
                last_layer = layer_name
            if alpha is not None:
                y = self.main[last_layer].up(x)
                y = self.main[last_layer].to_res(y)
            x = self.main[layer](x, w)
            x = self.main[layer].to_res(x)
            if alpha is not None:
                return x[0] * alpha + y[0] * (1-alpha), x[1] * alpha + y[1] * (1-alpha) 
            else:
                return x


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