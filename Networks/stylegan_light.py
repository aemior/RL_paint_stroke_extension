
import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class StyleBlock(nn.Module):
    def __init__(self, input_channel, output_channel, w_dim):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2)
        self.conv1 = nn.Conv2d(input_channel, input_channel, 3, 1, 1)
        self.A1 = AffineNet(w_dim, input_channel)
        self.conv2 = nn.Conv2d(input_channel, output_channel, 3, 1, 1)
        self.A2 = AffineNet(w_dim, output_channel)

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
    def __init__(self, action_size, w_dim=512):
        super().__init__()
        self.map = nn.Sequential(
            nn.Linear(action_size, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 512),
            nn.LeakyReLU(),
            nn.Linear(512, w_dim),
            nn.LeakyReLU(),
        )
    def forward(self, x):
        return self.map(x)

class StyleRenderLight(nn.Module):
    def __init__(self, action_size, noise_dim=4, dim=32, w_dim=512):
        super().__init__()
        self.noise_dim = noise_dim
        self.w_dim = w_dim
        self.dim = dim 
        self.const = nn.Parameter(
            torch.ones((1,dim*16,4,4))
        )
        self.MapNet = MappingNet(action_size=action_size, w_dim=w_dim)
        self.L1 = StyleBlock(dim*16, dim*16, w_dim=w_dim) #8*8 /32 init
        self.L2 = StyleBlock(dim*16, dim*8, w_dim=w_dim) #16*16
        self.L3 = StyleBlock(dim*8, dim*4, w_dim=w_dim) #32*32
        self.L4 = StyleBlock(dim*4, dim*2, w_dim=w_dim) #64*64
        self.L5 = StyleBlock(dim*2, dim, w_dim=w_dim) #128*128
        self.to_RGB = nn.Conv2d(dim, 3, 1, 1, 0)
        self.to_ALPHA = nn.Conv2d(dim, 1, 1, 1, 0)

    def main_forward(self, actions):
        w = self.MapNet(actions)
        x = self.const
        x = self.L1(x, w)
        x = self.L2(x, w)
        x = self.L3(x, w)
        x = self.L4(x, w)
        x = self.L5(x, w)
        return x, w

    def forward(self, actions):
        x, w = self.main_forward(actions)
        return self.to_RGB(x), self.to_ALPHA(x)

class StyleRenderLight_256(StyleRenderLight):
    def __init__(self, action_size, noise_dim=4, dim=32, w_dim=512):
        super().__init__(action_size, noise_dim, dim, w_dim)
        self.L6 = StyleBlock(dim, dim, w_dim=w_dim) #256*256

    def main_forward(self, actions):
        x, w = super().main_forward(actions)
        x = self.L6(x, w)
        return x, w


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