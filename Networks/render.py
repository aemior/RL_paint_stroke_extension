import torch
import torch.nn as nn
from torch.nn import init
import functools
import torch.nn.functional as F
import math

from Networks.stylegan_v2 import StyleLight, StyleRender

PI = math.pi
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

###############################################################################
# Helper Functions
###############################################################################


class Identity(nn.Module):
    def forward(self, x):
        return x

def get_norm_layer(norm_type='instance'):
    """Return a normalization layer
    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none
    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        norm_layer = lambda x: Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.
    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.
    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2
    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net


def define_R(rddim, shape_dim, netR, init_type='normal', init_gain=0.02, gpu_ids=[]):
    net = None
    if netR == 'dual-render':
        net = DualRender(rddim, shape_dim)
    elif netR == 'dual-render-noise':
        net = DualRender_noise(rddim, shape_dim)
    elif netR == 'only-ret':
        net = PixelShuffleNet_4C(rddim)
    elif netR == 'only-shad':
        net = DCGAN_4C(rddim)
    elif netR == 'NP-render':
        net = NPGenerator(rddim)
    elif netR == 'NP-dual':
        net = NPDualPath(rddim)
    elif netR == 'UP-conv':
        net = UPPConv(rddim)
    elif netR == 'Style-render':
        net = StyleRender(rddim)
    elif netR == 'Style-light':
        from Networks.stylegan_v2 import StyleLight
        net = StyleLight(rddim)
    elif netR == 'Style-Light':
        from Networks.stylegan_light import StyleRenderLight
        net = StyleRenderLight(rddim)
    elif netR == 'BRM-render':
        from Networks.ebrn import BRMRender
        net = BRMRender(rddim)
    elif netR == 'Res-render':
        from Networks.stylegan_res import SRRender
        net = SRRender(rddim)
    elif netR == 'MB-render':
        from Networks.multiband import MBRender
        net = MBRender(rddim)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % netR)
    return init_net(net, init_type, init_gain, gpu_ids)

###############################################################################
# Networks Define 
###############################################################################

class DCGAN_3C(nn.Module):
    def __init__(self, rddim, ngf=64):
        super(DCGAN_3C, self).__init__()
        input_nc = rddim
        self.out_size = 128
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(input_nc, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4

            nn.ConvTranspose2d(ngf * 8, ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8

            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16

            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 32 x 32

            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf*2) x 64 x 64

            nn.ConvTranspose2d(ngf, 3, 4, 2, 1, bias=False),
            # state size. (nc) x 128 x 128
        )

    def forward(self, input):
        return self.main(input)


class PixelShuffleNet(nn.Module):
    def __init__(self, input_nc):
        super(PixelShuffleNet, self).__init__()
        self.fc1 = (nn.Linear(input_nc, 512))
        self.fc2 = (nn.Linear(512, 1024))
        self.fc3 = (nn.Linear(1024, 2048))
        self.fc4 = (nn.Linear(2048, 4096))
        self.conv1 = (nn.Conv2d(16, 32, 3, 1, 1))
        self.conv2 = (nn.Conv2d(32, 32, 3, 1, 1))
        self.conv3 = (nn.Conv2d(8, 16, 3, 1, 1))
        self.conv4 = (nn.Conv2d(16, 16, 3, 1, 1))
        self.conv5 = (nn.Conv2d(4, 8, 3, 1, 1))
        self.conv6 = (nn.Conv2d(8, 4, 3, 1, 1))
        self.pixel_shuffle = nn.PixelShuffle(2)

    def forward(self, x):
        x = x.squeeze()
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = x.view(-1, 16, 16, 16)
        x = F.relu(self.conv1(x))
        x = self.pixel_shuffle(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pixel_shuffle(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.pixel_shuffle(self.conv6(x))
        x = x.view(-1, 1, 128, 128)
        return x


class DualRender(nn.Module):
    def __init__(self, rddim, shape_dim):
        super(DualRender, self).__init__()
        self.rddim = rddim
        self.shape_dim = shape_dim
        self.shading_path = DCGAN_3C(rddim)
        self.raster_path = PixelShuffleNet(shape_dim)
    
    def forward(self, a):
        foreground = self.shading_path(a.view(a.shape+(1,1)))
        alpha = self.raster_path(a[:,:self.shape_dim])
        return foreground, alpha

class DualRender_noise(nn.Module):
    def __init__(self, rddim, shape_dim):
        super(DualRender_noise, self).__init__()
        self.rddim = rddim+1
        self.shape_dim = shape_dim+1
        self.shading_path = DCGAN_3C(self.rddim)
        self.raster_path = PixelShuffleNet(self.shape_dim)
    
    def forward(self, a):
        noise = torch.rand((a.shape[0],1)).to(a.device)
        s_p = torch.cat((a.view(a.shape+(1,1)), noise.view(a.shape[0],1,1,1)), dim=1)
        r_p = torch.cat((a[:,:self.shape_dim-1], noise), dim=1)
        #foreground = F.sigmoid(self.shading_path(s_p))
        #alpha = F.sigmoid(self.raster_path(r_p))
        foreground = self.shading_path(s_p)
        alpha = self.raster_path(r_p)
        return foreground, alpha

class DCGAN_4C(nn.Module):
    def __init__(self, rddim, ngf=64):
        super(DCGAN_4C, self).__init__()
        input_nc = rddim
        self.out_size = 128
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(input_nc, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4

            nn.ConvTranspose2d(ngf * 8, ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8

            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16

            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 32 x 32

            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf*2) x 64 x 64

            nn.ConvTranspose2d(ngf, 4, 4, 2, 1, bias=False),
            # state size. (nc) x 128 x 128
        )

    def forward(self, input):
        res = self.main(input.view(input.shape+(1,1)))
        return res[:,:3,:,:],res[:,3,:,:].view(res.shape[0],1,128,128)




class PixelShuffleNet_4C(nn.Module):
    def __init__(self, input_nc):
        super(PixelShuffleNet_4C, self).__init__()
        self.fc1 = (nn.Linear(input_nc, 512))
        self.fc2 = (nn.Linear(512, 1024))
        self.fc3 = (nn.Linear(1024, 2048))
        self.fc4 = (nn.Linear(2048, 4096))
        self.conv1 = (nn.Conv2d(16, 32, 3, 1, 1))
        self.conv2 = (nn.Conv2d(32, 32, 3, 1, 1))
        self.conv3 = (nn.Conv2d(8, 16, 3, 1, 1))
        self.conv4 = (nn.Conv2d(16, 16, 3, 1, 1))
        self.conv5 = (nn.Conv2d(4, 8, 3, 1, 1))
        self.conv6 = (nn.Conv2d(8, 4*4, 3, 1, 1))
        self.pixel_shuffle = nn.PixelShuffle(2)

    def forward(self, x):
        x = x.squeeze()
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = x.view(-1, 16, 16, 16)
        x = F.relu(self.conv1(x))
        x = self.pixel_shuffle(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pixel_shuffle(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.pixel_shuffle(self.conv6(x))
        x = x.view(-1, 4, 128, 128)
        return x[:,:3,:,:], x[:,3,:,:].view(x.shape[0],1,128,128)

class NPGenerator(nn.Module):
  def __init__(self, action_size, dim=16, noise_dim=6, num_deterministic=0):
    super(NPGenerator, self).__init__()
    self.dim = dim
    self.noise_dim = noise_dim
    self.num_deterministic = num_deterministic

    self.fc1 = nn.Linear(action_size + noise_dim, 4*4*(dim*16))  # This seems.. wrong.  Should it be dim*8?
    self.bn1 = nn.BatchNorm2d(dim*16)
    self.deconv1 = nn.ConvTranspose2d(dim*16, dim*8, 4, stride=2, padding=1)
    self.bn2 = nn.BatchNorm2d(dim*8)
    self.deconv2 = nn.ConvTranspose2d(dim*8, dim*4, 4, stride=2, padding=1)
    self.bn3 = nn.BatchNorm2d(dim*4)
    self.deconv3 = nn.ConvTranspose2d(dim*4, dim*2, 4, stride=2, padding=1)
    self.bn4 = nn.BatchNorm2d(dim*2)
    self.deconv4 = nn.ConvTranspose2d(dim*2, dim, 4, stride=2, padding=1)
    self.bn5 = nn.BatchNorm2d(dim)
    self.deconv5 = nn.ConvTranspose2d(dim, 4, 4, stride=2, padding=1)
    self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)

  def forward(self, actions):
    if self.noise_dim > 0:
      batch_size = actions.shape[0]
      noise_concat = torch.randn(batch_size, self.noise_dim - self.num_deterministic).to(actions.device)
      actions = torch.cat([actions, noise_concat], dim=1)

    x = self.fc1(actions)
    x = x.view(-1, self.dim*16, 4, 4)
    x = F.relu(self.bn1(x))
    x = F.relu(self.bn2(self.deconv1(x)))
    x = F.relu(self.bn3(self.deconv2(x)))
    x = F.relu(self.bn4(self.deconv3(x)))
    x = F.relu(self.bn5(self.deconv4(x)))
    x = F.sigmoid(self.deconv5(x))
    import pdb
    pdb.set_trace()
    x = x.view(-1, 4, 128, 128)
    return x[:,:3,:,:], x[:,3,:,:].view(x.shape[0],1,128,128)

class NPSinglePath(nn.Module):
    def __init__(self, dim, out_dim):
        super(NPSinglePath, self).__init__()
        self.dim = dim
        self.deconv5 = nn.ConvTranspose2d(dim, out_dim, 4, stride=2, padding=1)

    def forward(self, x):
        x = F.sigmoid(self.deconv5(x))
        return x

class NPDualPath(nn.Module):
    def __init__(self, action_size, dim=16, noise_dim=6, num_deterministic=0):
        super(NPDualPath, self).__init__()
        self.dim = dim
        self.noise_dim = noise_dim
        self.num_deterministic = num_deterministic
        self.fc1 = nn.Linear(action_size + noise_dim, 4*4*(dim*16))  # This seems.. wrong.  Should it be dim*8?
        self.bn1 = nn.BatchNorm2d(dim*16)
        self.deconv1 = nn.ConvTranspose2d(dim*16, dim*8, 4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(dim*8)
        self.deconv2 = nn.ConvTranspose2d(dim*8, dim*4, 4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(dim*4)
        self.deconv3 = nn.ConvTranspose2d(dim*4, dim*2, 4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(dim*2)
        self.deconv4 = nn.ConvTranspose2d(dim*2, dim, 4, stride=2, padding=1)
        self.bn5 = nn.BatchNorm2d(dim)
        self.ret = NPSinglePath(dim, 1)
        self.shad = NPSinglePath(dim, 3)

    def forward(self, actions):
        if self.noise_dim > 0:
            batch_size = actions.shape[0]
            noise_concat = torch.randn(batch_size, self.noise_dim - self.num_deterministic).to(actions.device)
            actions = torch.cat([actions, noise_concat], dim=1)

        x = self.fc1(actions)
        x = x.view(-1, self.dim*16, 4, 4)
        x = F.relu(self.bn1(x))
        x = F.relu(self.bn2(self.deconv1(x)))
        x = F.relu(self.bn3(self.deconv2(x)))
        x = F.relu(self.bn4(self.deconv3(x)))
        x = F.relu(self.bn5(self.deconv4(x)))
        return self.shad(x), self.ret(x)

class UPBlock(nn.Module):
    def __init__(self, input_channel, output_channel):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2)
        self.conv1 = nn.Conv2d(input_channel, input_channel, 3, 1, 1)
        self.conv2 = nn.Conv2d(input_channel, output_channel, 3, 1, 1)
        self.RGB = nn.Sequential(
            nn.Conv2d(output_channel, 8, 3, 1, 1),
            nn.Conv2d(8, 3, 1, 1, 0),
        ) 
        self.ALPHA = nn.Sequential(
            nn.Conv2d(output_channel, 8, 3, 1, 1),
            nn.Conv2d(8, 1, 1, 1, 0),
        ) 

    def to_res(self, x):
        return torch.sigmoid(self.RGB(x)), torch.sigmoid(self.ALPHA(x))

    def forward(self, x):
        x = self.up(x)
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class UPPConv(nn.Module):
    def __init__(self, action_size, noise_dim=4, dim=16):
        super().__init__()
        self.noise_dim = noise_dim
        self.dim = dim 
        self.fc1 = nn.Linear(action_size+noise_dim, 4*4*dim)
        self.conv1 = nn.Conv2d(dim, dim*2, 3, 1, 1)
        self.main = nn.ModuleDict({
            "L1":UPBlock(dim*2, dim*4), #8*8
            "L2":UPBlock(dim*4, dim*8), #16*16
            "L3":UPBlock(dim*8, dim*16), #32*32
            "L4":UPBlock(dim*16, dim*8), #64*64
            "L5":UPBlock(dim*8, dim*4) #128*128
        })

    def forward(self, actions, layer=None):
        if self.noise_dim > 0:
            batch_size = actions.shape[0]
            noise_concat = torch.randn(batch_size, self.noise_dim).to(actions.device)
            actions = torch.cat([actions, noise_concat], dim=1)

        x = self.fc1(actions).view(-1,self.dim,4,4)
        x = self.conv1(x)

        if layer is None:
            for layer_name, UPLayer in self.main.items():
                x = UPLayer(x)
            return self.main["L5"].to_res(x)
        else:
            for layer_name, UPLayer in self.main.items():
                if layer_name == layer:
                    break
                x = UPLayer(x)
            x = self.main[layer](x)
            return self.main[layer].to_res(x)



