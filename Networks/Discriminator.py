from signal import Sigmasks
from turtle import forward
import torch
import torch.nn as nn
from torch import autograd
import torch.nn.functional as F

from Networks.translator import init_net

def define_D(cond_dim):
    #net = torch.nn.utils.spectral_norm(CondDescriminator(cond_dim))
    #net = CondDescriminator(cond_dim)
    #net = CondDescriminatorV5(cond_dim)
    #net = Descriminator(cond_dim)
    net = NPDiscriminator(cond_dim)
    #return net
    return init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[])

class CondDescriminator(nn.Module):
    def __init__(self, cond_dim):
        super(CondDescriminator, self).__init__()
        self.in_fc = nn.utils.spectral_norm(nn.Linear(cond_dim, 16)) 
        self.in_conv = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(3, 16, 4, 2, 1)),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.conv = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(16, 32, 4, 2, 1)),
            nn.LeakyReLU(negative_slope=0.2),
            nn.utils.spectral_norm(nn.Conv2d(32, 64, 4, 2, 1)),
            nn.LeakyReLU(negative_slope=0.2),
            nn.utils.spectral_norm(nn.Conv2d(64, 128, 4, 2, 1)),
            nn.LeakyReLU(negative_slope=0.2),
            nn.utils.spectral_norm(nn.Conv2d(128, 256, 4, 2, 1)),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.fc = nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(4096, 1)),
            nn.LeakyReLU(negative_slope=0.2),
        )

    def set_cond(self, cond):
        self.cond = F.relu(self.in_fc(cond))

    def forward(self, x):
        x = self.in_conv(x)
        x = x + self.cond.view(-1,16,1,1)
        x = self.conv(x)
        x = self.fc(x.view(-1,4096))
        return x

class CondDescriminatorV2(nn.Module):
    def __init__(self, cond_dim):
        super(CondDescriminatorV2, self).__init__()
        self.in_fc = nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(cond_dim, 512)),
            nn.LeakyReLU(),
            nn.utils.spectral_norm(nn.Linear(512, 1024)),
            nn.LeakyReLU(),
            nn.utils.spectral_norm(nn.Linear(1024, 2048)),
            nn.LeakyReLU(),
            nn.utils.spectral_norm(nn.Linear(2048, 4096)),
            nn.LeakyReLU(),
        )
        self.in_conv = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(3, 16, 4, 2, 1)),
            nn.LeakyReLU(),
            nn.utils.spectral_norm(nn.Conv2d(16, 32, 4, 2, 1)),
            nn.LeakyReLU(),
            nn.utils.spectral_norm(nn.Conv2d(32, 64, 4, 2, 1)),
            nn.LeakyReLU(),
        )
        self.conv = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(80, 128, 4, 2, 1)),
            nn.LeakyReLU(),
            nn.utils.spectral_norm(nn.Conv2d(128, 256, 4, 2, 1)),
            nn.LeakyReLU(),
        )
        self.out_fc = nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(4096, 1)),
            nn.Sigmoid(),
        )
    def set_cond(self, cond):
        self.cond = self.in_fc(cond).view(-1,16,16,16)
    def forward(self, x):
        x = self.in_conv(x)
        x = self.conv(torch.cat((x, self.cond), dim=1))
        x = self.out_fc(x.view(-1,4096))
        return x

class CondDescriminatorV3(nn.Module):
    def __init__(self, cond_dim):
        super(CondDescriminatorV3, self).__init__()
        self.in_fc = nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(cond_dim, 2048)),
            nn.LeakyReLU(),
        )
        self.merge_conv = nn.Sequential( 
            nn.utils.spectral_norm(nn.Conv2d(8, 16, 3, 1, 1)),
            nn.LeakyReLU(),
            nn.utils.spectral_norm(nn.Conv2d(16, 32, 3, 1, 1)),
            nn.LeakyReLU(),
            nn.utils.spectral_norm(nn.Conv2d(32, 64, 3, 1, 1)),
            nn.LeakyReLU(),
        )
        self.in_conv = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(3, 16, 5, 2, 2)),
            nn.LeakyReLU(),
            nn.utils.spectral_norm(nn.Conv2d(16, 32, 5, 2, 2)),
            nn.LeakyReLU(),
            nn.utils.spectral_norm(nn.Conv2d(32, 64, 5, 2, 2)),
            nn.LeakyReLU(),
        )
        self.conv = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(128, 128, 5, 2, 2)),
            nn.LeakyReLU(),
            nn.utils.spectral_norm(nn.Conv2d(128, 1, 5, 2, 2)),
            nn.LeakyReLU(),
        )
    def set_cond(self, cond):
        cond = self.in_fc(cond).view(-1,8,16,16)
        self.cond = self.merge_conv(cond)
    def forward(self, x):
        x = self.in_conv(x)
        x = self.conv(torch.cat((x, self.cond), dim=1))
        #x = self.conv(x)
        x = F.avg_pool2d(x, 4)
        x = F.sigmoid(x)
        return x.view(-1,1)

class CondDescriminatorV4(nn.Module):
    def __init__(self, cond_dim):
        super(CondDescriminatorV4, self).__init__()
        self.in_fc = nn.Sequential(
            nn.Linear(cond_dim, 2048),
            nn.LeakyReLU(),
        )
        self.merge_conv = nn.Sequential( 
            nn.Conv2d(8, 16, 3, 1, 1),
            nn.LeakyReLU(),
            nn.Conv2d(16, 32, 3, 1, 1),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.LeakyReLU(),
        )
        self.in_conv = nn.Sequential(
            nn.Conv2d(3, 16, 5, 2, 2),
            nn.LeakyReLU(),
            nn.Conv2d(16, 32, 5, 2, 2),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, 5, 2, 2),
            nn.LeakyReLU(),
        )
        self.conv = nn.Sequential(
            nn.Conv2d(128, 128, 5, 2, 2),
            nn.LeakyReLU(),
            nn.Conv2d(128, 1, 5, 2, 2),
            nn.LeakyReLU(),
        )
    def set_cond(self, cond):
        cond = self.in_fc(cond).view(-1,8,16,16)
        self.cond = self.merge_conv(cond)
    def forward(self, x):
        x = self.in_conv(x)
        x = self.conv(torch.cat((x, self.cond), dim=1))
        x = F.avg_pool2d(x, 4)
        x = F.sigmoid(x)
        return x.view(-1,1)

class CondDescriminatorV5(nn.Module):
    """
    这个鉴别器可以收敛，学习率2e-4， real-0.5*(fake+mis)
    """
    def __init__(self, cond_dim):
        super(CondDescriminatorV5, self).__init__()
        self.in_fc = nn.Sequential(
            nn.Linear(cond_dim, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 2048),
            nn.LeakyReLU(),
            nn.Linear(2048, 4096),
            nn.LeakyReLU(),
        )
        self.merge_conv = nn.Sequential( 
            nn.Conv2d(16, 16, 3, 1, 1),
            nn.LeakyReLU(),
            nn.Conv2d(16, 32, 3, 1, 1),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.LeakyReLU(),
        )
        self.in_conv = nn.Sequential(
            nn.Conv2d(3, 16, 5, 2, 2),
            nn.LeakyReLU(),
            nn.Conv2d(16, 32, 5, 2, 2),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, 5, 2, 2),
            nn.LeakyReLU(),
        )
        self.conv = nn.Sequential(
            nn.Conv2d(128, 128, 5, 2, 2),
            nn.LeakyReLU(),
            nn.Conv2d(128, 1, 5, 2, 2),
            nn.LeakyReLU(),
        )
    def set_cond(self, cond):
        cond = self.in_fc(cond).view(-1,16,16,16)
        self.cond = self.merge_conv(cond)
    def forward(self, x):
        x = self.in_conv(x)
        x = self.conv(torch.cat((x, self.cond), dim=1))
        x = F.avg_pool2d(x, 4)
        x = F.sigmoid(x)
        return x.view(-1,1)

class Descriminator(nn.Module):
    """
    使用这个鉴别器可以收敛 学习率 2e-4, init_net随机初始化
    """
    def __init__(self, cond_dim):
        super(Descriminator, self).__init__()
        self.in_conv = nn.Sequential(
            nn.Conv2d(3, 16, 5, 2, 2),
            nn.LeakyReLU(),
            nn.Conv2d(16, 32, 5, 2, 2),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, 5, 2, 2),
            nn.LeakyReLU(),
        )
        self.conv = nn.Sequential(
            nn.Conv2d(64, 128, 5, 2, 2),
            nn.LeakyReLU(),
            nn.Conv2d(128, 1, 5, 2, 2),
            nn.LeakyReLU(),
        )
    def set_cond(self, cond):
        pass
    def forward(self, x):
        x = self.in_conv(x)
        x = self.conv(x)
        x = F.avg_pool2d(x, 4)
        x = F.sigmoid(x)
        return x.view(-1,1)

class CondD(nn.Module):
    def __init__(self, cond_dim):
        super(CondD, self).__init__()
        self.in_cond = nn.Sequential(
            nn.Linear(cond_dim, 512),
            nn.Linear(512, 1024),
            nn.Linear(1024, 2048),
            nn.Linear(2048, 4096),
        )
        self.conv0 = nn.Conv2d(16, 512, 3, 1, 1)
        self.in_conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1),
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.Conv2d(256, 512, 3, 2, 1),
        )
        self.main = nn.Sequential(
            nn.Linear(512, 128),
            nn.Linear(128, 32),
            nn.Linear(32, 1),
        )
    def set_cond(self, cond):
        self.cond = self.in_cond(cond.squeeze())
        self.cond = self.conv0(self.cond.view(-1,16,16,16))
    def forward(self, x):
        x = self.in_conv(x)
        x = x + self.cond
        x = F.avg_pool2d(x, 16)
        x = self.main(x.squeeze())
        return F.sigmoid(x)

class NPDiscriminator(nn.Module):
  def __init__(self, action_size, dim=16):
    super(NPDiscriminator, self).__init__()
    self.dim = dim

    self.fc1 = nn.Linear(action_size, 8)
    self.conv0 = nn.Conv2d(3, 8, 4, 2, 1)
    self.conv1 = nn.Conv2d(8, dim, 4, stride=2, padding=1)
    self.bn1 = nn.BatchNorm2d(dim)
    self.conv2 = nn.Conv2d(dim, dim*2, 4, stride=2, padding=1)
    self.bn2 = nn.BatchNorm2d(dim*2)
    self.conv3 = nn.Conv2d(dim*2, dim*4, 4, stride=2, padding=1)
    self.bn3 = nn.BatchNorm2d(dim*4)
    self.conv4 = nn.Conv2d(dim*4, dim*8, 4, stride=2, padding=1)
    self.bn4 = nn.BatchNorm2d(dim*8)
    self.fc2 = nn.Linear(4*4*(dim*8), 1)
    self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)

  def forward(self, images, actions):
    actions = F.relu(self.fc1(actions))
    actions = actions.view(-1, 8, 1, 1)
    x = self.leaky_relu(self.conv0(images))

    x = x + actions
    x = self.leaky_relu(self.bn1(self.conv1(x)))
    x = self.leaky_relu(self.bn2(self.conv2(x)))
    x = self.leaky_relu(self.bn3(self.conv3(x)))
    x = self.leaky_relu(self.bn4(self.conv4(x)))
    x = x.flatten(start_dim=1)
    x = self.fc2(x)
    return x
