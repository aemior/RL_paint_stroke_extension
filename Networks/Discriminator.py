import imp
from turtle import forward
from torch.nn import init
import torch
import torch.nn as nn
import torch.nn.functional as F

from Networks.translator import init_net

def define_D():
	net = BinDescriminator()
	return init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[])

class TReLU(nn.Module):
    def __init__(self):
            super(TReLU, self).__init__()
            self.alpha = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
            self.alpha.data.fill_(0)

    def forward(self, x):
        x = F.relu(x - self.alpha) + self.alpha
        return x

class BinDescriminator(nn.Module):
	def __init__(self):
		super(BinDescriminator, self).__init__()
		self.conv = nn.Sequential(
			nn.Conv2d(3, 16, 3, 1, 1),
			TReLU(),
			nn.Conv2d(16, 32, 5, 2, 2),
			nn.BatchNorm2d(32),
			TReLU(),
			nn.Conv2d(32, 64, 5, 2, 2),
			nn.BatchNorm2d(64),
			TReLU(),
			nn.Conv2d(64, 128, 5, 2, 2),
			nn.BatchNorm2d(128),
			TReLU(),
			nn.Conv2d(128, 256, 5, 2, 2),
			nn.BatchNorm2d(256),
			TReLU(),
			nn.Conv2d(256, 512, 5, 2, 2),
			nn.BatchNorm2d(512),
			TReLU()
		)
		self.fc = nn.Sequential(
			nn.Linear(512, 1),
			#nn.Softmax(dim=1)
			nn.Sigmoid()
		)
	
	def forward(self, x):
		x = self.conv(x)
		x = F.avg_pool2d(x, 4)
		x = self.fc(x.view(-1,512))
		return x

