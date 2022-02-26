import torch
import torch.nn as nn
import torch.nn.functional as F

from Networks.translator import init_net

def define_D(cond_dim):
	#net = torch.nn.utils.spectral_norm(CondDescriminator(cond_dim))
	net = CondDescriminator(cond_dim)
	return init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[])

class CondDescriminator(nn.Module):
	def __init__(self, cond_dim):
		super(CondDescriminator, self).__init__()
		self.in_conv = nn.utils.spectral_norm(nn.Conv2d(3, 64, 3, 1, 1))
		self.in_fc = nn.utils.spectral_norm(nn.Linear(cond_dim, 64)) 
		self.conv = nn.Sequential(
			nn.utils.spectral_norm(nn.Conv2d(64, 128, 5, 2, 2)),
			#nn.BatchNorm2d(128),
			nn.ReLU(),
			nn.utils.spectral_norm(nn.Conv2d(128, 256, 5, 2, 2)),
			nn.ReLU(),
			nn.utils.spectral_norm(nn.Conv2d(256, 512, 5, 2, 2)),
			nn.ReLU(),
		)
		self.fc = nn.Sequential(
			nn.utils.spectral_norm(nn.Linear(512, 1)),
			nn.LeakyReLU(),
		)

	def set_cond(self, cond):
		self.cond = self.in_fc(cond)

	def forward(self, x):
		x = F.relu(self.in_conv(x))
		x = self.conv(x+self.cond.view(self.cond.shape+(1,1)))
		x = F.avg_pool2d(x, 16)
		x = self.fc(x.view(-1,512))
		return x

