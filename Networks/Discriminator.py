import torch
import torch.nn as nn
import torch.nn.functional as F

from Networks.translator import init_net

def define_D(cond_dim):
	#net = torch.nn.utils.spectral_norm(CondDescriminator(cond_dim))
	#net = CondDescriminator(cond_dim)
	net = CondDescriminatorV5(cond_dim)
	#net = Descriminator(cond_dim)
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
	def forward(self, x):
		x = self.in_conv(x)
		x = self.conv(x)
		x = F.avg_pool2d(x, 4)
		x = F.sigmoid(x)
		return x.view(-1,1)