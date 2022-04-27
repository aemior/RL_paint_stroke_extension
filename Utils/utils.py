import time
import os
from turtle import forward

import numpy as np
import torch
import torchvision.transforms.functional as TF
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
from torchvision import utils
from torch import autograd
import lpips
from zmq import device

from Renders import RealRenders
from Networks.render import define_R
from Networks.Discriminator import define_D
from Networks.translator import define_T

M_RENDERING_SAMPLES_PER_EPOCH = 50000
ACTION_SIZE = {'MyPaintWaterInk':12, 'MyPaintPencil':12, 'MyPaintCharcoal':12,\
	 'WaterColor':13, 'OilPaint':12, 'MarkPen':12, 'Rectangle':9}
ACTION_SHAPESIZE = {'MyPaintWaterInk':9, 'MyPaintPencil':9, 'MyPaintCharcoal':9,\
	 'WaterColor':10, 'OilPaint':6, 'MarkPen':9, 'Rectangle':6}

class StrokeDataset(Dataset):
	def __init__(self, args, is_train=True, batch_size=16):
		if args.StrokeType == 'MyPaintWaterInk':
			self.RENDER = RealRenders.MyPaintStroke(128, 'WaterInk')
		elif args.StrokeType == 'MyPaintPencil':
			self.RENDER = RealRenders.MyPaintStroke(128, 'Pencil')
		elif args.StrokeType == 'MyPaintCharcoal':
			self.RENDER = RealRenders.MyPaintStroke(128, 'Charcoal')
		elif args.StrokeType == 'WaterColor':
			self.RENDER = RealRenders.WaterColorStroke()
		elif args.StrokeType == 'OilPaint':
			self.RENDER = RealRenders.SNPStroke(cw=128, BrushType='oilpaintbrush')
		elif args.StrokeType == 'MarkPen':
			self.RENDER = RealRenders.SNPStroke(cw=128, BrushType='markerpen')
		elif args.StrokeType == 'Rectangle':
			self.RENDER = RealRenders.SNPStroke(cw=128, BrushType='rectangle')
		self.is_train = is_train

	def __len__(self):
		if self.is_train:
			return M_RENDERING_SAMPLES_PER_EPOCH
		else:
			return int(M_RENDERING_SAMPLES_PER_EPOCH / 20)

	def __getitem__(self, idx):
		action = np.random.uniform(size=[self.RENDER.action_size])
		image = self.RENDER.SingleStroke(action)

		params = torch.tensor(action.astype(np.float32))
		foreground = TF.to_tensor(image[:,:,:3].astype(np.float32)/255.)
		stroke_alpha_map = TF.to_tensor(image[:,:,3].astype(np.float32)/255.)
		data = {'A': params, 'B': foreground, 'ALPHA': stroke_alpha_map}
		return data

# Some get function

def get_stroke_dataset(args):
	training_set = StrokeDataset(args, is_train=True)

	dataloader = DataLoader(training_set, batch_size=args.batch_size,\
									shuffle=False, num_workers=6)#worker_init_fn= worker_init_fn_seed)
	return dataloader

def get_neural_render(stroke_type, net_R):
	return define_R(rddim=ACTION_SIZE[stroke_type],\
		 shape_dim=ACTION_SHAPESIZE[stroke_type], netR=net_R)

def get_discriminator(args):
	from Networks.Discriminator import PatchDiscriminator
	netD = PatchDiscriminator()
	netD.load_state_dict(torch.load(os.path.join(args.checkpoint_dir, 'wgan.pkl')))
	return netD
	#return define_D(ACTION_SIZE[stroke_type])

def get_translator(args):
	return define_T(ACTION_SIZE[args.StrokeType], ACTION_SIZE[args.TargetStroke], "FCN4L")

#=========================================
class RealDecoder(object):
	def __init__(self, args, cw=128):
		self.ST = args.StrokeType
		self.cw = cw
		self.set_render()
	def set_render(self):
		if self.ST == 'MyPaintWaterInk':
			self.RENDER = RealRenders.MyPaintStroke(128, 'WaterInk')
		elif self.ST == 'MyPaintPencil':
			self.RENDER = RealRenders.MyPaintStroke(128, 'Pencil')
		elif self.ST == 'MyPaintCharcoal':
			self.RENDER = RealRenders.MyPaintStroke(128, 'Charcoal')
		elif self.ST == 'WaterColor':
			self.RENDER = RealRenders.WaterColorStroke(cw=self.cw)
		elif self.ST == 'OilPaint':
			self.RENDER = RealRenders.SNPStroke(cw=self.cw, BrushType='oilpaintbrush')
		elif self.ST == 'MarkPen':
			self.RENDER = RealRenders.SNPStroke(cw=self.cw, BrushType='markerpen')
		elif self.ST == 'Rectangle':
			self.RENDER = RealRenders.SNPStroke(cw=self.cw, BrushType='rectangle')

	def __call__(self, canvas, actions):
		actions = actions.view(5, ACTION_SIZE[self.ST]).detach().cpu().numpy()
		canvas = canvas.astype(np.float32) / 255
		for i in range(5):
			stroke = self.RENDER.SingleStroke(actions[i]).astype(np.float32) / 255
			alpha = stroke[:,:,3].reshape(self.cw,self.cw,1)
			canvas = canvas * (1-alpha) + stroke[:,:,:3] * alpha
		return (np.clip(canvas, 0, 1) * 255).astype(np.uint8)
	def single_stroke(self, canvas, actions):
		actions = actions.view(1, ACTION_SIZE[self.ST]).detach().cpu().numpy()
		canvas = canvas.astype(np.float32) / 255
		stroke = self.RENDER.SingleStroke(actions[0]).astype(np.float32) / 255
		alpha = stroke[:,:,3].reshape(self.cw,self.cw,1)
		canvas = canvas * (1-alpha) + stroke[:,:,:3] * alpha
		return (np.clip(canvas, 0, 1) * 255).astype(np.uint8)
	def batch_stroke(self, actions):
		actions = actions.detach().cpu().numpy()
		foreground = np.zeros((actions.shape[0], 3, self.cw, self.cw)).astype(np.float32)
		alpha = np.zeros((actions.shape[0], 1, self.cw, self.cw)).astype(np.float32)
		for i in range(actions.shape[0]):
			stroke = self.RENDER.SingleStroke(actions[i]).astype(np.float32) / 255
			foreground[i] = np.transpose(stroke[:,:,:3].reshape((self.cw, self.cw, 3)), (2,0,1))
			alpha[i] = stroke[:,:,3]
		return torch.tensor(foreground), torch.tensor(alpha)


class RealDecoder_T(RealDecoder):
	def __init__(self, args, cw=128):
		self.ST = args.TargetStroke
		self.cw = cw
		self.set_render()

#=========================================

class LPIPS_loss(nn.Module):
	def __init__(self, device):
		super(LPIPS_loss, self).__init__()
		self.lpips_model = lpips.LPIPS(net='alex').to(device)
		self.lpips_model.eval()
	def forward(self, x, y):
		x = x * 2 - 1
		y = y * 2 - 1
		return self.lpips_model(x, y).mean()


class PatchGan_loss(nn.Module):
	def __init__(self, args, device):
		super().__init__()
		from Networks.Discriminator import PatchDiscriminator
		self.netD = PatchDiscriminator()
		self.netD.load_state_dict(torch.load(os.path.join(args.checkpoint_dir, 'wgan.pkl')))
		self.netD = self.netD.to(device)
	def forward(self, x, y):
		data = torch.cat((x,y), dim=1)
		return (-self.netD(data)).mean()

def cpt_batch_psnr(img, img_gt, PIXEL_MAX):
    mse = torch.mean((img - img_gt) ** 2)
    psnr = 20 * torch.log10(PIXEL_MAX / torch.sqrt(mse))
    return psnr

def worker_init_fn_seed(worker_id):
    seed = int(time.time())
    seed += worker_id
    np.random.seed(seed)

def make_numpy_grid(tensor_data, nr=8):
    tensor_data = tensor_data.detach()
    vis = utils.make_grid(tensor_data, nrow=nr)
    vis = np.array(vis.cpu()).transpose((1,2,0))
    if vis.shape[2] == 1:
        vis = np.stack([vis, vis, vis], axis=-1)
    return vis.clip(min=0, max=1)

def to_numpy(var):
    return var.cpu().data.numpy() if torch.cuda.is_available() else var.data.numpy()

def to_tensor(ndarray, device):
    return torch.tensor(ndarray, dtype=torch.float, device=device)

def calc_gradient_penalty(discriminator: nn.Module, real_data: torch.Tensor,
                          fake_data: torch.Tensor, actions: torch.Tensor,
                          device: torch.device, scale: float):
    batch_size = real_data.shape[0]
    epsilon = torch.rand(1, 1)
    epsilon = epsilon.expand(batch_size, real_data.nelement()//batch_size).contiguous().view(batch_size, 3, 128, 128)
    epsilon = epsilon.to(device)

    interpolates = epsilon * real_data + ((1.0 - epsilon) * fake_data)
    interpolates.requires_grad = True

    disc_interpolates = discriminator(interpolates, actions)
    gradients = autograd.grad(disc_interpolates, interpolates,
                            grad_outputs=torch.ones_like(disc_interpolates),
                            create_graph=True)[0]
    gradients = gradients.view(batch_size, -1)

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * scale

    return gradient_penalty


from torch import autograd
from torch.autograd import Variable
import torch.nn.functional as F
from torch.autograd import grad as torch_grad
import torch.nn.utils.weight_norm as weightNorm

dim = 128
LAMBDA = 10 # Gradient penalty lambda hyperparameter

def cal_gradient_penalty(netD, real_data, fake_data, batch_size, device):
    alpha = torch.rand(batch_size, 1)
    alpha = alpha.expand(batch_size, int(real_data.nelement()/batch_size)).contiguous()
    alpha = alpha.view(batch_size, 6, dim, dim)
    alpha = alpha.to(device)
    fake_data = fake_data.view(batch_size, 6, dim, dim)
    interpolates = Variable(alpha * real_data.data + ((1 - alpha) * fake_data.data), requires_grad=True)
    disc_interpolates = netD(interpolates)
    gradients = autograd.grad(disc_interpolates, interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                              create_graph=True, retain_graph=True)[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty