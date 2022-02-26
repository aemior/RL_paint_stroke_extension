import time

import numpy as np
import torch
import torchvision.transforms.functional as TF
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
from torchvision import utils
import lpips

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
									shuffle=True, num_workers=6, worker_init_fn= worker_init_fn_seed)
	return dataloader

def get_neural_render(stroke_type, net_R):
	return define_R(rddim=ACTION_SIZE[stroke_type],\
		 shape_dim=ACTION_SHAPESIZE[stroke_type], netR=net_R)

def get_discriminator(stroke_type):
	return define_D(ACTION_SIZE[stroke_type])

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
