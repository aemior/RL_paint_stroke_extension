import argparse
import os

import torch
import matplotlib.pyplot as plt

from Utils.utils import get_neural_render,get_stroke_dataset,make_numpy_grid

parser = argparse.ArgumentParser(description='展示渲染器的渲染结果')
parser.add_argument('--StrokeType', type=str, default=r'MyPaintWaterColor', metavar='str',
                    help='set stroke type:MyPaintWaterColor, MyPaintPencil, MyPaintCharcoal\
						OilPaint, SimOilPaint')
parser.add_argument('--net_R', type=str, default='dual-render', metavar='str',
                    help='net_R: plain-dcgan or plain-unet or huang-net,'
                         'zou-fusion-net, or zou-fusion-net-light')
parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 4)')
parser.add_argument('--checkpoint', type=str, default=r'./checkpoints_R/last_ckpt.pt', metavar='str',
                    help='dir to load checkpoints (default: ...)')
parser.add_argument('--nrows', type=int, default=8, metavar='N',
                    help='n_rows')
parser.add_argument('--save_path', type=str, default=r'./', metavar='str',
                    help='dir to save Res')
args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if not os.path.exists(args.save_path):
	os.mkdir(args.save_path)
if __name__ == '__main__':
	NeuralRender = get_neural_render(args.StrokeType, args.net_R) 
	ckpt = torch.load(args.checkpoint)
	NeuralRender.load_state_dict(ckpt['model_R_state_dict'])
	NeuralRender.eval().to(device)
	dataloader = get_stroke_dataset(args) 

	batch = next(iter(dataloader))
	
	gt_foreground, gt_alpha = batch['B'].to(device), batch['ALPHA'].to(device)
	GT_C_B = gt_alpha * gt_foreground
	GT_C_W = (1-gt_alpha) + gt_alpha * gt_foreground 

	a_in = batch['A'].to(device)

	pd_foreground = []
	pd_alpha = []
	for i in range(args.batch_size):
		f, a = NeuralRender(a_in[i].view((1,-1)))
		pd_foreground.append(f)
		pd_alpha.append(a)

	pd_foreground = torch.cat(pd_foreground)
	pd_alpha = torch.cat(pd_alpha)
	PD_C_B = pd_alpha * pd_foreground
	PD_C_W = (1-pd_alpha) + pd_alpha * pd_foreground 

	vis_GT_C_B = make_numpy_grid(GT_C_B, args.nrows)
	vis_PD_C_B = make_numpy_grid(PD_C_B, args.nrows)
	vis_GT_C_W = make_numpy_grid(GT_C_W, args.nrows)
	vis_PD_C_W = make_numpy_grid(PD_C_W, args.nrows)

	plt.imsave(os.path.join(args.save_path,'GT_C_B.png'), vis_GT_C_B)
	plt.imsave(os.path.join(args.save_path,'GT_C_W.png'), vis_GT_C_W)
	plt.imsave(os.path.join(args.save_path,'PD_C_B.png'), vis_PD_C_B)
	plt.imsave(os.path.join(args.save_path,'PD_C_W.png'), vis_PD_C_W)