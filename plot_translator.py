import argparse
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from Utils.utils import RealDecoder, get_translator,get_stroke_dataset,make_numpy_grid,RealDecoder_T
parser = argparse.ArgumentParser(description='展示单步笔触的翻译结果')
parser.add_argument('--StrokeType', type=str, default=r'MyPaintWaterColor', metavar='str',
                    help='set stroke type:MyPaintWaterColor, MyPaintPencil, MyPaintCharcoal\
						OilPaint, SimOilPaint')
parser.add_argument('--TargetStroke', type=str, default=r'MyPaintWaterColor', metavar='str',
                    help='set stroke type:MyPaintWaterColor, MyPaintPencil, MyPaintCharcoal\
						OilPaint, SimOilPaint')
parser.add_argument('--t_path', type=str, default=r'./checkpoints_R', metavar='str',
                    help='dir to Agent checkpoints (default: ...)')
parser.add_argument('--net_R', type=str, default='dual-render', metavar='str',
                    help='net_R: plain-dcgan or plain-unet or huang-net,'
                         'zou-fusion-net, or zou-fusion-net-light')
parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 4)')
parser.add_argument('--nrows', type=int, default=8, metavar='N',
                    help='n_rows')
parser.add_argument('--save_path', type=str, default=r'./', metavar='str',
                    help='dir to save Res')

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if __name__ == '__main__':
	translator = get_translator(args)
	translator.load_state_dict(torch.load(args.t_path)['model_T_state_dict'])
	translator = translator.to(device)
	dataloader = get_stroke_dataset(args) 
	RD = RealDecoder_T(args)

	if not os.path.exists('./'+args.StrokeType+'_batch.tensor'):
		batch = next(iter(dataloader))
		torch.save(batch, './'+args.StrokeType+'_batch.tensor')
	else:
		batch = torch.load('./'+args.StrokeType+'_batch.tensor')
	
	gt_foreground, gt_alpha = batch['B'].to(device), batch['ALPHA'].to(device)
	GT_C_B = gt_alpha * gt_foreground
	GT_C_W = (1-gt_alpha) + gt_alpha * gt_foreground 

	a_in = batch['A'].to(device)

	pd_foreground = []
	pd_alpha = []
	C_B = np.zeros((128,128,3), np.uint8)
	C_W = np.ones((128,128,3), np.uint8)*255
	for i in range(args.batch_size):
		a_t = translator(a_in[i].view((1,-1)))
		rb = RD.single_stroke(C_B, a_t)
		rb = np.transpose(rb.reshape(1, 128, 128, 3), (0,3,1,2))
		pd_foreground.append(torch.tensor(rb.astype(np.float32)/255))
		rw = RD.single_stroke(C_W, a_t)
		rw = np.transpose(rw.reshape(1, 128, 128, 3), (0,3,1,2))
		pd_alpha.append(torch.tensor(rw.astype(np.float32)/255))

	pd_foreground = torch.cat(pd_foreground)
	pd_alpha = torch.cat(pd_alpha)

	vis_GT_C_B = make_numpy_grid(GT_C_B, args.nrows)
	vis_PD_C_B = make_numpy_grid(pd_foreground, args.nrows)
	vis_GT_C_W = make_numpy_grid(GT_C_W, args.nrows)
	vis_PD_C_W = make_numpy_grid(pd_alpha, args.nrows)

	plt.imsave(os.path.join(args.save_path,'GT_C_B.png'), vis_GT_C_B)
	plt.imsave(os.path.join(args.save_path,'GT_C_W.png'), vis_GT_C_W)
	plt.imsave(os.path.join(args.save_path,'PD_C_B.png'), vis_PD_C_B)
	plt.imsave(os.path.join(args.save_path,'PD_C_W.png'), vis_PD_C_W)