import argparse
import os

import cv2
import numpy as np
import torch

from Networks.actor import ResNet
from Networks.render import define_R
from Utils.utils import RealDecoder_T, get_translator,make_numpy_grid,ACTION_SHAPESIZE,ACTION_SIZE

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='使用翻译器的绘画过程')
parser.add_argument('--StrokeType', type=str, default=r'MyPaintWaterColor', metavar='str',
                    help='set stroke type:MyPaintWaterColor, MyPaintPencil, MyPaintCharcoal\
						OilPaint, SimOilPaint')
parser.add_argument('--TargetStroke', type=str, default=r'MyPaintWaterColor', metavar='str',
                    help='set target stroke type:MyPaintWaterColor, MyPaintPencil, MyPaintCharcoal\
						OilPaint, SimOilPaint')
parser.add_argument('--r_path', type=str, default=r'./checkpoints_R', metavar='str',
                    help='dir to Render checkpoints (default: ...)')
parser.add_argument('--agent_path', type=str, default=r'./checkpoints_R', metavar='str',
                    help='dir to Agent checkpoints (default: ...)')
parser.add_argument('--t_path', type=str, default=r'./checkpoints_R', metavar='str',
                    help='dir to Translator checkpoints (default: ...)')
parser.add_argument('--img', type=str, default=r'./checkpoints_R', metavar='str',
                    help='dir to TargetImage')
parser.add_argument('--save_path', type=str, default=r'./', metavar='str',
                    help='dir to save res')
args = parser.parse_args()

STROKE_TYPE = args.StrokeType
Decoder = define_R(ACTION_SIZE[STROKE_TYPE],\
	ACTION_SHAPESIZE[STROKE_TYPE], 'dual-render')
Decoder.load_state_dict(torch.load(args.r_path)['model_R_state_dict'])
Decoder.eval().to(device)
def decode(x, canvas):
	x = x.view(-1, ACTION_SIZE[STROKE_TYPE])
	foreground, alpha = Decoder(x)
	foreground = foreground.view(-1, 5, 3, 128, 128)
	alpha = alpha.view(-1, 5, 1, 128, 128)
	for i in range(5):
		canvas = torch.clamp(canvas * (1 - alpha[:,i]) + \
			alpha[:,i] * foreground[:,i], 0.0, 1.0)
	return canvas
RD = RealDecoder_T(args)

translator = get_translator(args)
translator.load_state_dict(torch.load(args.t_path)['model_T_state_dict'])
translator = translator.to(device)

def trans_action(x):
	x = x.view(5,-1)
	x = translator(x)
	return x.view(1,-1)

agent = ResNet(9, 18, ACTION_SIZE[STROKE_TYPE]*5)
agent.load_state_dict(torch.load(args.agent_path))
agent = agent.eval().to(device)
coord = torch.zeros([1, 2, 128, 128])
for i in range(128):
    for j in range(128):
        coord[0, 0, i, j] = i / (128 - 1.)
        coord[0, 1, i, j] = j / (128 - 1.)
coord = coord.to(device) # Coordconv
T = torch.ones([1, 1, 128, 128], dtype=torch.float32).to(device)

C_n = torch.zeros([1, 3, 128, 128]).to(device)
C_r = np.zeros((128,128,3), np.uint8)
img = cv2.imread(args.img)
img = cv2.resize(img, (128, 128))
img = img.reshape(1, 128, 128, 3)
img = np.transpose(img, (0, 3, 1, 2))
I = torch.tensor(img).to(device).float() / 255.

N_process = []
R_process = []

if not os.path.exists(args.save_path):
	os.mkdir(args.save_path)
import time
if __name__ == '__main__':
	t1 = time.time()
	for t in range(40):
		t_N = T * t / 40
		actions = agent(torch.cat([C_n, I, t_N, coord], 1))
		C_n = decode(actions, C_n)
		actions = trans_action(actions)
		C_r = RD(C_r, actions)
		if (t) % 5 == 0 or t in range(1,10):
			N_process.append(C_n.clone().detach().cpu())
			R_s = np.transpose(C_r.reshape(1, 128, 128, 3), (0,3,1,2))
			R_process.append(torch.tensor(R_s.astype(np.float32)/255))
	print("消耗时间：", time.time()-t1, 's')
	vis_source = make_numpy_grid(torch.cat(N_process))
	vis_target = make_numpy_grid(torch.cat(R_process))
	vis_final = make_numpy_grid(R_process[-1], nr=1)
	IMG_NAME = args.save_path+args.img.split('/')[-1].split('.')[0]
	cv2.imwrite(IMG_NAME+'_source.png', vis_source*255)
	cv2.imwrite(IMG_NAME+'_translate.png', (vis_target*255))
	cv2.imwrite(IMG_NAME+'_final.png', (vis_final*255))

	


