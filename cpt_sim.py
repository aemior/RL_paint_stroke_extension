import os
import argparse

import cv2
from cv2 import COLOR_BGR2RGB
from cv2 import COLOR_GRAY2RGB
import torch

from Utils.utils import cpt_batch_psnr, LPIPS_loss

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
lpips_cal =  LPIPS_loss(device)

parser = argparse.ArgumentParser(description='计算两个文件夹中图片的PSNR、MSE、LPIPS，文件夹内图片的名称必须一一对应')
parser.add_argument('--A', type=str, default=r'./dataset/pngs/', metavar='str',
                    help='first_dir')
parser.add_argument('--B', type=str, default=r'./checkpoints_T', metavar='str',
                    help='second_dir')
args = parser.parse_args()

NAMES = os.listdir(args.A)

imgs_a = []
imgs_b = []

def read_img(path):
	try:
		img = cv2.resize(cv2.imread(path), (128,128))
	except Exception as e:
		import pdb
		pdb.set_trace()
	if len(img.shape) < 3:
		return cv2.cvtColor(img, COLOR_GRAY2RGB)
	else:
		return cv2.cvtColor(img, COLOR_BGR2RGB)
	

for i in NAMES:
	#a = cv2.cvtColor(cv2.resize(cv2.imread(os.path.join(args.A, i)), (128,128)), COLOR_BGR2RGB)
	#b = cv2.cvtColor(cv2.resize(cv2.imread(os.path.join(args.B, i)), (128,128)), COLOR_BGR2RGB)
	a = read_img(os.path.join(args.A, i))
	b = read_img(os.path.join(args.B, i))
	imgs_a.append((torch.tensor(a).float()/255.0).permute(2,0,1))
	imgs_b.append((torch.tensor(b).float()/255.0).permute(2,0,1))

batch_a = torch.stack(imgs_a).to(device)
batch_b = torch.stack(imgs_b).to(device)


with torch.no_grad():
	PSNR = cpt_batch_psnr(batch_b, batch_a, 1.0)
	MSE = ((batch_a - batch_b)**2).mean()
	LPIPS = lpips_cal(batch_b, batch_a) 

print('MSE', 'PSNR', 'LPIPS')
print(MSE.item(), PSNR.item(), LPIPS.item())


