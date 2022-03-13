import os
import cv2
from cv2 import COLOR_GRAY2BGR


path = './dataset/orig_imgs/'
target = './dataset/resize/'

names = os.listdir(path)

for i in names:
	img = cv2.imread(os.path.join(path, i))
	if len(img.shape) < 3:
		img = cv2.cvtColor(img, COLOR_GRAY2BGR)
	img = cv2.resize(img, (128,128))
	cv2.imwrite(os.path.join(target, i[:-4]+'png'), img)