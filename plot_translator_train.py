"""
画出翻译器的训练过程
"""

NAMES = {
	'水彩-水墨':'./back_up/translator/watercolor_waterink/checkpoints_T/val_acc.npy',
	'水彩-油画':'./back_up/translator/watercolor_oilpaint/checkpoints_T/val_acc.npy',
	'水彩-铅笔':'back_up/translator/watercolor_pencil/checkpoints_T/val_acc.npy',
	'水彩-粉笔':'back_up/translator/watercolor_charcoal/checkpoints_T_charcoal/val_acc.npy'
}
LINESTY = {
	'水彩-水墨':'^',
	'水彩-油画':'+',
	'水彩-铅笔':'*',
	'水彩-粉笔':'x'
}

import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt

mpl.rcParams[u'font.sans-serif'] = ['simhei']
mpl.rcParams['axes.unicode_minus'] = False

RAW_DATA = {}
for i in NAMES.keys():
	RAW_DATA[i] = np.load(NAMES[i])

plt.figure(dpi=300)
X_data = np.arange(1,151)
for i in NAMES.keys():
	plt.plot(X_data, RAW_DATA[i][:150], LINESTY[i], linestyle='-', markevery=8)


plt.xlim((0, 150))
plt.xlabel('Epochs')
plt.ylabel('PSNR/dB')

plt.legend(list(NAMES.keys()))
plt.grid()

plt.show()
