"""
渲染器的训练曲线
"""


NAMES = {
	'水彩':'back_up/oil_mse/checkpoints_R/val_acc.npy',
	'水墨':'back_up/watercolor_mse/checkpoints_R/val_acc.npy',
	'油画':'back_up/simoil_mse/checkpoints_R/val_acc.npy',
	'铅笔':'back_up/pencil_mse/checkpoints_R/val_acc.npy',
	'碳棒':'back_up/Charcoal_mse/checkpoints_R/val_acc.npy'
}
LINESTY = {
	'水彩':'o',
	'水墨':'^',
	'油画':'+',
	'铅笔':'*',
	'碳棒':'x'
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
