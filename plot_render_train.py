"""
渲染器的训练曲线
"""


NAMES = {
	'水彩':'./back_up/renders/watercolor_mse/checkpoints_R/val_acc.npy',
	'水墨':'./back_up/renders/waterink_mse/checkpoints_R/val_acc.npy',
	'油画':'./back_up/renders/oilpaint_mse/checkpoints_R/val_acc.npy',
	'铅笔':'./back_up/renders/pencil_mse/checkpoints_R/val_acc.npy',
	'粉笔':'./back_up/renders/charcoal_mse/checkpoints_R/val_acc.npy'
}
LINESTY = {
	'水彩':'o',
	'水墨':'^',
	'油画':'+',
	'铅笔':'*',
	'粉笔':'x'
}
"""
NAMES_3 = {
	'黑白画板+双路网络':'back_up/renders/oilpaint_mse/checkpoints_R/val_acc.npy',
	'黑画板+双路网络':'back_up/renders/oilpaint_only_black/val_acc.npy',
	'白画板+双路网络':'back_up/renders/oilpaint_only_white/val_acc.npy',
	'黑白画板+栅格网络':'back_up/renders/oilpaint_only_ret/val_acc.npy',
	'黑白画板+着色网络':'back_up/renders/oilpaint_only_shad/val_acc.npy'
}
LINESTY = {
	'黑白画板+双路网络':'o',
	'黑画板+双路网络':'^',
	'白画板+双路网络':'+',
	'黑白画板+栅格网络':'*',
	'黑白画板+着色网络':'x'
}

NAMES_2 = {
	'黑白画板+双路网络':'back_up/renders/watercolor_mse/checkpoints_R/val_acc.npy',
	'黑画板+双路网络':'back_up/renders/watercolor_only_black/val_acc.npy',
	'白画板+双路网络':'back_up/renders/watercolor_only_white/val_acc.npy',
	'黑白画板+栅格网络':'back_up/renders/watercolor_only_ret/val_acc.npy',
	'黑白画板+着色网络':'back_up/renders/watercolor_only_shad/val_acc.npy'
}
NAMES_1 = {
	'黑白画板+双路网络':'back_up/renders/waterink_mse/checkpoints_R/val_acc.npy',
	'黑画板+双路网络':'back_up/renders/waterink_only_black/val_acc.npy',
	'白画板+双路网络':'back_up/renders/waterink_only_white/val_acc.npy',
	'黑白画板+栅格网络':'back_up/renders/waterink_only_ret/val_acc.npy',
	'黑白画板+着色网络':'back_up/renders/waterink_only_shad/val_acc.npy'
}
LINESTY = {
	'黑白画板+双路网络':'o',
	'黑画板+双路网络':'^',
	'白画板+双路网络':'+',
	'黑白画板+栅格网络':'*',
	'黑白画板+着色网络':'x'
}

"""
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt

mpl.rcParams[u'font.sans-serif'] = ['simhei']
mpl.rcParams['axes.unicode_minus'] = False

RAW_DATA = {}
for i in NAMES.keys():
	RAW_DATA[i] = np.load(NAMES[i])

#plt.figure(dpi=300, figsize=(4, 7))
plt.figure(dpi=300)
#plt.title('水墨笔触')
X_data = np.arange(1,151)
for i in NAMES.keys():
	plt.plot(X_data, RAW_DATA[i][:150], LINESTY[i], linestyle='-', markevery=8)


plt.xlim((0, 150))
#plt.ylim((26, 38))
#my_y_ticks = np.arange(16, 36, 2)
#plt.yticks(my_y_ticks)
plt.xlabel('Epochs')
plt.ylabel('PSNR/dB')

plt.legend(list(NAMES.keys()), loc=4)
plt.grid()

plt.show()
"""
RAW_DATA = {}
for i in NAMES_1.keys():
	RAW_DATA[i] = np.load(NAMES_1[i])

plt.figure(dpi=200)
plt.subplot(131)
plt.title('水墨笔触')
X_data = np.arange(1,151)
for i in NAMES_1.keys():
	plt.plot(X_data, RAW_DATA[i][:150], LINESTY[i], linestyle='-', markevery=8)


plt.xlim((0, 150))
plt.ylim((26, 38))
#my_y_ticks = np.arange(16, 36, 2)
#plt.yticks(my_y_ticks)
#plt.xlabel('Epochs')
plt.ylabel('PSNR/dB')

#plt.legend(list(NAMES.keys()), loc=4)
plt.grid()

plt.subplot(132)
RAW_DATA = {}
for i in NAMES_2.keys():
	RAW_DATA[i] = np.load(NAMES_2[i])
plt.title('水彩笔触')
X_data = np.arange(1,151)
for i in NAMES_2.keys():
	plt.plot(X_data, RAW_DATA[i][:150], LINESTY[i], linestyle='-', markevery=8)


plt.xlim((0, 150))
plt.ylim((16, 35))
my_y_ticks = np.arange(16, 36, 2)
plt.yticks(my_y_ticks)
plt.xlabel('Epochs')
#plt.ylabel('PSNR/dB')

#plt.legend(list(NAMES.keys()), loc=4)
plt.grid()

plt.subplot(133)
RAW_DATA = {}
for i in NAMES_3.keys():
	RAW_DATA[i] = np.load(NAMES_3[i])
plt.title('油画笔触')
X_data = np.arange(1,151)
for i in NAMES_3.keys():
	plt.plot(X_data, RAW_DATA[i][:150], LINESTY[i], linestyle='-', markevery=8)


plt.xlim((0, 150))
plt.ylim((16, 34))
#my_y_ticks = np.arange(16, 36, 2)
#plt.yticks(my_y_ticks)
#plt.xlabel('Epochs')
#plt.ylabel('PSNR/dB')

plt.legend(list(NAMES_3.keys()), loc=4)
plt.grid()
plt.show()

"""