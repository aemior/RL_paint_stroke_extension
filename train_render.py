import argparse
from Utils import renderTrain

parser = argparse.ArgumentParser(description='Render Training Program')
parser.add_argument('--StrokeType', type=str, default=r'MyPaintWaterInk', metavar='str',
                    help='set stroke type:MyPaintWaterInk, MyPaintPencil, MyPaintCharcoal\
						WaterColor, SimOilPaint, MarkPen, Rectangle')
parser.add_argument('--canvas_width', type=int, default=128, metavar='N',
                    help='canvas_width')
parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 4)')
parser.add_argument('--net_R', type=str, default='dual-render', metavar='str',
                    help='net_R: plain-dcgan or plain-unet or huang-net,'
                         'zou-fusion-net, or zou-fusion-net-light')
parser.add_argument('--checkpoint_dir', type=str, default=r'./checkpoints_R', metavar='str',
                    help='dir to save checkpoints (default: ...)')
parser.add_argument('--render_ckpt', type=str, default=r'./checkpoints_RD/50_epoch_bakup', metavar='str',
                    help='dir to save checkpoints (default: ...)')
parser.add_argument('--vis_dir', type=str, default=r'./val_out_R', metavar='str',
                    help='dir to save results during training (default: ./val_out_R)')
parser.add_argument('--lr', type=float, default=3e-4,
                    help='learning rate (default: 0.0002)')
parser.add_argument('--lr_gan', type=float, default=3e-4,
                    help='learning rate (default: 0.0002)')
parser.add_argument('--max_num_epochs', type=int, default=500, metavar='N',
                    help='max number of training epochs (default 400)')
parser.add_argument('--vis_port', type=int, default=0, metavar='N',
                    help='Visdom port')
parser.add_argument('--GAN', type=bool, default=False, metavar='N',
                    help='GAN train?')
parser.add_argument('--debug', type=bool, default=False, metavar='N',
                    help='debug?')
parser.add_argument('--only_black', type=bool, default=False, metavar='N',
                    help='only backward black canvas')
parser.add_argument('--only_white', type=bool, default=False, metavar='N',
                    help='only backward white canvas')
parser.add_argument('--rand_c', type=bool, default=False, metavar='N',
                    help='noise canvas')
parser.add_argument('--patch_gan_loss', type=bool, default=False, metavar='N',
                    help='noise canvas')
args = parser.parse_args()

if __name__ == '__main__':
	if args.GAN:
		T = renderTrain.DisTrain(args)
	else:
		if args.net_R == 'Style-render':
			T = renderTrain.StyTrain(args)
		elif args.net_R == 'Style-Light':
			T = renderTrain.SLNTrain(args)
		else:
			T = renderTrain.Train(args)
	T.train()