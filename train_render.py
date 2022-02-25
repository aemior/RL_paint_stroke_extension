import argparse
from Utils import renderTrain

parser = argparse.ArgumentParser(description='Render Training Program')
parser.add_argument('--StrokeType', type=str, default=r'MyPaintWaterColor', metavar='str',
                    help='set stroke type:MyPaintWaterColor, MyPaintPencil, MyPaintCharcoal\
						OilPaint, SimOilPaint')
parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 4)')
parser.add_argument('--net_R', type=str, default='dual-render', metavar='str',
                    help='net_R: plain-dcgan or plain-unet or huang-net,'
                         'zou-fusion-net, or zou-fusion-net-light')
parser.add_argument('--checkpoint_dir', type=str, default=r'./checkpoints_R', metavar='str',
                    help='dir to save checkpoints (default: ...)')
parser.add_argument('--vis_dir', type=str, default=r'./val_out_R', metavar='str',
                    help='dir to save results during training (default: ./val_out_R)')
parser.add_argument('--lr', type=float, default=2e-4,
                    help='learning rate (default: 0.0002)')
parser.add_argument('--max_num_epochs', type=int, default=1000, metavar='N',
                    help='max number of training epochs (default 400)')
parser.add_argument('--vis_port', type=int, default=0, metavar='N',
                    help='Visdom port')
args = parser.parse_args()

if __name__ == '__main__':
	T = renderTrain.Train(args)
	T.train()