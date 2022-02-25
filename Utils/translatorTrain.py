import os
import resource
from turtle import forward, pd

import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import matplotlib.pyplot as plt
import visdom

from Utils.utils import get_neural_render,get_stroke_dataset,get_translator 
import Utils.utils as utils

# Decide which device we want to run on
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Train(object):
    def __init__(self, args):
        self.dataloader = get_stroke_dataset(args) 
        self.render = get_neural_render(args.TargetStroke, args.net_R).to(device)
        self.render.load_state_dict(torch.load(args.render_ckpt)['model_R_state_dict'])
        self.render.to(device)
        self.translator = get_translator(args).to(device)

        self.checkpoint_dir = args.checkpoint_dir
        self.vis_dir = args.vis_dir

        self.epoch_id = 0
        self.epoch_to_start = 0
        self.max_num_epochs = args.max_num_epochs
        self.is_training = False
        self.batch = None
        self.GT_C_B = None
        self.GT_C_W = None
        self.PD_C_B = None
        self.PD_C_W = None
        self.G_loss = None
        self.best_val_acc = 0.0
        self.best_epoch_id = 0
        
        self.vis_port = args.vis_port
        self.is_VIS = self.vis_port != 0
        self.vis = None

        self.lr = args.lr
        self.optimizer_T = optim.Adam(self.translator.parameters(), lr=self.lr, betas=(0.9,0.999))
        self.exp_lr_scheduler_T = lr_scheduler.StepLR(
            self.optimizer_T, step_size=100, gamma=0.1
        )
        self._pxl_loss = torch.nn.MSELoss()

        self.VAL_ACC = np.array([], np.float32)
        if os.path.exists(os.path.join(self.checkpoint_dir, 'val_acc.npy')):
            self.VAL_ACC = np.load(os.path.join(self.checkpoint_dir, 'val_acc.npy'))

        # check and create model dir
        if os.path.exists(self.checkpoint_dir) is False:
            os.mkdir(self.checkpoint_dir)
        if os.path.exists(self.vis_dir) is False:
            os.mkdir(self.vis_dir)


    def LoadCheckPoint(self):
        if os.path.exists(os.path.join(self.checkpoint_dir, 'last_ckpt.pt')):
            print('# loading last checkpoint R...')
            # load the entire checkpoint
            checkpoint = torch.load(os.path.join(self.checkpoint_dir, 'last_ckpt.pt'))

            # update render states
            self.translator.load_state_dict(checkpoint['model_T_state_dict'])
            self.optimizer_T.load_state_dict(checkpoint['optimizer_T_state_dict'])
            self.exp_lr_scheduler_T.load_state_dict(
                checkpoint['exp_lr_scheduler_T_state_dict'])
            self.translator.to(device)

            # update some other states
            self.epoch_to_start = checkpoint['epoch_id'] + 1
            self.best_val_acc = checkpoint['best_val_acc']
            self.best_epoch_id = checkpoint['best_epoch_id']

            print('Epoch_to_start = %d, Historical_best_acc = %.4f (at epoch %d)' %
                    (self.epoch_to_start, self.best_val_acc, self.best_epoch_id))

        else:
            print('# training from scratch...')

    def _save_checkpoint(self, ckpt_name):
        torch.save({
            'epoch_id': self.epoch_id,
            'best_val_acc': self.best_val_acc,
            'best_epoch_id': self.best_epoch_id,
            'model_T_state_dict': self.translator.state_dict(),
            'optimizer_T_state_dict': self.optimizer_T.state_dict(),
            'exp_lr_scheduler_T_state_dict': self.exp_lr_scheduler_T.state_dict()
        }, os.path.join(self.checkpoint_dir, ckpt_name))

    def _clear_cache(self):
        self.running_acc = []

    def _forward_pass(self, batch):
        self.batch = batch
        z_in = batch['A'].to(device)
        z_trans = self.translator(z_in)
        pred_foreground, pred_alpha = self.render(z_trans)
        self.PD_C_B = pred_alpha * pred_foreground
        self.PD_C_W = (1-pred_alpha) + pred_alpha * pred_foreground

    def _backward_R(self):
        gt_foreground, gt_alpha = self.batch['B'].to(device),self.batch['ALPHA'].to(device)
        self.GT_C_B = gt_alpha * gt_foreground
        self.GT_C_W = (1-gt_alpha) + gt_alpha * gt_foreground 

        pixel_loss1 = self._pxl_loss(self.PD_C_B, self.GT_C_B)
        pixel_loss2 = self._pxl_loss(self.PD_C_W, self.GT_C_W)
        self.T_loss = pixel_loss1 + pixel_loss2
        self.T_loss.backward()

    def _collect_running_batch_states(self):
        self.running_acc.append(self._compute_acc().item())

        m = len(self.dataloader)
        
        if np.mod(self.batch_id, 100) == 1:
            mem_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            # print("MEM USAGE:", mem_usage/(1024*1024))
            print('Is_training: %s. [%d,%d][%d,%d], T_loss: %.5f, running_acc(PSNR): %.5f'
                    % (self.is_training, self.epoch_id, self.max_num_epochs-1, self.batch_id, m,
                        self.T_loss.item(), np.mean(self.running_acc)))
            if self.is_VIS:
                self.vis.images(self.GT_C_B, win='GT_C_B')
                self.vis.images(self.PD_C_B, win='PD_C_B')
                self.vis.images(self.GT_C_W, win='GT_C_W')
                self.vis.images(self.PD_C_W, win='PD_C_W')

        if np.mod(self.batch_id, 1000) == 1:
            vis_pred_foreground = utils.make_numpy_grid(self.GT_C_B)
            vis_gt_foreground = utils.make_numpy_grid(self.PD_C_B)
            vis_pred_alpha = utils.make_numpy_grid(self.GT_C_W)
            vis_gt_alpha = utils.make_numpy_grid(self.PD_C_W)

            vis = np.concatenate([vis_pred_foreground, vis_gt_foreground,
                                    vis_pred_alpha, vis_gt_alpha], axis=0)
            vis = np.clip(vis, a_min=0.0, a_max=1.0)
            file_name = os.path.join(
                self.vis_dir, 'istrain_'+str(self.is_training)+'_'+
                                str(self.epoch_id)+'_'+str(self.batch_id)+'.jpg')
            plt.imsave(file_name, vis)

    def _compute_acc(self):
        target_C_B = self.GT_C_B.to(device).detach()
        target_C_W = self.GT_C_W.to(device).detach()
        C_B = self.PD_C_B.detach()
        C_W = self.PD_C_W.detach()

        psnr1 = utils.cpt_batch_psnr(C_B, target_C_B, PIXEL_MAX=1.0)
        psnr2 = utils.cpt_batch_psnr(C_W, target_C_W, PIXEL_MAX=1.0)
        return (psnr1 + psnr2)/2.0

    def _collect_epoch_states(self):

        self.epoch_acc = np.mean(self.running_acc)
        print('Is_training: %s. Epoch %d / %d, epoch_acc= %.5f' %
              (self.is_training, self.epoch_id, self.max_num_epochs-1, self.epoch_acc))

    def _update_lr_sechedulers(self):
        self.exp_lr_scheduler_T.step()
    
    def _update_checkpoints(self):
        self._save_checkpoint(ckpt_name='last_ckpt.pt')
        print('Lastest model updated. Epoch_acc=%.4f, Historical_best_acc=%.4f (at epoch %d)'
                % (self.epoch_acc, self.best_val_acc, self.best_epoch_id))
        self.VAL_ACC = np.append(self.VAL_ACC, [self.epoch_acc])
        np.save(os.path.join(self.checkpoint_dir, 'val_acc.npy'), self.VAL_ACC)

        # update the best model (based on eval acc)
        if self.epoch_acc > self.best_val_acc:
            self.best_val_acc = self.epoch_acc
            self.best_epoch_id = self.epoch_id
            self._save_checkpoint(ckpt_name='best_ckpt.pt')
            print('*' * 10 + 'Best model updated!')


    def train(self):
        print("Traning Translator. Device:", device)
        self.LoadCheckPoint()
        if self.is_VIS:
            self.vis = visdom.Visdom(port=self.vis_port)
        for self.epoch_id in range(self.epoch_to_start, self.max_num_epochs):
            self._clear_cache()
            self.is_training = True
            self.render.eval()
            self.translator.train()
            for self.batch_id, batch in enumerate(self.dataloader, 0):
                self._forward_pass(batch)
                self.optimizer_T.zero_grad()
                self._backward_R()
                self.optimizer_T.step()
                self._collect_running_batch_states()
            self._collect_epoch_states()
            self._update_lr_sechedulers()
            self._update_checkpoints()
