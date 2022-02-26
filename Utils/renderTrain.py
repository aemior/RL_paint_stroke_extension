import os
import resource
from turtle import forward, pd

import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import matplotlib.pyplot as plt
import visdom

from Utils.utils import get_neural_render,get_stroke_dataset,get_discriminator 
import Utils.utils as utils

# Decide which device we want to run on
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Train(object):
    def __init__(self, args):
        self.dataloader = get_stroke_dataset(args) 
        self.render = get_neural_render(args.StrokeType, args.net_R).to(device)

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
        self.LOSS = {}
        self.best_val_acc = 0.0
        self.best_epoch_id = 0
        
        self.vis_port = args.vis_port
        self.is_VIS = self.vis_port != 0
        self.vis = None

        self.lr = args.lr
        self.optimizer_R = optim.Adam(self.render.parameters(), lr=self.lr, betas=(0.9,0.999))
        self.exp_lr_scheduler_R = lr_scheduler.StepLR(
            self.optimizer_R, step_size=100, gamma=0.1
        )
        self._pxl_loss = torch.nn.MSELoss()
        self.only_black = args.only_black

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
            self.render.load_state_dict(checkpoint['model_R_state_dict'])
            self.optimizer_R.load_state_dict(checkpoint['optimizer_R_state_dict'])
            self.exp_lr_scheduler_R.load_state_dict(
                checkpoint['exp_lr_scheduler_R_state_dict'])
            self.render.to(device)

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
            'model_R_state_dict': self.render.state_dict(),
            'optimizer_R_state_dict': self.optimizer_R.state_dict(),
            'exp_lr_scheduler_R_state_dict': self.exp_lr_scheduler_R.state_dict()
        }, os.path.join(self.checkpoint_dir, ckpt_name))

    def _clear_cache(self):
        self.running_acc = []

    def _forward_pass(self, batch):
        self.batch = batch
        self.z_in = batch['A'].to(device)
        pred_foreground, pred_alpha = self.render(self.z_in)
        self.PD_C_B = pred_alpha * pred_foreground
        self.PD_C_W = (1-pred_alpha) + pred_alpha * pred_foreground
        gt_foreground, gt_alpha = self.batch['B'].to(device),self.batch['ALPHA'].to(device)
        self.GT_C_B = gt_alpha * gt_foreground
        self.GT_C_W = (1-gt_alpha) + gt_alpha * gt_foreground 

    def _backward_R(self):

        pixel_loss1 = self._pxl_loss(self.PD_C_B, self.GT_C_B)
        pixel_loss2 = self._pxl_loss(self.PD_C_W, self.GT_C_W)
        if self.only_black:
            self.LOSS['R_loss'] = pixel_loss1
        else:
            self.LOSS['R_loss'] = pixel_loss1 + pixel_loss2
        self.LOSS['RCW_loss'] = pixel_loss2
        self.LOSS['RCB_loss'] = pixel_loss1
        self.LOSS['R_loss'].backward()

    def _collect_running_batch_states(self):
        self.running_acc.append(self._compute_acc().item())

        m = len(self.dataloader)
        
        if np.mod(self.batch_id, 100) == 1:
            mem_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            # print("MEM USAGE:", mem_usage/(1024*1024))
            print('[%d,%d][%d,%d], running_acc(PSNR): %.5f, '
                    % (self.epoch_id, self.max_num_epochs-1, self.batch_id, m,
                     np.mean(self.running_acc)), end='')
            for loss_name in self.LOSS.keys():
                print(loss_name+':%.5f, '%(self.LOSS[loss_name].item()), end='')
            print("<")

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
        self.exp_lr_scheduler_R.step()
    
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
        print("Traning Render. Device:", device)
        self.LoadCheckPoint()
        if self.is_VIS:
            self.vis = visdom.Visdom(port=self.vis_port)
        for self.epoch_id in range(self.epoch_to_start, self.max_num_epochs):
            self._clear_cache()
            self.is_training = True
            self.render.train()
            for self.batch_id, batch in enumerate(self.dataloader, 0):
                self._forward_pass(batch)
                self.optimizer_R.zero_grad()
                self._backward_R()
                self.optimizer_R.step()
                self._collect_running_batch_states()
            self._collect_epoch_states()
            self._update_lr_sechedulers()
            self._update_checkpoints()


class DisTrain(Train):
    def __init__(self, args):
        super().__init__(args)
        self.D = get_discriminator(args.StrokeType).to(device)

        self.optimizer_D = optim.Adam(self.D.parameters(), lr=2e-5, betas=(0.9,0.999))
        self.exp_lr_scheduler_D = lr_scheduler.StepLR(
            self.optimizer_D, step_size=100, gamma=0.1
        )

        self.render_ckpt = args.render_ckpt

    def load_D_checkpoint(self):
        if os.path.exists(os.path.join(self.checkpoint_dir, 'last_D_ckpt.pt')):
            print('# loading last checkpoint D...')
            # load the entire checkpoint
            checkpoint = torch.load(os.path.join(self.checkpoint_dir, 'last_D_ckpt.pt'))

            # update render states
            self.D.load_state_dict(checkpoint['model_D_state_dict'])
            self.optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
            self.exp_lr_scheduler_D.load_state_dict(
                checkpoint['exp_lr_scheduler_D_state_dict'])
            self.D.to(device)

        else:
            print('# training D from scratch...')

    def _save_D_checkpoint(self, ckpt_name):
        torch.save({
            'model_D_state_dict': self.D.state_dict(),
            'optimizer_D_state_dict': self.optimizer_D.state_dict(),
            'exp_lr_scheduler_D_state_dict': self.exp_lr_scheduler_D.state_dict()
        }, os.path.join(self.checkpoint_dir, ckpt_name))

    def forward_pass_D(self):

        self.D.set_cond(self.z_in)
        #self.D_pd_real_W = self.D(self.GT_C_W)
        #self.D_pd_fake_W = self.D(self.PD_C_W)
        self.D_pd_real_B = self.D(self.GT_C_B)
        self.D_pd_fake_B = self.D(self.PD_C_B)

    def _backward_D(self):
        loss_real = self.D_pd_real_B.mean()# + self.D_pd_real_W.mean()
        loss_fake = self.D_pd_fake_B.mean()# + self.D_pd_fake_W.mean()
        self.LOSS['D_loss'] = loss_real - loss_fake
        self.LOSS['D_loss'].backward()

    def _backward_R(self):

        self.D.set_cond(self.z_in)
        #dloss_w = self.D(self.PD_C_W)
        dloss_b = self.D(self.PD_C_B)
        self.LOSS['R_loss'] = dloss_b.mean()# + dloss_w.mean()
        self.LOSS['R_loss'].backward()
    
    def _update_lr_sechedulers(self):
        super()._update_lr_sechedulers()
        self.exp_lr_scheduler_D.step()

    def upd(self, ckpt_name):
        super()._save_checkpoint(ckpt_name)
    
    def _update_checkpoints(self):
        super()._update_checkpoints()
        self._save_D_checkpoint('last_D_ckpt.pt')

    def train(self):
        print("Traning Render. Device:", device)
        if self.is_VIS:
            self.vis = visdom.Visdom(port=self.vis_port)
        self.LoadCheckPoint()
        self.load_D_checkpoint()
        self.D.to(device)
        for self.epoch_id in range(self.epoch_to_start, self.max_num_epochs):
            self._clear_cache()
            self.is_training = True
            self.render.train()
            self.D.train()
            for self.batch_id, batch in enumerate(self.dataloader, 0):
                self._forward_pass(batch)
                #if (self.batch_id+1) % 10 == 0:
                if (self.batch_id+1) % 2 == 0:
                    self.optimizer_R.zero_grad()
                    self._backward_R()
                    self.optimizer_R.step()
                else:
                    self.forward_pass_D()
                    self.optimizer_D.zero_grad()
                    self._backward_D()
                    self.optimizer_D.step()
                self._collect_running_batch_states()
            self._collect_epoch_states()
            self._update_lr_sechedulers()
            self._update_checkpoints()


    



