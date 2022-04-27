import os
import resource
from turtle import backward, forward, pd

import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import visdom

from Utils.utils import get_neural_render,get_stroke_dataset,get_discriminator,calc_gradient_penalty,cal_gradient_penalty
import Utils.utils as utils

# Decide which device we want to run on
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
        if args.patch_gan_loss:
            from Utils.utils import PatchGan_loss
            self._pxl_loss = PatchGan_loss(args, device)
        self.only_black = args.only_black
        self.only_white = args.only_white
        self.rand_c = args.rand_c

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
        gt_foreground, gt_alpha = self.batch['B'].to(device),self.batch['ALPHA'].to(device)
        if self.rand_c:
            C_R = torch.rand(pred_foreground.shape).to(device)
            self.PD_C_R = (1-pred_alpha) * C_R + pred_alpha * pred_foreground
            self.GT_C_R = (1-gt_alpha) * C_R + gt_alpha * gt_foreground
            return
        self.PD_C_B = pred_alpha * pred_foreground
        self.PD_C_W = (1-pred_alpha) + pred_alpha * pred_foreground
        self.old_GT_C_B = self.GT_C_B
        self.old_GT_C_W = self.GT_C_W
        self.GT_C_B = gt_alpha * gt_foreground
        self.GT_C_W = (1-gt_alpha) + gt_alpha * gt_foreground 

    def _backward_R(self):

        if self.rand_c:
            pixel_loss3 = self._pxl_loss(self.PD_C_R, self.GT_C_R)
            self.LOSS['R_loss'] = pixel_loss3
            self.LOSS['RCR_loss'] = pixel_loss3
            self.LOSS['R_loss'].backward()
            return
        pixel_loss1 = self._pxl_loss(self.PD_C_B, self.GT_C_B)
        pixel_loss2 = self._pxl_loss(self.PD_C_W, self.GT_C_W)
        if self.only_black and not self.only_white:
            self.LOSS['R_loss'] = pixel_loss1
        elif self.only_white and not self.only_black:
            self.LOSS['R_loss'] = pixel_loss2
        else:
            self.LOSS['R_loss'] = pixel_loss1 + pixel_loss2
        self.LOSS['RCW_loss'] = pixel_loss2
        self.LOSS['RCB_loss'] = pixel_loss1
        self.LOSS['R_loss'].backward()

    def _collect_running_batch_states(self):
        self.running_acc.append(self._compute_acc().item())

        m = len(self.dataloader)
        
        if np.mod(self.batch_id, 10) == 1 and self.is_VIS:
            for loss_name in self.LOSS.keys():
                self.vis.line(Y=[self.LOSS[loss_name].item()],
                X=[m*self.epoch_id+self.batch_id], update='append',
                win=loss_name, name=loss_name)

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
                if self.rand_c:
                    self.vis.images(self.GT_C_R, win='GT_C_R')
                    self.vis.images(self.PD_C_R, win='PD_C_R')
                else:
                    self.vis.images(self.GT_C_B, win='GT_C_B')
                    self.vis.images(self.PD_C_B, win='PD_C_B')
                    self.vis.images(self.GT_C_W, win='GT_C_W')
                    self.vis.images(self.PD_C_W, win='PD_C_W')

        if np.mod(self.batch_id, 1000) == 1:
            if self.rand_c:
                vis_pred = utils.make_numpy_grid(self.PD_C_R)
                vis_gt = utils.make_numpy_grid(self.GT_C_R)
                vis = np.concatenate([vis_gt, vis_pred], axis=0)
            else:
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
        if self.rand_c:
            target_C_R = self.GT_C_R.to(device).detach()
            C_R = self.PD_C_R.detach()
            psnr = utils.cpt_batch_psnr(C_R, target_C_R, PIXEL_MAX=1.0)
            return psnr
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


class IRTrain(Train):
    def _forward_pass(self, batch):
        self.batch = batch
        self.z_in = batch['A'].to(device)
        pred_foreground, pred_alpha, reg = self.render(self.z_in)
        gt_foreground, gt_alpha = self.batch['B'].to(device),self.batch['ALPHA'].to(device)
        if self.rand_c:
            C_R = torch.rand(pred_foreground.shape).to(device)
            self.PD_C_R = (1-pred_alpha) * C_R + pred_alpha * pred_foreground
            self.GT_C_R = (1-gt_alpha) * C_R + gt_alpha * gt_foreground
        self.REG = reg
        self.PD_C_B = pred_alpha * pred_foreground
        self.PD_C_W = (1-pred_alpha) + pred_alpha * pred_foreground
        self.old_GT_C_B = self.GT_C_B
        self.old_GT_C_W = self.GT_C_W
        self.GT_C_B = gt_alpha * gt_foreground
        self.GT_C_W = (1-gt_alpha) + gt_alpha * gt_foreground 
    def _backward_R(self):
        pixel_loss1 = self._pxl_loss(self.PD_C_B, self.GT_C_B)
        pixel_loss2 = self._pxl_loss(self.PD_C_W, self.GT_C_W)
        if self.only_black and not self.only_white:
            self.LOSS['R_loss'] = pixel_loss1
        elif self.only_white and not self.only_black:
            self.LOSS['R_loss'] = pixel_loss2
        elif self.rand_c:
            pixel_loss3 = self._pxl_loss(self.PD_C_R, self.GT_C_R)
            self.LOSS['R_loss'] = pixel_loss3
            self.LOSS['RCR_loss'] = pixel_loss3
        else:
            self.LOSS['R_loss'] = pixel_loss1 + pixel_loss2
        self.LOSS['RCW_loss'] = pixel_loss2
        self.LOSS['RCB_loss'] = pixel_loss1
        self.LOSS['REG_term'] = self.REG
        (self.LOSS['R_loss'] + 0.3 * self.REG).backward()
        

        



class DisTrain(Train):
    def __init__(self, args):
        super().__init__(args)
        #self.D = get_discriminator(args.StrokeType).to(device)
        self.D = get_discriminator(args).to(device)

        self.optimizer_D = optim.Adam(self.D.parameters(), lr=args.lr_gan, betas=(0.5,0.999))
        self.exp_lr_scheduler_D = lr_scheduler.StepLR(
            self.optimizer_D, step_size=100, gamma=0.1
        )

        self.render_ckpt = args.render_ckpt
        self.DLOSS = {}
        self.RLOSS = {}
        self.dataloader_2 = get_stroke_dataset(args) 

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
            """
            if os.path.exists(os.path.join(self.render_ckpt, 'last_ckpt.pt')):
                print("# load pretrain render checkpoint..")
                checkpoint = torch.load(os.path.join(self.render_ckpt, 'last_ckpt.pt'))
                self.render.load_state_dict(checkpoint['model_R_state_dict'])
                self.render.to(device)
            """
            print('# training D from scratch...')

    def _save_D_checkpoint(self, ckpt_name):
        torch.save({
            'model_D_state_dict': self.D.state_dict(),
            'optimizer_D_state_dict': self.optimizer_D.state_dict(),
            'exp_lr_scheduler_D_state_dict': self.exp_lr_scheduler_D.state_dict()
        }, os.path.join(self.checkpoint_dir, ckpt_name))

    def forward_pass_D(self):

        self.real_batch = torch.cat((torch.cat((self.GT_C_W, self.GT_C_B)), torch.cat((self.GT_C_W, self.GT_C_B))), dim=1)
        self.fake_batch = torch.cat((torch.cat((self.GT_C_W, self.GT_C_B)), torch.cat((self.PD_C_W, self.PD_C_B))), dim=1)
        self.D_pd_real = self.D(self.real_batch)
        self.D_pd_fake = self.D(self.fake_batch)
        self.gradient_penalty = cal_gradient_penalty(self.D, self.real_batch, self.fake_batch, self.D_pd_real.shape[0], device)

    def _backward_D(self):
        loss_real = self.D_pd_real.mean()
        loss_fake = self.D_pd_fake.mean()
        self.LOSS['D_loss'] = loss_fake - loss_real + self.gradient_penalty
        self.DLOSS['D_fake'] = loss_fake
        self.DLOSS['D_real'] = loss_real
        self.DLOSS['D_penalty'] = self.gradient_penalty
        self.LOSS['D_loss'].backward()

    def _backward_R(self):

        self.fake_batch = torch.cat((torch.cat((self.GT_C_W, self.GT_C_B)), torch.cat((self.PD_C_W, self.PD_C_B))), dim=1)
        self.LOSS['R_loss'] = (-self.D(self.fake_batch)).mean()
        self.LOSS['R_loss'].backward()
    
    def _update_lr_sechedulers(self):
        if self.epoch_id > 2:
            super()._update_lr_sechedulers()
        self.exp_lr_scheduler_D.step()

    def _collect_running_batch_states(self):
        super()._collect_running_batch_states()
        m = len(self.dataloader)
        if np.mod(self.batch_id, 10) == 1 and self.is_VIS:
            for loss_name in self.DLOSS.keys():
                self.vis.line(Y=[self.DLOSS[loss_name].item()],
                X=[m*self.epoch_id+self.batch_id], update='append',
                win='D LOSS', name=loss_name)
            for loss_name in self.RLOSS.keys():
                self.vis.line(Y=[self.RLOSS[loss_name].item()],
                X=[m*self.epoch_id+self.batch_id], update='append',
                win='R LOSS', name=loss_name)


    def _update_checkpoints(self):
        super()._update_checkpoints()
        self._save_D_checkpoint('last_D_ckpt.pt')

    def update_D(self, batch):
        self.D.train()
        self.render.eval()
        with torch.no_grad():
            self._forward_pass(batch)
        self.forward_pass_D()
        self.optimizer_D.zero_grad()
        self._backward_D()
        self.optimizer_D.step()

    def update_R(self, batch, recon=False):
        self.D.eval()
        self.render.train()
        self._forward_pass(batch)
        self.optimizer_R.zero_grad()
        if recon:
            super()._backward_R()
        else:
            self._backward_R()
        self.optimizer_R.step()

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
            w_batch = next(iter(self.dataloader_2))
            self._forward_pass(w_batch)
            for self.batch_id, batch in enumerate(self.dataloader, 0):
                #self._forward_pass(batch)
                #if (self.batch_id+1) % 10 == 0:
                if False:#self.epoch_id < 3:
                    self.update_D(batch)
                else:
                    if (self.batch_id) % 2 == 0:
                        self.update_D(batch)
                    else:
                        self.update_R(batch)
                self._collect_running_batch_states()
            self._collect_epoch_states()
            self._update_lr_sechedulers()
            self._update_checkpoints()
    def LoadCheckPoint(self):
        if os.path.exists(os.path.join(self.checkpoint_dir, 'last_ckpt.pt')):
            print('# loading last checkpoint R...')
            # load the entire checkpoint
            checkpoint = torch.load(os.path.join(self.checkpoint_dir, 'last_ckpt.pt'))

            # update render states
            self.render.load_state_dict(checkpoint['model_R_state_dict'])
            self.render.to(device)

            # update some other states
            self.epoch_to_start = checkpoint['epoch_id'] + 1
            self.best_val_acc = checkpoint['best_val_acc']
            self.best_epoch_id = checkpoint['best_epoch_id']

            print('Epoch_to_start = %d, Historical_best_acc = %.4f (at epoch %d)' %
                    (self.epoch_to_start, self.best_val_acc, self.best_epoch_id))

        else:
            print('# training from scratch...')


    



class PgTrain(Train):

    def __init__(self, args):
        super().__init__(args)
        self.resize = {
            "L5":transforms.Resize((128,128)),
            "L4":transforms.Resize((64,64)),
            "L3":transforms.Resize((32,32)),
            "L2":transforms.Resize((16,16)),
            "L1":transforms.Resize((8,8)),
        }
        self.train_range = {
            "L1":[0,50],
            "L2":[50,100],
            "L3":[100,150],
            "L4":[150,200],
            "L5":[200,250],
            "L5":[250,400],
        }
        self.cur_layer = "L0"
    
    def get_layer(self):
        for layer in self.train_range.keys():
            if self.epoch_id < self.train_range[layer][1] and\
                self.train_range[layer][0] <= self.epoch_id:
                if self.cur_layer == layer:
                    self.keep = True
                else:
                    self.cur_layer = layer
                    self.keep = False

    def set_optimizer(self):
        #for name, parm in self.render.named_parameters():
        #    parm.requires_grad = False
        if self.keep:
            return
        if self.cur_layer == "L1":
            self.render.conv1.requires_gard = True
            self.render.fc1.requires_gard = True
            self.render.main.L1.requires_gard = True
            for name, parm in self.render.named_parameters():
                if name in ["conv1", "fc1", "L1"]:
                    parm.requires_grad = True
            self.optimizer_R = optim.Adam([{"params":self.render.conv1.parameters()},\
                                           {"params":self.render.fc1.parameters()},\
                                        {"params":self.render.main.L1.parameters()}], self.lr)
        elif self.cur_layer == "L2":
            self.render.main.L2.requires_gard = True
            self.optimizer_R = optim.Adam(self.render.main.L2.parameters(), self.lr)
        elif self.cur_layer == "L3":
            self.render.main.L3.requires_gard = True
            self.optimizer_R = optim.Adam(self.render.main.L3.parameters(), self.lr)
        elif self.cur_layer == "L4":
            self.render.main.L4.requires_gard = True
            self.optimizer_R = optim.Adam(self.render.main.L4.parameters(), self.lr)
        elif self.cur_layer == "L5":
            self.render.main.L5.requires_gard = True
            self.optimizer_R = optim.Adam(self.render.main.L5.parameters(), self.lr)

    def _forward_pass(self, batch):
        self.batch = batch
        self.z_in = batch['A'].to(device)
        pred_foreground, pred_alpha = self.render(self.z_in, self.cur_layer)
        self.PD_C_B = pred_alpha * pred_foreground
        self.PD_C_W = (1-pred_alpha) + pred_alpha * pred_foreground
        gt_foreground, gt_alpha = self.batch['B'].to(device),self.batch['ALPHA'].to(device)
        self.old_GT_C_B = self.GT_C_B
        self.old_GT_C_W = self.GT_C_W
        org_gt_CB = gt_alpha * gt_foreground
        org_gt_CW = (1-gt_alpha) + gt_alpha * gt_foreground 
        self.GT_C_B = self.resize[self.cur_layer](org_gt_CB)
        self.GT_C_W = self.resize[self.cur_layer](org_gt_CW)


    def train(self):
        print("Traning Render. Device:", device)
        self.LoadCheckPoint()
        if self.is_VIS:
            self.vis = visdom.Visdom(port=self.vis_port)
        for self.epoch_id in range(self.epoch_to_start, self.max_num_epochs):
            self.get_layer()
            self._clear_cache()
            self.is_training = True
            self.render.train()
            self.set_optimizer()
            print("Current Layer:", self.cur_layer, "Keep:", self.keep)
            for self.batch_id, batch in enumerate(self.dataloader, 0):
                self._forward_pass(batch)
                self.optimizer_R.zero_grad()
                self._backward_R()
                self.optimizer_R.step()
                self._collect_running_batch_states()
            self._collect_epoch_states()
            if self.epoch_id >= 250:
                self._update_lr_sechedulers()
            self._update_checkpoints()

MINI_LR = 2e-6
class StyTrain(PgTrain):
    def set_optimizer(self):
        if self.keep:
            return
        if self.cur_layer == "L1":
            self.optimizer_R = optim.Adam([{"params":self.render.const},\
                                           {"params":self.render.MapNet.parameters()},\
                                        {"params":self.render.main.L1.parameters()}], self.lr)
        elif self.cur_layer == "L2":
            self.optimizer_R = optim.Adam([{"params":self.render.const},\
                        {"params":self.render.MapNet.parameters()},\
                        {"params":self.render.main.L1.parameters()},\
                        {"params":self.render.main.L2.parameters()}], self.lr)
        elif self.cur_layer == "L3":
            self.optimizer_R = optim.Adam([{"params":self.render.const},\
                        {"params":self.render.MapNet.parameters()},\
                        {"params":self.render.main.L1.parameters()},\
                        {"params":self.render.main.L2.parameters()},\
                        {"params":self.render.main.L3.parameters()}], self.lr)
        elif self.cur_layer == "L4":
            self.optimizer_R = optim.Adam([{"params":self.render.const},\
                        {"params":self.render.MapNet.parameters()},\
                        {"params":self.render.main.L1.parameters()},\
                        {"params":self.render.main.L2.parameters()},\
                        {"params":self.render.main.L3.parameters()},\
                        {"params":self.render.main.L4.parameters()}], self.lr)
        elif self.cur_layer == "L5":
            self.optimizer_R = optim.Adam(self.render.parameters(), lr=self.lr, betas=(0.9,0.999))
            self.exp_lr_scheduler_R = lr_scheduler.StepLR(
                self.optimizer_R, step_size=50, gamma=0.1
            )
        """
        elif self.cur_layer == "L2":
            self.optimizer_R = optim.Adam([{"params":self.render.const, 'lr': MINI_LR},\
                        {"params":self.render.MapNet.parameters(), 'lr': MINI_LR},\
                        {"params":self.render.main.L1.parameters(), 'lr': MINI_LR},\
                        {"params":self.render.main.L2.parameters()}], self.lr)
        elif self.cur_layer == "L3":
            self.optimizer_R = optim.Adam([{"params":self.render.const, 'lr': MINI_LR},\
                        {"params":self.render.MapNet.parameters(), 'lr': MINI_LR},\
                        {"params":self.render.main.L1.parameters(), 'lr': MINI_LR},\
                        {"params":self.render.main.L2.parameters(), 'lr': MINI_LR},\
                        {"params":self.render.main.L3.parameters()}], self.lr)
        elif self.cur_layer == "L4":
            self.optimizer_R = optim.Adam([{"params":self.render.const, 'lr': MINI_LR},\
                        {"params":self.render.MapNet.parameters(), 'lr': MINI_LR},\
                        {"params":self.render.main.L1.parameters(), 'lr': MINI_LR},\
                        {"params":self.render.main.L2.parameters(), 'lr': MINI_LR},\
                        {"params":self.render.main.L3.parameters(), 'lr': MINI_LR},\
                        {"params":self.render.main.L4.parameters()}], self.lr)
        elif self.cur_layer == "L5":
            self.optimizer_R = optim.Adam([{"params":self.render.const, 'lr': MINI_LR},\
                        {"params":self.render.MapNet.parameters(), 'lr': MINI_LR},\
                        {"params":self.render.main.L1.parameters(), 'lr': MINI_LR},\
                        {"params":self.render.main.L2.parameters(), 'lr': MINI_LR},\
                        {"params":self.render.main.L3.parameters(), 'lr': MINI_LR},\
                        {"params":self.render.main.L4.parameters(), 'lr': MINI_LR},\
                        {"params":self.render.main.L5.parameters()}], self.lr)
        """
    def get_alpha(self):
        b = (self.train_range[self.cur_layer][0] \
            + self.train_range[self.cur_layer][1]) // 2
        if self.epoch_id > b or self.cur_layer == "L1":
            return None
        else:
            a = self.epoch_id - self.train_range[self.cur_layer][0]
            b = b - self.train_range[self.cur_layer][0]
            return float(a)/float(b)

    def _forward_pass(self, batch):
        a = self.get_alpha()
        self.batch = batch
        self.z_in = batch['A'].to(device)
        pred_foreground, pred_alpha = self.render(self.z_in, self.cur_layer, a)
        self.PD_C_B = pred_alpha * pred_foreground
        self.PD_C_W = (1-pred_alpha) + pred_alpha * pred_foreground
        gt_foreground, gt_alpha = self.batch['B'].to(device),self.batch['ALPHA'].to(device)
        self.old_GT_C_B = self.GT_C_B
        self.old_GT_C_W = self.GT_C_W
        org_gt_CB = gt_alpha * gt_foreground
        org_gt_CW = (1-gt_alpha) + gt_alpha * gt_foreground 
        self.GT_C_B = self.resize[self.cur_layer](org_gt_CB)
        self.GT_C_W = self.resize[self.cur_layer](org_gt_CW)

    def LoadCheckPoint(self):
        if os.path.exists(os.path.join(self.checkpoint_dir, 'last_ckpt.pt')):
            print('# loading last checkpoint R...')
            # load the entire checkpoint
            checkpoint = torch.load(os.path.join(self.checkpoint_dir, 'last_ckpt.pt'))

            # update render states
            self.render.load_state_dict(checkpoint['model_R_state_dict'])
            self.render.to(device)

            # update some other states
            self.epoch_to_start = checkpoint['epoch_id'] + 1
            self.best_val_acc = checkpoint['best_val_acc']
            self.best_epoch_id = checkpoint['best_epoch_id']

            print('Epoch_to_start = %d, Historical_best_acc = %.4f (at epoch %d)' %
                    (self.epoch_to_start, self.best_val_acc, self.best_epoch_id))

        else:
            print('# training from scratch...')