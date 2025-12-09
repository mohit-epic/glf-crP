import torch
import torch.nn as nn

from model_base import *

from net_CR_RDN import *

from metrics import *

from torch.optim import lr_scheduler

class ModelCRNet(ModelBase):
    def __init__(self, opts):
        super(ModelCRNet, self).__init__()
        self.opts = opts
        
        # create network
        self.net_G = RDN_residual_CR(self.opts.crop_size)
        # Move to single GPU (use GPU 0 to avoid DataParallel nested wrapping issues)
        # For multi-GPU setups, increase batch_sz instead of using DataParallel
        gpu_ids = [int(g) for g in self.opts.gpu_ids.split(',')] if self.opts.gpu_ids else [0]
        device = torch.device(f'cuda:{gpu_ids[0]}')
        self.net_G = self.net_G.to(device)
        if len(gpu_ids) > 1:
            print(f"Single-GPU training on GPU {gpu_ids[0]} (use batch_sz >= 16 for better throughput)")
        self.print_networks(self.net_G)

        # initialize optimizers
        if self.opts.optimizer == 'Adam':
            self.optimizer_G = torch.optim.Adam(self.net_G.parameters(), lr=opts.lr)
            self.lr_scheduler = lr_scheduler.StepLR(self.optimizer_G, step_size=self.opts.lr_step, gamma=0.5)
            
        self.loss_fn=nn.L1Loss()
                        
    def set_input(self, _input):
        inputs = _input
        device = next(self.net_G.parameters()).device  # Get device from model
        self.cloudy_data = inputs['cloudy_data'].to(device)
        self.cloudfree_data = inputs['cloudfree_data'].to(device)
        self.SAR_data = inputs['SAR_data'].to(device)
        
    def forward(self):
        pred_CloudFree_data = self.net_G(self.cloudy_data, self.SAR_data)
        return pred_CloudFree_data

    def optimize_parameters(self):
                
        self.pred_Cloudfree_data = self.forward()

        self.loss_G = self.loss_fn(self.pred_Cloudfree_data, self.cloudfree_data)

        self.optimizer_G.zero_grad()
        self.loss_G.backward()
        
        # Clip gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.net_G.parameters(), max_norm=1.0)
        
        self.optimizer_G.step()  

        return self.loss_G.item()

    def get_current_scalars(self):
        losses = {}
        losses['PSNR_train']=PSNR(self.pred_Cloudfree_data.data, self.cloudfree_data)
        return losses

    def save_checkpoint(self, epoch):
        self.save_network(self.net_G, self.optimizer_G, epoch, self.lr_scheduler, self.opts.save_model_dir)
