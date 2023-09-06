# ---------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for Denoising Diffusion GAN. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------
import argparse
import torch
import os
import numpy as np
import torch.nn as nn
from torchdiffeq import odeint_adjoint as odeint
from models.util import get_flow_model
import torchvision
from pytorch_fid.fid_score import calculate_fid_given_paths
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn.functional as F
from copy import deepcopy

ADAPTIVE_SOLVER = ["dopri5", "dopri8", "adaptive_heun", "bosh3"]
FIXER_SOLVER = ["euler", "rk4", "midpoint", "heun"]


def heun(model, noise, dt):
    N = int(1/dt)
    x_t = noise
    for i in range(N):
        v_t = model(torch.tensor((N-i)/N), x_t)
        x_tp1 = x_t - v_t * dt
        v_tp1 = model(torch.tensor((N-(i+1))/N), x_tp1)
        v_t = 1/2 * (v_tp1 + v_t)
        x_t = x_t - v_t * dt
    return (noise, x_t)

def euler_curvature(model, noise, nfe):
    N = nfe
    dt = 1./nfe
    x_t = noise
    series = [x_t]
    pred_series = []
    for i in range(N):
        v_t = model(torch.tensor((N-i)/N), x_t)
        pred_series.append(x_t - v_t * torch.tensor((N-i)/N))
        x_t = x_t - v_t * dt
        series.append(x_t)
        
    return torch.stack(series), torch.stack(pred_series)


class Model_(nn.Module):
    def __init__(self, model):
        super(Model_, self).__init__()
        self.model = model

    def forward(self, t, x_0):
        out = self.model(t, x_0)
        return out
    
def sample_from_model(model, x_0, args):
    
    model_ = Model_(model)
    model_.eval()
    with torch.no_grad():        
        fake_image, pred_series = euler_curvature(model_, x_0, args.nfe)
    return fake_image, pred_series


def sample_and_test(args):
    torch.manual_seed(43)
    device = 'cuda:0'
    
    to_range_0_1 = lambda x: (x + 1.) / 2.

    model =  get_flow_model(args).to(device)
    ckpt = torch.load('./saved_info/flow_matching/{}/{}/model_{}.pth'.format(args.dataset, args.exp, args.epoch_id), map_location=device)
    print("Finish loading model")
    #loading weights from ddp in single gpu
    for key in list(ckpt.keys()):
        ckpt[key[7:]] = ckpt.pop(key)
    model.load_state_dict(ckpt)
    model.eval()
    
    x_0 = torch.randn(args.batch_size, 3, args.image_size, args.image_size).to(device)
    # x_0 = torch.randn(1, 3, args.image_size, args.image_size).to(device)
    # x_0_mod = deepcopy(x_0)
    # x_0_mod[0, :, 16, 17] = torch.tensor([1, 1, 1])
    # assert torch.sum(x_0 - x_0_mod) != 0, "use deep copy"
    # x_0 = torch.cat([x_0, x_0_mod])
    fake_sample, pred_series = sample_from_model(model, x_0, args)
    fake_sample, pred_series = to_range_0_1(fake_sample), to_range_0_1(pred_series)
    for i in range(x_0.size(0)):  
        torchvision.utils.save_image(fake_sample[:, i, :, :, :], "series_{}.png".format(i), nrow=10)
        torchvision.utils.save_image(pred_series[:, i, :, :, :], "pred_series_{}.png".format(i), nrow=10)
        
    residual_pred = -pred_series[:-1] + pred_series[1:]
    
    norm_residual = residual_pred**2
    norm_residual = torch.sqrt(torch.sum(norm_residual, dim=[2,3,4]))
    norm_residual = torch.mean(norm_residual, dim = 1)
    
    residual_pred = torch.mean(residual_pred, dim = 1)
    double_residual_pred = residual_pred[1:] - residual_pred[:-1]
    torchvision.utils.save_image(residual_pred, "pred_residual.png", nrow=10, normalize=True)
    torchvision.utils.save_image(double_residual_pred, "double_pred_residual.png", nrow=10, normalize=True)
    residual_pred = torch.mean(residual_pred, dim = [1,2,3])
    double_residual_pred = torch.mean(double_residual_pred, dim = [1,2,3])
    plt.figure()
    plt.plot(np.linspace(0, 1, residual_pred.size(0)), residual_pred.to("cpu"))
    plt.savefig('plot_residual.png')
    plt.figure()
    plt.plot(np.linspace(0, 1, double_residual_pred.size(0)), double_residual_pred.to("cpu"))
    plt.savefig('double_plot_residual.png')
    plt.figure()
    plt.plot(np.linspace(0, 1, norm_residual.size(0)), norm_residual.to("cpu"))
    plt.savefig('norm_residual.png')
    
    
    
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser('ddgan parameters')
    parser.add_argument('--seed', type=int, default=1024,
                        help='seed used for initialization')
    parser.add_argument('--compute_fid', action='store_true', default=False,
                            help='whether or not compute FID')
    parser.add_argument('--epoch_id', type=int,default=1000)

    parser.add_argument('--image_size', type=int, default=32,
                            help='size of image')
    parser.add_argument('--num_in_channels', type=int, default=3,
                            help='in channel image')
    parser.add_argument('--num_out_channels', type=int, default=3,
                            help='in channel image')
    parser.add_argument('--nf', type=int, default=256,
                            help='channel of image')
    parser.add_argument('--centered', action='store_false', default=True,
                            help='-1,1 scale')
    parser.add_argument("--resamp_with_conv", type=bool, default=True)
    parser.add_argument('--num_res_blocks', type=int, default=2,
                            help='number of resnet blocks per scale')
    parser.add_argument('--num_heads', type=int, default=4,
                            help='number of head')
    parser.add_argument('--num_head_upsample', type=int, default=-1,
                            help='number of head upsample')
    parser.add_argument('--num_head_channels', type=int, default=-1,
                            help='number of head channels')
    parser.add_argument('--attn_resolutions', nargs='+', type=int, default=(16,),
                            help='resolution of applying attention')
    parser.add_argument('--ch_mult', nargs='+', type=int, default=(1,2,2,2),
                            help='channel mult')
    parser.add_argument('--dropout', type=float, default=0.,
                            help='drop-out rate')
    parser.add_argument('--num_classes', type=int, default=None,
                            help='num classes')
    parser.add_argument("--use_scale_shift_norm", type=bool, default=True)
    parser.add_argument("--resblock_updown", type=bool, default=False)
    parser.add_argument("--use_new_attention_order", type=bool, default=False)
    
    #######################################
    parser.add_argument('--exp', default='experiment_cifar_default', help='name of experiment')
    parser.add_argument('--real_img_dir', default='./pytorch_fid/cifar10_train_stat.npy', help='directory to real images for FID computation')
    parser.add_argument('--dataset', default='cifar10', help='name of dataset')
    parser.add_argument('--nfe', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=2, help='sample generating batch size')
    
    # sampling argument
    parser.add_argument('--atol', type=float, default=1e-5, help='absolute tolerance error')
    parser.add_argument('--rtol', type=float, default=1e-5, help='absolute tolerance error')
    parser.add_argument('--method', type=str, default='dopri5', help='solver_method', choices=["dopri5", "dopri8", "adaptive_heun", "bosh3", "euler", "midpoint", "rk4", "heun"])
    parser.add_argument('--step_size', type=float, default=0.04, help='step_size')
    parser.add_argument('--perturb', action='store_true', default=False)
        



   
    args = parser.parse_args()
    
    sample_and_test(args)
    