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

def euler(model, nested_model, noise, args):
    step_noise = (1 - args.t_noise)/args.nfe_noise
    step_data = args.t_data/args.nfe_data
    step_middle = 1/args.nfe_middle
    x_t = noise
    t_noise = 1.0
    
    for _ in range(args.nfe_noise):
        v_t = model(torch.tensor(t_noise), x_t)
        x_t = x_t - v_t * step_noise
        t_noise = t_noise - step_noise
    
    t_middle = 1.0
    for _ in range(args.nfe_middle):
        v_t = nested_model(torch.tensor(t_middle), x_t)
        x_t = x_t - v_t * step_middle
        t_middle = t_middle - step_middle
    
    t_data = args.t_data
    for _ in range(args.nfe_data):
        v_t = model(torch.tensor(t_data), x_t)
        x_t = x_t - v_t * step_data
        t_data = t_data - step_data
        
    return (noise, x_t)


class Model_(nn.Module):
    def __init__(self, model):
        super(Model_, self).__init__()
        self.model = model

    def forward(self, t, x_0):
        out = self.model(t, x_0)
        return out
    
def sample_from_model(model, nested_model, x_0, args, reverse=False):
    if args.method in ADAPTIVE_SOLVER:
        options = {
            "dtype": torch.float64,
        }
    else:
        options = {
            "step_size": args.step_size,
            "perturb": args.perturb
        }
    if not args.compute_fid:
        model.count_nfe = True
    model_ = Model_(model)
    nested_model = Model_(nested_model)
    model_.eval()
    nested_model.eval()
    
    with torch.no_grad():
        if args.method == "heun":
            fake_image = heun(model_, x_0, args.step_size)
        else:
            fake_image = euler(model_, nested_model, x_0, args)
    return fake_image


def sample_and_test(args):
    torch.manual_seed(42)
    device = 'cuda:0'
    
    if args.dataset == 'cifar10':
        real_img_dir = 'pytorch_fid/cifar10_train_stat.npy'
    elif args.dataset == 'celeba_256':
        real_img_dir = 'pytorch_fid/celeba_256_stat.npy'
    elif args.dataset == 'lsun':
        real_img_dir = 'pytorch_fid/lsun_church_stat.npy'
    else:
        real_img_dir = args.real_img_dir
    
    to_range_0_1 = lambda x: (x + 1.) / 2.

    model =  get_flow_model(args).to(device)
    ckpt = torch.load(args.init_model, map_location=device)
    print("Finish loading init model")
    #loading weights from ddp in single gpu
    for key in list(ckpt.keys()):
        ckpt[key[7:]] = ckpt.pop(key)
    model.load_state_dict(ckpt)
    model.eval()
    
    nested_model =  get_flow_model(args).to(device)
    nested_ckpt = torch.load(args.nest_model, map_location=device)
    print("Finish loading nested model")
    #loading weights from ddp in single gpu
    for key in list(nested_ckpt.keys()):
        nested_ckpt[key[7:]] = nested_ckpt.pop(key)
    nested_model.load_state_dict(nested_ckpt)
    nested_model.eval()
    
    
    iters_needed = 50000 //args.batch_size
    
    save_dir = "./generated_samples/{}".format(args.dataset)
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    if args.compute_fid:
        for i in tqdm(range(iters_needed)):
            with torch.no_grad():
                x_0 = torch.randn(args.batch_size, 3, args.image_size, args.image_size).to(device)
                fake_sample = sample_from_model(model, x_0, args)[-1]
                fake_sample = to_range_0_1(fake_sample)
                for j, x in enumerate(fake_sample):
                    index = i * args.batch_size + j 
                    torchvision.utils.save_image(x, './generated_samples/{}/{}.jpg'.format(args.dataset, index))
                print('generating batch ', i)
        
        paths = [save_dir, real_img_dir]
    
        kwargs = {'batch_size': 200, 'device': device, 'dims': 2048}
        fid = calculate_fid_given_paths(paths=paths, **kwargs)
        print('FID = {}'.format(fid))
    else:
        x_0 = torch.randn(args.batch_size, 3, args.image_size, args.image_size).to(device)
        fake_sample = sample_from_model(model, nested_model, x_0, args)[-1]
        fake_sample = to_range_0_1(fake_sample)
        torchvision.utils.save_image(fake_sample, './samples_{}_{}_{}_{}_{}.jpg'.format(args.dataset, args.method, args.nfe_noise, args.nfe_middle, args.nfe_data))
   
    
            

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
    
    
    ###nest
    parser.add_argument('--t_noise', type=float, default=0.75,
                            help='beta1 for adam')
    parser.add_argument('--t_data', type=float, default=0.25,
                            help='beta1 for adam')
    parser.add_argument('--nfe_noise', type=int, default=5,
                            help='beta1 for adam')
    parser.add_argument('--nfe_middle', type=int, default=10,
                            help='beta1 for adam')
    parser.add_argument('--nfe_data', type=int, default=5,
                            help='beta1 for adam')
    parser.add_argument('--init_model', default='init model', help='name of experiment')
    parser.add_argument('--nest_model', default='init model', help='name of experiment')
    
    
    #######################################
    parser.add_argument('--exp', default='experiment_cifar_default', help='name of experiment')
    parser.add_argument('--real_img_dir', default='./pytorch_fid/cifar10_train_stat.npy', help='directory to real images for FID computation')
    parser.add_argument('--dataset', default='cifar10', help='name of dataset')
    parser.add_argument('--num_timesteps', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=100, help='sample generating batch size')
    
    
    # sampling argument
    parser.add_argument('--atol', type=float, default=1e-5, help='absolute tolerance error')
    parser.add_argument('--rtol', type=float, default=1e-5, help='absolute tolerance error')
    parser.add_argument('--method', type=str, default='dopri5', help='solver_method', choices=["dopri5", "dopri8", "adaptive_heun", "bosh3", "euler", "midpoint", "rk4", "heun"])
    parser.add_argument('--step_size', type=float, default=0.04, help='step_size')
    parser.add_argument('--perturb', action='store_true', default=False)
        



   
    args = parser.parse_args()
    
    sample_and_test(args)