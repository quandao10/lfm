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

from datasets_prep import get_dataset
from tqdm import tqdm
import time
from matplotlib import pyplot as plt

ADAPTIVE_SOLVER = ["dopri5", "dopri8", "adaptive_heun", "bosh3"]
FIXER_SOLVER = ["euler", "rk4", "midpoint"]


class Model_(nn.Module):
    def __init__(self, model):
        super(Model_, self).__init__()
        self.model = model

    def forward(self, t, x):
        return self.model(t, x)

    # x1*t + x0*(1-t) = xt

def sample_from_noise(model, x_1, args):
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
    t = torch.tensor([0., 1.], device="cuda") ##
    noise = odeint(model_, 
                        x_1, 
                        t, 
                        method=args.method, 
                        atol = args.atol, 
                        rtol = args.rtol,
                        adjoint_method=args.method,
                        adjoint_atol= args.atol,
                        adjoint_rtol= args.rtol,
                        options=options
                        )
    return noise


def sample_and_test(args):
    torch.manual_seed(24)
    device = 'cuda:0'

    
    to_range_0_1 = lambda x: (x + 1.) / 2.

    args.layout = False
    model =  get_flow_model(args).to(device)
    ckpt = torch.load('./saved_info/flow_matching/{}/{}/model_{}.pth'.format(args.dataset, args.exp, args.epoch_id), map_location=device)
    print("Finish loading model")
    #loading weights from ddp in single gpu
    for key in list(ckpt.keys()):
        ckpt[key[7:]] = ckpt.pop(key)
    model.load_state_dict(ckpt)
    model.eval()
        
    iters_needed = 1000 //args.batch_size
    
    save_dir = "./generated_samples/{}".format(args.dataset)
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    dataset = get_dataset(args)
    data_loader = torch.utils.data.DataLoader(dataset,
                                               batch_size=args.batch_size,
                                               shuffle=False,
                                               num_workers=4,
                                               pin_memory=True,
                                               drop_last = True)

    saved = []
    saved2 = []
    for i, (image, _) in enumerate(data_loader):
        clock = time.time()
        if i >= iters_needed:
            break
        image = image.to(device)
        with torch.no_grad():
            noise = sample_from_noise(model, image, args)[-1]
            for j, x in enumerate(noise):
                index = i * args.batch_size + j 
                saved.append(torch.mean(x).item())
                for p in range(3):
                    for q in range(32):
                        for r in range(32):
                            saved2.append(x[p][q][r].item())
                torchvision.utils.save_image(x, './generated_samples/{}/{}.jpg'.format(args.dataset, index))
            print('generating batch ', i)
        print('Time: ', time.time()-clock)
        clock = time.time()
    
    plt.hist(saved, bins=100)
    plt.savefig('test_noise.png')
    plt.hist(saved2, bins=100)
    plt.savefig('test_noise2.png')
    print('Mean for mean of image: ', np.mean(saved))
    print('Var for mean of image: ', np.var(saved))
    print('Mean for pixels of image: ', np.mean(saved2))
    print('Var for pixels of image: ', np.var(saved2))    
    # paths = [save_dir, real_img_dir]

    # kwargs = {'batch_size': 200, 'device': device, 'dims': 2048}
    # fid = calculate_fid_given_paths(paths=paths, **kwargs)
    # print('FID = {}'.format(fid))
    
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser('ddgan parameters')
    parser.add_argument('--seed', type=int, default=1024,
                        help='seed used for initialization')
    parser.add_argument('--compute_fid', action='store_true', default=False,
                            help='whether or not compute FID')
    parser.add_argument('--epoch_id', type=int,default=25)

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
    parser.add_argument('--exp', default='exp1', help='name of experiment')
    parser.add_argument('--real_img_dir', default='./pytorch_fid/cifar10_train_stat.npy', help='directory to real images for FID computation')
    parser.add_argument('--dataset', default='cifar10', help='name of dataset')
    parser.add_argument('--num_timesteps', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=200, help='sample generating batch size')
    
    # sampling argument
    parser.add_argument('--atol', type=float, default=1e-5, help='absolute tolerance error')
    parser.add_argument('--rtol', type=float, default=1e-5, help='absolute tolerance error')
    parser.add_argument('--method', type=str, default='euler', help='solver_method', choices=["dopri5", "dopri8", "adaptive_heun", "bosh3", "euler", "midpoint", "rk4"])
    parser.add_argument('--step_size', type=float, default=0.01, help='step_size')
    parser.add_argument('--perturb', action='store_true', default=False)
        



   
    args = parser.parse_args()
    
    sample_and_test(args)
    
   
                