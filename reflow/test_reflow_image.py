# ---------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for Denoising Diffusion GAN. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------
import argparse
import math
import os

import torch
import torch.distributed as dist
from diffusers.models import AutoencoderKL
from models import create_network
from PIL import Image
from pytorch_fid.fid_score import calculate_fid_given_paths
from sampler.random_util import get_generator
from tqdm import tqdm
import torchvision
from torchdiffeq import odeint_adjoint as odeint
from torch import nn
 

ADAPTIVE_SOLVER = ["dopri5", "dopri8", "adaptive_heun", "bosh3"]
FIXER_SOLVER = ["euler", "rk4", "midpoint", "stochastic"]

class Model_(nn.Module):
    def __init__(self, model):
        super(Model_, self).__init__()
        self.model = model

    def forward(self, t, x_0):
        out = self.model(t, x_0)
        return out


def sample_from_model(model, x_1, nfe):
    t = torch.tensor([1., 0.], device="cuda")
    model_ = Model_(model)
    fake_image = odeint(model_, x_1, t, method="euler", options = {"step_size": 1./nfe})
    return fake_image

def main(args):
    device = "cuda"
    torch.manual_seed(24)
    args.layout=False

    if args.dataset == "cifar10":
        real_img_dir = "pytorch_fid/cifar10_train_stat.npy"
    elif args.dataset == "celeba_256":
        real_img_dir = "pytorch_fid/celebahq_stat.npy"
    elif args.dataset == "lsun_church":
        real_img_dir = "pytorch_fid/lsun_church_stat.npy"
    elif args.dataset == "ffhq_256":
        real_img_dir = "pytorch_fid/ffhq_stat.npy"
    elif args.dataset == "lsun_bedroom":
        real_img_dir = "pytorch_fid/lsun_bedroom_stat.npy"
    elif args.dataset in ["latent_imagenet_256", "imagenet_256"]:
        real_img_dir = "pytorch_fid/imagenet_stat.npy"
    else:
        real_img_dir = args.real_img_dir
        
    print(real_img_dir)

    to_range_0_1 = lambda x: (x + 1.0) / 2.0

    model = create_network(args).to(device)

    first_stage_model = AutoencoderKL.from_pretrained(args.pretrained_autoencoder_ckpt).to(device)
    first_stage_model.eval()

    ckpt = torch.load("./saved_info/reflow/{}/{}/model_{}.pth".format(args.dataset, args.exp, args.epoch_id))
    print("Finish loading model")
    # loading weights from ddp in single gpu
    for key in list(ckpt.keys()):
        ckpt[key[7:]] = ckpt.pop(key)
    model.load_state_dict(ckpt, strict=True)
    model.eval()

    del ckpt
    
    x_0 = torch.randn(4, 4, 32, 32).to(device)
    fake_sample = sample_from_model(model, x_0, args.nfe)[-1]
    fake_sample = first_stage_model.decode(fake_sample / args.scale_factor).sample
    fake_sample = to_range_0_1(fake_sample)
    torchvision.utils.save_image(fake_sample, './samples_{}_{}_{}_{}_{}.jpg'.format(args.dataset, args.method, args.atol, args.rtol, args.nfe))
    
    
    

    


if __name__ == "__main__":
    parser = argparse.ArgumentParser("flow-matching parameters")
    parser.add_argument(
        "--generator",
        type=str,
        default="determ",
        help="type of seed generator",
        choices=["dummy", "determ", "determ-indiv"],
    )
    parser.add_argument("--seed", type=int, default=1024, help="seed used for initialization")
    parser.add_argument("--compute_fid", action="store_true", default=False, help="whether or not compute FID")
    parser.add_argument("--use_origin_adm", action="store_true", default=False, help="whether or not compute FID")
    parser.add_argument("--compute_nfe", action="store_true", default=False, help="whether or not compute NFE")
    parser.add_argument("--measure_time", action="store_true", default=False, help="wheter or not measure time")
    parser.add_argument("--epoch_id", type=int, default=1000)

    parser.add_argument(
        "--model_type",
        type=str,
        default="adm",
        help="model_type",
        choices=["adm", "ncsn++", "ddpm++", "DiT-B/2", "DiT-L/2", "DiT-XL/2"],
    )
    parser.add_argument("--image_size", type=int, default=256, help="size of image")
    parser.add_argument("--f", type=int, default=8, help="downsample rate of input image by the autoencoder")
    parser.add_argument("--scale_factor", type=float, default=0.18215, help="size of image")
    parser.add_argument("--num_in_channels", type=int, default=4, help="in channel image")
    parser.add_argument("--num_out_channels", type=int, default=4, help="in channel image")
    parser.add_argument("--nf", type=int, default=256, help="channel of image")
    parser.add_argument("--n_sample", type=int, default=50000, help="number of sampled images")
    parser.add_argument("--centered", action="store_false", default=True, help="-1,1 scale")
    parser.add_argument("--resamp_with_conv", type=bool, default=True)
    parser.add_argument("--num_res_blocks", type=int, default=2, help="number of resnet blocks per scale")
    parser.add_argument("--num_heads", type=int, default=4, help="number of head")
    parser.add_argument("--num_head_upsample", type=int, default=-1, help="number of head upsample")
    parser.add_argument("--num_head_channels", type=int, default=-1, help="number of head channels")
    parser.add_argument(
        "--attn_resolutions", nargs="+", type=int, default=(16,8), help="resolution of applying attention"
    )
    parser.add_argument("--ch_mult", nargs="+", type=int, default=(1, 2, 2, 2), help="channel mult")
    parser.add_argument("--label_dim", type=int, default=0, help="label dimension, 0 if unconditional")
    parser.add_argument("--augment_dim", type=int, default=0, help="dimension of augmented label, 0 if not used")
    parser.add_argument("--dropout", type=float, default=0.0, help="drop-out rate")
    parser.add_argument("--num_classes", type=int, default=None, help="num classes")
    parser.add_argument(
        "--label_dropout",
        type=float,
        default=0.0,
        help="Dropout probability of class labels for classifier-free guidance",
    )
    parser.add_argument("--cfg_scale", type=float, default=1.0, help="Scale for classifier-free guidance")

    parser.add_argument("--use_scale_shift_norm", type=bool, default=True)
    parser.add_argument("--resblock_updown", type=bool, default=False)
    parser.add_argument("--use_new_attention_order", type=bool, default=False)

    parser.add_argument("--pretrained_autoencoder_ckpt", type=str, default="stabilityai/sd-vae-ft-mse")
    parser.add_argument("--output_log", type=str, default="")

    #######################################
    parser.add_argument("--exp", default="experiment_cifar_default", help="name of experiment")
    parser.add_argument(
        "--real_img_dir",
        default="./pytorch_fid/cifar10_train_stat.npy",
        help="directory to real images for FID computation",
    )
    parser.add_argument("--dataset", default="cifar10", help="name of dataset")
    parser.add_argument("--nfe", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=200, help="sample generating batch size")

    # sampling argument
    parser.add_argument("--use_karras_samplers", action="store_true", default=False)
    parser.add_argument("--atol", type=float, default=1e-5, help="absolute tolerance error")
    parser.add_argument("--rtol", type=float, default=1e-5, help="absolute tolerance error")
    parser.add_argument(
        "--method",
        type=str,
        default="dopri5",
        help="solver_method",
        choices=[
            "dopri5",
            "dopri8",
            "adaptive_heun",
            "bosh3",
            "euler",
            "midpoint",
            "rk4",
            "heun",
            "multistep",
            "stochastic",
            "dpm",
        ],
    )
    parser.add_argument("--step_size", type=float, default=0.01, help="step_size")
    parser.add_argument("--perturb", action="store_true", default=False)

    # ddp
    parser.add_argument("--num_proc_node", type=int, default=1, help="The number of nodes in multi node env.")
    parser.add_argument("--num_process_per_node", type=int, default=1, help="number of gpus")
    parser.add_argument("--node_rank", type=int, default=0, help="The index of node.")
    parser.add_argument("--local_rank", type=int, default=0, help="rank of process in the node")
    parser.add_argument("--master_address", type=str, default="127.0.0.1", help="address for master")
    parser.add_argument("--master_port", type=str, default="6000", help="port for master")

    args = parser.parse_args()
    main(args)