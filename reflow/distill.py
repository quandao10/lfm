# ---------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for Denoising Diffusion GAN. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import os
import shutil
import argparse
from functools import partial
from omegaconf import OmegaConf
from copy import deepcopy
import torch
from torchdiffeq import odeint_adjoint as odeint
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torch.distributed as dist
from torch.multiprocessing import Process
from collections import OrderedDict
from datasets_prep import get_dataset
from models import create_network
from EMA import EMA
from tqdm import tqdm
import numpy as np
import sys
import lpips
from glob import glob
import torch.utils.data as data
from test_flow_latent import ADAPTIVE_SOLVER, FIXER_SOLVER



class NumpyDataset(data.Dataset):
    def __init__(self, root, transform=None):
        self.transform = transform
        self.root = root
        self.z1_paths = glob('{}/z1_*.npy'.format(root))
        self.z0_paths = glob('{}/z0_*.npy'.format(root))
        assert len(self.z1_paths) == len(self.z0_paths), "check again"

    def __getitem__(self, index):
        z1 = np.load('{}/z1_{}.npy'.format(self.root, index), allow_pickle=True)
        z0 = np.load('{}/z0_{}.npy'.format(self.root, index), allow_pickle=True)
        
        z1 = torch.from_numpy(z1)
        z0 = torch.from_numpy(z0)
        
        return z1, z0

    def __len__(self):
        return len(self.z1_paths)

def copy_source(file, output_dir):
    shutil.copyfile(file, os.path.join(output_dir, os.path.basename(file)))

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag

def get_weight(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    return size_all_mb


def broadcast_params(params):
    for param in params:
        dist.broadcast(param.data, src=0)

def sample_from_model(model, x_0, args):
    if args.method in ADAPTIVE_SOLVER:
        options = {
            "dtype": torch.float64,
        }
    else:
        options = {
            "step_size": args.step_size,
            "perturb": args.perturb
        }

    t = torch.tensor([1., 0.], device="cuda")

    fake_image = odeint(model,
                        x_0,
                        t,
                        method=args.method,
                        atol = args.atol,
                        rtol = args.rtol,
                        adjoint_method=args.method,
                        adjoint_atol= args.atol,
                        adjoint_rtol= args.rtol,
                        options=options,
                        adjoint_params=model.parameters(),
                        )
    
    return fake_image

# def sample_from_model(model, x_0):
#     t = torch.tensor([1., 0.], dtype=x_0.dtype, device="cuda")
#     fake_image = odeint(model, x_0, t, atol=1e-5, rtol=1e-5, adjoint_params=model.func.parameters())
#     return fake_image

def train(rank, gpu, args):
    from diffusers.models import AutoencoderKL
    torch.manual_seed(args.seed + rank)
    torch.cuda.manual_seed(args.seed + rank)
    torch.cuda.manual_seed_all(args.seed + rank)

    device = torch.device('cuda:{}'.format(gpu))

    batch_size = args.batch_size
    
    dataset = NumpyDataset(args.datadir)
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset,
                                                                    num_replicas=args.world_size,
                                                                    rank=rank)
    data_loader = torch.utils.data.DataLoader(dataset,
                                               batch_size=batch_size,
                                               shuffle=False,
                                               num_workers=4,
                                               pin_memory=True,
                                               sampler=train_sampler,
                                               drop_last = True)
    

    model = create_network(args).to(device)
    if args.use_grad_checkpointing and "DiT" in args.model_type:
        model.set_gradient_checkpointing()

    first_stage_model = AutoencoderKL.from_pretrained(args.pretrained_autoencoder_ckpt).to(device)
    first_stage_model = first_stage_model.eval()
    first_stage_model.train = False
    for param in first_stage_model.parameters():
        param.requires_grad = False
        
    
    lpips_model = lpips.LPIPS(net='vgg')
    lpips_model = lpips_model.cuda()
    for p in lpips_model.parameters():
        p.requires_grad = False

    print('AutoKL size: {:.3f}MB'.format(get_weight(first_stage_model)))
    print('FM size: {:.3f}MB'.format(get_weight(model)))

    broadcast_params(model.parameters())

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)

    if args.use_ema:
        ema = deepcopy(model).to(device)
        requires_grad(ema, False)
    else:
        ema = None

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.num_epoch, eta_min=1e-5)

    #ddp
    model = nn.parallel.DistributedDataParallel(model, device_ids=[gpu], find_unused_parameters=False)

    exp = args.exp
    parent_dir = "./saved_info/reflow/{}".format(args.dataset)

    exp_path = os.path.join(parent_dir, exp)
    if rank == 0:
        if not os.path.exists(exp_path):
            os.makedirs(exp_path)
            config_dict = vars(args)
            OmegaConf.save(config_dict, os.path.join(exp_path, "config.yaml"))
    print("Exp path:", exp_path)

    if args.resume:
        checkpoint_file = os.path.join(exp_path, 'content.pth')
        checkpoint = torch.load(checkpoint_file, map_location=device)
        init_epoch = checkpoint['epoch']
        epoch = init_epoch
        model.load_state_dict(checkpoint['model_dict'])
        if args.use_ema:
            ema.load_state_dict(checkpoint['ema_dict'])
        # load G
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        global_step = checkpoint["global_step"]

        print("=> resume checkpoint (epoch {})"
                  .format(checkpoint['epoch']))
        del checkpoint
    else:
        checkpoint_file = args.model_init
        checkpoint = torch.load(checkpoint_file, map_location=device)
        model.load_state_dict(checkpoint)
        global_step, epoch, init_epoch = 0, 0, 0
        if args.use_ema:
            update_ema(ema, model.module, decay=0)  # Ensure EMA is initialized with synced weights
            ema.eval()  # EMA model should always be in eval mode
        del checkpoint
        
    
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    
    for epoch in range(init_epoch, args.num_epoch+1):

        for iteration, (z_1, z_0) in tqdm(enumerate(data_loader)):
            
            #sample t
            z_1 = z_1.to(device)
            z_0 = z_0.to(device)
            t = torch.ones((z_0.size(0),), device=device)
            t = t.view(-1, 1, 1, 1)
            # corrected notation: 1 is real noise, 0 is real data
            v_t = (1 - t) * z_0 + (1e-5 + (1 - 1e-5) * t) * z_1
            u = (1 - 1e-5) * z_1 - z_0
            # alternative notation (similar to flow matching): 1 is data, 0 is real noise
            # v_t = (1 - (1 - 1e-5) * t) * z_0 + t * z_1
            # u = z_1 - (1 - 1e-5) * z_0
            v = model(t.squeeze(), v_t, None)
            # loss = F.mse_loss(v, u)
            
            estimated = first_stage_model.decode((-v + z_1) / args.scale_factor).sample
            target = first_stage_model.decode(z_0 / args.scale_factor).sample
            
            
            loss = lpips_model(estimated, target)
            loss = torch.mean(loss.reshape(loss.shape[0], -1))
            
            # print(loss.shape)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if args.use_ema:
                update_ema(ema, model.module)
            
            global_step += 1
            if iteration % 100 == 0:
                if rank == 0:
                    print('epoch {} iteration{}, Loss: {}'.format(epoch,iteration, loss.item()))

        if not args.no_lr_decay:
            scheduler.step()

        if rank == 0:
            if epoch % args.plot_every == 0:

                with torch.no_grad():
                    rand = torch.randn_like(z_0)[:4]
                
                    # sample_model = partial(model, y=None)
                    fake_sample = sample_from_model(model, rand, args)[-1]
                    fake_image = first_stage_model.decode(fake_sample / args.scale_factor).sample
                torchvision.utils.save_image(fake_image, os.path.join(exp_path, 'image_epoch_{}.png'.format(epoch)), normalize=True, value_range=(-1, 1))
                print("Finish sampling")

            if args.save_content:
                if epoch % args.save_content_every == 0:
                    print('Saving content.')
                    content = {'epoch': epoch + 1, 
                               'global_step': global_step, 
                               'args': args,
                               "ema_dict": ema.state_dict() if args.use_ema else None,
                               'model_dict': model.state_dict(), 'optimizer': optimizer.state_dict(),
                               'scheduler': scheduler.state_dict()}

                    torch.save(content, os.path.join(exp_path, 'content.pth'))

            if epoch % args.save_ckpt_every == 0:
                if args.use_ema:
                    torch.save(ema.state_dict(), os.path.join(exp_path, 'ema_{}.pth'.format(epoch)))
                torch.save(model.state_dict(), os.path.join(exp_path, 'model_{}.pth'.format(epoch)))
    



def init_processes(rank, size, fn, args):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = args.master_address
    os.environ['MASTER_PORT'] = args.master_port
    torch.cuda.set_device(args.local_rank)
    gpu = args.local_rank
    dist.init_process_group(backend='nccl', init_method='env://', rank=rank, world_size=size)
    fn(rank, gpu, args)
    dist.barrier()
    cleanup()


def cleanup():
    dist.destroy_process_group()


#%%
if __name__ == '__main__':
    parser = argparse.ArgumentParser('ddgan parameters')
    parser.add_argument('--seed', type=int, default=1024,
                        help='seed used for initialization')

    parser.add_argument('--resume', action='store_true', default=False)
    parser.add_argument('--precomputed', action='store_true', default=False)
    parser.add_argument('--model_ckpt', type=str, default=None,
                            help="Model ckpt to init from")

    parser.add_argument('--model_type', type=str, default="adm",
                            help='model_type', choices=['adm', 'ncsn++', 'ddpm++', 'DiT-B/2', 'DiT-L/2', 'DiT-L/4', 'DiT-XL/2'])
    parser.add_argument('--image_size', type=int, default=32,
                            help='size of image')
    parser.add_argument('--f', type=int, default=8,
                            help='downsample rate of input image by the autoencoder')
    parser.add_argument('--scale_factor', type=float, default=0.18215,
                            help='size of image')
    parser.add_argument('--num_in_channels', type=int, default=4,
                            help='in channel image')
    parser.add_argument('--num_out_channels', type=int, default=4,
                            help='in channel image')
    parser.add_argument('--nf', type=int, default=256,
                            help='channel of model')
    parser.add_argument('--num_res_blocks', type=int, default=2,
                            help='number of resnet blocks per scale')
    parser.add_argument('--attn_resolutions', nargs='+', type=int, default=(16,8),
                            help='resolution of applying attention')
    parser.add_argument('--ch_mult', nargs='+', type=int, default=(1,2,2,2),
                            help='channel mult')
    parser.add_argument('--dropout', type=float, default=0.,
                            help='drop-out rate')
    parser.add_argument('--label_dim', type=int, default=0,
                            help='label dimension, 0 if unconditional')
    parser.add_argument('--augment_dim', type=int, default=0,
                            help='dimension of augmented label, 0 if not used')
    parser.add_argument('--num_classes', type=int, default=None,
                            help='num classes')
    parser.add_argument('--label_dropout', type=float, default=0.,
                            help='Dropout probability of class labels for classifier-free guidance')

    # Original ADM
    parser.add_argument('--layout', action='store_true')
    parser.add_argument('--use_origin_adm', action='store_true')
    parser.add_argument("--use_scale_shift_norm", type=bool, default=True)
    parser.add_argument("--resblock_updown", type=bool, default=False)
    parser.add_argument("--use_new_attention_order", type=bool, default=False)
    parser.add_argument('--centered', action='store_false', default=True,
                            help='-1,1 scale')
    parser.add_argument("--resamp_with_conv", type=bool, default=True)
    parser.add_argument('--num_heads', type=int, default=4,
                            help='number of head')
    parser.add_argument('--num_head_upsample', type=int, default=-1,
                            help='number of head upsample')
    parser.add_argument('--num_head_channels', type=int, default=-1,
                            help='number of head channels')

    parser.add_argument('--pretrained_autoencoder_ckpt', type=str, default="stabilityai/sd-vae-ft-mse")

    # training
    parser.add_argument('--exp', default='experiment_cifar_default', help='name of experiment')
    parser.add_argument('--model_init', default='experiment_cifar_default', help='name of experiment')
    
    parser.add_argument('--dataset', default='cifar10', help='name of dataset')
    parser.add_argument('--datadir', default='./dataset')
    parser.add_argument('--num_timesteps', type=int, default=200)
    parser.add_argument('--use_bf16', action='store_true', default=False)
    parser.add_argument('--use_grad_checkpointing', action='store_true', default=False,
        help="Enable gradient checkpointing for mem saving")

    parser.add_argument('--batch_size', type=int, default=128, help='input batch size')
    parser.add_argument('--num_epoch', type=int, default=1200)

    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate g')

    parser.add_argument('--beta1', type=float, default=0.5,
                            help='beta1 for adam')
    parser.add_argument('--beta2', type=float, default=0.9,
                            help='beta2 for adam')
    parser.add_argument('--no_lr_decay',action='store_true', default=False)

    parser.add_argument('--use_ema', action='store_true', default=False,
                            help='use EMA or not')
    parser.add_argument('--ema_decay', type=float, default=0.9999, help='decay rate for EMA')


    parser.add_argument('--save_content', action='store_true', default=False)
    parser.add_argument('--save_content_every', type=int, default=10, help='save content for resuming every x epochs')
    parser.add_argument('--save_ckpt_every', type=int, default=25, help='save ckpt every x epochs')
    parser.add_argument('--plot_every', type=int, default=1, help='plot every x epochs')
    
    
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

    ###ddp
    parser.add_argument('--num_proc_node', type=int, default=1,
                        help='The number of nodes in multi node env.')
    parser.add_argument('--num_process_per_node', type=int, default=1,
                        help='number of gpus')
    parser.add_argument('--node_rank', type=int, default=0,
                        help='The index of node.')
    parser.add_argument('--local_rank', type=int, default=0,
                        help='rank of process in the node')
    parser.add_argument('--master_address', type=str, default='127.0.0.1',
                        help='address for master')
    parser.add_argument('--master_port', type=str, default='6000',
                        help='port for master')

    args = parser.parse_args()
    args.world_size = args.num_proc_node * args.num_process_per_node
    size = args.num_process_per_node

    if size > 1:
        processes = []
        for rank in range(size):
            args.local_rank = rank
            global_rank = rank + args.node_rank * args.num_process_per_node
            global_size = args.num_proc_node * args.num_process_per_node
            args.global_rank = global_rank
            print('Node rank %d, local proc %d, global proc %d' % (args.node_rank, rank, global_rank))
            p = Process(target=init_processes, args=(global_rank, global_size, train, args))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
    else:
        print('starting in debug mode')

        init_processes(0, size, train, args)