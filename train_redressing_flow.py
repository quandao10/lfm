# ---------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for Denoising Diffusion GAN. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import argparse
from copy import deepcopy
import torch
import numpy as np
from torchdiffeq import odeint_adjoint as odeint
import os
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from datasets_prep import get_dataset
from models.util import get_flow_model
from torch.multiprocessing import Process
import torch.distributed as dist
import shutil
from glob import glob
from test_noise import sample_from_noise
from tqdm import tqdm

class Model_(nn.Module):
    def __init__(self, model):
        super(Model_, self).__init__()
        self.model = model

    def forward(self, t, x_0):
        return self.model(t, x_0)

def copy_source(file, output_dir):
    shutil.copyfile(file, os.path.join(output_dir, os.path.basename(file)))
            
def broadcast_params(params):
    for param in params:
        dist.broadcast(param.data, src=0)

def sample_from_model(model, x_0, nfes = [1, 5, 10, 25, 50, 100]):
    t = torch.tensor([1., 0.], device="cuda")
    model_ = Model_(model)
    
    fake_images = {
    }
    for nfe in nfes:
        fake_image = odeint(model_, x_0, t, method="euler", options = {
                "step_size": 1./nfe,
            })[-1]
        fake_images[nfe] = fake_image
    return fake_images

def create_coupling_dataset(args, rank, gpu, device):
    to_range_0_1 = lambda x: (x + 1.) / 2.

    model =  get_flow_model(args).to(device)
    ckpt = torch.load(args.coupling_model, map_location=device)
    print("Finish loading model")
    #loading weights from ddp in single gpu
    for key in list(ckpt.keys()):
        ckpt[key[7:]] = ckpt.pop(key)
    model.load_state_dict(ckpt)
    model.eval()
            
    save_dir = "./generated_couplings/{}/seed_{}".format(args.dataset, args.seed)
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    dataset = get_dataset(args)
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset,
                                                                    num_replicas=args.world_size,
                                                                    rank=rank)
    data_loader = torch.utils.data.DataLoader(dataset,
                                               batch_size=args.batch_size,
                                               shuffle=False,
                                               num_workers=4,
                                               pin_memory=True,
                                               sampler=train_sampler,
                                               drop_last = True)
    
    model = nn.parallel.DistributedDataParallel(model, device_ids=[gpu],find_unused_parameters=True)
    global_iter = 0
    for i, (image, _) in enumerate(tqdm(data_loader)):
        image = image.to(device)
        with torch.no_grad():
            noise = sample_from_noise(model, image, args)[-1]
            for j, (x0, x1) in enumerate(zip(image, noise)):
                index = (i * args.batch_size + j)*args.num_process_per_node + rank
                x0 = x0.to("cpu").numpy()
                x1 = x1.to("cpu").numpy()
                np.save(os.path.join(save_dir, "image_{}.npy".format(index)), x0)
                np.save(os.path.join(save_dir, "noise_{}.npy".format(index)), x1)
            print('generating batch ', i)
    return save_dir

class CouplingDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform=None):
        self.train = train
        self.root = root
        self.transform = transform
        self.image_paths = glob(f'{root}/image_*.npy')
        self.noise_paths = glob(f'{root}/noise_*.npy')
        assert len(self.image_paths) == len(self.noise_paths), 'size not equal'


    def __getitem__(self, index):
        image = np.load(os.path.join(self.root, "image_{}.npy".format(index)))
        noise = np.load(os.path.join(self.root, "noise_{}.npy".format(index)))
        image = torch.from_numpy(image)
        noise = torch.from_numpy(noise)

        if self.transform is not None:
            x = self.transform(x)

        return noise, image

    def __len__(self):
        return len(self.image_paths)




def train(rank, gpu, args):
    
    from EMA import EMA
        
    torch.manual_seed(args.seed + rank)
    torch.cuda.manual_seed(args.seed + rank)
    torch.cuda.manual_seed_all(args.seed + rank)
    device = torch.device('cuda:{}'.format(gpu))
    
    
    batch_size = args.batch_size
    
    root = create_coupling_dataset(args, rank, gpu, device) if args.coupling_dir is None else args.coupling_dir
    dataset = CouplingDataset(root)
    if args.keep_training == False: return

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

    model = get_flow_model(args).to(device)
    deep_model = deepcopy(model).to(device)
    deep_model.eval()
    for p in deep_model.parameters():
        p.requires_grad_(False)
    broadcast_params(model.parameters())
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas = (args.beta1, args.beta2))
    
    if args.use_ema:
        optimizer = EMA(optimizer, ema_decay=args.ema_decay)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.num_epoch, eta_min=1e-5)
    
    #ddp
    model = nn.parallel.DistributedDataParallel(model, device_ids=[gpu], find_unused_parameters=False)

    exp = args.exp
    parent_dir = "./saved_info/flow_matching/{}".format(args.dataset)

    exp_path = os.path.join(parent_dir, exp)
    if rank == 0:
        if not os.path.exists(exp_path):
            os.makedirs(exp_path)
    
    if args.resume:
        checkpoint_file = os.path.join(exp_path, 'content.pth')
        checkpoint = torch.load(checkpoint_file, map_location=device)
        init_epoch = checkpoint['epoch']
        epoch = init_epoch
        model.load_state_dict(checkpoint['model_dict'])
        # load G
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        
        print("=> loaded checkpoint (epoch {})"
                  .format(checkpoint['epoch']))
    else:
        ckpt = torch.load(args.coupling_model, map_location=device)
        print("Finish loading model")
        #loading weights from ddp in single gpu
        # for key in list(ckpt.keys()):
        #     ckpt[key[7:]] = ckpt.pop(key)
        model.load_state_dict(ckpt)
        epoch, init_epoch = 0, 0
        del ckpt
    
    
    for epoch in range(init_epoch, args.num_epoch+1):
        train_sampler.set_epoch(epoch)
       
        for iteration, (x_1, x_0) in enumerate(data_loader):
            x_1 = x_1.to(device, non_blocking=True)
            x_0 = x_0.to(device, non_blocking=True)
            model.zero_grad()
            
            x_r1 = x_1 - (1-args.tau1) * deep_model(torch.ones((x_1.size(0),), device = device), x_1)
            x_r0 = x_0 + args.tau0 * deep_model(torch.zeros((x_1.size(0),), device = device), x_0)
            u = (x_r1 - x_r0)/(args.tau1 - args.tau0)
            x_r1 = x_r1 + (1-args.tau1) * u
            x_r0 = x_r0 - args.tau0 * u
            
            #sample t
            t = args.tau0+(args.tau1-args.tau0)*torch.rand((x_1.size(0),) , device=device)
            t = t.view(-1, 1, 1, 1)
            v_t = t * x_r1 + (1-t) * x_r0
            out = model(t.squeeze(), v_t)
            loss = F.mse_loss(out, u)
            loss.backward()
            optimizer.step()
            
            if iteration % 100 == 0:
                if rank == 0:
                    print('epoch {} iteration{}, Loss: {}'.format(epoch,iteration, loss.item()))
        
        if not args.no_lr_decay:
            scheduler.step()
        
        if rank == 0:
            rand = torch.randn_like(x_1)[:16]
            fake_samples = sample_from_model(model, rand)
            for nfe in fake_samples.keys():
                torchvision.utils.save_image(fake_samples[nfe], os.path.join(exp_path, 'sample_epoch_{}_{}.png'.format(epoch, nfe)), normalize=True)
            
            if args.save_content:
                if epoch % args.save_content_every == 0:
                    print('Saving content.')
                    content = {'epoch': epoch + 1, 'args': args,
                               'model_dict': model.state_dict(), 'optimizer': optimizer.state_dict(),
                               'scheduler': scheduler.state_dict()}
                    
                    torch.save(content, os.path.join(exp_path, 'content.pth'))
                
            if epoch % args.save_ckpt_every == 0:
                if args.use_ema:
                    optimizer.swap_parameters_with_ema(store_params_in_ema=True)
                    
                torch.save(model.state_dict(), os.path.join(exp_path, 'model_{}.pth'.format(epoch)))
                if args.use_ema:
                    optimizer.swap_parameters_with_ema(store_params_in_ema=True)
            


def init_processes(rank, size, fn, args):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = args.master_address
    os.environ['MASTER_PORT'] = '6021'
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
    parser.add_argument('--seed', type=int, default=25,
                        help='seed used for initialization')
    
    parser.add_argument('--resume', action='store_true',default=False)
    
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
    
    
    #geenrator and training
    parser.add_argument('--exp', default='experiment_cifar_default', help='name of experiment')
    parser.add_argument('--dataset', default='cifar10', help='name of dataset')

    parser.add_argument('--batch_size', type=int, default=250, help='input batch size')
    parser.add_argument('--num_epoch', type=int, default=800)

    parser.add_argument('--lr', type=float, default=5e-4, help='learning rate g')
    
    parser.add_argument('--beta1', type=float, default=0.5,
                            help='beta1 for adam')
    parser.add_argument('--beta2', type=float, default=0.9,
                            help='beta2 for adam')
    parser.add_argument('--no_lr_decay',action='store_true', default=False)
    
    parser.add_argument('--use_ema', action='store_true', default=False,
                            help='use EMA or not')
    parser.add_argument('--ema_decay', type=float, default=0.9999, help='decay rate for EMA')
    parser.add_argument('--perturb_rate', type=float, default=0.2, help='decay rate for EMA')
    

    parser.add_argument('--save_content', action='store_true',default=False)
    parser.add_argument('--save_content_every', type=int, default=10, help='save content for resuming every x epochs')
    parser.add_argument('--save_ckpt_every', type=int, default=25, help='save ckpt every x epochs')
   
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

    ### LTC
    parser.add_argument('--coupling_dir', type=str, default=None,
                        help='Root directory for coupling data')
    parser.add_argument('--coupling_model', type=str, default=None,
                        help='Coupling model to generate the couplings')
    parser.add_argument('--keep_training', type=bool, default=True,
                        help='Option wheather to train a model based on generated coupling dataset')
    parser.add_argument('--tau1', type=float, default=0.9)
    parser.add_argument('--tau0', type=float, default=0.1)
    
    
    # sampling argument
    parser.add_argument('--atol', type=float, default=1e-5, help='absolute tolerance error')
    parser.add_argument('--rtol', type=float, default=1e-5, help='absolute tolerance error')
    parser.add_argument('--method', type=str, default='euler', help='solver_method', choices=["dopri5", "dopri8", "adaptive_heun", "bosh3", "euler", "midpoint", "rk4"])
    parser.add_argument('--step_size', type=float, default=0.01, help='step_size')
    parser.add_argument('--perturb', action='store_true', default=False)
        

   
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