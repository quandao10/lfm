import argparse
import os
 os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
import numpy as np
import torch.nn as nn
from torchdiffeq import odeint_adjoint as odeint
# from models.util import get_flow_model
from improved_diffusion.unet import UNetModel
import torchvision
from pytorch_fid.fid_score import calculate_fid_given_paths
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn.functional as F

ADAPTIVE_SOLVER = ["dopri5", "dopri8", "adaptive_heun", "bosh3"]
FIXER_SOLVER = ["euler", "rk4", "midpoint", "heun"]


def to_range_0_1(x): return (x + 1.) / 2.


class Model_(nn.Module):
    def __init__(self, model):
        super(Model_, self).__init__()
        self.model = model

    def forward(self, t, x_0):
        out = self.model(t, x_0)
        # return -out[:,:3,:,:] + out[:,3:,:,:]
        return out


def create_model(
    image_size,
    num_channels,
    num_res_blocks,
    class_cond,
    use_checkpoint,
    attention_resolutions,
    num_heads,
    num_heads_upsample,
    use_scale_shift_norm,
    dropout,
):
    if image_size == 256:
        channel_mult = (1, 1, 2, 2, 4, 4)
    elif image_size == 64:
        channel_mult = (1, 2, 3, 4)
    elif image_size == 32:
        channel_mult = (1, 2, 2, 2)
        num_classes = 10
    else:
        raise ValueError(f"unsupported image size: {image_size}")

    # attention_ds = []
    # for res in attention_resolutions.split(","):
    #     attention_ds.append(image_size // int(res))

    return UNetModel(
        in_channels=3,
        model_channels=num_channels,
        out_channels=3,
        num_res_blocks=num_res_blocks,
        attention_resolutions=attention_resolutions,
        dropout=dropout,
        channel_mult=channel_mult,
        num_classes=num_classes if class_cond else None,
        use_checkpoint=use_checkpoint,
        num_heads=num_heads,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
    )


parser = argparse.ArgumentParser('ddgan parameters')
parser.add_argument('--seed', type=int, default=25,
                    help='seed used for initialization')

parser.add_argument('--resume', action='store_true', default=False)

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
parser.add_argument('--ch_mult', nargs='+', type=int, default=(1, 2, 2, 2),
                    help='channel mult')
parser.add_argument('--dropout', type=float, default=0.,
                    help='drop-out rate')
parser.add_argument('--num_classes', type=int, default=None,
                    help='num classes')
parser.add_argument("--use_scale_shift_norm", type=bool, default=True)
parser.add_argument("--resblock_updown", type=bool, default=False)
parser.add_argument("--use_new_attention_order", type=bool, default=False)


#geenrator and training
parser.add_argument(
    '--exp', default='experiment_cifar_default', help='name of experiment')
parser.add_argument('--dataset', default='cifar10', help='name of dataset')

parser.add_argument('--batch_size', type=int,
                    default=250, help='input batch size')
parser.add_argument('--num_epoch', type=int, default=800)

parser.add_argument('--lr', type=float, default=5e-4, help='learning rate g')

parser.add_argument('--beta1', type=float, default=0.5,
                    help='beta1 for adam')
parser.add_argument('--beta2', type=float, default=0.9,
                    help='beta2 for adam')
parser.add_argument('--no_lr_decay', action='store_true', default=False)

parser.add_argument('--use_ema', action='store_true', default=False,
                    help='use EMA or not')
parser.add_argument('--ema_decay', type=float,
                    default=0.9999, help='decay rate for EMA')
parser.add_argument('--perturb_rate', type=float,
                    default=0.2, help='decay rate for EMA')


parser.add_argument('--save_content', action='store_true', default=False)
parser.add_argument('--save_content_every', type=int, default=10,
                    help='save content for resuming every x epochs')
parser.add_argument('--save_ckpt_every', type=int,
                    default=25, help='save ckpt every x epochs')

# ddp
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

# LTC
parser.add_argument('--coupling_dir', type=str, default=None,
                    help='Root directory for coupling data')
parser.add_argument('--coupling_model', type=str, default=None,
                    help='Coupling model to generate the couplings')
parser.add_argument('--keep_training', type=bool, default=True,
                    help='Option wheather to train a model based on generated coupling dataset')
parser.add_argument('--tau1', type=float, default=0.9)
parser.add_argument('--tau0', type=float, default=0.1)


# sampling argument
parser.add_argument('--atol', type=float, default=1e-5,
                    help='absolute tolerance error')
parser.add_argument('--rtol', type=float, default=1e-5,
                    help='absolute tolerance error')
parser.add_argument('--method', type=str, default='euler', help='solver_method',
                    choices=["dopri5", "dopri8", "adaptive_heun", "bosh3", "euler", "midpoint", "rk4"])
parser.add_argument('--step_size', type=float, default=0.01, help='step_size')
parser.add_argument('--perturb', action='store_true', default=False)
parser.add_argument("--class_cond", type=bool, default=False)
parser.add_argument('--epoch_id', type=int,default=500)


args = parser.parse_args()
args.world_size = args.num_proc_node * args.num_process_per_node
size = args.num_process_per_node

# setting back to what cifar10 model was trained
args.batch_size = 128
args.num_epoch = 600
args.image_size = 32
args.num_channels = 256
args.attn_resolution = 8
args.num_res_blocks = 2
args.lr = 1e-4
args.epoch_id = 500

device = torch.device('cuda:0')

args.layout = False

model = create_model(image_size=args.image_size,
                     num_channels=args.num_channels,
                     num_res_blocks=args.num_res_blocks,
                     class_cond=args.class_cond,
                     use_checkpoint=False,
                     attention_resolutions=args.attn_resolutions,
                     num_heads=4,
                     num_heads_upsample=-1,
                     use_scale_shift_norm=True,
                     dropout=args.dropout).to(device)

ckpt = torch.load(
    f'./saved_info/flow_matching/{args.dataset}/{args.exp}/model_{args.epoch_id}.pth', map_location=device)
print("Finish loading model")

for key in list(ckpt.keys()):
    ckpt[key[7:]] = ckpt.pop(key)
model.load_state_dict(ckpt)
model.eval()

iters_needed = 50000 // args.batch_size

## Try sampling stuffs
x_0 = torch.randn(16, 3, args.image_size, args.image_size).to(device)
fake_image = odeint(model, x_0, t, atol=1e-5, rtol=1e-5)  # just as simple as this

# Need to create 3 NFE sampling for the trained redressing
