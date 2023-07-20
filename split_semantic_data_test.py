import os
import shutil
import torch
import torch.nn as nn
import torchvision
from datasets_prep.ade20k import ADE20kTrain, ADE20kValidation
from datasets_prep.celeb_mask import CelebAMaskTrain, CelebAMaskValidation

colorize = torch.randn(3, 19, 1, 1)

def to_rgb(x):
    x = x.float()
    x = nn.functional.conv2d(x, weight=colorize.to(x))
    x = 2. * (x - x.min()) / (x.max() - x.min()) - 1.
    return x

def run(dataset):
    save_path = "fid_dataset/{}".format(dataset)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    os.makedirs(os.path.join(save_path, "mask"), exist_ok=True)
    os.makedirs(os.path.join(save_path, "image"), exist_ok=True)

    if dataset == "ade20k":
        dataset = ADE20kValidation(size=256, crop_size=256, random_crop=False)
        num_cls = 151
    else:
        dataset = CelebAMaskValidation(size=256, crop_size=256)
        num_cls = 19
        
    data_loader = torch.utils.data.DataLoader(dataset,
                                            batch_size=64,
                                            shuffle=False,
                                            pin_memory=True,
                                            drop_last = False)
    global_step = 0
    for iteration, (image, segmentation) in enumerate(data_loader):
        segmentation = torch.nn.functional.one_hot(segmentation, num_cls).permute(0, 3, 1, 2)
        seg = segmentation.float()
        for i in range(image.size(0)):
            seg = to_rgb(segmentation)
            torchvision.utils.save_image(seg[i], os.path.join(save_path, "mask", 'seg_{}.png'.format(global_step)), normalize=True)
            torchvision.utils.save_image(image[i], os.path.join(save_path, "image", 'image_{}_gt.png'.format(global_step)), normalize=True)
            global_step += 1
            
run("a")