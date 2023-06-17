# type: ignore
# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Generate images using pretrained network pickle."""

import os
import re
from typing import List, Optional
from pathlib import Path

import click
from tqdm import tqdm
import random

import cv2
import dnnlib
import numpy as np
import PIL.Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import legacy
from facenet_pytorch import MTCNN
import torchvision.utils as torch_utils

# Own imports
from align_trans import norm_crop


# Global variables
device = torch.device('cuda')
mtcnn = MTCNN(select_largest=True, post_process=False, device=device)

def pytorch_to_cv2_RGB(images):
    
    if len(images.shape) == 3:
        images = images.unqueeze(0)
        
    swapped = images.permute(0, 2, 3, 1)
    imgs_uint8 = (swapped * 127.5 + 127.5).clamp(0, 255).to(torch.uint8)
    imgs_uint8 = imgs_uint8.numpy()
    
    return imgs_uint8


def save_image(image_arr, image_path):
    bgr_img = cv2.cvtColor(image_arr, cv2.COLOR_RGB2BGR)
    return cv2.imwrite(image_path, bgr_img)

    
    
def align_image(img):
    if len(img) != 3:
        img = img[0]
    
    if img.shape[0] == 3:
        img = img.permute(1, 2, 0)
        
    if img.dtype != torch.uint8:
        img = (img * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        
    boxes, probs, landmarks = mtcnn.detect(img, landmarks=True)
    
    if landmarks is None:
        return None

    facial5points = landmarks[0]

    warped_face = norm_crop(img.cpu().numpy(), landmark=facial5points, createEvalDB=True)
    warped_face = ((warped_face / 255.0) - 0.5) / 0.5
    warped_face = torch.from_numpy(warped_face).permute(2, 0, 1).to(img.device)
    
    return warped_face


@torch.no_grad()
def synthesize_images(G, 
                    trunc, 
                    noise, 
                    seed, 
                    amount_to_generate,
                    ID
                    ):
    
    label = torch.zeros([1, G.c_dim], device=device)
    label[:, ID] = 1.0
    
    generated_images = torch.zeros((amount_to_generate, 3, 112, 112), device="cpu")
    
    generator = np.random.RandomState(int(seed))
    
    amount_generated = 0
    while amount_generated < amount_to_generate:
        
        z = generator.randn(1, G.z_dim)
        z = torch.from_numpy(z).to(device)
            
        generated_image = G(z, label, truncation_psi=trunc, noise_mode=noise)
        aligned_image   = align_image(generated_image)
            
        if aligned_image is None:
            continue
        
        generated_images[amount_generated] = aligned_image.clone().cpu()
        amount_generated += 1
    
    return generated_images


@click.command()
@click.pass_context
@click.option('--network', 'network_pkl', help='Network pickle filename', default="network-snapshot-012096.pkl", required=True)
@click.option('--count', 'count', help='amount of images with label-idx=1', required=True, default=100)
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=1, show_default=True)
@click.option('--noise-mode', help='Noise mode', type=click.Choice(['const', 'random', 'none']), default='random', show_default=True)
@click.option('--outdir', help='Where to save te output images', type=str, required=True, metavar='DIR', default="./")
@click.option('--start-idx', 'start_idx', help='Start Index for IDs', required=True, default=-1)
@click.option('--end-idx', 'end_idx', help='End Index for IDs (Including End Index)', required=True, default=-1)
def generate_images(
    ctx: click.Context,
    count: int,
    network_pkl: str,
    truncation_psi: float,
    noise_mode: str,
    outdir: str,
    count_mixture,
    start_idx,
    end_idx
):

    os.makedirs(outdir, exist_ok=True)
    
    with open(f"{outdir}/settings_{start_idx}_{end_idx}.txt", "w") as f:
        f.write(f"count={count}\n")
        f.write(f"network_pkl={network_pkl}\n")
        f.write(f"truncation_psi={truncation_psi}\n")
        f.write(f"noise_mode={noise_mode}\n")
        f.write(f"outdir={outdir}\n")
        f.write(f"start_idx={start_idx}\n")
        f.write(f"end_idx={end_idx}\n")
        

    print('Loading networks from "%s"...' % network_pkl)
    
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore

    if start_idx == -1:
        start_idx = 0
    
    if end_idx == -1:
        end_idx = G.c_dim - 1
    
    if start_idx >= end_idx:
        raise ValueError("Start index can not be equal or greater than end index!")
    
    
    print(f"Create synthetic dataset for ID classes [{start_idx}-{end_idx}]\n")
    index_range = np.arange(start_idx, end_idx+1, dtype=int)
    
    
    for ID in tqdm(index_range):
        
        os.makedirs(outdir+"/"+f'{ID:06d}', exist_ok=True)
        
        normal_seed =  np.random.RandomState(ID*2).randint(
                            low=0, 
                            high=2**30
                        )
        
        # Generatore images
        images_normal =  synthesize_images(  
                            G, 
                            truncation_psi, 
                            noise_mode, 
                            normal_seed,
                            count,
                            ID,
                            apply_mixture=False
                         )
        images_normal = pytorch_to_cv2_RGB(images_normal)
        
        # Saving normal images
        for i in range(0, len(images_normal)):
            image = images_normal[i] 
            save_image(image, f'{outdir}/{ID:06d}/{ID:06d}_normal_{i:03d}.jpg')
        

#----------------------------------------------------------------------------

if __name__ == "__main__":
    generate_images() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
