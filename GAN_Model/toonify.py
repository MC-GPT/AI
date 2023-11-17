import os
import torch
import numpy as np
import tensorflow
from model import Generator
from util import *
import matplotlib.pyplot as plt

def save_image(image, save_path, size=None, mode='nearest', unnorm=False, title=''):
    # image is [3,h,w] or [1,3,h,w] tensor [0,1]
    if image.is_cuda:
        image = image.cpu()
    if size is not None and image.size(-1) != size:
        image = F.interpolate(image, size=(size, size), mode=mode)
    if image.dim() == 4:
        image = image[0]

    converted_image = ((image.clamp(-1, 1) + 1) / 2).cpu().permute(1, 2, 0).detach().numpy()

    plt.figure()
    plt.axis('off')
    plt.imshow(converted_image)
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)

def img_toonify(latent1,save_path):

    device = 'mps' #@param ['cuda', 'cpu','mps'] mps for arm structure

    generator1 = Generator(256, 512, 8, channel_multiplier=2).eval().to(device)
    generator2 = Generator(256, 512, 8, channel_multiplier=2).to(device).eval()

    load_model(generator1, '/Users/hoon/Desktop/AI/GAN_Model/Models/face.pt')
    mean_latent1 = load_model(generator2, '/Users/hoon/Desktop/AI/GAN_Model/Models/disney.pt')

    num_swap =  6
    alpha =  0.3

    early_alpha = 0

    with torch.no_grad():
        noise1 = [getattr(generator1.noises, f'noise_{i}') for i in range(generator1.num_layers)]
        noise2 = [getattr(generator2.noises, f'noise_{i}') for i in range(generator2.num_layers)]

        out1 = generator1.input(latent1[0])
        out2 = generator2.input(mean_latent1[0])
        out = (1-early_alpha)*out1 + early_alpha*out2

        out1, _ = generator1.conv1(out, latent1[0], noise=noise1[0])
        out2, _ = generator2.conv1(out, mean_latent1[0], noise=noise2[0])
        out = (1-early_alpha)*out1 + early_alpha*out2

        skip1 = generator1.to_rgb1(out, latent1[1])
        skip2 = generator2.to_rgb1(out, mean_latent1[1])
        skip = (1-early_alpha)*skip1 + early_alpha*skip2

        i = 2
        for conv1_1, conv1_2, noise1_1, noise1_2, to_rgb1, conv2_1, conv2_2, noise2_1, noise2_2, to_rgb2 in zip(
            generator1.convs[::2], generator1.convs[1::2], noise1[1::2], noise1[2::2], generator1.to_rgbs,
            generator2.convs[::2], generator2.convs[1::2], noise2[1::2], noise2[2::2], generator2.to_rgbs
        ):


            conv_alpha = early_alpha if i < num_swap else alpha
            out1, _ = conv1_1(out, latent1[i], noise=noise1_1)
            out2, _ = conv2_1(out, mean_latent1[i], noise=noise2_1)
            out = (1-conv_alpha)*out1 + conv_alpha*out2
            i += 1

            conv_alpha = early_alpha if i < num_swap else alpha
            out1, _ = conv1_2(out, latent1[i], noise=noise1_2)
            out2, _ = conv2_2(out, mean_latent1[i], noise=noise2_2)
            out = (1-conv_alpha)*out1 + conv_alpha*out2
            i += 1

            conv_alpha = early_alpha if i < num_swap else alpha
            skip1 = to_rgb1(out, latent1[i], skip)
            skip2 = to_rgb2(out, mean_latent1[i], skip)
            skip = (1-conv_alpha)*skip1 + conv_alpha*skip2

            i += 1

    image = skip.clamp(-1,1)
    
    save_image(image, save_path)

