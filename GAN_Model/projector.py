import torch
import os
from pathlib import Path
import lpips
from model import Generator
from op import fused_leaky_relu
import numpy as np
from util import *
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def noise_regularize(noises):
    loss = 0

    for noise in noises:
        size = noise.shape[2]

        while True:
            loss = (
                    loss
                    + (noise * torch.roll(noise, shifts=1, dims=3)).mean().pow(2)
                    + (noise * torch.roll(noise, shifts=1, dims=2)).mean().pow(2)
            )

            if size <= 8:
                break
            noise = noise.reshape([1, 1, size // 2, 2, size // 2, 2])
            noise = noise.mean([3, 5])
            size //= 2

    return loss


def noise_normalize_(noises):
    for noise in noises:
        mean = noise.mean()
        std = noise.std()

        noise.data.add_(-mean).div_(std)


def get_lr(t, initial_lr, rampdown=0.25, rampup=0.05):
    lr_ramp = min(1, (1 - t) / rampdown)
    lr_ramp = 0.5 - 0.5 * math.cos(lr_ramp * math.pi)
    lr_ramp = lr_ramp * min(1, t / rampup)

    return initial_lr * lr_ramp


def latent_noise(latent, strength):
    noise = torch.randn_like(latent) * strength

    return latent + noise


def make_image(tensor):
    return (
            tensor.detach()
            .clamp_(min=-1, max=1)
            .add(1)
            .div_(2)
            .mul(255)
            .type(torch.uint8)
            .permute(0, 2, 3, 1)
            .to('cpu')
            .numpy()
    )


def image_inversion(input_path, output_path):
    device = 'mps'

    out_dir = './inversion_codes'
    os.makedirs(out_dir, exist_ok=True)

    face_img = input_path

    n_mean_latent = 10000  # You can adjust this value
    resize = 256

    transform = transforms.Compose(
        [
            transforms.Resize(resize),
            transforms.CenterCrop(resize),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

    imgs = []

    img = transform(Image.open(face_img).convert('RGB'))
    imgs.append(img)

    imgs = torch.stack(imgs, 0).to(device)

    g_ema = Generator(256, 512, 8)
    g_ema.load_state_dict(torch.load('/Users/hoon/Desktop/AI/GAN_Model/Models/face.pt', map_location='mps')['g_ema'], strict=False)
    g_ema = g_ema.to(device).eval()

    with torch.no_grad():
        latent_mean = g_ema.mean_latent(50000)
        latent_in = list2style(latent_mean)

    # get gaussian stats
    if not os.path.isfile('inversion_stats.npz'):
        with torch.no_grad():
            source = list2style(g_ema.get_latent(torch.randn([10000, 512], device='mps'))).cpu().numpy()
            gt_mean = source.mean(0)
            gt_cov = np.cov(source, rowvar=False)

        # We show that style space follows gaussian distribution
        # An extension from this work https://arxiv.org/abs/2009.06529
        np.savez('inversion_stats.npz', mean=gt_mean, cov=gt_cov)

    data = np.load('inversion_stats.npz')
    gt_mean = torch.tensor(data['mean'], device='mps').view(1, -1).float()
    gt_cov_inv = torch.tensor(data['cov'], dtype=torch.float32)

    # Only take diagonals
    mask = torch.eye(*gt_cov_inv.size())
    gt_cov_inv = torch.inverse(gt_cov_inv * mask).float()

    percept = lpips.LPIPS(net='vgg', spatial=True).to(device)
    latent_in.requires_grad = True

    optimizer = torch.optim.Adam([latent_in], lr=0.05, betas=(0.9, 0.999))

    pbar = tqdm(range(3000))
    latent_path = []

    def gaussian_loss(v):
        # [B, 9088]
        loss = (v - gt_mean.to('mps')) @ gt_cov_inv.to('mps') @ (v - gt_mean.to('mps')).transpose(1, 0)
        return loss.mean()

    for i in pbar:
        t = i / 3000
        lr = get_lr(t, 0.05)
        latent_n = latent_in

        img_gen, _ = g_ema(style2list(latent_n))

        batch, channel, height, width = img_gen.shape

        if height > 256:
            img_gen = F.interpolate(img_gen, size=(256, 256), mode='area')

        p_loss = 20 * percept(img_gen, imgs).mean()
        mse_loss = 1 * F.mse_loss(img_gen, imgs)
        g_loss = 1e-3 * gaussian_loss(latent_n.to('mps'))

        loss = p_loss + mse_loss + g_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            latent_path.append(latent_in.detach().clone())

        pbar.set_description(
            (
                    f'perceptual: {p_loss.item():.4f};'
                    f' mse: {mse_loss.item():.4f}; gaussian: {g_loss.item():.4f} lr: {lr:.4f}'
            )
        )

    result_file = {}

    latent_path.append(latent_in.detach().clone())
    img_gen, _ = g_ema(style2list(latent_path[-1]))

    filename = f'{out_dir}/{os.path.splitext(os.path.basename(input_path))[0]}.pt'

    img_ar = make_image(img_gen)

    real_file = str(input_path)
    '''
    for i, input_name in enumerate(real_file):
        result_file['latent'] = latent_in[i]
    '''
    torch.save({'latent': latent_in}, output_path)
