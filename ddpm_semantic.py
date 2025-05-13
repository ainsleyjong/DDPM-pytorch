import torch
import torch.nn as nn
from torch import optim
from utils import get_data, save_images
from ddpm import Diffusion
from unet_semantic import UNetSemantic
import logging
import random

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")

def train(args):
    dataloader = get_data(args)  # Adjust get_data so it yields (image, mask)
    device = args.device
    model = UNetSemantic(in_channels=3, out_channels=3, label_nc=args.label_nc).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    mse = nn.MSELoss()
    diffusion = Diffusion(img_size=args.image_size, device=device)
    
    for epoch in range(args.epochs):
        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.to(device)  # semantic mask
            t = diffusion.sample_timesteps(images.shape[0]).to(device)
            x_t, noise = diffusion.noise_images(images, t)
            
            if random.random() < args.drop_prob:
                mask_input = torch.zeros_like(masks)
            else:
                mask_input = masks
            
            predicted_noise = model(x_t, mask_input)
            loss = mse(noise, predicted_noise)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
if __name__ == "__main__":
    class Args:
        device = "cuda"
        epochs = 100
        image_size = 64
        lr = 3e-4
        drop_prob = 0.1
        label_nc = 35  # adjust to your dataset
    train(Args())
