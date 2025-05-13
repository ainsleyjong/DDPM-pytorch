'''
Based on https://github.com/dome272/Diffusion-Models-pytorch 
'''
import os
import datetime
import contextlib
import comet_ml 
import torch
import torch.nn as nn
import numpy as np
import logging
import argparse
from collections import defaultdict
from pathlib import Path
from matplotlib import pyplot as plt
from tqdm import tqdm
from torch import optim
from utils import *
from modules import UNet
from is_connected import is_connected

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")

class Diffusion:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=256, device="cuda"):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.device = device

        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)
        
    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def noise_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        Ɛ = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def sample(self, model, n):
        logging.info(f"Sampling {n} new images....")
        model.eval()
        with torch.no_grad():
            x = torch.randn((n, 3, self.img_size, self.img_size)).to(self.device)
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = model(x, t)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
        model.train()
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)
        return x


def train(args,experiment):
    scaler = torch.cuda.amp()
    
    dataloader = get_data(args)
    device = args.device
    model = UNet()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    if args.load_checkpoint is not None:
      if Path(f"{args.load_checkpoint}/ckpt.pt").is_file():
        checkpoint = torch.load(f"{args.load_checkpoint}/ckpt.pt")
        #,                map_location = lambda storage, loc: storage.cuda(torch.cuda.current_device()))
        model.load_state_dict(checkpoint["model"])
        print("loading checkpoint")
        #optimizer.load_state_dict(checkpoint["optimizer"]) # This one causes problems. Don't know why
        #scaler.load_state_dict(checkpoint["scaler"]) # No point loading it if optimizer state is not loaded
      else:
          print(f"cannot find checkpoint file {args.load_checkpoint}/ckpt.pt")

    else:
        print("checkpoint is None")
    model = model.to(device)
    mode  = torch.compile(model)
    mse   = nn.MSELoss()
    diffusion = Diffusion(img_size=args.image_size, device=device)
    
    l = len(dataloader)

    if device == "cuda":
        cm = torch.autocast(device_type='cuda', dtype=torch.float16)
    else:
        cm = contextlib.nullcontext()
    for epoch in range(args.epochs):
        metrics = defaultdict(list)
        pbar = tqdm(dataloader,disable=True if args.offline else False)
      
        for i, (images, _) in enumerate(pbar):
            images=images.to(device,non_blocking=True)
            t = diffusion.sample_timesteps(images.shape[0]).to(device)
            x_t, noise = diffusion.noise_images(images, t)
            with cm:
                predicted_noise = model(x_t, t)
                loss = mse(noise, predicted_noise)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
            metrics["mse-loss"]+=[loss]
        
        
        m=np.mean([x.item() for x in metrics["mse-loss"]])
        print(f"Epoch={epoch}, ms={m}")
        #pbar.set_postfix(MSE=m)
        experiment.log_metrics({"loss":m},epoch=epoch)    
        if(epoch%args.save_frequency==0):
            #with cm:
            sampled_images = diffusion.sample(model, n=images.shape[0])
            checkpoint={"model":model.state_dict(),"optimizer":optimizer.state_dict(),
                    "scaler":scaler.state_dict()}
            torch.save(checkpoint, os.path.join("models", args.run_name, f"ckpt.pt"))
            imgs=save_images(sampled_images, os.path.join(args.saved_images_path, args.run_name, f"{epoch}.jpg"))
            experiment.log_image(image_data=imgs,name=f"{epoch}.jpg")
    experiment.end()        


def setup():
    torch.set_float32_matmul_precision('medium')
    parser = argparse.ArgumentParser(description="Training a DDPM model to generate high-fidelity images.")
    parser.add_argument("--epochs",type=int,default=500,
                        help="Number of epochs to train the model")
    parser.add_argument("--batch-size",type=int,choices=[4,8,12,16,20,24],default=4,
                        help="Number of images to generate per batch")
    parser.add_argument("--image-size",type=int,default=64,
                        help="Image resolution (e.g., 64)")
    parser.add_argument("--dataset-path",type=str,
                        help="Path to the dataset directory (e.g., artwork-cover/resized)")
    parser.add_argument("--mask-path", type=str, default=None,
                        help="Path to the semantic mask directory")
    parser.add_argument("--dataset",type=str,choices=["custom_semantic","custom","cifar10"],default="custom",
                        help="The TYPE of dataset being used to train")
    parser.add_argument("--saved-images-path",type=str,default="./results",
                        help="The dedicated directory to store generated images (e.g., /results)")
    parser.add_argument("--save-frequency",type=int, default=10,
                        help="Number of epochs between saving model checkpoints and generated images (e.g., 10)")
    parser.add_argument("--device",type=str,choices=["cpu","cuda"],default="cuda",
                        help="Device to run on")
    parser.add_argument("--lr",type=float,default=3e-4,
                        help="Learning rate of the model (e.g., 0.0003)")
    parser.add_argument("--run-name",type=str,
                        help="Name for the experiment run in Comet ML")
    parser.add_argument("--load-checkpoint",type=str,
                        help="Path to the model checkpoint directory or file (e.g., models/Artwork-Image_1.0/ckpt.pt)")
    parser.add_argument("--offline",action=argparse.BooleanOptionalAction,
                        help="Flag to use online Comet ML logging")
    args = parser.parse_args()
    if args.offline or not is_connected() :
        experiment = comet_ml.OfflineExperiment(project_name="denoising-diffusion", workspace="diff-model",
                                             auto_metric_logging=False, auto_output_logging=False,
                                             auto_param_logging=False,
                                              offline_directory="comet_ml")
    else:
        experiment = comet_ml.Experiment(api_key="7cLmRTEUBOdCDPIJHa0cvzGOy", project_name="denoising-diffusion", workspace="diff-model",
                                             auto_metric_logging=False, auto_output_logging=False,
                                             auto_param_logging=False)
        
    print(vars(args))
    if args.run_name is None:
       now=datetime.datetime.now()
       args.run_name=f"{now.year}-{now.month}-{now.day},{now.hour}-{now.minute}"
    experiment.set_name(args.run_name)
    experiment.log_parameters(vars(args))
    models_dir=Path(os.path.join("models", args.run_name))
    os.makedirs(models_dir,exist_ok=True)
    saved_dir=Path(os.path.join(args.saved_images_path, args.run_name))
    os.makedirs(saved_dir,exist_ok=True)
    train(args,experiment)


if __name__ == '__main__':
    setup()
    # device = "cuda"
    # model = UNet().to(device)
    # ckpt = torch.load("./models/DDPM_Uncondtional/ckpt.pt")
    # model.load_state_dict(ckpt)
    # diffusion = Diffusion(img_size=64, device=device)
    # x = diffusion.sample(model, 8)
    # print(x.shape)
    # plt.figure(figsize=(32, 32))
    # plt.imshow(torch.cat([
    #     torch.cat([i for i in x.cpu()], dim=-1),
    # ], dim=-2).permute(1, 2, 0).cpu())
    # plt.show()
