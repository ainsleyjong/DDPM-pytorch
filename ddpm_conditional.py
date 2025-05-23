import os
import copy
import numpy as np
import comet_ml
import torch
import torch.nn as nn
import argparse
import logging
from pathlib import Path
from tqdm import tqdm
from torch import optim
from utils import *
from modules import UNet_conditional, EMA
from torch.utils.tensorboard import SummaryWriter
from is_connected import is_connected

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")

class Diffusion:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=256, device="cuda"):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end

        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

        self.img_size = img_size
        self.device = device

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def noise_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        Ɛ = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def sample(self, model, n, labels, cfg_scale=3):
        logging.info(f"Sampling {n} new images....")
        model.eval()
        with torch.no_grad():
            x = torch.randn((n, 3, self.img_size, self.img_size)).to(self.device)
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = model(x, t, labels)
                if cfg_scale > 0:
                    uncond_predicted_noise = model(x, t, None)
                    predicted_noise = torch.lerp(uncond_predicted_noise, predicted_noise, cfg_scale)
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


def train(args):
    setup_logging(args.run_name)
    device = args.device
    dataloader = get_data(args)
    model = UNet_conditional(num_classes=args.num_classes).to(device)
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
    mse = nn.MSELoss()
    diffusion = Diffusion(img_size=args.image_size, device=device)
    logger = SummaryWriter(os.path.join("runs", args.run_name))
    l = len(dataloader)
    ema = EMA(0.995)
    ema_model = copy.deepcopy(model).eval().requires_grad_(False)

    for epoch in range(args.epochs):
        logging.info(f"Starting epoch {epoch}:")
        pbar = tqdm(dataloader)
        for i, (images, labels) in enumerate(pbar):
            images = images.to(device)
            labels = labels.to(device)
            t = diffusion.sample_timesteps(images.shape[0]).to(device)
            x_t, noise = diffusion.noise_images(images, t)
            if np.random.random() < 0.1:
                labels = None
            predicted_noise = model(x_t, t, labels)
            loss = mse(noise, predicted_noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ema.step_ema(ema_model, model)

            pbar.set_postfix(MSE=loss.item())
            logger.add_scalar("MSE", loss.item(), global_step=epoch * l + i)

        if epoch % 10 == 0:
            labels = torch.arange(10).long().to(device)
            sampled_images = diffusion.sample(model, n=len(labels), labels=labels)
            ema_sampled_images = diffusion.sample(ema_model, n=len(labels), labels=labels)
            plot_images(sampled_images)
            save_images(sampled_images, os.path.join("results", args.run_name, f"{epoch}.jpg"))
            save_images(ema_sampled_images, os.path.join("results", args.run_name, f"{epoch}_ema.jpg"))
            torch.save(model.state_dict(), os.path.join("models", args.run_name, f"ckpt.pt"))
            torch.save(ema_model.state_dict(), os.path.join("models", args.run_name, f"ema_ckpt.pt"))
            torch.save(optimizer.state_dict(), os.path.join("models", args.run_name, f"optim.pt"))


def setup():
    torch.set_float32_matmul_precision('medium')
    parser = argparse.ArgumentParser(description="Training a DDPM model to generate images based on conditional labels.")
    parser.add_argument("--epochs",type=int,default=500,
                        help="Number of epochs to train the model")
    parser.add_argument("--batch-size",type=int,choices=[4,8,12,16,20,24],default=4,
                        help="Number of images to generate per batch")
    parser.add_argument("--image-size",type=int,default=64,
                        help="Image resolution (e.g., 64)")
    parser.add_argument("--num-classes",type=int,default=10,
                        help="Number of classes (e.g., 10)")
    parser.add_argument("--dataset-path",type=str,
                        help="Path to the dataset directory (e.g., artwork-cover/resized)")
    parser.add_argument("--dataset",type=str,choices=["custom","cifar10"],default="custom",
                        help="The TYPE of dataset being used to train")
    parser.add_argument("--device",type=str,choices=["cpu","cuda"],default="cuda",
                        help="Device to run on")
    parser.add_argument("--lr",type=float,default=3e-4,
                        help="Learning rate of the model (e.g., 0.0003)")
    parser.add_argument("--run-name",type=str,default="DDPM_Conditional",
                        help="Name for the experiment run in Comet ML")
    parser.add_argument("--load-checkpoint",type=str,
                        help="Path to the model checkpoint directory or file (e.g., models/conditional_ema_ckpt.pt)")
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
    experiment.set_name(args.run_name)
    experiment.log_parameters(vars(args))
    train(args)


if __name__ == '__main__':
    setup()
