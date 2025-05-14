import torch
from matplotlib import pyplot as plt
from modules import UNet, UNet_conditional
from ddpm import Diffusion
from ddpm_conditional import Diffusion as DiffusionCond

if __name__ == "__main__":
    n = 10
    device = "cuda" if torch.cuda.is_available() else "cpu"

    #-----Unconditional-----#
    # model = UNet().to(device)
    # ckpt = torch.load("models/Artwork-Resized-7.0/ckpt.pt", map_location=device)
    
    # if isinstance(ckpt, dict) and "model" in ckpt:
    #     model.load_state_dict(ckpt["model"])
    # else:
    #     model.load_state_dict(ckpt)
    # model.eval()

    # diffusion = Diffusion(img_size=64, device=device)

    # with torch.no_grad():
    #     samples = diffusion.sample(model, n=8)

    # grid = torch.cat([img for img in samples.cpu()], dim=-1)
    # img = grid.permute(1, 2, 0).numpy()

    # plt.figure(figsize=(16, 2))
    # plt.imshow(img)
    # plt.axis("off")
    # plt.show()
    #-----     End     -----#


    #----- Conditional -----#
    model = UNet_conditional(num_classes=10).to(device)
    ckpt = torch.load("models/conditional_ema_ckpt.pt", map_location=device)
    model.load_state_dict(ckpt)
    model.eval()

    diffusion = DiffusionCond(img_size=64, device=device)
    y = torch.Tensor([6] * n).long().to(device)  # Class 6, repeated 'n' times
    with torch.no_grad():
        x = diffusion.sample(model, n, y, cfg_scale=3)

    # Convert images to numpy and make a grid
    grid = torch.cat([img for img in x.cpu()], dim=-1)
    img = grid.permute(1, 2, 0).numpy()

    plt.figure(figsize=(16, 2))
    plt.imshow(img)
    plt.axis("off")
    plt.show()
    #-----     End     -----# 
