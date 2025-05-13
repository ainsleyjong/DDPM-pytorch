import comet_ml 
import torch
from matplotlib import pyplot as plt
from modules import UNet, UNet_conditional
from ddpm import Diffusion

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = UNet().to(device)
    ckpt = torch.load("models/Artwork-Resized-7.0/ckpt.pt", map_location=device)
    
    if isinstance(ckpt, dict) and "model" in ckpt:
        model.load_state_dict(ckpt["model"])
    else:
        model.load_state_dict(ckpt)
    model.eval()

    diffusion = Diffusion(img_size=64, device=device)

    with torch.no_grad():
        samples = diffusion.sample(model, n=8)

    grid = torch.cat([img for img in samples.cpu()], dim=-1)
    img = grid.permute(1, 2, 0).numpy()

    plt.figure(figsize=(16, 2))
    plt.imshow(img)
    plt.axis("off")
    plt.show()
