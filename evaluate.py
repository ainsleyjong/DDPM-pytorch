import argparse
import glob
import torch
from torchvision.io import read_image
import torchvision.transforms.functional as TF
from torchmetrics.image.fid import FrechetInceptionDistance

def load(img_path, image_size):
    img = read_image(img_path)
    img = TF.resize(img, [image_size]*2)

    # if it’s grayscale, replicate to 3 channels
    if img.shape[0] == 1:
        img = img.repeat(3, 1, 1)

    assert img.dtype == torch.uint8
    return img

def main():
    parser = argparse.ArgumentParser("Compute FID")
    parser.add_argument("--real-path", type=str, required=True, help="Folder of real images")
    parser.add_argument("--image-path", type=str, required=True, help="Folder of generated images")
    parser.add_argument("--image-size", type=int, default=64, help="Resize H x W for FID")
    parser.add_argument("--device", type=str, default="cuda", help="cpu or cuda")
    args = parser.parse_args()

    fid = FrechetInceptionDistance(feature=64).to(args.device)

    # real images
    for p in glob.glob(f"{args.real_path}/*"):
        img = load(p, args.image_size).unsqueeze(0).to(args.device)
        fid.update(img, real=True)

    # generated images
    for p in glob.glob(f"{args.image_path}/*"):
        img = load(p, args.image_size).unsqueeze(0).to(args.device)
        fid.update(img, real=False)

    score = fid.compute()
    print(f"FID: {score:.3f}")

if __name__=="__main__":
    main()
