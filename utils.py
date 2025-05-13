import os
import torch
import torchvision
from torch.utils.data import Dataset
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

class SemanticImageFolder(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, mask_transform=None):
        """
        image_dir: Path to the folder of images.
        mask_dir: Path to the folder of semantic masks.
        transform: Transformations to apply to the image.
        mask_transform: Transformations to apply to the mask (e.g., resizing, converting to tensor).
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.mask_transform = mask_transform
        
        # Assume that images in image_dir have corresponding mask files in mask_dir.
        # The filenames (minus extension) should be identical.
        self.image_names = sorted(os.listdir(image_dir))
    
    def __len__(self):
        return len(self.image_names)
    
    def __getitem__(self, index):
        img_name = self.image_names[index]
        img_path = os.path.join(self.image_dir, img_name)
        # Here we assume masks are stored with the same name but with a .png extension
        mask_name = os.path.splitext(img_name)[0] + ".png"
        mask_path = os.path.join(self.mask_dir, mask_name)
        
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)  # Depending on how your masks are stored, you might use convert("L")
        
        if self.transform is not None:
            image = self.transform(image)
        if self.mask_transform is not None:
            mask = self.mask_transform(mask)
            
        return image, mask

def plot_images(images):
    plt.figure(figsize=(32, 32))
    plt.imshow(torch.cat([
        torch.cat([i for i in images.cpu()], dim=-1),
    ], dim=-2).permute(1, 2, 0).cpu())
    plt.show()


def save_images(images, path, **kwargs):
    grid = torchvision.utils.make_grid(images, **kwargs)
    ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
    im = Image.fromarray(ndarr)
    im.save(path)
    return im


def get_data(args):
    # Define image transformation (same as before)
    image_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(args.image_size + args.image_size // 4), 
        torchvision.transforms.RandomResizedCrop(args.image_size, scale=(0.8, 1.0)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Define mask transformation; note that you might want a simpler transform.
    mask_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(args.image_size),
        torchvision.transforms.ToTensor()
    ])

    if args.dataset == "custom_semantic":
        dataset = SemanticImageFolder(
            image_dir=args.dataset_path,
            mask_dir=args.mask_path,
            transform=image_transforms,
            mask_transform=mask_transforms
        )
    elif args.dataset == "custom":
        dataset = torchvision.datasets.ImageFolder(args.dataset_path, transform=image_transforms)

    elif args.dataset == "cifar10":
        dataset = torchvision.datasets.CIFAR10(
            root=args.dataset_path, train=False,
            download=False, transform=image_transforms
        )
    else:
        raise ValueError("Unsupported dataset type")
    
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    return dataloader



def setup_logging(run_name):
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs(os.path.join("models", run_name), exist_ok=True)
    os.makedirs(os.path.join("results", run_name), exist_ok=True)
