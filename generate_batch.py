import os
import torch
import math
import argparse
import logging
import comet_ml
from modules import UNet  # Replace with UNet_conditional if needed
from ddpm import Diffusion
from utils import save_images
from is_connected import is_connected

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")

def main():
    parser = argparse.ArgumentParser(description="Generate image batches from a trained DDPM model.")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to the model checkpoint directory or file")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Number of images to generate in one batch")
    parser.add_argument("--number-images", type=int, default=4,
                        help="Total number of images to generate (spread over multiple batches)")
    parser.add_argument("--offline",action=argparse.BooleanOptionalAction,
                         help="Flag to use online Comet ML logging")
    parser.add_argument("--run-name", type=str, default="generate_run",
                        help="Name for the experiment run in Comet ML")
    parser.add_argument("--image-size", type=int, default=64,
                        help="Image resolution (e.g., 64)")
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"],
                        help="Device to run on")
    args = parser.parse_args()

    if args.offline or not is_connected():
        experiment = comet_ml.OfflineExperiment(project_name="denoising-diffusion", workspace="diff-model",
                                             auto_metric_logging=False, auto_output_logging=False,
                                             auto_param_logging=False,
                                              offline_directory="comet_ml")
    else:
        experiment = comet_ml.Experiment(api_key="7cLmRTEUBOdCDPIJHa0cvzGOy", project_name="denoising-diffusion", workspace="diff-model",
                                             auto_metric_logging=False, auto_output_logging=False,
                                             auto_param_logging=False)
    print(vars(args))
    experiment.set_name(args.run_name)
    experiment.log_parameters(vars(args))

    device = args.device
    model = UNet().to(device)

    # Determine the checkpoint file path
    checkpoint_path = args.checkpoint
    if os.path.isdir(checkpoint_path):
        checkpoint_file = os.path.join(checkpoint_path, "ckpt.pt")
    else:
        checkpoint_file = checkpoint_path

    if not os.path.isfile(checkpoint_file):
        raise ValueError(f"Checkpoint file not found: {checkpoint_file}")

    checkpoint = torch.load(checkpoint_file, map_location=device)
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    diffusion = Diffusion(img_size=args.image_size, device=device)

    output_dir = os.path.join("generated", args.run_name)
    os.makedirs(output_dir, exist_ok=True)

    total_images = args.number_images
    batch_size = args.batch_size
    num_batches = math.ceil(total_images / batch_size)

    logging.info(f"Generating {total_images} images in {num_batches} batch(es). Each batch = {batch_size} image(s).")

    generated_count = 0
    for batch_idx in range(num_batches):
        current_batch_size = min(batch_size, total_images - (batch_idx * batch_size))
        logging.info(f"Sampling batch {batch_idx + 1}/{num_batches} with {current_batch_size} image(s)...")

        # Sample images for this batch
        images = diffusion.sample(model, n=current_batch_size)

        # Save as a single grid (one image file) for the batch
        out_file = os.path.join(output_dir, f"batch_{batch_idx}.jpg")
        pil_img = save_images(images, out_file)
        logging.info(f"Saved batch grid to {out_file}")

        # Log the entire grid to Comet ML as one image
        experiment.log_image(pil_img, name=f"batch_{batch_idx}.jpg")

        generated_count += current_batch_size

    logging.info(f"Done! Generated {generated_count} images in total.")
    logging.info(f"All images saved to: {output_dir}")
    experiment.end()

if __name__ == "__main__":
    main()