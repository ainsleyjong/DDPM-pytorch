# 🌀 DDPM-Based Models for Conditional Generation Tasks

This repository is an adaptation of the original [Diffusion-Models-pytorch](https://github.com/dome272/Diffusion-Models-pytorch) implementation of Denoising Diffusion Probabilistic Models (DDPM). It supports unconditional, label-conditional, and an experimental form of semantic-conditional image generation.

## 📁 Project Structure

- `ddpm.py`: Standard unconditional DDPM training and sampling.
- `ddpm_conditional.py`: Implements class-conditional DDPM.
- `ddpm_semantic.py`: **(Work in progress)** Experimental script for semantic-conditional DDPM.
- `modules.py`: Base model architecture (UNet) and diffusion logic.
- `modules_semantic.py`: **(In development)** Planned extensions for supporting semantic inputs in the model.
- `unet_semantic.py`: **(In development)** Modified UNet to support semantic features.
- `evaluate.py`: Script to evaluate generated outputs, including visual or metric-based assessments.
- `generate_batch.py`: Generates batches of images for evaluation or preview.
- `preview.py`: Script to preview generated samples from the trained model.
- `utils.py`: Helper functions used across scripts.
- `is_connected.py`: Checks network connectivity (useful for runtime environments).
- `env.yml`: Conda environment configuration.
- `README.md`: Project documentation (this file).

## 🖼️ Dataset

All experiments in this repository are conducted on the [Best Artworks of All Time dataset](https://www.kaggle.com/datasets/ikarus777/best-artworks-of-all-time) from Kaggle. This dataset includes thousands of classical paintings and is used to explore artistic generation and semantic control through diffusion models.

## 🧪 Experimental: Semantic Conditioning (WIP)

We're extending DDPM to support semantic conditioning using high-level inputs (like segmentation maps or descriptive embeddings). The following files support this ongoing work:

- `ddpm_semantic.py`: Training and sampling script under development.
- `modules_semantic.py`: Semantic feature integration (in progress).
- `unet_semantic.py`: Planned semantic-aware UNet implementation.

⚠️ These features are **not yet complete** and the scripts are not guaranteed to run without errors.

## 🧪 Evaluation

Use `evaluate.py` to analyze generated outputs. This may include:

- Comparing real and generated image distributions.
- Visual inspection of conditional or semantic adherence.
- Quantitative metrics (FID computation).

## 🛠️ Next Steps

- Finalize semantic conditioning architecture and training loop.
- Add support for custom semantic inputs in the dataloader.
- Evaluate semantic-DDPM results and compare with label-conditional outputs.

---

*Note: This repository is actively being developed. Semantic DDPM features are still experimental and may not yet produce usable results.*
