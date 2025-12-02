# Learned Superpixel Generator for Explainable Medical Imaging

<img src="pipeline.svg" width="750" alt="RepresentationUNet training and superpixel generation pipeline">

---

## Overview

RepresentationUNet is a U-Net–based architecture that learns to produce superpixels using **self-supervised contrastive representation learning**.  
A DINO ResNet-50 encoder provides target features, and the network is trained so that each predicted superpixel corresponds to a coherent semantic region.

The repository includes:

- **RepresentationUNet model** (`model.py`)
- **Training script** (`train.py`) with **RepresentationLoss** combining similarity and contrastiveness
- **Inference script** for generating superpixels (`generate_superpixels.py`)
- **Variance-based aggregation algorithm** (`agregation_algorithm.py`) for merging initial regions into the final set of superpixels
