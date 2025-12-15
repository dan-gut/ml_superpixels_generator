# Information-Guided Partitioning (IGP)

This repository provides a reference implementation of **Information-Guided Partitioning (IGP)**, an unsupervised method for superpixel generation designed for explainability-driven image analysis, with a particular focus on medical imaging.

IGP generates superpixels by **guiding image partitioning with information captured by learned representations**, rather than relying on visual similarity, color homogeneity, or geometric regularity.

<img src="pipeline.svg" width="750" alt="RepresentationUNet training and superpixel generation pipeline">

---

## Overview

**Information-Guided Partitioning (IGP)** follows a two-stage design:

1. **Learning informative region assignments**  
   A neural network learns dense region assignment maps using **self-supervised contrastive representation learning**, guided by a pretrained teacher network (DINO ResNet-50).

2. **Region aggregation into superpixels**  
   The learned region primitives are optionally aggregated into a target number of superpixels  using a variance-based merging strategy.

The resulting superpixels are intended to serve as atomic units for downstream explainability methods (e.g., LIME, SHAP), rather than as visually homogeneous image regions.


The repository contains the following main components:
- **`model.py`** - definition of the U-Net-based model with an additional representation head,
- **`train.py`** - training script for self-supervised contrastive learning of region representations,
- **`generate_superpixels.py`** - inference script that produces region assignment maps and initial superpixel partitions,
- **`agregation_algorithm.py`** - variance-based aggregation algorithm for merging region primitives into a desired number of superpixels,
- **`evaluate_superpixels.py`** - evaluation utilities for computing standard superpixel quality metrics (e.g., Boundary Recall, ASA, Undersegmentation Error, Explained Variation),
- **Additional utility scripts** for data preprocessing and resizing.