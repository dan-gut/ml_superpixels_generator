from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import PIL
from skimage.measure import regionprops
from skimage.segmentation import find_boundaries


def boundary_pixels(segmentation):
    """Zwraca piksele graniczne segmentacji."""
    return find_boundaries(segmentation, mode='outer')


def region_intersections(superpixels, ground_truth):
    """Oblicza przecinające się obszary między superpikselami a segmentami ground truth."""
    superpixel_ids = np.unique(superpixels)
    gt_ids = np.unique(ground_truth)
    intersection_matrix = np.zeros((len(superpixel_ids), len(gt_ids)))

    for i, s_id in enumerate(superpixel_ids):
        for j, gt_id in enumerate(gt_ids):
            intersection_matrix[i, j] = np.sum((superpixels == s_id) & (ground_truth == gt_id))

    return intersection_matrix


def boundary_recall(superpixels, ground_truth):
    boundaries_sp = boundary_pixels(superpixels)
    boundaries_gt = boundary_pixels(ground_truth)

    intersection = np.sum(boundaries_sp & boundaries_gt)
    total_gt_boundary = np.sum(boundaries_gt)

    return intersection / total_gt_boundary


def achievable_segmentation_accuracy(superpixels, ground_truth):
    intersection_matrix = region_intersections(superpixels, ground_truth)
    max_intersections = np.max(intersection_matrix, axis=1)  # Max dla każdego superpiksela
    total_pixels = ground_truth.size

    return np.sum(max_intersections) / total_pixels


def undersegmentation_error(superpixels, ground_truth):
    intersection_matrix = region_intersections(superpixels, ground_truth)
    total_pixels = ground_truth.size

    ue = 0
    for i in range(intersection_matrix.shape[0]):
        ue += np.sum(intersection_matrix[i, :]) - np.max(intersection_matrix[i, :])

    return ue / total_pixels


def explained_variation(superpixels, image):
    mean_intensity = np.mean(image)
    total_variation = np.sum((image - mean_intensity) ** 2)

    explained_variation = 0
    for region in regionprops(superpixels, intensity_image=image):
        region_mean = region.mean_intensity
        explained_variation += region.area * np.sum((region_mean - mean_intensity) ** 2)

    return explained_variation / total_variation


def compactness(superpixels):
    total_pixels = superpixels.size
    regions = regionprops(superpixels)

    total_compactness = 0

    for region in regions:
        normalization_factor = region.num_pixels / total_pixels # bigger superpixels should contribute more (https://www.sciencedirect.com/science/article/pii/S0167865513003498)
        if region.perimeter == 0:
            isoperimetric_quotient = 1
        else:
            isoperimetric_quotient = (4 * np.pi * region.num_pixels) / (region.perimeter ** 2)

        total_compactness += normalization_factor * isoperimetric_quotient

    return total_compactness


def calculate_metrics(superpixels, ground_truth, image):
    br = boundary_recall(superpixels, ground_truth)
    asa = achievable_segmentation_accuracy(superpixels, ground_truth)
    ue = undersegmentation_error(superpixels, ground_truth)
    ev = explained_variation(superpixels, image)
    comp = compactness(superpixels)

    print(f"Boundary Recall: {br}")
    print(f"ASA: {asa}")
    print(f"Undersegmentation Error: {ue}")
    print(f"Explained Variation: {ev}")
    print(f"Compactness: {comp}")

if __name__ == "__main__":
    image_path = Path("Data/images_resized/image_001.nii.gz")
    sp_path = Path("Data/superpixels/bs8_lr0.001_c1024_cr0.5_ps40_40_sample_ns40/image_001.png")
    gt_path = Path("Data/labels_resized/label_001.nii.gz")

    img = nib.load(image_path).get_fdata().squeeze()
    sp_img = np.asarray(PIL.Image.open(sp_path))
    gt_seg = nib.load(gt_path).get_fdata().astype(np.uint16).squeeze()

    calculate_metrics(sp_img, gt_seg, img)