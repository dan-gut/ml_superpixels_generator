from collections import Counter, defaultdict

import numpy as np
from scipy.ndimage import label
from skimage.segmentation import find_boundaries
from skimage.util import img_as_float64


def relabel_small_regions(labeled_image, threshold=5, connectivity=1):
    """
    Reassigns labels of small isolated regions to the most frequent neighboring label.

    This function scans all connected components within each labeled region
    and reassigns those that are smaller than the given `threshold` to the most
    frequent label among their neighboring regions.

    Parameters
    ----------
    labeled_image : ndarray of int, shape (H, W)
        Input image where each integer represents a labeled region (e.g., output of segmentation).
    threshold : int, optional, default=5
        Minimum component size. Connected components with fewer pixels than this value
        will be relabeled to a neighboring region.
    connectivity : {1, 2}, optional, default=1
        Pixel connectivity used for component detection:
            - 1: 4-connectivity (up/down/left/right),
            - 2: 8-connectivity (including diagonals).

    Returns
    -------
    output : ndarray of int, shape (H, W)
        A relabeled image where small components are reassigned to larger neighboring regions.
    """
    output = labeled_image.copy()

    for current_label in np.unique(labeled_image):
        # Create binary mask for current label
        region_mask = (labeled_image == current_label)

        # Label connected components within this region
        structure = np.ones((3, 3)) if connectivity == 2 else None
        component_labels, num_components = label(region_mask, structure=structure)

        for component_id in range(1, num_components + 1):
            component_mask = (component_labels == component_id)
            component_size = component_mask.sum()

            if component_size < threshold:
                # Find boundary of the small component
                boundary_mask = find_boundaries(component_mask, connectivity=connectivity, mode='thick')
                neighbor_labels = labeled_image[boundary_mask & ~component_mask]

                # Exclude background and self-labels
                valid_neighbors = neighbor_labels[(neighbor_labels != current_label)]

                if len(valid_neighbors) == 0:
                    continue  # No valid neighbor to reassign to

                # Count frequency of neighboring labels
                neighbor_counts = Counter(valid_neighbors)
                target_label = max(neighbor_counts.items(), key=lambda x: x[1])[0]

                # Relabel the small component
                output[component_mask] = target_label

    return output


def compute_region_features(image, mask):
    """
    Computes per-region (superpixel) features based on pixel intensity or color.

    Parameters
    ----------
    image : ndarray of shape (H, W) or (H, W, C)
        Input grayscale or RGB image.
    mask : ndarray of shape (H, W)
        Integer mask where each pixel contains a superpixel label.

    Returns
    -------
    features : dict
        Dictionary mapping each label to a dict with:
            - 'pixels': ndarray of shape (N, C) containing pixel values,
            - 'mean': mean color/intensity vector,
            - 'var': average variance across channels.
    """
    features = {}
    labels = np.unique(mask)

    for label in labels:
        pixels = image[mask == label]

        # Ensure shape is (N, C) regardless of grayscale or single-pixel case
        if pixels.ndim == 1 or (pixels.ndim == 2 and pixels.shape[1] == 1):
            pixels = pixels.reshape(-1, 1)

        mean = pixels.mean(axis=0)
        var = pixels.var(axis=0).mean()
        features[label] = {
            'pixels': pixels,
            'mean': mean,
            'var': var,
        }

    return features


def compute_neighbors(mask):
    """
    Builds a neighborhood graph of superpixels based on 8-connectivity.

    Parameters
    ----------
    mask : ndarray of shape (H, W)
        Integer mask where each value represents a superpixel label.

    Returns
    -------
    neighbors : dict of sets
        Dictionary where keys are superpixel labels and values are sets of neighboring labels.
    """
    boundaries = find_boundaries(mask, connectivity=1)
    neighbors = defaultdict(set)
    h, w = mask.shape

    for y in range(h):
        for x in range(w):
            if not boundaries[y, x]:
                continue

            label = mask[y, x]

            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w:
                        neighbor = mask[ny, nx]
                        if neighbor != label:
                            neighbors[label].add(neighbor)

    return neighbors


def merge_regions(f1, f2):
    """
    Merges two region feature dicts and computes updated statistics.

    Parameters
    ----------
    f1, f2 : dict
        Feature dictionaries from `compute_region_features`.

    Returns
    -------
    merged : dict
        New feature dictionary for the merged region.
    """
    pixels = np.vstack([f1['pixels'], f2['pixels']])
    return {
        'pixels': pixels,
        'mean': pixels.mean(axis=0),
        'var': pixels.var(axis=0).mean()
    }


def resolve_label(label, label_map):
    """
    Resolves the final label in case of chained merges.
    """
    while label != label_map.get(label, label):
        label = label_map[label]
    return label


def superpixel_variance_merge(image, superpixel_mask, target_segments):
    """
    Iteratively merges superpixels to reach a target number of segments,
    prioritizing merges that maximize internal variance after combining
    and starting from the most homogeneous regions.

    Parameters
    ----------
    image : ndarray of shape (H, W) or (H, W, C)
        Input image (grayscale or RGB).
    superpixel_mask : ndarray of shape (H, W)
        Initial superpixel segmentation mask.
    target_segments : int
        Desired number of final segments.

    Returns
    -------
    final_mask : ndarray of shape (H, W)
        Integer mask containing final segment labels, relabeled from 0.
    """
    image = img_as_float64(image)
    mask = superpixel_mask.copy()

    features = compute_region_features(image, mask)
    neighbors = compute_neighbors(mask)

    current_labels = set(features.keys())
    label_map = {l: l for l in current_labels}
    current_label = max(current_labels) + 1

    while len(features) > target_segments:
        # 1. Select the most homogeneous superpixel (lowest internal variance)
        min_label = min(features, key=lambda l: features[l]['var'])

        # 2. Among its neighbors, choose the one that maximizes variance after merging
        best_neighbor = None
        max_merge_var = -np.inf
        for neighbor in neighbors[min_label]:
            if neighbor not in features:
                continue
            merged = merge_regions(features[min_label], features[neighbor])
            if merged['var'] > max_merge_var:
                max_merge_var = merged['var']
                best_neighbor = neighbor

        if best_neighbor is None:
            break  # No valid neighbor to merge with

        # 3. Perform the merge
        new_feature = merge_regions(features[min_label], features[best_neighbor])
        features[current_label] = new_feature

        # 4. Update label mappings and remove obsolete entries
        for key in list(label_map):
            if resolve_label(label_map[key], label_map) in {min_label, best_neighbor}:
                label_map[key] = current_label

        features.pop(min_label, None)
        features.pop(best_neighbor, None)

        # 5. Update the neighborhood graph
        new_neighbors = (neighbors[min_label] | neighbors[best_neighbor]) - {min_label, best_neighbor}
        neighbors[current_label] = new_neighbors

        for n in new_neighbors:
            neighbors[n].discard(min_label)
            neighbors[n].discard(best_neighbor)
            neighbors[n].add(current_label)

        del neighbors[min_label]
        del neighbors[best_neighbor]

        current_label += 1

    # Replace old labels with final merged ones
    final_mask = mask.copy()
    for old_label in np.unique(mask):
        final_mask[mask == old_label] = label_map[old_label]

    # Relabel all segments to consecutive integers starting from 0
    unique_labels = np.unique(final_mask)
    relabel_map = {old: new for new, old in enumerate(unique_labels)}
    for old, new in relabel_map.items():
        final_mask[final_mask == old] = new

    print(f"Unique labels: {len(unique_labels)}")

    return final_mask
