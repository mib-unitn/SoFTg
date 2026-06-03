import numpy as np
import scipy
import scipy.ndimage
import skimage
import skimage.morphology
import skimage.segmentation
import skimage.measure
from tqdm import tqdm
import soft.soft as st
from scipy.ndimage import label


def segment_solar_granules(sample_img, plate_scale, k=0.165, lane_width_km=150, reconnect_ratio=4.0, lane_intensity_fraction=.4):
    """
    Segments solar granules and intergranular lanes from an image.

    Parameters:
        sample_img: The input solar image.
        plate_scale: The spatial scale of the image pixels.
        k: Thresholding parameter for dynamic thresholding.
        lane_width_km: Expected width of the intergranular lanes in km.
        reconnect_ratio: Parameter controlling the morphological reconnection.
        lane_intensity_fraction: Fraction used in the watershed step.

    Returns:
        labels: The final segmented regions (granules) from the watershed algorithm.
        seeds: The seed points used in watershed.
        skeleton_pruned: The pruned skeleton of lanes.
    """

    # STEP 0: Pre-processing (New)
    # This ensures bright points in lanes don't get labeled as granules.
    processed_img = remove_bright_points(sample_img, plate_scale)

    # STEP 1: Dynamic Thresholding
    binary_mask = dynamic_threshold(processed_img, plate_scale, k)

    # STEP 2: Morphological Cleaning
    cleaned = clean_lane_mask(binary_mask, plate_scale, lane_width_km=lane_width_km, reconnect_ratio=reconnect_ratio)
    cleaned_filtered = mode1_detection(cleaned, plate_scale)

    # STEP 3: Skeletonization and Pruning
    skeleton = skeletonize_lanes(cleaned_filtered)
    skeleton_pruned = prune_skeleton(skeleton)

    # STEP 4: Seeding Enclosed Regions
    seeds = seed_granules(cleaned_filtered, plate_scale)

    # STEP 5: Watershed Segmentation
    labels = watershed_granules(processed_img, seeds, skeleton_pruned, cleaned_filtered,
                                lane_intensity_fraction=lane_intensity_fraction)

    return labels, seeds, skeleton_pruned


class color:
    CYAN = '\033[96m'
    BOLD = '\033[1m'
    END = '\033[0m'

def _gaussian_footprint(window_size: int) -> np.ndarray:
    """Create a 2D Gaussian footprint of given window size."""
    half = window_size // 2
    y, x = np.ogrid[-half:half+1, -half:half+1]
    sigma = window_size / 6.0 # 3-sigma rule
    footprint = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    return footprint > 0

def _window_size(granule_size_km: float, plate_scale: float) -> int:
    """Compute the window size in pixels based on granule size and plate scale."""
    window_size = int(np.round(granule_size_km / plate_scale))
    if window_size % 2 == 0:
        window_size += 1
    return max(window_size, 3)


def remove_bright_points(img: np.ndarray, plate_scale: float,
                          bp_size_km: float = 200.0) -> np.ndarray:
    """
    Removes small, intense bright points (magnetic elements) from the image
    to prevent them from interfering with lane detection.
    """
    # 1. Calculate footprint for the bright points
    radius = max(1, int(np.round((bp_size_km / plate_scale) / 2)))
    footprint = skimage.morphology.disk(radius)

    # 2. Use a morphological opening to find the "background"
    background = skimage.morphology.opening(img, footprint)

    # 3. Identify where the original image is significantly brighter than the background
    spikes = img - background

    # 4. Create a cleaned image:
    img_cleaned = np.where(spikes > 0, background, img)

    return img_cleaned


def dynamic_threshold(img: np.ndarray, plate_scale: float, k: float,
                      granule_size_km: float = 1250.0) -> np.ndarray:
    """Dynamic thresholding using local standard deviation and median."""
    img = np.array(img, dtype=np.float64)

    window_size = _window_size(granule_size_km, plate_scale)
    sigma = window_size / 6.0
    local_mean = scipy.ndimage.gaussian_filter(img, sigma=sigma)
    local_mean_sq = scipy.ndimage.gaussian_filter(img ** 2, sigma=sigma)

    local_std = np.sqrt(np.clip(local_mean_sq - local_mean ** 2, 0, None))

    footprint = _gaussian_footprint(window_size)
    local_median = scipy.ndimage.percentile_filter(img, percentile=50, footprint=footprint)

    threshold = local_median - k * local_std
    binary = (img > threshold).astype(np.uint8)

    binary = scipy.ndimage.binary_fill_holes(binary).astype(np.uint8)

    return binary


def _lane_footprint(plate_scale: float, lane_width_km: float = 150.0) -> np.ndarray:
    """Generates a disk structuring element scaled to expected lane width."""
    radius = max(1, int(np.round((lane_width_km / plate_scale) / 2)))
    return skimage.morphology.disk(radius)


def clean_lane_mask(binary_mask: np.ndarray, plate_scale: float,
                    lane_width_km: float = 150.0,
                    reconnect_ratio: float = 3.0) -> np.ndarray:
    """Aggressively cleans the lane mask to widen lanes and split overlapping granules."""
    open_footprint = _lane_footprint(plate_scale, lane_width_km)

    # Step A: Erode granules (widens lanes significantly)
    eroded = skimage.morphology.binary_erosion(
        binary_mask.astype(bool), footprint=open_footprint)

    # Step B: Small Dilation
    recovery_footprint = skimage.morphology.disk(max(1, open_footprint.shape[0] // 4))
    cleaned = skimage.morphology.binary_dilation(eroded, footprint=recovery_footprint)

    # Reconnect nearby granules that were split too far
    close_footprint = skimage.morphology.disk(
        max(1, int(open_footprint.shape[0] // reconnect_ratio))
    )
    cleaned = skimage.morphology.closing(cleaned, footprint=close_footprint)

    # Fill internal holes
    cleaned = scipy.ndimage.binary_fill_holes(cleaned)

    return cleaned.astype(np.uint8)


def _granule_min_size(plate_scale: float, granule_min_diam_km: float = 400) -> int:
    """Calculate the minimum granule size in pixels."""
    granule_min_area_pixels = (np.pi * (granule_min_diam_km / 2) ** 2) / (plate_scale ** 2)
    return int(granule_min_area_pixels)


def mode1_detection(cleaned: np.ndarray, plate_scale: float) -> np.ndarray:
    """Prune small detected granules using soft package identification."""
    labeled_mask, num_features = label(cleaned)
    min_size = _granule_min_size(plate_scale, granule_min_diam_km=400)
    pruned_mask = st.identification(labeled_mask, min_size=min_size)

    return pruned_mask


def skeletonize_lanes(cleaned_pruned: np.ndarray) -> np.ndarray:
    """Extracts the intergranular lane skeleton."""
    lane_mask = np.invert(cleaned_pruned.astype(bool)).astype(np.uint8)
    skeleton = skimage.morphology.skeletonize(lane_mask)

    return skeleton.astype(np.uint8)


def prune_skeleton(skeleton: np.ndarray) -> np.ndarray:
    """Removes dead-end branches from the lane skeleton network."""
    skel = skeleton.astype(bool).copy()

    # Identify junction pixels: 3+ neighbours in the 3x3 neighbourhood
    neighbour_count = scipy.ndimage.convolve(
        skel.astype(np.uint8),
        np.ones((3, 3), dtype=np.uint8),
        mode="constant", cval=0
    )
    junctions = skel & (neighbour_count >= 4)

    # Temporarily remove junctions to isolate individual branches
    skel_no_junctions = skel & ~junctions

    # Label individual branches
    labeled_branches, _ = scipy.ndimage.label(
        skel_no_junctions,
        structure=np.ones((3, 3))
    )

    # Keep only branches that touch at least two junction pixels
    junction_coords = np.argwhere(junctions)
    junction_touch_count = {}
    for jy, jx in junction_coords:
        neighbourhood = labeled_branches[
            max(0, jy-1):jy+2,
            max(0, jx-1):jx+2
        ]
        for bid in set(np.unique(neighbourhood)) - {0}:
            junction_touch_count[bid] = junction_touch_count.get(bid, 0) + 1

    keep = {bid for bid, count in junction_touch_count.items() if count >= 2}

    # Reconstruct skeleton from kept branches + junctions
    pruned = junctions.copy()
    for bid in keep:
        pruned |= labeled_branches == bid

    return pruned.astype(np.uint8)


def seed_granules(cleaned_pruned: np.ndarray, plate_scale: float) -> np.ndarray:
    """Places one seed per granule at the distance transform maximum."""
    binary_mask = cleaned_pruned.astype(bool)
    distance    = scipy.ndimage.distance_transform_edt(binary_mask)
    low_labels  = skimage.measure.label(binary_mask)
    all_comps   = set(np.unique(low_labels)) - {0}

    seeds = np.zeros_like(cleaned_pruned, dtype=np.int32)

    for seed_id, comp in enumerate(all_comps, start=1):
        comp_points = np.argwhere(low_labels == comp)
        vals        = distance[comp_points[:, 0], comp_points[:, 1]]
        best        = comp_points[np.argmax(vals)]
        seeds[best[0], best[1]] = seed_id

    return seeds


def watershed_granules(img: np.ndarray, seeds: np.ndarray,
                       skeleton: np.ndarray,
                       cleaned_pruned: np.ndarray,
                       lane_intensity_fraction: float = 0.5) -> np.ndarray:
    """Performs watershed segmentation using seed points and skeletal constraints."""
    relief = np.max(img) - img

    skel_dilated = skimage.morphology.dilation(
        skeleton.astype(bool), footprint=skimage.morphology.disk(1)
    )

    mask = cleaned_pruned.astype(bool) & ~skel_dilated

    labels = skimage.segmentation.watershed(
        relief, markers=seeds, mask=mask, compactness=0
    )

    # Restore lanes based on absolute intensity thresholding
    intensity_threshold = np.mean(img) * lane_intensity_fraction
    labels[img < intensity_threshold] = 0

    return labels
