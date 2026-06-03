import torch
import numpy as np
import scipy.ndimage
import torch.nn.functional as F
from tqdm import tqdm

EPS = 1e-10

# ---------------------------------------------------------------------------
# 1. EVALUATION METRICS
# ---------------------------------------------------------------------------
def acc(label, predicted):
    """Overall pixel accuracy (fraction of matching pixels)."""
    seg_acc = (label.cpu() == torch.argmax(predicted, axis=1).cpu()).sum() / torch.numel(label.cpu())
    return seg_acc

def nanmean(x):
    """Computes the arithmetic mean ignoring any NaNs."""
    return torch.mean(x[x == x])

def _fast_hist(true, pred, num_classes):
    """Generates the confusion matrix histogram."""
    mask = (true >= 0) & (true < num_classes)
    hist = torch.bincount(
        num_classes * true[mask] + pred[mask],
        minlength=num_classes ** 2,
    ).reshape(num_classes, num_classes).float()
    return hist

def overall_pixel_accuracy(hist):
    correct = torch.diag(hist).sum()
    total = hist.sum()
    overall_acc = correct / (total + EPS)
    return overall_acc

def per_class_pixel_accuracy(hist):
    correct_per_class = torch.diag(hist)
    total_per_class = hist.sum(dim=1)
    per_class_acc = correct_per_class / (total_per_class + EPS)
    avg_per_class_acc = nanmean(per_class_acc)
    return avg_per_class_acc

def jaccard_index(hist):
    """Computes the Jaccard index (Intersection over Union - IoU)."""
    A_inter_B = torch.diag(hist)
    A = hist.sum(dim=1)
    B = hist.sum(dim=0)
    jaccard = A_inter_B / (A + B - A_inter_B + EPS)
    avg_jacc = nanmean(jaccard)
    return avg_jacc

def dice_coefficient(hist):
    """Computes the Sørensen–Dice coefficient (F1 score)."""
    A_inter_B = torch.diag(hist)
    A = hist.sum(dim=1)
    B = hist.sum(dim=0)
    dice = (2 * A_inter_B) / (A + B + EPS)
    avg_dice = nanmean(dice)
    return avg_dice

def eval_metrics_sem(true, pred, num_classes, device):
    """Computes various semantic segmentation metrics on 2D maps."""
    hist = torch.zeros((num_classes, num_classes)).to(device)
    for t, p in zip(true, pred):
        hist += _fast_hist(t.flatten(), p.flatten(), num_classes)
    overall_acc = overall_pixel_accuracy(hist)
    avg_per_class_acc = per_class_pixel_accuracy(hist)
    avg_jacc = jaccard_index(hist)
    avg_dice = dice_coefficient(hist)
    return overall_acc, avg_per_class_acc, avg_jacc, avg_dice

# ---------------------------------------------------------------------------
# 2. TILING & IMAGING UTILITIES
# ---------------------------------------------------------------------------
def blockshaped(arr, nrows, ncols):
    """
    Return an array of shape (n, nrows, ncols) representing subblocks of arr.
    """
    h, w = arr.shape[0], arr.shape[1] 
    assert h % nrows == 0, f"{h} rows is not evenly divisible by {nrows}"
    assert w % ncols == 0, f"{w} cols is not evenly divisible by {ncols}"
    return (arr.reshape(h//nrows, nrows, -1, ncols)
               .swapaxes(1,2)
               .reshape(-1, nrows, ncols))

def slice_into_tiles(img, tile_size=256):
    """Slice image into non-overlapping tiles. Only complete tiles are returned."""
    h, w = img.shape
    tiles = []
    for r in range(0, h, tile_size):
        for c in range(0, w, tile_size):
            tile = img[r:r+tile_size, c:c+tile_size]
            if tile.shape == (tile_size, tile_size):  # skip incomplete edge tiles
                tiles.append(tile)
    return tiles

def _tile_image_gen(img, tile_size, overlap=32):
    """Generator to yield overlapping tiles and their starting coordinates."""
    stride = tile_size - overlap
    h, w = img.shape
    for r in range(0, h, stride):
        for c in range(0, w, stride):
            r_end = min(r + tile_size, h)
            c_end = min(c + tile_size, w)
            r_start = max(0, r_end - tile_size)
            c_start = max(0, c_end - tile_size)
            yield img[r_start:r_start + tile_size, c_start:c_start + tile_size], r_start, c_start

def _preprocess_tile(tile, tile_size):
    """Normalizes and prepares a single tile image for deep learning model input."""
    import skimage.transform
    if tile.shape[:2] != (tile_size, tile_size):
        tile = skimage.transform.resize(tile, (tile_size, tile_size), order=1, preserve_range=True)
    p_min, p_max = np.percentile(tile, (1.0, 99.0))
    span = p_max - p_min if p_max > p_min else 1.0
    tile_norm = np.clip((tile - p_min) / span, 0.0, 1.0)
    return torch.from_numpy(tile_norm[np.newaxis, np.newaxis, ...]).float()

def predict_granules(img, model, tile_size=256, overlap=32, threshold=0.5):
    """
    Tiled model inference: runs model over overlapping tiles, stitches predictions,
    and returns a clean binary mask.
    """
    model.eval()
    device = next(model.parameters()).device

    data = np.asarray(img, dtype=np.float64)
    tiles = list(_tile_image_gen(data, tile_size, overlap))

    # Accumulate granule probability
    h, w = data.shape
    accum = np.zeros((h, w), dtype=np.float32)
    count = np.zeros((h, w), dtype=np.float32)

    with torch.no_grad():
        with torch.amp.autocast(device_type='cuda' if 'cuda' in str(device) else 'cpu'):
            for tile, r, c in tqdm(tiles, desc="Predicting tiles", leave=False):
                tile_t = _preprocess_tile(tile, tile_size).to(device)
                output = model(tile_t)
                # Channel 1 = granule probability
                prob = F.softmax(output, dim=1)[0, 1].cpu().numpy()  # (H, W)
                accum[r:r + tile_size, c:c + tile_size] += prob
                count[r:r + tile_size, c:c + tile_size] += 1.0
                del output, tile_t

    prob_map = accum / np.maximum(count, 1.0)

    # Threshold → binary mask
    binary_mask = (prob_map >= threshold).astype(np.uint8)

    # Fill interior holes
    binary_mask = scipy.ndimage.binary_fill_holes(binary_mask).astype(np.uint8)

    return binary_mask
