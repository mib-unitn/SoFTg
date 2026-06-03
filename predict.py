#!/usr/bin/env python3
"""
predict.py — Runs tiled inference on solar images using a trained UNet++ model.

Supports:
  • Single FITS file, directory of FITS files, or NumPy (.npy) arrays
  • Automatic contrast enhancement (CLAHE + sharpening)
  • Tiled prediction with overlap stitching
  • GPU-accelerated post-inference cleanup of small components
  • Outputs saved as binary FITS, NumPy, or visual PNG files

Usage:
    python predict.py --input data/test.fits --checkpoint model_params/best_model.pt --output-dir outputs/
"""

import os
import glob
import argparse
import numpy as np
import cv2
import torch
import scipy.ndimage
from astropy.io import fits
from tqdm import tqdm
from PIL import Image

# Local imports
from model import UNetPlusPlus
import utils

# Device selection
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Pre-processing & Post-processing
# ---------------------------------------------------------------------------
def enhance_solar_contrast(frames):
    """Stack, CLAHE-enhance, and sharpen a sequence of solar frames."""
    master = np.median(frames, axis=0).astype(np.float32)
    norm   = cv2.normalize(master, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")

    clahe     = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced  = clahe.apply(norm)

    blurred   = cv2.GaussianBlur(enhanced, (0, 0), 2.0)
    sharpened = cv2.addWeighted(enhanced, 1.5, blurred, -0.5, 0)

    # Normalize to [0, 1]
    sharpened = (sharpened - sharpened.min()) / (sharpened.max() - sharpened.min() + 1e-8)
    return sharpened.astype(np.float32)


def _identification(labels_arr, min_size: int, device: torch.device) -> np.ndarray:
    """GPU-accelerated component size filtering."""
    labels_t = torch.from_numpy(labels_arr).to(device).to(torch.int64)
    counts = torch.bincount(labels_t.view(-1))
    valid_mask = counts >= min_size
    labels_t = labels_t * valid_mask[labels_t]
    return labels_t.cpu().numpy()


def prediction_cleanup(mask: np.ndarray, min_size: int, device: torch.device) -> np.ndarray:
    """Remove foreground connected components smaller than min_size pixels."""
    if mask.max() == 0:
        return mask
    labeled, _ = scipy.ndimage.label(mask > 0)
    filtered = _identification(labeled, min_size, device)
    return (filtered > 0).astype(np.uint8)


# ---------------------------------------------------------------------------
# Model Loading
# ---------------------------------------------------------------------------
def load_trained_model(model_path: str, device: torch.device) -> torch.nn.Module:
    """Instantiates UNetPlusPlus and loads the weights."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model weights not found at: {model_path}")

    # Build UNet++ model for binary segmentation (n_classes=2)
    net = UNetPlusPlus(
        n_channels=1,
        n_classes=2,
        scale=1,
        bilinear=False,
        dropout=False,  # No dropout during inference
    )

    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    
    # Handle state-dict format: either lightning module wrapper, torch.compile wrapper, or raw state_dict
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    
    # Strip lightning/torch.compile prefixes
    clean_state_dict = {}
    for k, v in state_dict.items():
        key = k
        if key.startswith("net."):
            key = key[4:]
        if key.startswith("_orig_mod."):
            key = key[10:]
        clean_state_dict[key] = v

    net.load_state_dict(clean_state_dict)
    net.to(device).eval()
    print(f"  ✅ Loaded trained UNet++ model on {device}")
    return net


# ---------------------------------------------------------------------------
# Main Inference Logic
# ---------------------------------------------------------------------------
def run_inference(input_path, model, output_dir, tile_size, overlap, threshold, min_size, preprocess, device):
    # Load input data
    if input_path.endswith(".fits") or input_path.endswith(".fit"):
        with fits.open(input_path) as hdul:
            data = hdul[0].data.astype(np.float32)
            if data.ndim == 3:  # If stack, convert or use first frame
                print(f"  ℹ️ Found 3D stack shape {data.shape}. Stacking median...")
                frames = [data[i] for i in range(data.shape[0])]
            else:
                frames = [data]
            header = hdul[0].header
    elif input_path.endswith(".npy"):
        data = np.load(input_path).astype(np.float32)
        if data.ndim == 3:
            frames = [data[i] for i in range(data.shape[0])]
        else:
            frames = [data]
        header = None
    else:
        raise ValueError("Unsupported input format. Must be a .fits or .npy file.")

    # Contrast enhancement
    if preprocess:
        print("  ⚙  Enhancing contrast (CLAHE + sharpening)...")
        img = enhance_solar_contrast(frames)
    else:
        img = np.median(frames, axis=0)
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)

    # Run tiled inference
    print("  ⚙  Running overlapping tiled inference...")
    mask = utils.predict_granules(
        img,
        model,
        tile_size=tile_size,
        overlap=overlap,
        threshold=threshold,
    )

    # Clean small components
    if min_size > 0:
        print(f"  ⚙  Removing connected components smaller than {min_size} pixels...")
        mask = prediction_cleanup(mask, min_size, device)

    # Save output
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    
    # Save as NumPy array
    npy_path = os.path.join(output_dir, f"{base_name}_mask.npy")
    np.save(npy_path, mask)
    print(f"  💾 Saved binary array → {npy_path}")

    # Save as PNG image
    png_path = os.path.join(output_dir, f"{base_name}_mask.png")
    # Scale binary {0, 1} to grayscale {0, 255}
    Image.fromarray((mask * 255).astype(np.uint8)).save(png_path)
    print(f"  💾 Saved PNG mask     → {png_path}")

    # Save as FITS if header is available
    if header is not None:
        fits_path = os.path.join(output_dir, f"{base_name}_mask.fits")
        hdu = fits.PrimaryHDU(data=mask, header=header)
        hdu.writeto(fits_path, overwrite=True)
        print(f"  💾 Saved FITS mask    → {fits_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Predict solar granule binary masks using trained UNet++.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", type=str, required=True, help="Path to input FITS file, npy file, or a directory of files.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the trained best_model.pt weights file.")
    parser.add_argument("--output-dir", type=str, default="predictions", help="Directory where masks will be saved.")
    
    parser.add_argument("--tile-size", type=int, default=512, help="Side length (px) of each inference tile.")
    parser.add_argument("--overlap", type=int, default=64, help="Tile overlap (px) to prevent stitch boundary issues.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Probability threshold for classification.")
    parser.add_argument("--min-size", type=int, default=500, help="Minimum connected component size (px) to keep.")
    parser.add_argument("--no-preprocess", action="store_true", help="Skip CLAHE/sharpening pre-processing step.")

    args = parser.parse_args()

    # Load model
    model = load_trained_model(args.checkpoint, DEVICE)

    # Identify inputs
    if os.path.isdir(args.input):
        inputs = sorted(glob.glob(os.path.join(args.input, "*.fits")) +
                        glob.glob(os.path.join(args.input, "*.fit")) +
                        glob.glob(os.path.join(args.input, "*.npy")))
        print(f"📂 Found {len(inputs)} files in directory '{args.input}'")
    else:
        inputs = [args.input]

    for item in inputs:
        print(f"\nProcessing: {os.path.basename(item)}")
        try:
            run_inference(
                input_path=item,
                model=model,
                output_dir=args.output_dir,
                tile_size=args.tile_size,
                overlap=args.overlap,
                threshold=args.threshold,
                min_size=args.min_size,
                preprocess=not args.no_preprocess,
                device=DEVICE,
            )
        except Exception as e:
            print(f"❌ Error processing {item}: {e}")

    print("\n Inference Complete!")


if __name__ == "__main__":
    main()
