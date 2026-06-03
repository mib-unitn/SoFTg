#!/usr/bin/env python3
"""
label_tool.py — Interactive solar granule review and labeling tool.

Combines classic adaptive thresholding & watershed segmentation with an
interactive GUI to accept, reject, or manually fix granules. Output is
saved as NPZ files ready for deep learning training.

Usage:
    python label_tool.py --fits-path "data/*.fits" --out-dir "labels"
"""

import os
import sys
import glob
import argparse
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
import cv2
from scipy.ndimage import label

# Set backend to support Matplotlib interactive features
try:
    import matplotlib
    matplotlib.use('Qt5Agg')
except Exception:
    try:
        matplotlib.use('TkAgg')
    except Exception:
        pass

from utils import slice_into_tiles
from granule_detection_lib import segment_solar_granules


def enhance_solar_contrast(frames):
    """Stack, CLAHE-enhance, and sharpen a sequence of solar frames."""
    master     = np.median(frames, axis=0).astype(np.float32)
    norm       = cv2.normalize(master, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
    enhanced   = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(norm)
    blurred    = cv2.GaussianBlur(enhanced, (0, 0), 2.0)
    sharpened  = cv2.addWeighted(enhanced, 1.5, blurred, -0.5, 0).astype(np.float32)
    return (sharpened - sharpened.min()) / (sharpened.max() - sharpened.min() + 1e-8)


class BlobLabeler:
    """
    Binary granule reviewer — accept / reject / fix.

    ── REVIEW MODE (default) ─────────────────────────────────────────────────
      Left-click    Accept blob as granule (turns green)
      Right-click   Delete blob (removes from mask)
      'a'           Accept ALL blobs at once
      'f'           Enter POLYGON FIX mode
      Enter         Save and go to next tile
      'n'           Skip tile
      Esc           Quit

    ── POLYGON FIX MODE (f) ──────────────────────────────────────────────────
      Left-click    Add vertex
      Right-click   Remove last vertex
      Enter         Confirm: everything inside = granule (merges overlapping blobs)
      Esc           Cancel / Return to review mode
    """

    _COL_ACCEPT  = (0.20, 0.90, 0.20, 0.90)
    _COL_FILL_OK = (0.20, 0.90, 0.20, 0.12)
    _COL_PEND    = (1.00, 1.00, 0.00, 0.75)
    _COL_FIXED   = (0.00, 0.90, 0.90, 0.45)   # Hand-drawn cyan overlay

    def __init__(self, img, instance_mask):
        self.img           = img
        self.instance_mask = instance_mask.copy()
        self.h, self.w     = img.shape
        self.binary_mask   = np.zeros((self.h, self.w), dtype=np.uint8)
        self.accepted_ids  = set()
        self.mode          = "review"
        self.poly_pts      = []
        self.done = False
        self.skip = False

        self.fig, self.ax = plt.subplots(figsize=(11, 11))
        self.ax.imshow(self.img, cmap='gray')
        self.visual_mask = np.zeros((self.h, self.w, 4), dtype=np.float32)
        self._refresh_overlay(initial=True)
        self.im_overlay = self.ax.imshow(self.visual_mask, interpolation='nearest')

        # Polygon drawing preview
        self.poly_line,  = self.ax.plot([], [], 'o-',  color='cyan', lw=1.8, ms=4)
        self.close_line, = self.ax.plot([], [], '--',  color='cyan', lw=1.0, alpha=0.6)

        self._set_title()
        self.fig.canvas.mpl_connect('button_press_event',  self._on_press)
        self.fig.canvas.mpl_connect('motion_notify_event', self._on_motion)
        self.fig.canvas.mpl_connect('key_press_event',     self._on_key)

    # ── Overlay ─────────────────────────────────────────────────────────────
    def _refresh_overlay(self, initial=False):
        self.visual_mask[:] = 0.0
        k = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

        def border(m):
            f = m.astype(np.float32)
            return (cv2.erode(f, k) != f) & m

        # Accepted blobs
        if self.accepted_ids:
            acc = np.isin(self.instance_mask, list(self.accepted_ids))
            self.visual_mask[acc]          = self._COL_FILL_OK
            self.visual_mask[border(acc)]  = self._COL_ACCEPT

        # Unassigned/pending blobs (yellow border)
        unasgn = (self.instance_mask > 0) & (~np.isin(self.instance_mask, list(self.accepted_ids)))
        if unasgn.any():
            self.visual_mask[border(unasgn)] = self._COL_PEND

        # Hand-drawn custom polygon areas
        fixed = (self.binary_mask > 0) & (self.instance_mask == 0)
        self.visual_mask[fixed] = self._COL_FIXED

        if not initial:
            self.im_overlay.set_data(self.visual_mask)
            self.fig.canvas.draw_idle()

    def _commit(self):
        self.binary_mask[:] = 0
        if self.accepted_ids:
            self.binary_mask[np.isin(self.instance_mask, list(self.accepted_ids))] = 1
        # Keep hand-drawn/merged pixels
        self.binary_mask[(self.binary_mask == 0) & (self.instance_mask == 0) &
                         (self.visual_mask[:, :, 3] > 0.3)] = 1

    # ── Title ────────────────────────────────────────────────────────────────
    def _set_title(self):
        n_total = max(0, len(np.unique(self.instance_mask)) - 1)
        n_ok    = len(self.accepted_ids)
        if self.mode == "polygon":
            msg = (f"FIX POLYGON ({len(self.poly_pts)} pts) — "
                   f"left-click=add vertex │ right-click=undo │ Enter=confirm │ Esc=cancel")
        else:
            msg = (f"Accepted {n_ok}/{n_total} │ left=accept │ right=delete │ "
                   f"'a'=accept all │ 'f'=fix polygon │ Enter=save │ 'n'=skip │ Esc=quit")
        self.ax.set_title(msg, fontsize=11, fontweight='bold')
        self.fig.canvas.draw_idle()

    # ── Polygon helpers ──────────────────────────────────────────────────────
    def _poly_mask(self):
        verts = np.array(self.poly_pts, dtype=np.float64)
        path  = MplPath(verts)
        ny, nx = np.indices((self.h, self.w))
        return path.contains_points(
            np.column_stack([nx.ravel(), ny.ravel()])
        ).reshape(self.h, self.w)

    def _draw_preview(self, mouse_xy=None):
        pts = self.poly_pts
        if pts:
            xs, ys = zip(*pts)
            self.poly_line.set_data(xs, ys)
            end = mouse_xy if mouse_xy else pts[0]
            self.close_line.set_data([pts[-1][0], end[0]], [pts[-1][1], end[1]])
        else:
            self.poly_line.set_data([], [])
            self.close_line.set_data([], [])
        self.fig.canvas.draw_idle()
        self._set_title()

    def _cancel_poly(self):
        self.poly_pts = []
        self._draw_preview()
        self.mode = "review"
        self._set_title()

    def _confirm_poly(self):
        if len(self.poly_pts) < 3:
            print("⚠  Need at least 3 vertices.")
            return
        pmask = self._poly_mask()

        # Collect overlapping blob IDs inside the polygon
        ids = np.unique(self.instance_mask[pmask])
        ids = ids[ids > 0]

        # Merge them into one new blob ID
        new_id = int(self.instance_mask.max()) + 1
        for oid in ids:
            self.instance_mask[self.instance_mask == oid] = new_id
        self.instance_mask[pmask] = new_id

        # Auto-accept the new merged granule region
        self.accepted_ids -= set(ids.tolist())
        self.accepted_ids.add(new_id)

        self._cancel_poly()
        self._refresh_overlay()
        print(f"  ✅ Fixed: merged {len(ids)} blob(s) into single granule.")

    # ── Key handler ──────────────────────────────────────────────────────────
    def _on_key(self, event):
        if self.mode == "polygon":
            if event.key == 'enter':
                self._confirm_poly()
            elif event.key == 'escape':
                self._cancel_poly()
            return

        # Review mode keyboard controls
        if event.key == 'a':
            for uid in np.unique(self.instance_mask):
                if uid > 0:
                    self.accepted_ids.add(uid)
            self._refresh_overlay()
            self._set_title()
        elif event.key == 'f':
            self.mode = "polygon"
            self.poly_pts = []
            self._draw_preview()
        elif event.key == 'enter':
            self._commit()
            self.done = True
            plt.close(self.fig)
        elif event.key == 'n':
            self.skip = True
            plt.close(self.fig)
        elif event.key == 'escape':
            sys.exit(0)

    # ── Mouse handler ────────────────────────────────────────────────────────
    def _on_press(self, event):
        if event.inaxes != self.ax or event.xdata is None:
            return
        x, y = int(event.xdata + 0.5), int(event.ydata + 0.5)

        if self.mode == "polygon":
            if event.button == 1:
                self.poly_pts.append((x, y))
                self._draw_preview()
            elif event.button == 3 and self.poly_pts:
                self.poly_pts.pop()
                self._draw_preview()
            return

        # Review mode: click individual blobs
        if 0 <= x < self.w and 0 <= y < self.h:
            bid = self.instance_mask[y, x]
            if bid > 0:
                if event.button == 1:
                    self.accepted_ids.add(bid)
                elif event.button == 3:
                    self.accepted_ids.discard(bid)
                    self.instance_mask[self.instance_mask == bid] = 0
                self._refresh_overlay()
                self._set_title()

    def _on_motion(self, event):
        if self.mode != "polygon" or event.inaxes != self.ax or event.xdata is None:
            return
        if self.poly_pts:
            self.close_line.set_data(
                [self.poly_pts[-1][0], event.xdata],
                [self.poly_pts[-1][1], event.ydata])
            self.fig.canvas.draw_idle()

    def show(self):
        plt.tight_layout()
        plt.show(block=True)


def main():
    parser = argparse.ArgumentParser(
        description="Binary granule labeling: classic detection + interactive review."
    )
    parser.add_argument("--fits-path",   type=str,   default="data/*.fits", help="Glob pattern for FITS files.")
    parser.add_argument("--out-dir",     type=str,   default="labels", help="Directory where labels will be saved.")
    parser.add_argument("--tile-size",   type=int,   default=1024, help="Size of tiles to slice FITS images into.")
    parser.add_argument("--plate-scale", type=float, default=10.875, help="Spatial resolution scale (km/pixel).")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    fits_files = sorted(glob.glob(args.fits_path))
    if not fits_files:
        print(f"❌ No FITS files found for pattern: {args.fits_path}")
        return

    print(f"📂 Found {len(fits_files)} FITS files.")
    print("\n=== Classic Adaptive Segmentation Parameters ===")
    k                   = float(input("k (threshold param) [0.165]: ").strip() or 0.165)
    lane_width_km       = float(input("lane_width_km [150.0]: ").strip() or 150.0)
    reconnect_ratio     = float(input("reconnect_ratio [4.0]: ").strip() or 4.0)
    lane_intensity_frac = float(input("lane_intensity_fraction [0.4]: ").strip() or 0.4)
    chunk_size          = int(input("How many frames to stack/median for CLAHE [5]: ").strip() or 5)

    # Resume loading checks
    resume_chunk_idx, resume_tile_idx = -1, -1
    for i in range(0, len(fits_files), chunk_size):
        chunk_f = fits_files[i:i + chunk_size]
        if not chunk_f:
            continue
        b_name   = os.path.splitext(os.path.basename(chunk_f[len(chunk_f) // 2]))[0]
        existing = glob.glob(os.path.join(args.out_dir, f"{b_name}_stack_tile_*_binary.npz"))
        if existing:
            resume_chunk_idx = i
            resume_tile_idx  = max(
                int(os.path.basename(f).split('_stack_tile_')[1].split('_binary.npz')[0])
                for f in existing
            )

    if resume_chunk_idx >= 0:
        print(f"🔄 Resuming at stack {resume_chunk_idx // chunk_size + 1}, beyond tile {resume_tile_idx + 1}")

    for i in range(0, len(fits_files), chunk_size):
        if i < resume_chunk_idx:
            continue
        chunk_files = fits_files[i:i + chunk_size]
        if not chunk_files:
            continue

        ref_file = chunk_files[len(chunk_files) // 2]
        print(f"\n=== Stack {i // chunk_size + 1}: {len(chunk_files)} files (Ref: {os.path.basename(ref_file)}) ===")

        frames = []
        for ff in chunk_files:
            img = fits.getdata(ff)
            while img.ndim > 2:
                img = img[0]
            frames.append(img[100:-100, 100:-100].astype(np.float32))

        print("Enhancing contrast…")
        img_norm = enhance_solar_contrast(frames)
        tiles    = slice_into_tiles(img_norm, tile_size=args.tile_size)
        print(f"  → Sliced into {len(tiles)} tile(s) of {args.tile_size}x{args.tile_size}")

        for tile_idx, tile in enumerate(tiles):
            base_name = os.path.splitext(os.path.basename(ref_file))[0]
            out_path  = os.path.join(args.out_dir, f"{base_name}_stack_tile_{tile_idx}_binary.npz")

            if i == resume_chunk_idx and tile_idx <= resume_tile_idx:
                print(f"  Skipping tile {tile_idx + 1} (already saved)")
                continue

            print(f"\n  Tile {tile_idx + 1}/{len(tiles)} — running classic segmentation…")
            instance_mask, _, _ = segment_solar_granules(
                tile,
                plate_scale=args.plate_scale,
                k=k,
                lane_width_km=lane_width_km,
                reconnect_ratio=reconnect_ratio,
                lane_intensity_fraction=lane_intensity_frac,
            )
            n_blobs = len(np.unique(instance_mask)) - 1
            print(f"  Detected {n_blobs} blobs. Opening interactive reviewer…")

            labeler = BlobLabeler(tile, instance_mask)
            labeler.show()

            if labeler.skip:
                print("  Tile skipped.")
                continue

            binary = labeler.binary_mask
            if binary.any():
                np.savez_compressed(out_path, img=tile, mask=binary, smap=tile, cmask_map=binary)
                print(f"  💾 Saved → {out_path} ({binary.sum()} granule pixels)")
            else:
                print("  No granules accepted — skipping save.")

        del frames, img_norm, tiles


if __name__ == "__main__":
    main()
