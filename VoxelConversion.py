"""
Build a 3-region label volume (marrow / trabecular volume / trabecular
surface) from a 100x100x100 crop of the ESA29-99-L3 microCT slice stack.

Segmentation method follows Tronchin et al. 2025, Physica Medica 133:104966
(https://doi.org/10.1016/j.ejmp.2025.104966), Section 2.1: bone is separated
from marrow by intensity thresholding, then the outer 1-voxel layer of the
bone mask is labelled "trabecular surface" and the remaining interior bone
is labelled "trabecular volume". This is a geometric (morphological)
definition, not an intensity band.
"""

import glob
import os

import itk
import numpy as np
from PIL import Image
from scipy import ndimage

# ---- Parameters ----
SLICE_FOLDER = "/home/greggy/INTERNSHIP/Slices_Noise_Removed/Slices700_799"
CROP_BOX = (1300, 1200, 1400, 1300)  # (left, upper, right, lower) -> 100x100 px
VOXEL_SIZE_MM = 0.037  # 37 um, per Tronchin et al. 2025 Sec 2.1. Confirmed
# against the dataset source (Beller et al. 2005, bone3d.zib.de/data/2005/
# ESA29-99-L3): native voxel size is 37 um isotropic, 2048x2048 px/slice,
# 970 slices -- matches this crop's source data exactly.
TRIM_TO_SYMMETRIC_CUBE = True  # paper Sec 2.1: drop one outer layer per axis
# (100 -> 99 voxels/side) so the model is symmetric about the middle voxel.

OUTPUT_LABEL_IMAGE = "/home/greggy/INTERNSHIP/Voxel Models/trabecular_bone_labels.mhd"

SAVE_LABELED_SLICES = True  # write each z-slice as a color-coded PNG, for
# documentation/visual spot-checking without needing a 3D viewer (napari).
LABELED_SLICES_DIR = "/home/greggy/INTERNSHIP/Voxel Models/labeled_slices"

LABEL_MARROW = 0
LABEL_TRAB_VOLUME = 1
LABEL_TRAB_SURFACE = 2

LABEL_COLORS = {
    LABEL_MARROW: (255, 255, 255),  # white
    LABEL_TRAB_VOLUME: (70, 130, 180),  # steelblue
    LABEL_TRAB_SURFACE: (220, 20, 60),  # crimson
}


def load_cropped_volume(folder, crop_box):
    file_paths = sorted(glob.glob(os.path.join(folder, "*.png")))
    assert len(file_paths) == 100, f"Expected 100 PNGs in {folder}, found {len(file_paths)}"
    slices = [np.array(Image.open(fp).convert("L").crop(crop_box)) for fp in file_paths]
    return np.stack(slices, axis=0)  # shape (z, y, x)


def otsu_threshold(volume):
    """Data-driven bone/marrow cutoff -- avoids hand-picked thresholds on a
    continuous intensity distribution that has no obviously separated peaks."""
    hist, _ = np.histogram(volume, bins=256, range=(0, 255))
    hist = hist.astype(np.float64)
    total = hist.sum()
    sum_all = np.dot(hist, np.arange(256))
    weight_bg = 0.0
    sum_bg = 0.0
    best_thresh, best_variance = 0, -1.0
    for t in range(256):
        weight_bg += hist[t]
        sum_bg += t * hist[t]
        weight_fg = total - weight_bg
        if weight_bg == 0 or weight_fg == 0:
            continue
        mean_bg = sum_bg / weight_bg
        mean_fg = (sum_all - sum_bg) / weight_fg
        between_class_variance = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
        if between_class_variance > best_variance:
            best_variance = between_class_variance
            best_thresh = t
    return best_thresh


def segment_bone_marrow_surface(volume):
    threshold = otsu_threshold(volume)
    bone_mask = volume >= threshold
    # border_value=1: treat the crop's outer faces as if bone continued past
    # them, so voxels at the edge of this sub-volume aren't misclassified as
    # "surface" merely because the crop cut them off from their true 3D
    # neighbours. Real surface should only come from actual bone/marrow
    # boundaries inside the crop.
    interior = ndimage.binary_erosion(bone_mask, border_value=1)

    labels = np.full(volume.shape, LABEL_MARROW, dtype=np.uint8)
    labels[bone_mask] = LABEL_TRAB_SURFACE  # all bone starts as surface...
    labels[interior] = LABEL_TRAB_VOLUME  # ...interior (survives erosion) becomes volume
    return labels, threshold


def trim_to_symmetric_cube(labels):
    return labels[:-1, :-1, :-1]


def save_label_image(labels, spacing_mm, out_path):
    itk_image = itk.GetImageFromArray(labels.astype(np.uint8))
    itk_image.SetSpacing([spacing_mm] * 3)
    itk.imwrite(itk_image, out_path)


def save_region_activity_image(labels, spacing_mm, label_value, out_path):
    """Binary activity map (1 where labels==label_value, else 0) for use as
    an OpenGATE VoxelSource image -- source uniformly distributed over one
    labeled region (e.g. trabecular surface)."""
    activity = (labels == label_value).astype(np.float32)
    itk_image = itk.GetImageFromArray(activity)
    itk_image.SetSpacing([spacing_mm] * 3)
    itk.imwrite(itk_image, out_path)


def save_labeled_slice_images(labels, out_dir):
    """Save each z-slice of the label volume as a color-coded PNG (marrow=white,
    trabecular volume=blue, trabecular surface=red), one file per slice, for
    documentation/visual review of the 2D segmentation result -- e.g. to embed
    in a report or spot-check a specific slice without loading the volume into
    napari."""
    os.makedirs(out_dir, exist_ok=True)
    lut = np.zeros((3, 3), dtype=np.uint8)
    for label_value, color in LABEL_COLORS.items():
        lut[label_value] = color

    for z in range(labels.shape[0]):
        rgb = lut[labels[z]]
        Image.fromarray(rgb, mode="RGB").save(os.path.join(out_dir, f"slice_{z:03d}.png"))
    return labels.shape[0]


def region_volume_fractions(labels, voxel_size_mm):
    voxel_vol = voxel_size_mm**3
    counts = {
        "marrow": int(np.sum(labels == LABEL_MARROW)),
        "trabecular_volume": int(np.sum(labels == LABEL_TRAB_VOLUME)),
        "trabecular_surface": int(np.sum(labels == LABEL_TRAB_SURFACE)),
    }
    volumes_mm3 = {k: v * voxel_vol for k, v in counts.items()}
    return counts, volumes_mm3


if __name__ == "__main__":
    volume = load_cropped_volume(SLICE_FOLDER, CROP_BOX)
    labels, threshold = segment_bone_marrow_surface(volume)

    if TRIM_TO_SYMMETRIC_CUBE:
        labels = trim_to_symmetric_cube(labels)

    save_label_image(labels, VOXEL_SIZE_MM, OUTPUT_LABEL_IMAGE)

    if SAVE_LABELED_SLICES:
        n_slices = save_labeled_slice_images(labels, LABELED_SLICES_DIR)
        print(f"Saved {n_slices} labeled slice PNGs to {LABELED_SLICES_DIR}")

    counts, volumes_mm3 = region_volume_fractions(labels, VOXEL_SIZE_MM)
    print(f"Source slices: {SLICE_FOLDER}")
    print(f"Crop box: {CROP_BOX}, cube shape after processing: {labels.shape}")
    print(f"Otsu threshold used: {threshold}")
    for name in ["marrow", "trabecular_volume", "trabecular_surface"]:
        print(f"  {name:20s} {counts[name]:7d} voxels  ({volumes_mm3[name]:.3f} mm^3)")
    print(f"Saved label volume to {OUTPUT_LABEL_IMAGE}")
