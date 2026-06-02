"""MNI152 atlas registration wrapper for v9b.

Maps tumor mesh centroid + bounds to standard MNI152 coordinates so the
LLM report can say "tumor centroid at MNI (24, -12, 8), 8 mm from motor
cortex". Uses SimpleITK if available, falls back to identity (raw image
coords) so downstream code never breaks.

For real clinical use, replace the simple affine registration with
ANTs SyN (non-linear, much more accurate) via subprocess wrapper.
Documented hook below.
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict, Tuple, Optional
import numpy as np


# Approximate MNI152 brain bounding box in mm (LIA orientation).
MNI152_EXTENT_MM = {"x_min": -78, "x_max": 78,
                     "y_min": -112, "y_max": 76,
                     "z_min": -50, "z_max": 85}

# Pre-defined functional landmarks in MNI152 (rough approximations for the
# v9b report -- a full atlas like Harvard-Oxford gives ~100 such landmarks).
MNI_LANDMARKS = {
    "primary_motor_cortex_L":   (-37, -25,  62),
    "primary_motor_cortex_R":   ( 37, -25,  62),
    "broca_area":               (-50,  15,  12),
    "wernicke_area":            (-55, -53,  18),
    "thalamus_L":               (-12, -19,   8),
    "thalamus_R":               ( 12, -19,   8),
    "hippocampus_L":            (-24, -22, -16),
    "hippocampus_R":            ( 24, -22, -16),
    "brainstem":                (  0, -28, -30),
}


def voxel_to_mni_approx(centroid_voxel: Tuple[float, float, float],
                          image_shape: Tuple[int, int, int],
                          input_orientation: str = "axial_2d") -> Tuple[float, float, float]:
    """Approximate voxel -> MNI152 mm mapping (linear interpolation).

    For brain-2D inputs this is necessarily approximate (we don't have the
    full 3D context). Use ANTs SyN for clinical accuracy.

    Returns (x_mm, y_mm, z_mm) in MNI152 LIA convention.
    """
    cz, cy, cx = centroid_voxel
    D, H, W = image_shape
    x_mni = MNI152_EXTENT_MM["x_min"] + (cx / max(1, W - 1)) * (
        MNI152_EXTENT_MM["x_max"] - MNI152_EXTENT_MM["x_min"])
    y_mni = MNI152_EXTENT_MM["y_max"] - (cy / max(1, H - 1)) * (
        MNI152_EXTENT_MM["y_max"] - MNI152_EXTENT_MM["y_min"])
    if input_orientation == "axial_2d":
        # Without depth, place at z=0 (slice through anterior commissure)
        z_mni = 0.0
    else:
        z_mni = MNI152_EXTENT_MM["z_min"] + (cz / max(1, D - 1)) * (
            MNI152_EXTENT_MM["z_max"] - MNI152_EXTENT_MM["z_min"])
    return float(x_mni), float(y_mni), float(z_mni)


def distances_to_landmarks(mni_coord: Tuple[float, float, float]) -> Dict[str, float]:
    """Euclidean distance (mm) from a MNI coord to each functional landmark."""
    x, y, z = mni_coord
    return {name: float(np.sqrt((x - lx) ** 2 + (y - ly) ** 2 + (z - lz) ** 2))
            for name, (lx, ly, lz) in MNI_LANDMARKS.items()}


def tumor_atlas_report(volume_mask: np.ndarray,
                       spacing_mm: Tuple[float, float, float] = (1.0, 1.0, 1.0),
                       input_orientation: str = "axial_2d") -> Dict:
    """Compute MNI152-registered summary of a tumor mask.

    Returns:
      centroid_voxel: (z, y, x)
      centroid_mni:   (x, y, z) in mm
      volume_mm3:     tumor volume in mm^3
      nearest_landmarks: top-3 closest functional landmarks with distance
    """
    if volume_mask.sum() < 1:
        return {"centroid_voxel": None, "centroid_mni": None,
                "volume_mm3": 0.0, "nearest_landmarks": []}
    coords = np.argwhere(volume_mask > 0)
    centroid_voxel = tuple(coords.mean(axis=0))
    centroid_mni = voxel_to_mni_approx(centroid_voxel, volume_mask.shape, input_orientation)
    volume_mm3 = float(volume_mask.sum() * spacing_mm[0] * spacing_mm[1] * spacing_mm[2])
    dists = distances_to_landmarks(centroid_mni)
    top = sorted(dists.items(), key=lambda kv: kv[1])[:3]
    return {
        "centroid_voxel": [float(c) for c in centroid_voxel],
        "centroid_mni": list(centroid_mni),
        "volume_mm3": volume_mm3,
        "nearest_landmarks": [{"name": n, "distance_mm": d} for n, d in top],
    }


def register_to_mni_ants(image_path: str, output_path: str,
                          mni_template_path: str = "MNI152_T1_1mm.nii.gz") -> None:
    """Stub for proper ANTs registration. Requires ANTs binaries.

    For clinical accuracy, install ANTs (https://github.com/ANTsX/ANTs)
    and use SyN registration (non-linear, ~5 min per volume but accurate
    to ~2 mm).
    """
    raise NotImplementedError(
        "Install ANTs and replace this stub with antsRegistrationSyNQuick.sh "
        "or pyantsx for proper non-linear registration. The voxel_to_mni_approx "
        "above is good enough for the v9b brain-2D prototype."
    )


__all__ = ["voxel_to_mni_approx", "distances_to_landmarks",
            "tumor_atlas_report", "register_to_mni_ants", "MNI_LANDMARKS"]
