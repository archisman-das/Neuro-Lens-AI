"""3D tumor mesh extraction via marching cubes.

Takes a binary anomaly mask (from conformal head) and extracts a 3D
mesh suitable for clinical visualization. For brain-2D scope we
stack 2D slices into a pseudo-3D volume; true 3D inputs work the same way.
"""
from __future__ import annotations
from typing import Tuple, Dict
import numpy as np


def extract_tumor_mesh(volume_mask: np.ndarray,
                        spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
                        level: float = 0.5,
                        smoothing: int = 0) -> Dict:
    """Marching cubes on a 3D binary tumor mask.

    volume_mask: (D, H, W) bool/float array, 1 inside tumor, 0 outside
    spacing: voxel spacing in mm (z, y, x)
    level: iso-surface threshold (0.5 for binary masks)
    smoothing: optional Gaussian smoothing iterations (0 = no smoothing)

    Returns dict:
      verts:   (N, 3) vertex coords in mm
      faces:   (M, 3) triangle vertex indices
      normals: (N, 3) per-vertex normals
      values:  (N,) per-vertex sampled values (for coloring)
      n_verts, n_faces, volume_mm3, surface_mm2: summary stats
    """
    try:
        from skimage import measure
    except ImportError as exc:
        raise RuntimeError("scikit-image required for mesh extraction") from exc

    vol = volume_mask.astype(np.float32)
    if smoothing > 0:
        try:
            from scipy.ndimage import gaussian_filter
            vol = gaussian_filter(vol, sigma=smoothing)
        except ImportError:
            pass

    # Pad by 1 voxel on each side so marching cubes closes boundary surfaces.
    vol_padded = np.pad(vol, 1, mode="constant", constant_values=0)
    verts, faces, normals, values = measure.marching_cubes(
        vol_padded, level=level, spacing=spacing,
    )
    # Shift verts back to remove padding offset
    verts = verts - np.array(spacing)

    # Compute volume + surface area
    volume_mm3 = float(vol.sum() * spacing[0] * spacing[1] * spacing[2])
    # Surface = sum of triangle areas
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    surface_mm2 = float(0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1).sum())

    return {
        "verts": verts.astype(np.float32),
        "faces": faces.astype(np.int32),
        "normals": normals.astype(np.float32),
        "values": values.astype(np.float32),
        "n_verts": int(verts.shape[0]),
        "n_faces": int(faces.shape[0]),
        "volume_mm3": volume_mm3,
        "surface_mm2": surface_mm2,
    }


def save_mesh_obj(mesh: Dict, path: str) -> None:
    """Save mesh as Wavefront OBJ (simplest interchange format)."""
    verts = mesh["verts"]; faces = mesh["faces"]
    with open(path, "w") as f:
        for v in verts:
            f.write(f"v {v[0]:.4f} {v[1]:.4f} {v[2]:.4f}\n")
        for tri in faces:
            f.write(f"f {tri[0]+1} {tri[1]+1} {tri[2]+1}\n")


def stack_2d_to_pseudo_3d(masks_2d: np.ndarray) -> np.ndarray:
    """Stack a list of 2D masks into a (D, H, W) pseudo-3D volume.

    For brain-2D v9b scope: gives a working mesh even though depth is
    synthesized (each 2D slice gets stacked). True 3D inputs skip this
    and pass the volume directly to extract_tumor_mesh.
    """
    if masks_2d.ndim == 2:
        masks_2d = masks_2d[None]  # single-slice -> 1xHxW
    return masks_2d.astype(np.float32)


__all__ = ["extract_tumor_mesh", "save_mesh_obj", "stack_2d_to_pseudo_3d"]
