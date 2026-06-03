"""v9c CrossJEPA — Cross-Modal Conformal-JEPA for Brain MRI.

Implements the proposal in proposals/v9c_crossjepa_modal.md. Two
single-direction, frozen-teacher CrossJEPA towers:

  Method 1 (3D volume -> 2D slice):
    learnable 3D ViT-S encodes a brain volume; predictor reconstructs the
    embedding of any chosen 2D slice as seen by the FROZEN v8 ConvNeXt-Tiny
    teacher (the same encoder running in production). Conditioning =
    (plane, slice_idx, voxel_spacing, intensity_histogram).

  Method 2 (modality -> modality):
    learnable subset-encoder takes any non-empty S subset of
    {T1, T1c, T2, FLAIR}; predictor reconstructs the embedding of a
    held-out modality m as seen by the FROZEN per-modality I-JEPA teacher
    (4 separate ViT-S encoders, one per modality, pretrained on healthy
    brains in that modality only). Conditioning = (target_modality_id,
    slice-pose, per-context intensity_histogram).

Anomaly score at inference = weighted-conformal-certified residual from
one or both towers, plugged into the existing v9b two-tower combiner
(src/research/two_tower_anomaly.py).

Design constraints from CrossJEPA (Nazar et al. 2511.18424, Nov 2025):
  1. NO masking — the cross-modal split IS the context/target split
  2. NO dual-direction — single-direction with frozen teacher only
  3. Frozen teacher is MANDATORY (every learnable-teacher variant in the
     paper collapsed)
  4. Cache target embeddings once — drops epoch time from O(forward)
     to O(lookup); proposal reports 16h -> 2min per epoch from this
"""

__all__ = [
    'conditioning',
    'caching',
    'v8_teacher',
    'vit_3d',
    'volume_to_slice',
    'modality_to_modality',
    'dataset_3d',
]
