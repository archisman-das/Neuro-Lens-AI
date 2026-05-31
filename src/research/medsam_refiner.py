"""MedSAM cascade refiner for brain tumor segmentation.

Architecture
------------
Our 2D segmenters (v5/v7) are joint-trained on positives + healthy-brain
negatives, which gives them strong *localization* (they find the tumor
and respect empty-mask discipline) but middling *boundary precision*
(the joint sampler dilutes the per-pixel Dice gradient on positives).

MedSAM (Ma et al., Nature Communications Jan 2024) is the inverse: a
SAM ViT-B image encoder + mask decoder, finetuned on 1.5M medical
image-mask pairs *including BraTS brain tumor slices*. It is excellent
at boundary refinement when prompted with a bounding box, but lacks
the FP discipline to localize on its own (it will always produce a
mask inside the prompt region).

The cascade composes them:
    image -> v5/v7 -> coarse_mask -> bbox(largest_cc) -> MedSAM -> refined_mask

The bbox prompt scopes MedSAM to where the joint-trained segmenter
already said "tumor here", then MedSAM redraws the boundary at higher
resolution. This pattern is published as "SAM-as-refiner" in multiple
MICCAI 2024 papers (it's an established approach, not novel; the
novelty in this project is the conformal-counterfactual head, not the
refiner). Realistic gain on BraTS-style data: +3-5 micro-Dice points
without any retraining.

Weights
-------
We load `flaviagiammarino/medsam-vit-base` from the HF Hub - a
transformers-compatible mirror of the original MedSAM weights. ~358 MB,
fits in CPU memory comfortably, ~3-5 sec per image on CPU and
~150 ms on a 4060.

Lazy loading
------------
The refiner is constructed cheaply (no model load). The actual
350 MB weight download + ViT-B instantiation happens on the first
refine() call, behind a class-level singleton lock. On the HF Space
(cpu-basic, 16 GB RAM) this adds ~30-60 s to the first inference but
amortises to ~3-5 s after. Subsequent requests pay only the forward.
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


_LOAD_LOCK = threading.Lock()
_INSTANCE: Optional["MedSAMRefiner"] = None


@dataclass
class RefineResult:
    refined_mask: np.ndarray            # HxW bool
    coarse_mask: np.ndarray             # HxW bool (input)
    bbox_used: tuple[int, int, int, int] | None  # (x0, y0, x1, y1)
    score: float                        # MedSAM's IoU prediction
    refiner_area_px: int
    coarse_area_px: int
    delta_area_px: int                  # refined - coarse
    elapsed_ms: float
    skipped_reason: Optional[str] = None  # set when bypass triggered


class MedSAMRefiner:
    """Box-prompted SAM-style boundary refiner.

    Use:
        r = MedSAMRefiner.get()
        out = r.refine(image_rgb_uint8, coarse_mask_bool)
        if out.refined_mask is not None:
            ...
    """

    HF_REPO = os.environ.get("MEDSAM_HF_REPO", "flaviagiammarino/medsam-vit-base")
    EXPAND_FRAC = 0.10  # 10% box padding for context

    def __init__(self):
        self._model = None
        self._processor = None
        self._device = None

    # ----- singleton ---------------------------------------------------

    @classmethod
    def get(cls) -> "MedSAMRefiner":
        global _INSTANCE
        with _LOAD_LOCK:
            if _INSTANCE is None:
                _INSTANCE = cls()
        return _INSTANCE

    def is_loaded(self) -> bool:
        return self._model is not None

    # ----- lazy load ---------------------------------------------------

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        with _LOAD_LOCK:
            if self._model is not None:
                return
            try:
                import torch
                from transformers import SamModel, SamProcessor  # type: ignore
            except ImportError as exc:
                raise RuntimeError(
                    f"MedSAM refiner needs transformers + torch. "
                    f"Install with: pip install transformers torch. ({exc})"
                ) from exc
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            self._processor = SamProcessor.from_pretrained(self.HF_REPO)
            self._model = SamModel.from_pretrained(self.HF_REPO).to(self._device).eval()

    # ----- refine ------------------------------------------------------

    def refine(
        self,
        image_rgb_uint8: np.ndarray,
        coarse_mask: np.ndarray,
        threshold: float = 0.5,
        min_coarse_pixels: int = 16,
    ) -> RefineResult:
        """Return a refined mask using MedSAM with a bbox prompt.

        If coarse_mask is empty or too small, returns the coarse mask
        unchanged with `skipped_reason` set. This is important: we
        never invent tumor where the joint-trained segmenter said
        there was none - the FP discipline is preserved by the
        cascade, not the refiner.
        """
        import time
        t0 = time.perf_counter()
        assert image_rgb_uint8.dtype == np.uint8
        assert image_rgb_uint8.ndim == 3 and image_rgb_uint8.shape[2] == 3
        assert coarse_mask.shape == image_rgb_uint8.shape[:2]
        coarse_bool = coarse_mask.astype(bool)
        coarse_area = int(coarse_bool.sum())

        if coarse_area < min_coarse_pixels:
            return RefineResult(
                refined_mask=coarse_bool,
                coarse_mask=coarse_bool,
                bbox_used=None,
                score=0.0,
                refiner_area_px=coarse_area,
                coarse_area_px=coarse_area,
                delta_area_px=0,
                elapsed_ms=(time.perf_counter() - t0) * 1000,
                skipped_reason=(
                    "empty_coarse_mask" if coarse_area == 0
                    else f"coarse_mask_too_small ({coarse_area}px < {min_coarse_pixels})"
                ),
            )

        # Pick the largest connected component for the bbox prompt, so a
        # speckled coarse mask doesn't drag the bbox across the whole brain.
        bbox = self._largest_component_bbox(coarse_bool)
        if bbox is None:
            return RefineResult(
                refined_mask=coarse_bool, coarse_mask=coarse_bool,
                bbox_used=None, score=0.0,
                refiner_area_px=coarse_area, coarse_area_px=coarse_area,
                delta_area_px=0,
                elapsed_ms=(time.perf_counter() - t0) * 1000,
                skipped_reason="no_connected_components",
            )
        h, w = coarse_bool.shape
        bbox = self._expand_bbox(bbox, h, w, self.EXPAND_FRAC)

        try:
            self._ensure_loaded()
        except RuntimeError as exc:
            return RefineResult(
                refined_mask=coarse_bool, coarse_mask=coarse_bool,
                bbox_used=bbox, score=0.0,
                refiner_area_px=coarse_area, coarse_area_px=coarse_area,
                delta_area_px=0,
                elapsed_ms=(time.perf_counter() - t0) * 1000,
                skipped_reason=f"medsam_unavailable: {exc}",
            )

        import torch
        # transformers SamProcessor expects PIL image + input_boxes
        from PIL import Image as _PIL
        pil = _PIL.fromarray(image_rgb_uint8)
        inputs = self._processor(
            pil,
            input_boxes=[[list(bbox)]],  # batch of 1 image, 1 box
            return_tensors="pt",
        ).to(self._device)
        with torch.no_grad():
            outputs = self._model(**inputs, multimask_output=False)
        # Post-process: SamProcessor handles upsampling + thresholding when
        # given the original image size.
        masks = self._processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu(),
        )
        mask_t = masks[0][0]  # (num_masks, H, W) -> pick first
        if mask_t.ndim == 3:
            mask_t = mask_t[0]
        refined = mask_t.numpy().astype(bool)
        score_t = outputs.iou_scores[0, 0, 0].cpu().item() if outputs.iou_scores is not None else 0.0

        return RefineResult(
            refined_mask=refined,
            coarse_mask=coarse_bool,
            bbox_used=bbox,
            score=float(score_t),
            refiner_area_px=int(refined.sum()),
            coarse_area_px=coarse_area,
            delta_area_px=int(refined.sum()) - coarse_area,
            elapsed_ms=(time.perf_counter() - t0) * 1000,
            skipped_reason=None,
        )

    # ----- bbox helpers -------------------------------------------------

    @staticmethod
    def _largest_component_bbox(mask: np.ndarray) -> Optional[tuple[int, int, int, int]]:
        """Return (x0, y0, x1, y1) for the largest connected component of mask."""
        try:
            import cv2 as _cv2  # type: ignore
            m8 = mask.astype(np.uint8) * 255
            n, labels, stats, _ = _cv2.connectedComponentsWithStats(m8, connectivity=8)
            if n <= 1:
                return None
            areas = stats[1:, _cv2.CC_STAT_AREA]
            idx = int(areas.argmax()) + 1
            x0 = int(stats[idx, _cv2.CC_STAT_LEFT])
            y0 = int(stats[idx, _cv2.CC_STAT_TOP])
            w = int(stats[idx, _cv2.CC_STAT_WIDTH])
            h = int(stats[idx, _cv2.CC_STAT_HEIGHT])
            return (x0, y0, x0 + w, y0 + h)
        except Exception:
            # Pure-numpy fallback if cv2 missing: use full mask bbox.
            ys, xs = np.where(mask)
            if ys.size == 0:
                return None
            return (int(xs.min()), int(ys.min()),
                    int(xs.max()) + 1, int(ys.max()) + 1)

    @staticmethod
    def _expand_bbox(bbox: tuple[int, int, int, int], h: int, w: int,
                      frac: float) -> tuple[int, int, int, int]:
        x0, y0, x1, y1 = bbox
        pad_x = int((x1 - x0) * frac)
        pad_y = int((y1 - y0) * frac)
        return (
            max(0, x0 - pad_x),
            max(0, y0 - pad_y),
            min(w, x1 + pad_x),
            min(h, y1 + pad_y),
        )


__all__ = ["MedSAMRefiner", "RefineResult"]
