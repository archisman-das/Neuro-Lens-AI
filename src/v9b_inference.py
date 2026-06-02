"""v9b end-to-end inference CLI.

Loads pretrained JEPA + Stage-2 (DDPM+SDF) + optional conformal calibration,
runs inference on an input image, produces:
  - appearance/geometry/combined anomaly maps (PNG)
  - certified binary mask
  - healthy counterfactual image (PNG)
  - tumor residual (PNG)
  - 3D pseudo-mesh (OBJ)
  - MNI152-registered tumor report (JSON)

Usage:
  python src/v9b_inference.py --jepa_ckpt ... --stage2_ckpt ...
      --conformal ... --image input.png --output_dir out/
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import torch
from PIL import Image

try:
    from .research.v9b_model import V9BModel  # type: ignore
    from .research.mesh_extraction import extract_tumor_mesh, save_mesh_obj, stack_2d_to_pseudo_3d  # type: ignore
    from .research.mni152_registration import tumor_atlas_report  # type: ignore
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.research.v9b_model import V9BModel  # type: ignore
    from src.research.mesh_extraction import extract_tumor_mesh, save_mesh_obj, stack_2d_to_pseudo_3d  # type: ignore
    from src.research.mni152_registration import tumor_atlas_report  # type: ignore


def load_image(path: Path, image_size: int) -> torch.Tensor:
    img = Image.open(path).convert("RGB").resize((image_size, image_size), Image.BILINEAR)
    x = np.asarray(img, dtype=np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    x = (x - mean) / std
    return torch.from_numpy(x.transpose(2, 0, 1).copy()).float().unsqueeze(0)


def save_heatmap(arr: np.ndarray, path: Path) -> None:
    arr = (arr - arr.min()) / max(arr.max() - arr.min(), 1e-6)
    arr8 = (arr * 255).astype(np.uint8)
    Image.fromarray(arr8).save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jepa_ckpt", required=True)
    ap.add_argument("--stage2_ckpt", required=True)
    ap.add_argument("--conformal", default=None,
                    help="JSON from JepaConformalCalibrator.save()")
    ap.add_argument("--image", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--image_size", type=int, default=256)
    ap.add_argument("--combine_mode", default="weighted_sum",
                    choices=["weighted_sum", "and", "or"])
    ap.add_argument("--lambda_app", type=float, default=0.6)
    ap.add_argument("--lambda_geo", type=float, default=0.4)
    ap.add_argument("--ddpm_steps", type=int, default=50)
    args = ap.parse_args()

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = V9BModel.from_checkpoints(args.jepa_ckpt, args.stage2_ckpt,
                                       args.conformal, args.image_size, device)
    x = load_image(Path(args.image), args.image_size).to(device)
    print(f"[v9b-infer] loaded image {args.image} -> {tuple(x.shape)}", flush=True)

    res = model.infer(x, combine_mode=args.combine_mode,
                       lambda_app=args.lambda_app, lambda_geo=args.lambda_geo,
                       ddpm_num_steps=args.ddpm_steps)

    # Save anomaly maps
    save_heatmap(res["appearance_anomaly"][0, 0].cpu().numpy(), out / "appearance.png")
    if res["geometry_anomaly"] is not None:
        save_heatmap(res["geometry_anomaly"][0, 0].cpu().numpy(), out / "geometry.png")
    save_heatmap(res["combined_anomaly"][0, 0].cpu().numpy(), out / "combined.png")

    # Save counterfactual + residual
    if res["counterfactual"] is not None:
        cf = res["counterfactual"][0].cpu().numpy()
        cf = (cf - cf.min()) / max(cf.max() - cf.min(), 1e-6)
        Image.fromarray((cf.transpose(1, 2, 0) * 255).astype(np.uint8)).save(out / "counterfactual.png")
        save_heatmap(res["residual"][0, 0].cpu().numpy(), out / "residual.png")

    # Certified mask
    if res["certified_mask"] is not None:
        m = res["certified_mask"][0, 0].cpu().numpy().astype(np.uint8) * 255
        Image.fromarray(m).save(out / "certified_mask.png")
        # 3D mesh + MNI report on the binary mask
        vol = stack_2d_to_pseudo_3d(m > 127)
        try:
            mesh = extract_tumor_mesh(vol)
            save_mesh_obj(mesh, str(out / "tumor_mesh.obj"))
            atlas_report = tumor_atlas_report(vol)
            (out / "tumor_atlas_report.json").write_text(
                json.dumps({"mesh_stats": {k: v for k, v in mesh.items()
                                             if k in ("n_verts", "n_faces",
                                                       "volume_mm3", "surface_mm2")},
                            "atlas": atlas_report}, indent=2))
            print(f"[v9b-infer] saved mesh + atlas report (n_verts={mesh['n_verts']}, "
                  f"volume_mm3={mesh['volume_mm3']:.1f})", flush=True)
        except Exception as exc:
            print(f"[v9b-infer] mesh extraction skipped: {exc}", flush=True)

    print(f"[v9b-infer] done. outputs in {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
