# v9b training runbook (one-shot, do these in order)

Everything below uses the artefacts that are already in this repo. No
new code to write — just upload, run, paste results back.

## 0. Pre-flight (do once, takes 5 min)

```
[ ] Google Colab Pro+ subscription active (you already have this).
[ ] Google Drive free space ≥ 4 GB (for dataset + model checkpoints).
[ ] dataset_v8.zip exists at e:\Neuro-Lens-AI-main\Neuro-Lens-AI-main\dataset_v8.zip  (811 MB) — confirmed.
[ ] colab_bundle.zip exists at e:\Neuro-Lens-AI-main\Neuro-Lens-AI-main\colab_bundle.zip  (54 KB)  — confirmed.
```

## 1. Upload artefacts to Google Drive (~10 min, one-time)

1. Open Google Drive in your browser.
2. Create folder `MyDrive/neurolens/`.
3. Upload **both** zips into that folder:
   - `colab_bundle.zip` (54 KB — instant)
   - `dataset_v8.zip` (811 MB — depends on your upload speed, ~5 min on a normal connection)

> **Why both**: `colab_bundle.zip` is the v9b training code (3 stages + Colab notebook); `dataset_v8.zip` is the data those scripts need. The notebook unpacks both on the Colab VM at boot.

## 2. Open the Colab notebook (~1 min)

1. Inside Colab, **File → Upload notebook** → pick `e:\Neuro-Lens-AI-main\Neuro-Lens-AI-main\colab_bundle\v9b_colab_train.ipynb`.
   *(Alternative: unzip colab_bundle.zip locally and the .ipynb is right inside.)*
2. **Runtime → Change runtime type → A100 GPU** (or H100 if available). T4 also works but will roughly double the wall-clock.

## 3. Run cells 1–5 (setup, ~5 min total)

These cells:
- Print `nvidia-smi` (confirms you got A100/H100, not a downgraded T4 — abort the runtime and retry if downgraded).
- Mount Google Drive.
- Unzip `dataset_v8.zip` + `colab_bundle.zip` onto the local VM disk (faster I/O than Drive).
- `pip install` the three extra deps (`segmentation-models-pytorch`, `timm`, `scikit-image`).

**Sanity checkpoint after cell 5**: you should see `cuda True A100-SXM4-40GB` (or H100).

## 4. STAGE 1 — I-JEPA pretrain (~6 h A100, ~12 h T4)

Cell 7 runs `src/train_v9b_stage1_jepa.py`. What happens:

- 200 epochs of self-supervised I-JEPA on **only the healthy (no_tumor)** slices of dataset_v8 — that's how the model learns "what a normal brain latent looks like" without ever seeing a tumor.
- Saves a `last.pt` checkpoint to `MyDrive/neurolens/v9b_jepa/` every 200 steps.
- **Auto-resume on disconnect**: if Colab kicks you off, just re-run the same cell — `--resume auto` picks up from `last.pt`.

**Watch for** (in the cell output):
- `loss = 0.05` order of magnitude by epoch 5 — falling steadily.
- Per-epoch wall-clock ~1.5 min on A100, ~3 min on T4. If you're seeing 10 min/epoch, double-check the runtime type didn't get downgraded.
- `epoch xxx | loss x.xx | ema_momentum x.xxx` lines stream every epoch.

**Done when** you see `epoch 199 | loss ... | saved last.pt`.

## 5. STAGE 2 — DDPM counterfactual decoder + SDF geometric tower (~3 h A100, ~6 h T4)

Cell 9 runs `src/train_v9b_stage2.py`. What happens:

- Freezes the Stage-1 JEPA encoder; uses its global latent as the conditioning signal for two new heads:
  - **Latent-conditioned DDPM** → learns to generate healthy counterfactual images.
  - **SDF geometric tower** → learns the healthy-brain signed-distance prior.
- 100 epochs. Saves to `MyDrive/neurolens/v9b_stage2/last.pt` every 200 steps.
- Same `--resume auto` story.

**Watch for**:
- `ddpm_loss` and `sdf_loss` both falling (separate columns in the per-epoch log).
- DDPM loss should reach ~0.02–0.05 by the end. If it's stuck at 0.2 the JEPA latent isn't informative enough — usually means Stage 1 didn't run long enough.

## 6. STAGE 3 — conformal calibration (~2 min on Colab CPU, GPU optional)

Cell 11 runs in-notebook (no separate script). What happens:

- Walks the healthy held-out split.
- For each healthy slice, computes the JEPA per-pixel prediction-error map → takes the 95th percentile as the per-scan score.
- Calibrates a weighted-conformal quantile at α = 0.10 (90 % coverage) on those scores.
- Saves the calibration to `MyDrive/neurolens/v9b_conformal.json` (~2 KB).

**Watch for**: `coverage at alpha=0.10: 0.9X` (X close to 0.90 = good). If coverage is way off (0.7 or 0.99), the JEPA model is probably under-trained.

## 7. End-to-end inference smoke test (~30 sec)

Cell 13 runs `src/v9b_inference.py` on one test image with `--combine_mode weighted_sum --ddpm_steps 50`. Output written to `/content/v9b_inference_out/`:

- `anomaly_appearance.png` — JEPA residual map (red = high)
- `anomaly_geometry.png` — SDF residual map
- `anomaly_combined.png` — weighted-sum two-tower
- `counterfactual_healthy.png` — DDPM-generated "what this brain would look like if healthy"
- `report.json` — atlas landmarks, mesh stats, conformal verdict

If those files all appear and the combined anomaly looks plausibly aligned with the tumor region in the source image, **Stage 1–3 are done and the model is ready to evaluate**.

## 8. Bring the artefacts back (~3 min)

1. Run a small zip cell at the bottom (or just download the files manually from Drive):
   ```
   !zip -r /content/v9b_artefacts.zip {JEPA_OUT} {S2_OUT} {CONF}
   !cp /content/v9b_artefacts.zip /content/drive/MyDrive/neurolens/
   ```
2. Download `v9b_artefacts.zip` from Drive → `e:\Neuro-Lens-AI-main\Neuro-Lens-AI-main\v9b_artefacts\`.
3. The repo expects:
   - `v9b_artefacts/v9b_jepa/last.pt`
   - `v9b_artefacts/v9b_stage2/last.pt`
   - `v9b_artefacts/v9b_conformal.json`

## 9. After v9b: what to do next (in our local environment)

1. **Re-run the OOD eval** with the v9b two-tower anomaly score added to the cascade (script doesn't exist yet — would be a short patch to `scripts/eval_ood_cascade.py` adding a v9b column next to the existing v8 / current / view-aware columns). The expected payoff is the OpenNeuro coronal-T1 false-positive cohort should drop further, because v9b is by construction a "healthy-brain normality detector" and coronal-T1 healthy brains are still healthy brains regardless of view.
2. **Optionally wire v9b into the dashboard** as a third opinion gated by an env var (e.g., `V9B_TOWER_DISABLE=0`), mirroring how view_router is wired today. Same Pareto-improvement-with-killswitch pattern.
3. **Decide on v9c**. v9c (CrossJEPA-derived, see [proposals/v9c_crossjepa_modal.md](v9c_crossjepa_modal.md)) replaces v9b's mask-based JEPA with cross-modal JEPA — it's a strict superset of v9b's research idea using empirically-validated CrossJEPA tricks (gradient sink + frozen teacher). v9c would benefit from v9b having run successfully because:
   - v9c Method 2 needs frozen per-modality teachers — those can be the four modality-specific I-JEPA encoders trained the same way Stage 1 trains the single-modality one here.
   - The conformal / mesh / atlas layers are shared.

## Troubleshooting cheat-sheet

| Symptom | Likely cause | Fix |
|---|---|---|
| Colab assigned T4 instead of A100 | Pro+ load balancer | Disconnect & reconnect runtime until A100 appears |
| OOM at `batch_size=64` in Stage 1 | T4 has only 15 GB | Re-run cell 7 with `--batch_size 32` (or 16) |
| Stage 1 loss flat | LR too high | Add `--lr 1e-4` to cell 7 |
| Stage 2 DDPM samples look like noise | Under-trained | Stage 2 needs ~50+ epochs before counterfactuals look plausible — let it run |
| `coverage at alpha=0.10` is 0.99 (way over) | JEPA still memorising → low residuals on healthy | Stage 1 likely needs more epochs |
| `coverage at alpha=0.10` is 0.70 (way under) | JEPA noisy / under-trained | Same — let Stage 1 finish or extend with another 50 epochs |
| Disconnected mid-Stage-1 | normal Colab behaviour | Re-run cell 7; `--resume auto` picks up from `last.pt` |
| Tests fail locally before Colab | code drift | `python -m pytest tests/test_v9b_components.py -v` should be 18/18 |

## Time budget summary

| Stage | A100 wall-clock | T4 wall-clock | Active attention needed |
|---|---|---|---|
| Pre-flight + upload | 10 min | 10 min | Yes |
| Cells 1–5 setup | 5 min | 5 min | Yes |
| Stage 1 (cell 7) | ~6 h | ~12 h | No (auto-resume) |
| Stage 2 (cell 9) | ~3 h | ~6 h | No (auto-resume) |
| Stage 3 (cell 11) | 2 min | 2 min | Yes |
| Smoke test (cell 13) | 30 s | 30 s | Yes |
| Download artefacts | 5 min | 5 min | Yes |
| **Total elapsed** | **~10 h** | **~19 h** | **~30 min of your time** |

The long-running cells are unattended; you can leave them and come back.
