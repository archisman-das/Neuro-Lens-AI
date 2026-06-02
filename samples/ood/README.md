# OOD test samples for NeuroLens

Every image in this folder is genuinely out-of-distribution with respect
to v8 / v5 / v3 / cnn / transfer / vit — i.e. **NOT** drawn from
BraTS 2020, Mateusz Buda's kaggle_3m LGG, Cheng et al. 2017 Figshare,
or the Kaggle 4-class brain-tumor MRI set (the four sources our cascade
trained, validated, or tested on).

Built by `scripts/fetch_ood_samples.py`.

## Sources

### `healthy_coronal_T1_openneuro/` (12 PNGs)
- **Origin**: HF `g4m3r/T1w_MRI_Brain_Slices`, derived from OpenNeuro
  `ds003592` (Spreng et al., *Neurocognitive aging data release*).
- **Modality**: T1-weighted MPRAGE.
- **View**: coronal (axial in our training → coronal is a real shift).
- **Subjects**: 12 different healthy adults, one mid-coronal slice each.
- **License**: MIT.
- **Expected classifier verdict**: `no_tumor`.

### `tumor_proprietary_multimodal_unidata/` (20 PNGs)
- **Origin**: HF `UniDataPro/brain-cancer-dataset` (private clinical
  DICOM study converted to PNG via pydicom).
- **Modalities (read from DICOM SeriesDescription)**:
  - SE000001, SE000008 → T1W_FFE tra (axial T1, fast-field-echo)
  - SE000002 → T2W_TSE (axial T2, turbo-spin-echo)
  - SE000003 → T2 COR (coronal T2)
  - SE000004, SE000007 → T1 SAG (sagittal T1)
  - SE000005 → DWI (diffusion-weighted)
  - SE000006 → Survey_MST (localiser scout, low diagnostic value)
  - SE000009 → **T2W_FLAIR** (axial FLAIR)
  - SE000010 → T1W_Cor (coronal T1)
- **License**: CC-BY-NC-ND-4.0. Originals preserved alongside PNGs.
- **Expected classifier verdict**: tumor present (per dataset metadata).

### Wanted, but blocked
- `FOMO25/FOMO-MRI` (OASIS T1/T2/FLAIR multimodal NIfTI): **gated** —
  download fails with `GatedRepoError`. Auto-approval requires the user
  to accept terms on the HF web UI for the active token. Once accepted,
  re-running `fetch_ood_samples.py` will pull middle slices per modality
  into `multimodal_oasis_fomo/`.
