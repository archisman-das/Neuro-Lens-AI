# v9b — Conformal-Bounded Normative JEPA Anomaly Detection for Brain Tumor Diagnosis

**Working title (paper):** "Normative JEPA with Conformal Coverage for Unsupervised Brain Tumor Detection and Localization"

**Companion to:** v9 (Universal Causal-Hyperbolic Tumor Foundation Model). v9b is a narrower, brain-focused alternative path. Pick one as your primary; the other is a follow-up.

**Status:** Research proposal, May 2026
**Estimated effort:** 4-5 months focused work
**Estimated cloud compute:** $3-7K (much lighter than v9)

---

## 1. Problem statement

Tumor segmentation models today require **supervised pixel-level tumor annotations**, which are:
- Expensive (radiologists at $300-1000/hr)
- Inconsistent (inter-rater Dice ~0.85 — radiologists disagree)
- Bias-inducing (models inherit annotator quirks)
- Modality-specific (annotations for T1c don't help T2/FLAIR/CT directly)

**Normative learning** sidesteps this by training only on **healthy brain MRI** and detecting tumors as **deviations from the learned healthy manifold**. The core idea is well-established (49 published studies 2018-2025), and the field has converged on three architectures:
- **VAEs**: reconstruct healthy → tumor = reconstruction residual
- **GANs**: discriminator-aware reconstruction → tumor = high anomaly score
- **Diffusion models**: progressive denoising of healthy prior → tumor = denoising residual

Current limitations of the field:
1. **All operate in pixel space.** Pixel-level reconstruction is computationally expensive and conflates fine-grained appearance variation with true anomalies.
2. **Anomaly scores are raw — no coverage guarantees.** "This voxel scored 0.73, is it tumor?" has no formal answer.
3. **No counterfactual** — you get an anomaly map, not "what would this scan look like with no tumor?"
4. **No clinical integration** — anomaly maps are research artifacts, not diagnostic reports.

This proposal addresses all four jointly by using **JEPA (Joint-Embedding Predictive Architecture)** for latent-space normative learning + extending our Gap B (weighted conformal prediction) to JEPA prediction-error maps + integrating with the existing NeuroLens clinical pipeline.

---

## 2. Why JEPA specifically

JEPA (LeCun et al. 2023) is fundamentally different from VAE/GAN/diffusion:

| | VAE / GAN / Diffusion | JEPA |
|---|---|---|
| Reconstruction | Pixel-level (high-dimensional, noisy) | Latent-level (compact, semantic) |
| Loss | Pixel-MSE / adversarial / score-matching | Latent prediction error |
| Anomaly signal | Pixel residual (noisy, conflates style + content) | Latent residual (semantically meaningful) |
| Computational cost | High (pixel-space generation) | Lower (latent-space prediction) |
| Generative quality | Strong (samples look realistic) | Not directly generative |

**The key insight for normative anomaly detection:**
- A JEPA model trained on healthy brains learns to **predict the latent representation of masked regions given context**
- On a healthy brain: context predicts the mask well → low latent prediction error
- On a tumor scan: tumor region's latent is **out of distribution** → JEPA's prediction (based on healthy-trained context model) **fails** → high latent prediction error
- The prediction error in latent space is a **semantically meaningful** anomaly signal, not just an appearance residual

This addresses your specific desire: "a model that UNDERSTANDS the underlying geometry BEHIND the pixels." JEPA literally predicts representations, not pixels.

### What has been published

| Existing JEPA medical work | Scope | Citation |
|---|---|---|
| US-JEPA | Medical ultrasound, feature reconstruction | arXiv:2602.19322 (2026) |
| I-JEPA for brain tumor segmentation | Supervised seg with I-JEPA backbone, BRISC-2025 | Blog/informal 2025 |
| MTS-JEPA | Multi-resolution time-series anomaly | arXiv:2602.04643 (2026) |
| Var-JEPA | Variational JEPA bridging predictive+generative | arXiv:2603.20111 (2026) |
| V-JEPA 2.1 | Video temporal features | March 2026 |

### What is genuinely novel

| Novel piece | Why nobody's done it |
|---|---|
| **JEPA-based unsupervised normative learning on brain MRI** | US-JEPA does ultrasound; I-JEPA for brain is supervised. Nobody has trained JEPA on healthy brain MRI only and used latent prediction error for anomaly detection. |
| **Conformal coverage on JEPA prediction-error maps** | Conformal prediction has classification/regression/segmentation versions, but never on JEPA prediction residuals. Extending Gap B to JEPA latents is genuinely first-of-kind mathematics. |
| **Two-tower: JEPA latent tower + INR/SDF geometric tower** | Latent appearance prediction (JEPA) + geometric structure prediction (SDF/INR), combined as a single anomaly score. Neither tower exists in current normative anomaly work. |
| **Counterfactual healthy generation via JEPA latent inversion** | JEPA is not directly generative, but we can invert from healthy latent → image via a small decoder. "What would the healthy version look like?" with formal bounds on the difference. |
| **End-to-end normative-JEPA → conformal → counterfactual → LLM hallucination-bounded report** | Each piece exists; the integration with formal guarantees throughout is new. |

---

## 3. Architecture

```
Input: 3D brain MRI volume (multi-modal or single-modal)
                            │
                            ▼
        ┌──────────────────────────────────────────────┐
        │ JEPA encoder f_θ (3D ViT)                      │ → z_context (latent of visible regions)
        │ Trained on healthy brain MRI only             │
        └──────────────────────────────────────────────┘
                            │
                            ▼
        ┌──────────────────────────────────────────────┐
        │ JEPA predictor g_φ                            │ → ẑ_masked (predicted latent of masked regions)
        │ Predicts latents of masked patches            │
        │ from visible context                          │
        └──────────────────────────────────────────────┘
                            │
                            ▼
        ┌──────────────────────────────────────────────┐
        │ JEPA target encoder f_θ' (EMA of f_θ)         │ → z_true (true latent of masked regions)
        │ Computes "ground truth" latents               │
        └──────────────────────────────────────────────┘
                            │
                            ▼
        ┌──────────────────────────────────────────────┐
        │ Latent prediction error: e_v = ||ẑ_v - z_v||  │ → per-voxel anomaly score (appearance tower)
        └──────────────────────────────────────────────┘

                  ╔═══════════════ second tower ═══════════════╗
                  ║                                              ║
                  ▼                                              ▼
        ┌──────────────────────────────────────────────┐  ┌────────────────────────┐
        │ INR/SDF head h_ψ                              │  │ Pre-computed MNI152    │
        │ Predicts cortical SDF from input volume       │  │ healthy SDF template   │
        └──────────────────────────────────────────────┘  └────────────────────────┘
                            │                                       │
                            └───────── deviation: SDF_pred ─ SDF_template ──→
                                                                                    │
                                                                                    ▼
                                                              per-voxel geometric anomaly score
                                                              (geometry tower)
                            │
                            ▼
        ┌──────────────────────────────────────────────┐
        │ Combined anomaly map: λ·e_appearance +        │ → unified anomaly score
        │ (1-λ)·e_geometry                              │
        └──────────────────────────────────────────────┘
                            │
                            ▼
        ┌──────────────────────────────────────────────┐
        │ Weighted conformal calibration on JEPA        │ → coverage-bounded
        │ latent prediction error (extends Gap B)       │   prediction sets
        │ Score: s = ||ẑ - z||_2 in latent space        │
        │ Calibrated on held-out healthy scans          │
        └──────────────────────────────────────────────┘
                            │
                            ▼
        ┌──────────────────────────────────────────────┐
        │ Counterfactual healthy generation              │ → "what would this look like
        │ Invert healthy latent → small decoder D_ψ      │    with no tumor?"
        │ residual = input - counterfactual = tumor     │
        └──────────────────────────────────────────────┘
                            │
                            ▼
        ┌──────────────────────────────────────────────┐
        │ Tumor mesh (marching cubes) + MNI152 reg      │ → 3D clinical visualization
        └──────────────────────────────────────────────┘
                            │
                            ▼
        ┌──────────────────────────────────────────────┐
        │ LLM Pattern A/B/C/D reporter                   │ → hallucination-bounded report
        └──────────────────────────────────────────────┘
```

### 3.1 The hard math: conformal coverage on JEPA prediction-error maps

Classical conformal prediction (Romano et al. 2019, Tibshirani et al. 2019) requires a nonconformity score $s(x, y)$. For JEPA:

$$ s_v = \|\hat{z}_v - z_v\|_2^2 $$

where $\hat{z}_v = g_\phi(f_\theta(\text{context}))_v$ is the predicted latent and $z_v = f_{\theta'}(\text{full})_v$ is the EMA-target latent for voxel $v$.

**Calibration step (on healthy held-out set):**
1. For each calibration scan $i$ and each voxel $v$ in the voxel pool, compute $s_{i,v}$
2. Compute weighted (1-α)-quantile $q$ over all $\{s_{i,v}\}$
3. Under modality intervention $do(M=m)$: weight by $w_i = P(M=m | \text{covariates}_i) / P(M=m_i | \text{covariates}_i)$

**Test-time anomaly decision:**
- Compute $s_v^{\text{test}}$ for test scan
- $s_v^{\text{test}} > q$ → voxel is anomalous (tumor candidate) with $\geq (1-\alpha)$ coverage
- $s_v^{\text{test}} \leq q$ → voxel is healthy

**Coverage guarantee** (from Tibshirani et al. 2019 Thm. 2): under exchangeability of test and calibration in the post-intervention distribution Q,
$$ P_Q\left[s_v^{\text{test}} \leq q\right] \geq 1 - \alpha $$
which translates to **provable false-positive control on healthy voxels** at level α.

This is mathematically clean and **has not been published for JEPA latents in any domain, medical or otherwise**.

---

## 4. Phased plan with milestones

| Phase | Duration | Deliverables | Risks |
|---|---|---|---|
| **0. Foundations** | 1 week | Download healthy MRI datasets (IXI 581 scans, OASIS-3 ~600 scans, UK Biobank if accessible); cloud setup; literature finalization | Data access |
| **1. JEPA pretraining on healthy brain MRI** | 4-6 weeks | I-JEPA / V-JEPA encoder + predictor trained on ~3-10K healthy volumes; verify reconstruction quality on held-out healthy | JEPA training stability on 3D medical data; first-of-kind for brain MRI |
| **2. Tumor anomaly evaluation (qualitative)** | 1-2 weeks | Run JEPA on BraTS test set; visualize latent prediction error maps; verify tumor voxels have high error | Anomaly signal may be noisy; need denoising/post-processing |
| **3. INR/SDF geometric tower** | 3-4 weeks | Per-organ SDF head; train on cortical surface labels from FreeSurfer | Cortical surface generation pipeline complexity |
| **4. Combined two-tower anomaly map** | 1-2 weeks | Weighted combination of appearance + geometry towers; tune λ on val set | Combination may not be additive |
| **5. Weighted conformal extension to JEPA latents** | 4-6 weeks | Math derivation + implementation; empirical coverage verification | **Highest research risk** — novel mathematics |
| **6. Counterfactual healthy generation** | 3-4 weeks | Small decoder D_ψ inverting healthy latent → image; FID + radiologist eval on healthy counterfactuals | Generative quality may be low (JEPA isn't generative-first) |
| **7. 3D mesh + MNI152 registration** | 2 weeks | Marching cubes on conformal-thresholded anomaly map → tumor mesh → MNI152 atlas registration via ANTs | Existing tools, low risk |
| **8. LLM Pattern A/B/C/D integration** | 1-2 weeks | Extend current LLM reporter to consume 3D mesh + conformal sets + counterfactual | Already have Pattern A-D, just adapt input |
| **9. Evaluation** | 3-4 weeks | BraTS test set Dice (compare to supervised baselines); coverage validation; radiologist eval; ablations | Need radiologists for clinical eval |
| **10. Paper writing + open-source release** | 4 weeks | TMI / Nature Methods / MICCAI submission | Time-sensitive |
| **Total** | **~5-6 months** | Publishable system + paper | — |

---

## 5. Resource requirements

### 5.1 Compute

- **JEPA pretraining (Phase 1)**: 2-4× A100 for ~1-2 weeks. Cost: $2-4K cloud, or $0 with academic grant.
- **Fine-tuning + ablations (Phases 2-7)**: 1× A100/RTX 4090 intermittent for ~2 months. Cost: $1-3K.
- **Total cloud budget**: $3-7K (significantly less than v9's $8-20K).

### 5.2 Data

| Dataset | Type | # volumes | Use |
|---|---|---|---|
| **IXI** | Healthy brain MRI (T1+T2+PD+MRA) | 581 subjects | Primary JEPA pretraining |
| **OASIS-3** | Longitudinal healthy + AD brain | ~1000 subjects | Additional healthy pretraining |
| **UK Biobank** (if accessible) | Healthy brain MRI | ~50K subjects | Large-scale pretraining (optional, requires application) |
| **HCP (Human Connectome Project)** | Healthy adult brain | ~1100 subjects | Pretraining diversity |
| **BraTS 2024** | Tumor brain (T1+T1c+T2+FLAIR + masks) | ~5K | Tumor evaluation only (NOT for training) |
| **BraTS-PEDs** | Pediatric tumor brain | ~700 | Generalization eval |
| **Total** | | **~58K healthy + 5.7K tumor** | All free academic |

**Critical:** This is **unsupervised** — no tumor labels used in training. BraTS is only used for **evaluation** of how well anomaly detection finds tumors.

### 5.3 Human

- **You**: 50-70% time for 5-6 months
- **Math collaborator**: 5-10% time for Phase 5 (conformal extension proof)
- **2-3 board-certified radiologists**: ~10 hours each for Phase 9 evaluation

---

## 6. Evaluation methodology

### 6.1 Quantitative

| Metric | Target | Why |
|---|---|---|
| **BraTS Dice (unsupervised)** | ≥ 0.65 | Competitive with current normative methods (state-of-art ~0.70) |
| **AUROC for tumor vs healthy classification** | ≥ 0.95 | Discrimination quality |
| **Conformal empirical coverage at α=0.1** | 0.88-0.92 | Validates the math |
| **Coverage under modality intervention** | 0.85-0.92 | Validates weighted conformal extension |
| **Counterfactual FID (healthy counterfactual vs real healthy)** | ≤ 25 | Generative quality |
| **Inference latency** | ≤ 3 sec on A100, ≤ 15 sec on RTX 4090 | Clinical feasibility |

### 6.2 Comparison baselines

- AnoDDPM (CVPR 2022) — diffusion anomaly
- MAEDiff (2024) — masked autoencoder + diffusion
- Pinaya et al. (Nature Methods 2022) — VAE normative
- GAN-based (PubMed 39131566, 2024) — adversarial normative
- Sanchez et al. (MICCAI 2023) — diffusion anomaly for tumor

Our system needs to **match or beat** these on BraTS Dice, plus uniquely provide **coverage guarantees** (which none of them have).

### 6.3 Ablations

- JEPA without geometric tower: pure appearance anomaly
- Geometric tower without JEPA: pure SDF deviation
- Combined towers without conformal: raw anomaly scores
- Combined towers with conformal: full system (this work)
- No counterfactual: anomaly maps only
- No LLM report: numbers only

Each ablation isolates one contribution.

---

## 7. Target publication venues

| Venue | Fit | Timing |
|---|---|---|
| **TMI** (rolling) | Strong fit — long paper, methodology + clinical | Submit Nov 2026, accept Mar-Jun 2027 |
| **Nature Methods** (rolling) | Strong fit if results are exceptional | Submit Dec 2026, 6-9 mo review |
| **MICCAI 2027** (March 2027) | Strong fit — well-suited venue | Submit March 2027 |
| **NeurIPS 2027** (May 2027) | If conformal-JEPA math is the headline | Submit May 2027 |
| **ICLR 2027** (Sept 2026) | If JEPA + medical framing | Tight — need Phase 1-7 by Aug 2026 |

**Recommended primary target: MICCAI 2027** (cleaner story, faster turnaround) **with TMI extended version** as follow-up.

---

## 8. Risk assessment

| Risk | Likelihood | Mitigation |
|---|---|---|
| JEPA training doesn't converge on 3D medical | Medium | Use US-JEPA / I-JEPA recipes as starting point; reduce 3D to 2.5D patches if needed |
| Latent prediction error is too noisy for clean anomaly | Medium-High | Average across patches; combine with geometric tower; use multiple JEPA scales |
| Conformal extension math doesn't yield clean proof | Medium | Fallback: use empirical coverage validation only (still publishable, less theoretical) |
| Counterfactual generation is poor quality (JEPA not generative) | High | Train small auxiliary decoder; alternatively use Stable-Diffusion-Med (latent diffusion) for the generation step only |
| Field publishes "normative JEPA" before you do | Medium | Open-source from day one; release preprint early; aim for fast Phase 1 |
| BraTS Dice doesn't reach competitive level | Medium | Multiple existing normative methods get ~0.65-0.75; targeting that range |
| Compute exhausted | Low (lighter than v9) | Smaller budget needed; cloud grants likely sufficient |
| Radiologist availability for evaluation | Low | Start outreach Phase 6 |

---

## 9. Connection to current NeuroLens AI codebase

### Direct reuse

| v9b component | From current project |
|---|---|
| Conformal coverage extension | **`src/research/conformal_counterfactual_seg.py`** (Gap B) |
| Counterfactual generation pattern | Extends our interventions to "healthy counterfactual" |
| LLM Pattern A/B/C/D reporter | **`src/llm_explain.py`** |
| Dashboard UI for visualization | Extends current `web_dashboard/` with 3D viewer |
| HF Spaces deployment | Current Space architecture |
| Calibration script template | **`scripts/calibrate_conformal_counterfactual.py`** |

### New code needed

| Module | LoC | Difficulty |
|---|---|---|
| 3D JEPA encoder + predictor + target | ~2,000 | Medium-High (adapt I-JEPA / V-JEPA to 3D MRI) |
| JEPA pretraining loop | ~500 | Medium |
| INR/SDF geometric head | ~1,000 | Medium |
| Combined two-tower anomaly | ~300 | Low |
| Conformal extension for JEPA latents | ~600 | **High** (novel math) |
| Healthy counterfactual decoder | ~800 | Medium |
| MNI152 registration pipeline | ~500 | Medium (use ANTs / SimpleITK) |
| 3D viewer for dashboard | ~1,500 | Medium |
| Evaluation framework | ~800 | Medium |
| **Total new code** | **~8,000 LoC** | — |

---

## 10. Comparison: v9 vs v9b — which to pick?

| Dimension | **v9** (Universal Causal-Hyperbolic) | **v9b** (Normative JEPA Conformal) |
|---|---|---|
| Scope | Multi-organ (brain + liver + kidney + ...) | Brain only |
| Supervision | Supervised tumor seg labels needed | **Unsupervised** (only healthy data labeled-ly) |
| Novelty pillars | 5-way integration: universal + causal + hyperbolic + geometric + conformal | JEPA normative + conformal + geometric + counterfactual + LLM |
| Hard math | Hyperbolic conformal | JEPA-latent conformal |
| Compute | $8-20K cloud | $3-7K cloud |
| Time | 10-11 months | 5-6 months |
| Data | ~50-100K volumes across 8 datasets | ~58K healthy + 5.7K tumor (BraTS-only eval) |
| Risk | High (5 integration points) | Medium (4 integration points, narrower) |
| Paper impact | Very high if successful | High |
| Likelihood of completion in 2027 | 60% | 80% |
| Clinical adoption path | Multi-organ deployment | Brain-only deployment |

**My honest recommendation: start with v9b.** Reasons:
1. **Faster** (5-6 mo vs 10-11 mo) — can publish at MICCAI 2027 even with delays
2. **Unsupervised** — sidesteps annotation bottleneck, novel pitch
3. **JEPA is the trending architecture** — papers using JEPA get attention right now
4. **Lower compute cost** — feasible without major grant
5. **Brain focus** matches your existing project scope
6. **v9 is a natural follow-up** — after v9b ships, extend to universal as v10

If v9b succeeds and you want the "go big" follow-up, v9's universal scope becomes a natural extension paper. If you start with v9 and it fails partway, you've spent more time + money.

---

## 11. What to do this week to start

| Task | Time | Output |
|---|---|---|
| Read I-JEPA + V-JEPA original papers | 4 hrs | Foundational understanding |
| Read US-JEPA + MTS-JEPA (closest medical/anomaly work) | 3 hrs | Methodology examples |
| Read AnoDDPM + Pinaya 2022 (closest competitor baselines) | 3 hrs | What you're competing against |
| Set up cloud account + Apply for grants | 3 hrs | Compute secured |
| Download IXI dataset (581 healthy MRI) | 2 hrs | Starting data |
| Sketch JEPA-conformal proof outline + math collaborator outreach | 4 hrs | Phase 5 feasibility check |
| Fork I-JEPA codebase from Meta, adapt for 3D | 4 hrs | Phase 1 code skeleton |
| Set up WandB / Neptune for experiment tracking | 1 hr | Infrastructure |
| **Total** | **~24 hrs** | Phase 0 launched |

---

## 12. Honest summary

**v9b = "JEPA normative anomaly + conformal coverage + brain tumor diagnosis."**

The story is clean:
1. Train JEPA only on healthy brains → learns to predict healthy latents from context
2. On tumor scan, JEPA prediction fails on tumor voxels → that's your anomaly signal in latent space
3. Apply weighted conformal prediction (our Gap B) on the latent error → provable coverage on "this voxel is anomalous"
4. Generate counterfactual healthy version + tumor mesh + MNI152 registration → clinically actionable output
5. LLM Pattern A/B/C/D for hallucination-bounded reporting

**Novelty defensible at review:**
- First normative JEPA for brain MRI (US-JEPA is ultrasound only, I-JEPA for brain is supervised)
- First conformal coverage on JEPA prediction-error maps (any domain)
- First two-tower JEPA-latent + INR/SDF for tumor detection
- First end-to-end normative-JEPA → conformal → counterfactual → LLM-report system

**Realistic and shippable in 2027.** Less ambitious than v9 but higher completion probability.

Execute well. Open-source from day one. Get a math collaborator for Phase 5. Talk to radiologists early.
