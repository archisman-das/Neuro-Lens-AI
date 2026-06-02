# v9 — Universal Causal-Hyperbolic-Geometric Tumor Foundation Model with Conformal Coverage

**Working title (paper):** "UCHF-Tumor: Universal Causal-Hyperbolic Foundation Model with Conformal Coverage for Multi-Organ Tumor Diagnosis"

**Status:** Research proposal, May 2026
**Estimated effort:** 8-12 months focused work, ~2-3 papers worth of contribution
**Estimated cloud compute:** $8-20K (or $0 with academic grant)

---

## 1. Problem statement

Current tumor detection systems share four limitations:

1. **Organ-siloed.** A clinician hospital uses 4-7 separate models (one for brain, one for liver, one for kidney, etc.). Maintenance, validation, and clinical integration scale linearly with organ count.
2. **No causal disentanglement.** "Tumor signal", "anatomical variation", and "scanner artifact" are entangled in the latent representation. A model trained on Siemens 3T scans degrades on GE 1.5T because the latent encodes scanner identity as tumor evidence.
3. **No formal uncertainty.** Outputs are point estimates (a mask, a probability). Clinicians need prediction sets with coverage guarantees, especially for treatment decisions.
4. **Euclidean latent only.** Brain anatomy and tumor taxonomy are both hierarchical (cortex → regions → networks → voxels; glioma → GBM → IDH-wt → grade IV). Euclidean embeddings distort these hierarchies; hyperbolic embeddings preserve them with provably lower distortion.

We propose a single foundation model that addresses all four jointly.

---

## 2. Prior art and what is genuinely novel

### 2.1 Each component already exists in isolation

| Component | Existing work | Reference |
|---|---|---|
| Universal whole-body tumor segmentation | SAT3D — 17,075 3D volumes, multi-modality, multi-organ | arXiv:2511.09592 (Nov 2025) |
| Causal SCM for brain tumor | CausalX-Net — SCM + interventional reasoning, brain only | Frontiers Medicine 2025 |
| Latent SCM (general) | Learning Latent Structural Causal Models | arXiv:2210.13583 |
| Causal foundation model | "Towards Causal Foundation Model: Duality Between Causal Inference and Attention" | arXiv:2310.00809 |
| Hyperbolic medical segmentation | Hierarchical Compositionality in Hyperbolic Space | ResearchGate 374724459 |
| Poincaré-UNet for medical | Poincaré-guided geometric UNet for cardiac MRI | PMC 12263985 (2025) |
| Hyperbolic for pathology | Multi-scale follicular lymphoma in hyperbolic space | arXiv:2506.18523 |
| INR/SDF cortical surface | Weakly-supervised cortical surface reconstruction | arXiv:2406.12650 |
| Conformal segmentation | CONSeg — voxelwise conformal prediction sets | AJNR 2025 |
| Weighted conformal under shift | Tibshirani, Foygel Barber, Candes, Ramdas | NeurIPS 2019 |
| Counterfactual segmentation | CausalX-Net (brain), CounterFactualSeg (general) | 2024-2025 |
| Conformal-counterfactual segmentation | **Our Gap B (this project)** | This work, May 2026 |
| Hallucination-bounded LLM medical reports | **Our Pattern A/B/C/D (this project)** | This work, this session |

### 2.2 The 5-way intersection has not been done

Verified through literature search: **no published work integrates all five of:**

1. Universal multi-organ tumor scope (SAT3D has this; others don't)
2. Latent SCM disentanglement (CausalX-Net has this for brain; nobody extends to multi-organ)
3. Hyperbolic geometric embedding of hierarchies (organ + tumor taxonomy)
4. Conformal prediction in hyperbolic space + on causal counterfactuals (open mathematically)
5. Counterfactual healthy generation + LLM hallucination bounds (we have B+D in 2D; not in 3D + universal)

The genuinely novel contribution is **the integration + the hyperbolic-conformal mathematics**.

### 2.3 The mathematically hard part

**Conformal prediction in hyperbolic space.** Classical conformal prediction assumes a Euclidean nonconformity score. When the latent representation lives on the Poincaré ball, the natural distance is the hyperbolic geodesic distance, not Euclidean. We extend weighted split conformal prediction (Tibshirani et al. 2019) to nonconformity scores defined as hyperbolic geodesic distances:

$$ s_i = d_{\mathbb{H}^n}(z_i, \hat{y}_i) = 2\,\text{arctanh}\!\left(\left\| (-z_i) \oplus \hat{y}_i \right\|\right) $$

where $\oplus$ is the Möbius addition on the Poincaré ball. The coverage guarantee carries over because conformal prediction only requires score exchangeability, not Euclidean structure — but the practical implementation (quantile estimation, weight computation under modality intervention) needs new analysis. **No published work has done this for medical imaging or in general.**

### 2.4 Honest novelty claim

> First foundation model that jointly addresses universal-organ tumor detection with causal-disentangled hyperbolic latent representation, organ-conditioned geometric priors, weighted conformal coverage guarantees on hyperbolic non-conformity scores, counterfactual healthy generation, and hallucination-bounded textual reporting in a single end-to-end system.

The framing avoids overclaiming any single piece as novel while defending the integration + the hyperbolic-conformal mathematics as genuinely first-of-kind.

---

## 3. Architecture

```
Input: multi-modal 3D volume (any organ, any modality)
                            │
                            ▼
        ┌──────────────────────────────────────────┐
        │ Organ classifier (SAT3D-style routing)    │ → organ_id
        └──────────────────────────────────────────┘
                            │
                            ▼
        ┌──────────────────────────────────────────┐
        │ Multi-modal 3D ViT encoder (organ-shared) │ → z_euclidean
        └──────────────────────────────────────────┘
                            │
                            ▼
        ┌──────────────────────────────────────────┐
        │ Organ-specific geometric prior (SDF/INR)  │ → g_organ conditioning
        │ (per-organ SDF template + INR head)       │
        └──────────────────────────────────────────┘
                            │
                            ▼
        ┌──────────────────────────────────────────┐
        │ Hyperbolic projector (exp map → Poincaré) │ → z_hyperbolic ∈ 𝔹ⁿ
        │ Hierarchy: tissue → region → network →    │
        │            tumor presence → type → grade  │
        └──────────────────────────────────────────┘
                            │
                            ▼
        ┌──────────────────────────────────────────┐
        │ Latent SCM (Möbius-aware GNN)             │ → disentangled:
        │ Nodes: z_tumor, z_anatomy, z_scanner      │   z_tumor, z_anatomy,
        │ Edges learned via DAG-NoTears extension   │   z_scanner
        │ to hyperbolic space                       │
        └──────────────────────────────────────────┘
                            │
                            ▼
        ┌──────────────────────────────────────────┐
        │ 3D segmentation decoder (organ-aware)     │ → tumor mask M
        └──────────────────────────────────────────┘
                            │
                            ▼
        ┌──────────────────────────────────────────┐
        │ Weighted hyperbolic conformal calibration │ → coverage-bounded
        │ s = d_𝔹ⁿ(z, ŷ), Möbius quantile,         │   prediction sets
        │ importance weights for intervention shift  │
        └──────────────────────────────────────────┘
                            │
                            ▼
        ┌──────────────────────────────────────────┐
        │ Counterfactual decoder: do(z_tumor = ∅)   │ → healthy organ render
        │ Renders "no-tumor" version of input        │
        └──────────────────────────────────────────┘
                            │
                            ▼
        ┌──────────────────────────────────────────┐
        │ LLM Pattern A/B/C/D reporter              │ → hallucination-
        │ Polish / differential / visual /          │   bounded report
        │ negative-case reasoning                    │
        └──────────────────────────────────────────┘
```

### 3.1 Key architectural choices

- **Encoder**: 3D Vision Transformer pretrained on healthy multi-organ MRI+CT (modeled on SAT3D with hyperbolic projection head added)
- **Geometric prior**: per-organ pre-computed SDF templates (brain: MNI152 cortical; liver: standard atlas; etc.) + small INR head learned per-organ to capture patient-specific anatomy
- **Hyperbolic dimension**: $n=64$ Poincaré ball, deep enough to encode hierarchies, shallow enough for efficient Möbius operations
- **SCM structure**: 3-node DAG ($z_{tumor} \to M \leftarrow z_{anatomy} \leftarrow z_{scanner}$), edges learned via differentiable DAG constraint
- **Decoder**: SMP-style 3D UNet with hyperbolic skip connections (log/exp maps at each level)
- **Conformal head**: extends our Gap B to hyperbolic distances + multi-organ calibration sets

---

## 4. Phased plan with milestones

| Phase | Duration | Deliverables | Risks |
|---|---|---|---|
| **0. Foundations** | 2 weeks | Lit review final; math derivations for hyperbolic-conformal; cloud-compute account; data pipeline | Mathematical correctness of hyperbolic conformal — needs proof |
| **1. Universal tumor encoder** | 6 weeks | SAT3D-style encoder trained on BraTS + LiTS + KiTS + AMOS (~30K 3D volumes); baseline universal Dice numbers | Multi-organ data harmonization; modality variance |
| **2. Hyperbolic projection** | 4 weeks | Poincaré projector head; verify hierarchy preservation via embedding analysis | First-of-kind for tumors; numerical stability of Möbius ops |
| **3. SCM extension** | 5 weeks | Latent SCM head; verify causal disentanglement on synthetic interventions | DAG learning is unstable; needs careful regularization |
| **4. Geometric prior integration** | 3 weeks | Per-organ SDF/INR templates; verify they improve segmentation Dice over baseline | SDF templates exist for brain (MNI152); need to build for other organs |
| **5. Hyperbolic conformal extension** | 6 weeks | Math + implementation; empirical coverage verification on held-out organs | **Highest research risk** — novel mathematics |
| **6. Counterfactual healthy generation** | 4 weeks | Diffusion decoder for do(z_tumor=∅); FID + radiologist eval | Generative quality on multi-organ |
| **7. End-to-end integration + LLM Pattern A/B/C/D** | 3 weeks | Full pipeline; clinical reports with hallucination bounds | Already have Pattern A-D in 2D; extending to 3D + multi-organ |
| **8. Multi-organ evaluation** | 4 weeks | Held-out test sets across 5+ organs; coverage validation; radiologist comparison | Need ~3-5 board-certified radiologists for eval |
| **9. Write paper + open-source release** | 4 weeks | TMI / Nature Methods / MICCAI submission; code + weights public | Time-sensitive: field moves fast |
| **Total** | **~10-11 months** | Publishable system + paper | — |

---

## 5. Resource requirements

### 5.1 Compute

- **Pretraining (Phase 1)**: 4-8× A100/H100 for ~2 weeks. Cloud cost: $5-12K. Free with NIH NCATS or NVIDIA Academic Hardware Grant.
- **Fine-tuning + ablations (Phases 2-7)**: 1-2× A100/H100 for ~3 months intermittent. Cost: $2-5K.
- **Local development & eval**: laptop fine (your Legion Pro 5 once thermal is fixed) for code dev, small experiments.

### 5.2 Data

| Dataset | Organ | Modality | Volumes | License |
|---|---|---|---|---|
| BraTS 2024 | Brain | MRI multi-modal | ~5K | Free academic |
| Figshare (Cheng) | Brain | T1c | 3K | Free |
| LiTS | Liver | CT | ~200 | Free academic |
| KiTS23 | Kidney | CT | ~600 | Free academic |
| AMOS | Multi-organ abdomen | CT + MRI | ~600 | Free academic |
| TotalSegmentator | Whole-body | CT | ~1,200 | Free CC-BY |
| MSD (Medical Segmentation Decathlon) | 10 organs | Mixed | ~2K | Free academic |
| Prostate158 / PROSTATEx | Prostate | MRI | ~300 | Free academic |
| **Total** | | | **~13K labeled 3D volumes** | All free |

For pretraining (unsupervised normative): can add ~50K unlabeled volumes from UK Biobank (requires application), IXI, ADNI.

### 5.3 Human

- **You** (project lead): 40-60% time for 10-11 months
- **Math collaborator** (highly recommended): 10-20% time for hyperbolic conformal proofs (Phase 5). Find via your university math/stats department.
- **3-5 board-certified radiologists**: 10-20 hours each for Phase 8 evaluation (compensate via co-authorship or honorarium)

### 5.4 Other

- WandB Pro / Neptune for experiment tracking ($50-100/mo)
- Overleaf paper drafting ($0-100)
- Conference travel for MICCAI presentation (~$3-5K)

---

## 6. Evaluation methodology

### 6.1 Quantitative metrics

| Metric | Why | Target |
|---|---|---|
| Per-organ Dice (macro + micro) | Standard segmentation | ≥ 0.85 micro on brain, ≥ 0.80 on others |
| Universal Dice (averaged across organs) | Foundation model quality | ≥ 0.80 |
| Conformal empirical coverage at α=0.1 | Validates the math | 0.88-0.92 (within 2% of nominal) |
| Coverage under modality shift | Validates weighted conformal | 0.85-0.92 (some slack, intervention-aware) |
| Counterfactual fidelity (FID) | Generative quality | ≤ 20 |
| Causal disentanglement score | SCM head working | Measure via known synthetic interventions |
| Hyperbolic hierarchy preservation | Hyperbolic embedding works | Distortion ratio ≤ 1.2 vs ideal tree embedding |
| Inference latency per volume | Clinical feasibility | ≤ 5 sec on A100, ≤ 30 sec on RTX 4090 |

### 6.2 Clinical evaluation (Phase 8)

- Blind comparison with 3 board-certified radiologists
- 100 randomly-sampled cases across 5+ organs
- For each case: (a) model prediction with conformal sets + report, (b) radiologist independent diagnosis
- Measure: agreement rate, time-to-diagnosis with vs without model, radiologist trust in conformal bounds

### 6.3 Ablation studies

- Without hyperbolic head (Euclidean only): measure Dice drop
- Without SCM: measure cross-scanner generalization drop
- Without geometric prior: measure organ-specific accuracy drop
- Without conformal head: shows raw point estimates
- Without counterfactual decoder: shows segmentation only
- Without LLM reports: shows numbers only

Each ablation isolates one contribution.

---

## 7. Target publication venues

| Venue | Fit | Timing |
|---|---|---|
| **MICCAI 2027** (March 2027 deadline) | Strong fit (medical imaging) | Submit March 2027 if Phase 1-7 complete by Feb |
| **Nature Methods** | Strong fit (methodology) | Rolling submission, 4-6 mo review |
| **IEEE TMI (Transactions on Medical Imaging)** | Strong fit, longer paper | Rolling submission, 6-12 mo review |
| **Medical Image Analysis** | Strong fit | Rolling, 6-9 mo |
| **NeurIPS 2026** (May 2026 deadline — past!) | If hyperbolic-conformal math is strong | Miss this cycle; aim 2027 |
| **ICML 2027** (Jan 2027 deadline) | If math is the headline | Possible if Phase 5 strong |
| **ICLR 2027** (Sept 2026 deadline) | If foundation model + integration is headline | Possible if Phase 1-4 complete |

**Recommended primary target: TMI** (longer paper format suits the integrated system) with **MICCAI 2027 short companion paper** (focused on hyperbolic-conformal math).

---

## 8. Risk assessment

| Risk | Likelihood | Mitigation |
|---|---|---|
| Hyperbolic conformal math doesn't yield clean theorem | Medium | Fallback: Euclidean conformal on tangent-space projections; still publishable as "first hybrid hyperbolic-Euclidean conformal" |
| Multi-organ data harmonization fails (modality/spacing/normalization mismatches) | High | Established preprocessing pipelines (nnU-Net preprocessing, TorchIO); allocate Phase 1 buffer |
| SCM disentanglement doesn't actually disentangle (latent collapse) | Medium | Use established disentanglement metrics (Pearl, Locatello); regularize via interventional augmentation |
| Hyperbolic embeddings numerically unstable (gradient explosion near boundary) | Medium | Use Lorentz model instead of Poincaré ball (more stable); gradient clipping |
| Foundation model pretraining doesn't converge on multi-organ | Low-Medium | SAT3D demonstrates this is feasible; reuse their pretraining recipe |
| Radiologists unavailable for evaluation | Low | Start outreach Phase 6, offer co-authorship |
| Compute budget exhausted | Medium | Apply for NIH/NVIDIA/AWS academic grants Phase 0; budget conservatively |
| Field publishes "everything" before you do | High in 2026 | **Mitigation**: prioritize integration + execution quality + clinical eval (hard to scoop those); release code+weights publicly |
| You burn out | Real risk | Realistic time budget (40-60% time, not 100%); milestone check-ins every 6 weeks |

---

## 9. Connection to current NeuroLens AI codebase

### What carries forward directly

| v9 component | Reuses from current project |
|---|---|
| Conformal coverage extension | **`src/research/conformal_counterfactual_seg.py`** (Gap B, this session) |
| Counterfactual decoder | Extends our `IdentityIntervention` / `ModalityIntervention` / etc. battery |
| LLM Pattern A/B/C/D reporter | **`src/llm_explain.py`** (this session) |
| Per-modality intervention battery | `standard_intervention_battery()` |
| Dashboard UI for visualization | Extends current `web_dashboard/` with 3D volumetric viewer |
| HF Spaces deployment infra | Current Space architecture scales — just need larger weights repo |
| Conformal calibration script | **`scripts/calibrate_conformal_counterfactual.py`** as template |
| MedSAM cascade refiner | **`src/research/medsam_refiner.py`** as the "refine after universal segmenter" step |

### What needs to be built new

| New component | Estimated lines | Difficulty |
|---|---|---|
| 3D ViT encoder (multi-organ pretrain) | ~2,000 LoC | Medium (existing libraries) |
| Hyperbolic projection layer + Möbius ops | ~500 LoC | High (numerical care) |
| Latent SCM with DAG-NoTears extension | ~1,500 LoC | High (research code) |
| Per-organ SDF/INR template library | ~1,000 LoC | Medium |
| Hyperbolic-aware conformal calibration | ~800 LoC | **Very high** (novel math) |
| Counterfactual diffusion decoder | ~2,500 LoC | High (3D diffusion is heavy) |
| Multi-organ data pipeline | ~1,500 LoC | Medium |
| Evaluation framework | ~1,000 LoC | Medium |
| **Total new code** | **~10,800 LoC** | — |

### Branch strategy

```
main                  ← v8 (current production)
  └─ research/v9      ← v9 development
      ├─ phase-1/universal-encoder
      ├─ phase-2/hyperbolic-projection
      ├─ phase-3/scm-head
      ├─ phase-4/geometric-prior
      ├─ phase-5/conformal-hyperbolic
      ├─ phase-6/counterfactual-diffusion
      └─ phase-7/integration
```

Merge research/v9 back to main only when ready to deploy; v8 stays in production through the entire research period.

---

## 10. Honest recommendations

1. **Don't try to do this on your laptop.** Cloud GPU is mandatory. Apply for academic compute grants Phase 0.
2. **Find a math collaborator** for the hyperbolic conformal extension. Without this, that phase becomes implementation-only and the novelty claim weakens.
3. **Phase 0-1 are go/no-go.** If the universal encoder doesn't converge on multi-organ in Phase 1, scope down to single-organ (brain only) for v9 and save universal scope for v10.
4. **Publish incrementally.** Don't wait until Phase 9 to write. Each phase can yield a workshop paper or arXiv preprint that establishes priority and gathers feedback.
5. **Open-source from day one.** Publish code + checkpoints under permissive license. This is your defense against being scooped — community adoption of your model becomes the moat.
6. **Connect to clinical practice.** Start radiologist outreach in Phase 4 (not Phase 8). Clinical buy-in throughout shapes the system better than evaluation at the end.

---

## 11. What to do this week to start

| Task | Time | Output |
|---|---|---|
| Read SAT3D paper end-to-end + their code | 4 hrs | Understand the universal-organ baseline you're extending |
| Read CausalX-Net + Latent SCM papers | 4 hrs | Understand the causal head you'll generalize |
| Read Poincaré-UNet + Hierarchical Hyperbolic papers | 4 hrs | Understand existing hyperbolic medical work |
| Read Tibshirani et al. 2019 (weighted conformal) | 2 hrs | Math foundation for Phase 5 |
| Sketch hyperbolic conformal proof outline | 4 hrs | Decide if Phase 5 math is tractable; consult math collaborator |
| Apply for NIH NCATS or NVIDIA Academic Hardware Grant | 4 hrs | Compute budget secured |
| Download BraTS 2024 + LiTS + KiTS23 + AMOS | 4 hrs | Data ready for Phase 1 |
| Set up cloud account (RunPod/Lambda Labs/Modal) | 1 hr | Compute infra |
| Total | **~27 hrs** | Phase 0 launched |

If you do this Phase 0 prep work in week 1 and the math sketch (item 5) yields a tractable proof outline, **Phase 1 starts week 2 and the full project is 10-11 months from today**.

---

**This is a serious, publishable, paradigm-relevant project.** The integration + the hyperbolic-conformal mathematics + the clinical pipeline are genuinely first-of-kind. The 5-way intersection has not been published as of May 2026, and even partial completion (e.g., brain-only Phase 1-7) is a strong contribution.

Execute well. Don't rush. Get a math collaborator. Open-source. Talk to radiologists early.
