"""Research-grade extensions for NeuroLens AI.

Modules in this package contain novel methodological contributions on top
of the production segmentation/classification stack. They are deliberately
kept dependency-light (numpy + optional torch/onnxruntime) so they can run
on the HF Space inference container without GPU.

Currently implemented:
  - conformal_counterfactual_seg: joint conformal + counterfactual brain
    tumor segmentation with provable post-intervention coverage. Combines
    CONSeg-style voxelwise conformal prediction sets with CausalX-Net-style
    counterfactual segmentations under modality / intensity / contrast
    interventions, using weighted conformal prediction (Tibshirani et al.
    2019) to lift coverage from the factual to the post-intervention
    distribution. As of the last literature pass (May 2026) the two are
    not unified in a single segmentation framework anywhere we could find.
"""
