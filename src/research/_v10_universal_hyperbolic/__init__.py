"""v10 (parked) Universal Causal-Hyperbolic Tumor Foundation Model code.

Status: parked for future v10 follow-up paper. Not used by current production.
v9b (Normative JEPA + Conformal) is the active research direction.

Reusable from this package (if v10 ever resumes):
  - hyperbolic.py        Poincare ball math + Mobius ops + HyperbolicLinear
  - causal_scm.py        Latent SCM with NOTEARS DAG constraint
  - hyperbolic_conformal.py  Weighted conformal with hyperbolic geodesic scores
  - v10_model.py         Integrated v10 model (uses parent-level geometric_prior +
                         counterfactual_decoder)

Shared with v9b at the parent src/research/ level:
  - geometric_prior.py
  - counterfactual_decoder.py
"""
