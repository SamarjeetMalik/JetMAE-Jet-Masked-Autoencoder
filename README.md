# JetMAE — Jet Masked Autoencoder for Quark/Gluon Tagging

A masked autoencoder pretrained on IRC-safe jet observables, with follow-on interpretability showing that colour-factor observables explain ~96% of the tagger's decisions.

---

## Highlights

| Model | AUC | Accuracy |
|---|---|---|
| **JetMAE (pretrained → fine-tuned)** | **0.8769** | **80.02%** |
| Baseline (same architecture, no pretraining) | 0.8731 | 79.53% |
| BDT (300 trees, top-12 observables) | 0.8719 | — |
| Multiplicity only | 0.8409 | — |
| PySR symbolic regression | 0.8409 | — |

- Pretraining provides a consistent **+0.38% AUC gain** (mean +1.35% across 3 seeds).
- A 6-node symbolic equation `(n_mult × −0.229 + 0.663)²` recovers **95.9% of JetMAE's raw AUC** and **90.5% of its excess over the multiplicity baseline**.
- Top-5 SHAP features: `mult`, `C1_b02`, `C1_b10`, `LHA`, `width`.
- Typed mask tokens outperform shared mask tokens by +0.0039 AUC.
- Background rejection at signal efficiency 30%: JetMAE **72.2**, Baseline 71.2, BDT 69.4.

---

## Repository Structure

```
PAI26_submission_patched.ipynb          main Colab notebook (all sections)
pai26_submission_patched.py             script export of the notebook
ALL_results.json                        consolidated metrics for all experiments

# Model checkpoints
mae_pretrained.pt                       pretrained MAE encoder weights
clf_pretrained.pt                       MAE-initialized JetClassifier
clf_baseline.pt                         from-scratch JetClassifier (no pretraining)
abl_mae_mr{15,30,45}.pt                 MAE checkpoints for mask-ratio ablation
abl_clf_mr{15,30,45}.pt                 classifier checkpoints for mask-ratio ablation

# Data
observables.npz                         precomputed 12 jet observables (train/val/test)
seed_study.npz                          per-seed AUC arrays for stability study
shap__shap_values.npy                   SHAP values on the test set

# PySR symbolic regression
pysr_pretrained__eqs_pretrained.csv     full equation table (MAE-pretrained)
pysr_pretrained__pareto_pretrained.csv  Pareto-front table (MAE-pretrained)
pysr_baseline__eqs_baseline.csv         full equation table (baseline)
pysr_baseline__pareto_baseline.csv      Pareto-front table (baseline)

# Figures (PDF)
roc_full.pdf                            ROC curves for all models
roc_restframe.pdf                       rest-frame vs lab-frame ROC
confusion_matrix.pdf                    confusion matrix (rest-frame BDT)
shap_importance.pdf / shap_beeswarm.pdf SHAP bar and beeswarm plots
ablation_mask.pdf                       mask-ratio ablation AUC vs ratio
pysr_pareto_pretrained.pdf              PySR Pareto front (pretrained)
pysr_pareto_baseline.pdf                PySR Pareto front (baseline)
observable_distributions.pdf           12 observable distributions Q vs G
mult_distribution.pdf                  constituent multiplicity Q vs G
leading_constituent.pdf                leading-constituent pT and η
com_scatter.pdf                        constituent momenta in jet rest frame
pretrain_loss.pdf                      MAE pretraining loss curve
```

---

## Dataset

The notebook loads **Pythia8 quark/gluon jet files** (`QG_jets.npz`, `QG_jets_1.npz` … `QG_jets_10.npz`) with constituent arrays of shape `(N_jets, M_constituents, 4)` — columns `(pT, η, φ, pdgid)`. Labels are binary: 1 = quark, 0 = gluon.

> **TODO**: confirm the exact dataset version/DOI (EnergyFlow QG dataset vs private Pythia8 generation).

The 12 precomputed observables stored in `observables.npz` are:

`mass`, `width`, `pTD`, `mult`, `LHA`, `thrust`, `C1_β=0.2`, `C1_β=1.0`, `C1_β=2.0`, `ang12`, `ang20`, `ECF3`

---

## Model Architecture

**ObservableMAE** — a Transformer encoder operating on observable tokens:

| Hyperparameter | Value |
|---|---|
| `d_model` | 128 |
| `n_heads` | 8 |
| `n_layers` | 6 |
| `dropout` | 0.10 |
| `mask_ratio` | 0.30 |
| `mask_token_mode` | `typed` (observable-index embedding added to mask token) |

Each of the 12 observables is projected to `d_model` via a shared linear layer and summed with a learned type embedding. A CLS token is prepended. The encoder is a standard Pre-LN Transformer. For MAE pretraining, masked positions are replaced by `mask_token + type_embed[j]`; the reconstruction head (LayerNorm → Linear → GELU → Linear) predicts the scalar value of each masked observable.

**JetClassifier** reuses the MAE encoder and appends a classification head on the CLS token. Fine-tuning: encoder frozen for 5 epochs (`lr=5e-5`), then unfrozen.

---

## Pipeline

```
1. Pretrain
   ObservableMAE on training jets, 30% random masking per jet,
   MSE reconstruction loss. Warmup 8 epochs, AdamW lr=3e-4, wd=1e-4.
   → mae_pretrained.pt

2. Fine-tune (MAE-init)
   JetClassifier initialised from pretrained encoder.
   Freeze encoder 5 epochs → unfreeze. BCE + label-smoothing 0.05.
   → clf_pretrained.pt

3. Baseline
   Same JetClassifier trained from scratch (no pretraining).
   → clf_baseline.pt

4. Ablation
   Mask ratios {0.15, 0.30, 0.45} × {pretrain, fine-tune}.
   → abl_mae_mr*.pt, abl_clf_mr*.pt

5. Interpret
   a. SHAP (KernelExplainer) on clf_pretrained → shap_values.npy
   b. PySR symbolic regression on top-5 SHAP features
      → pareto_*.csv, pysr_*.pdf

6. Evaluate
   AUC, accuracy, background rejection at ε_S ∈ {0.3, 0.5},
   rest-frame vs lab-frame comparison, multi-seed stability.
   → ALL_results.json, *.pdf
```

---

## Getting Started

```bash
# 1. Clone
git clone https://github.com/SamarjeetMalik/JetMAE-Jet-Masked-Autoencoder.git
cd JetMAE-Jet-Masked-Autoencoder

# 2. Create environment (Python ≥ 3.10 recommended)
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install torch numpy matplotlib scikit-learn tqdm pandas shap pysr

# 4. Set paths
#    Edit DATA_DIR and OUTPUT_DIR at the top of pai26_submission_patched.py
#    (or the first notebook cell) to point to your QG_jets*.npz files
#    and the pre-computed outputs (observables.npz, *.pt, etc.).

# 5. Run
#    Notebook (recommended, GPU T4 on Colab):
jupyter notebook PAI26_submission_patched.ipynb

#    Script:
python pai26_submission_patched.py
```

> All heavy computations (MAE pretraining, SHAP, PySR) are cached.
> With the provided `.pt` and `.npy` files the full notebook reruns in ~30 min on a T4 GPU.

---

## Results

### ROC and Rejection

| Model | AUC | Rejection @ ε_S=0.3 | Rejection @ ε_S=0.5 |
|---|---|---|---|
| JetMAE | **0.8769** | **72.2** | **25.5** |
| Baseline | 0.8731 | 71.2 | 24.2 |
| BDT | 0.8719 | 69.4 | 23.8 |
| Multiplicity | 0.8409 | 56.7 | 18.4 |
| Symbolic (PySR) | 0.8409 | — | — |

See [`roc_full.pdf`](roc_full.pdf).

### Mask-Ratio Ablation

| Mask ratio | AUC |
|---|---|
| 0.15 | 0.8739 |
| **0.30** | **0.8769** (default) |
| 0.45 | 0.8708 |

See [`ablation_mask.pdf`](ablation_mask.pdf).

### Symbolic Regression (PySR)

Best Pareto-elbow equation for the **MAE-pretrained** SHAP scores (complexity 6):

```
score = (n_mult × −0.22871 + 0.66309)²
```

AUC = 0.8409 → **95.9% raw recovery**, **90.5% excess recovery** over multiplicity baseline.  
See [`pysr_pareto_pretrained.pdf`](pysr_pareto_pretrained.pdf).

### Multi-Seed Stability (n=3)

| Model | Mean AUC | Std |
|---|---|---|
| JetMAE | 0.8660 | 0.0021 |
| Baseline | 0.8525 | 0.0007 |

---

## Citation

If you use this work, please cite:

```bibtex
@misc{jetmae2025,
  title        = {{JetMAE}: Masked Autoencoder Pretraining for Quark/Gluon Jet Tagging
                  with Symbolic Interpretability},
  author       = {Malik, Samarjeet},
  year         = {2025},
  howpublished = {\url{https://github.com/SamarjeetMalik/JetMAE-Jet-Masked-Autoencoder}},
  note         = {PAI26 submission}
}
```
