# ML Classification Pipeline — make_moons

A from-scratch machine learning pipeline that benchmarks five classifiers on the **make_moons** dataset, including a hand-rolled logistic regression implementation trained via batch gradient descent.

---

## Overview

| Step | Model | Notes |
|------|-------|-------|
| 1 | **Custom Logistic Regression** | Implemented from scratch in `reglog.py` — no sklearn |
| 2 | **Decision Tree** | Grid search over depth (2–None) and criterion (Gini / Entropy) |
| 3 | **Random Forest** | Sweep over 1–200 estimators |
| 4 | **SVM (RBF kernel)** | Trained on standardised features |
| 5 | **Ensemble (Soft Voting)** | LR + RF + SVM combined via probability averaging |

---

## File Structure

```
.
├── main.py          # Full pipeline: data → models → evaluation → plots
├── reglog.py        # Custom logistic regression (batch gradient descent)
└── outputs/
    ├── decision_boundaries.png        # Decision boundary grid + accuracy bar chart
    ├── decision_tree_analysis.png     # Accuracy vs depth & overfitting analysis
    ├── random_forest_analysis.png     # Accuracy vs number of trees
    └── logistic_regression_loss.png   # Training loss curve
```

---

## Custom Logistic Regression (`reglog.py`)

Implements binary logistic regression from first principles:

- **Numerically stable sigmoid** — uses `e^z / (1 + e^z)` for negative inputs to prevent overflow
- **Binary cross-entropy loss** with probability clipping (`eps = 1e-15`)
- **Xavier weight initialisation** — samples from `N(0, 1/√n_features)`
- **Early stopping** — halts training when `|Δloss| < tol`
- **sklearn-compatible API** — exposes `fit`, `predict`, `predict_proba`, and `score`

```python
from reglog import LogisticRegressionCustom

model = LogisticRegressionCustom(lr=0.5, n_iter=2000, random_state=42)
model.fit(X_train, y_train)
print(model.score(X_test, y_test))
```

---

## Requirements

```
numpy
matplotlib
scikit-learn
```

Install with:

```bash
pip install numpy matplotlib scikit-learn
```

---

## Usage

```bash
python main.py
```

Plots are saved to `outputs/`. The directory is created automatically if it does not exist.

---

## Dataset

`sklearn.datasets.make_moons` — 1 000 samples, noise = 0.25, 80/20 stratified train/test split. Features are standardised with `StandardScaler` (fit on train only to prevent data leakage) before passing to gradient-descent-based or kernel-based models.

---

## Results

| Model | Test Accuracy |
|-------|:-------------:|
| Custom Logistic Regression | 0.9200 |
| Decision Tree (best) | 0.9750 |
| Random Forest (best) | 0.9750 |
| SVM (RBF) | 0.9650 |
| Ensemble (Soft Voting) | 0.9650 |

> The make_moons dataset is non-linearly separable, so models with flexible decision boundaries (Decision Tree, Random Forest) outperform the linear logistic regression. The custom implementation converges cleanly — training loss drops from ~0.47 to ~0.33 in under 150 iterations before early stopping kicks in.

---

## Visualisations

### Decision Boundaries

<img width="2384" height="1520" alt="decision_boundaries" src="https://github.com/user-attachments/assets/fab82cf3-e362-4473-a16a-46e0a1f8a29e" />

The linear boundary of the custom logistic regression clearly struggles with the curved moon shapes. Decision Tree and Random Forest carve more accurate, non-linear regions. SVM with an RBF kernel produces the smoothest curved boundary, while the soft-voting ensemble blends all three approaches.

### Decision Tree — Depth & Criterion Analysis

<img width="2084" height="740" alt="decision_tree_analysis" src="https://github.com/user-attachments/assets/5679bdb5-6599-4c3e-be7a-ebc51ca29055" />

Entropy peaks at depth 7 (0.9750), after which test accuracy drops as the tree overfits. The overfitting plot (right) shows train accuracy reaching 1.00 at `None` depth while test accuracy falls — a textbook bias-variance tradeoff.

### Random Forest — Effect of Number of Trees

<img width="1334" height="730" alt="random_forest_analysis" src="https://github.com/user-attachments/assets/ee0fce4c-8920-497a-a937-178cdbcb2b16" />

Test accuracy stabilises around **0.975** from ~10 trees onward. Notably, test accuracy consistently exceeds train accuracy — a healthy sign that bagging is reducing variance without overfitting.

### Custom Logistic Regression — Training Loss

<img width="1334" height="732" alt="logistic_regression_loss" src="https://github.com/user-attachments/assets/32207237-a024-49da-ba5d-935884a9e84e" />

Loss falls steeply in the first 20 iterations and plateaus by iteration ~60. Early stopping triggers around iteration 150, confirming convergence well before the 2 000-iteration cap.
