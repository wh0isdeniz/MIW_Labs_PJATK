# Project 04 — CIFAR-10 CNN

Two convolutional architectures benchmarked on **CIFAR-10**: a shallow baseline (Model A) and a deeper ResNet-style network (Model B). The goal is to compare a straightforward 3-block CNN against a modern residual architecture and quantify the impact of skip connections, deeper feature hierarchies, and staged regularisation.

---

## Results

| Model | Params | Test Loss | Test Accuracy |
|-------|:------:|:---------:|:-------------:|
| **Model A** — 3-block CNN | 324 K | 0.6551 | **79.16 %** |
| **Model B** — ResNet-style | 11.3 M | 0.3565 | **90.13 %** |

Model B comfortably clears the 80 % target and improves over the baseline by **~11 percentage points**.

---

## Architectures

### Model A — 3-block CNN (baseline)

Three convolutional blocks with progressively more filters and dropout:

- `Conv → BN → Conv → BN → MaxPool → Dropout` × 3 (filters: 32 → 64 → 128)
- `GlobalAveragePooling2D → Dense(256, L2) → Dropout(0.5) → Softmax`

### Model B — ResNet-style

Stem + four residual stages with skip connections:

- Stem: `Conv(64) → BN → ReLU`
- Four stages of two residual blocks each (filters: 64 → 128 → 256 → 512), with stride-2 downsampling between stages
- Projection shortcuts when filter count or stride changes
- Staged dropout (0.1 → 0.4) after each stage
- `GlobalAveragePooling2D → Dense(256, L2) → Dropout(0.5) → Softmax`

Both models use **Adam (lr = 1e-3)**, **sparse categorical cross-entropy**, **EarlyStopping**, and **ReduceLROnPlateau**.

---

## Dataset & Pipeline

- **CIFAR-10** — 60 000 colour images (32×32×3) across 10 classes
- Custom **60 / 20 / 20** split (36 000 train / 12 000 val / 12 000 test) for a cleaner three-way evaluation
- Pixel values normalised to `[0, 1]`
- **In-graph augmentation** applied only at training time: horizontal flip, rotation (±10 %), zoom (±10 %), translation (±10 %)

<img width="2147" height="724" alt="fig0_samples" src="https://github.com/user-attachments/assets/aa8683a0-c56d-422a-aa2d-52d9b7146375" />

---

## Training

Both models trained for **80 epochs** with the same optimiser settings.

<img width="2234" height="1475" alt="fig1_training_curves" src="https://github.com/user-attachments/assets/c57057ec-9a1b-4a10-b547-02511c4dd2a9" />

Model A plateaus around 80 % validation accuracy with train and validation curves staying close — a healthy fit with no significant overfitting, but limited capacity. Model B keeps climbing past 90 % validation accuracy; the larger train/val gap reflects its capacity, but BatchNorm and staged dropout keep it under control.

---

## Per-Class Accuracy

<img width="2082" height="884" alt="fig3_per_class_accuracy" src="https://github.com/user-attachments/assets/d95bc6a5-bd0d-4f39-a8c3-10eebb40283c" />

Model A's weakest classes — **cat (0.54)**, **dog (0.67)**, **bird (0.68)** — are exactly the visually similar animal categories that benefit most from deeper features. Model B lifts all three above the 80 % threshold (cat: 0.80, dog: 0.81, bird: 0.89).

---

## Confusion Matrices

<img width="2877" height="1183" alt="fig2_confusion_matrices" src="https://github.com/user-attachments/assets/baab6b33-f956-4ca6-af57-3e2a4a65dcb4" />

The dominant off-diagonal cells in Model A — `cat ↔ dog`, `bird → frog`, `deer → frog` — shrink substantially under Model B. The remaining confusion concentrates on the semantically closest pairs (cat/dog, deer/horse), which is consistent with human-level confusion on 32×32 thumbnails.

---

## Misclassified Examples (Model B)

<img width="2985" height="1621" alt="fig4_misclassifications" src="https://github.com/user-attachments/assets/58cb0f8c-1556-4092-aaa3-981abf47b79b" />

Inspecting the errors reveals that most are genuinely ambiguous at 32×32 resolution — partially occluded animals, unusual poses, or strong background bias (e.g. a cat against a red flag predicted as a dog). Few errors look like obvious model failures.

---

## Summary

<img width="2082" height="887" alt="fig5_summary" src="https://github.com/user-attachments/assets/1a072974-e3f1-4318-a360-f090e5a0d71c" />

**Why Model B wins:**

- **Skip connections** keep gradients flowing through deeper layers
- **Deeper feature hierarchy** captures finer-grained visual cues
- **BatchNorm + staged dropout** controls overfitting despite 35× more parameters
- **GlobalAveragePooling** replaces a large fully-connected layer, keeping the parameter count reasonable

---

## Stack

Python · TensorFlow / Keras · NumPy · scikit-learn · Matplotlib · Seaborn

---

## Usage

```bash
pip install tensorflow numpy matplotlib seaborn scikit-learn
python Project04_CIFAR10_CNN.py
```

All six figures (`fig0_samples.png` through `fig5_summary.png`) are saved to the working directory.
