# Neural Network from Scratch — Function Approximation

A single-hidden-layer neural network built entirely with **NumPy** (no PyTorch, no TensorFlow), trained to approximate a noisy cubic function **y ≈ x³ + x + noise** on `dane2.txt` (x ∈ [−3, 3]).

---

## Architecture

```
Input (1)  ──►  Hidden Layer (H neurons)  ──►  Output (1)
                       │
                  activation:
               tanh / sigmoid / ReLU
```

- Forward pass: `z1 = x·W1 + b1`, `a1 = act(z1)`, `z2 = a1·W2 + b2` (linear output)
- Loss: **Mean Squared Error (MSE)**
- Weight init: **He** for ReLU, **Xavier** for tanh/sigmoid

---

## Tasks

| Task | Description |
| ---- | ----------- |
| **1** | Load `dane2.txt`, 80/20 train/test split, Z-score normalisation |
| **2** | Batch backprop + `tanh`, H=15 — optimal fit |
| **3** | Underfitting demo (H=1) and overfitting demo (H=60, 40k epochs) |
| **4** | Online (stochastic) training vs. batch training comparison |
| **5** | Replace `tanh` with `ReLU` — piecewise-linear approximation |

---

## Results

| Model | Train MSE | Test MSE | Assessment |
| ----- | --------- | -------- | ---------- |
| Batch + tanh, H=15 (optimal) | 0.01760 | 0.10630 | OVERFITTING (ratio=6.06) |
| Batch + tanh, H=1 (underfit) | 37.53180 | 46.82700 | UNDERFITTING |
| Batch + tanh, H=60, 40k ep. | 0.01770 | 0.13520 | OVERFITTING (ratio=7.65) |
| Online + tanh, H=15 | 0.01280 | 0.01280 | GOOD FIT (ratio=2.21) |
| Batch + ReLU, H=15 | 0.06120 | 0.12910 | GOOD FIT (ratio=2.11) |

**Tasks 2 – 4:**

<img width="1934" height="2278" alt="results_tasks_2_3_4" src="https://github.com/user-attachments/assets/97bcf3dd-762f-498e-8fd9-844855ec7e5a" />

**Task 5 — ReLU:**

<img width="1784" height="740" alt="results_task5_relu" src="https://github.com/user-attachments/assets/7eff27e9-cc84-44e5-a229-d3c15726eca4" />

---

## Key Takeaways

- `tanh` outperforms `ReLU` for smooth function approximation — its smooth, bounded output maps naturally onto continuous regression targets
- **Online SGD** generalises better despite noisier updates; converges faster and stabilises at a lower test MSE than batch
- **Underfitting** (H=1) is immediately visible in the fit plot — one neuron cannot model a cubic relationship
- **Overfitting** is best detected via the test/train MSE ratio, not visual inspection alone
- **Z-score normalisation** is critical — raw x ∈ [−3, 3] causes much slower gradient descent convergence

---

## Stack

Python · NumPy · Matplotlib
