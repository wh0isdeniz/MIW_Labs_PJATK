# Knowledge Engineering Methods (MIW)

Project assignments for the course **Knowledge Engineering Methods (MIW)**  
at the **Polish-Japanese Academy of Information Technology (PJATK)**.

- All assignments were **graded 10/10**

---

## Topics Covered

- **Bayesian inference** — probabilistic belief updating from sequential observations
- **Markov models** — state-based transition modeling for sequential data
- **Adaptive strategies** — agents that learn opponent behaviour and optimise decisions over time
- **Supervised learning** — classical classification algorithms benchmarked on non-linear data

---

## Projects

### 01 · Rock–Paper–Scissors with Bayesian Learning

An adaptive agent that learns an opponent's strategy in real time:

- Opponent modelled as a **Markov chain** — moves depend only on the previous move
- Agent applies **Bayesian updating** to refine its estimate of the transition matrix after each round
- Predicts the opponent's next move and selects the optimal counter-action
- Simulated over **1 000 rounds** with win-rate tracking throughout

**Output:** learned transition matrix + score evolution plot

---

### 02 · ML Classification Pipeline — make_moons

A benchmark pipeline comparing five classifiers on a non-linearly separable dataset:

| Model | Test Accuracy |
|-------|:-------------:|
| Custom Logistic Regression | 0.9200 |
| Decision Tree (best) | 0.9750 |
| Random Forest (best) | 0.9750 |
| SVM (RBF) | 0.9650 |
| Ensemble (Soft Voting) | 0.9650 |

- **Custom logistic regression** implemented from scratch — batch gradient descent, Xavier init, early stopping
- **Decision Tree** grid-searched over depth (2–None) and criterion (Gini / Entropy)
- **Random Forest** swept over 1–200 estimators
- **SVM** with RBF kernel on standardised features
- **Soft-voting ensemble** combining LR + RF + SVM via probability averaging

**Output:** decision boundary plots, depth & overfitting analysis, training loss curve

---

## Stack

Python · NumPy · scikit-learn · Matplotlib

---

## Notes

These are my own solutions to the laboratory tasks — shared as study material and reference implementations, not for direct submission.
