# Knowledge Engineering Methods (MIW)

Project assignments for the course **Knowledge Engineering Methods (MIW)**  
at the **Polish-Japanese Academy of Information Technology (PJATK)**.

- Tasks 01–05 — graded **10/10**
- Final — graded **18/18**

---

## Topics Covered

- **Bayesian inference** — probabilistic belief updating from sequential observations
- **Markov models** — state-based transition modeling for sequential data
- **Adaptive strategies** — agents that learn opponent behaviour and optimise decisions over time
- **Supervised learning** — classical classification algorithms benchmarked on non-linear data
- **Neural networks from scratch** — backpropagation, activation functions, batch vs online training
- **Convolutional networks** — image classification, residual connections, regularisation
- **Recurrent networks** — sequence modeling, LSTM gating, time-series forecasting
- **Symbolic AI / Prolog** — knowledge representation, spatial reasoning, recursive pathfinding

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

| Model                      | Test Accuracy |
| -------------------------- | :-----------: |
| Custom Logistic Regression |    0.9200     |
| Decision Tree (best)       |    0.9750     |
| Random Forest (best)       |    0.9750     |
| SVM (RBF)                  |    0.9650     |
| Ensemble (Soft Voting)     |    0.9650     |

- **Custom logistic regression** implemented from scratch — batch gradient descent, Xavier init, early stopping
- **Decision Tree** grid-searched over depth (2–None) and criterion (Gini / Entropy)
- **Random Forest** swept over 1–200 estimators
- **SVM** with RBF kernel on standardised features
- **Soft-voting ensemble** combining LR + RF + SVM via probability averaging

**Output:** decision boundary plots, depth & overfitting analysis, training loss curve

---

### 03 · Neural Network from Scratch — Function Approximation

A single-hidden-layer neural network built entirely with **NumPy** (no PyTorch, no TensorFlow), trained to approximate the noisy cubic **y ≈ x³ + x + noise**:

| Model                       | Train MSE | Test MSE | Verdict           |
| --------------------------- | :-------: | :------: | ----------------- |
| Batch + tanh, H=15          |  0.0176   |  0.1063  | Overfitting       |
| Batch + tanh, H=1           |  37.532   |  46.827  | Underfitting      |
| Batch + tanh, H=60, 40k ep. |  0.0177   |  0.1352  | Overfitting       |
| **Online + tanh, H=15**     |  0.0128   |  0.0128  | **Good fit**      |
| Batch + ReLU, H=15          |  0.0612   |  0.1291  | Good fit          |

- Forward pass + backpropagation implemented **manually**
- Compared **tanh**, **sigmoid**, and **ReLU** activations with He / Xavier initialisation
- **Batch** vs **online (stochastic)** training — online generalises better despite noisier updates
- Five experimental scenarios spanning optimal fit, underfitting (H=1), and overfitting (H=60)

**Output:** function fit plots, train/test loss curves, activation comparison

---

### 04 · CIFAR-10 CNN — Baseline vs ResNet-style

Two convolutional architectures benchmarked on **CIFAR-10**: a shallow 3-block CNN vs a deeper ResNet-style network with skip connections.

| Model                       | Params  | Test Accuracy |
| --------------------------- | :-----: | :-----------: |
| Model A — 3-block CNN       |  324 K  |    79.16 %    |
| **Model B — ResNet-style**  | 11.3 M  |  **90.13 %**  |

- Custom **60 / 20 / 20** train/val/test split on 60 000 images
- In-graph augmentation: horizontal flip, rotation, zoom, translation (training only)
- Adam + EarlyStopping + ReduceLROnPlateau, 80 epochs
- Per-class accuracy + confusion matrices reveal residual blocks lift the visually-similar categories (cat / dog / bird) from sub-70 % to above 80 %
- Model B improves over the baseline by **~11 percentage points** while keeping overfitting controlled via BatchNorm and staged dropout

**Output:** training curves, per-class accuracy, confusion matrices, misclassification gallery

---

### 05 · Stock Price Prediction with RNNs — AAPL

Three recurrent architectures benchmarked on **Apple Inc. (AAPL)** daily prices (2018–2024): a univariate SimpleRNN, a univariate LSTM, and a multivariate LSTM fed with OHLCV data plus technical indicators.

| Model                       | Input       | MSE (USD²) | MAE (USD) |
| --------------------------- | ----------- | :--------: | :-------: |
| **SimpleRNN — univariate**  | Close only  |  **10.22** | **2.46**  |
| LSTM — univariate           | Close only  |   50.46    |   5.69    |
| LSTM — multivariate         | 10 features |   80.03    |   6.95    |

- Chronological **60 / 20 / 20** split (no shuffling) to prevent look-ahead **data leakage**
- MinMax normalisation fit on training data only; **sliding-window** sequence generation
- Exploratory analysis: moving averages, rolling volatility, seasonal decomposition, **ADF stationarity test**
- SimpleRNN hyperparameters chosen by grid search over window {20, 40} and units {32, 64}
- Multivariate features: OHLCV + **EMA, MACD, RSI, Bollinger-band width**
- Counter-intuitively the **simplest model wins** — on a strongly trending asset, the SimpleRNN's short memory suffices while the deeper, feature-richer LSTMs add more variance than signal

**Output:** price/volatility plots, seasonal decomposition, training curves, predictions vs. actual

---

### Final · Apartment Map — Prolog Knowledge Base + Python GUI

A knowledge-based model of an apartment built in **Prolog**, with a **Python + Tkinter** GUI driving the engine through `pyswip`:

- 5 rooms and 16 furniture items, related by `on`, `next_to`, `between`, `same_room`
- **Dynamic** add/move/remove of objects via `assertz` / `retract`
- Shortest-path navigation through **recursive** Prolog rules
- Animated GUI — agent walks room-to-room along the computed path
- Free-form Prolog query console embedded in the GUI

**Output:** interactive apartment map with animated navigation and live query support  
**Bonus:** language integration — Python (Tkinter) ↔ `pyswip` ↔ SWI-Prolog

---

## Stack

Python · NumPy · pandas · scikit-learn · TensorFlow / Keras · statsmodels · yfinance · Matplotlib · Seaborn · SWI-Prolog · pyswip · Tkinter

---

## Notes

These are my own solutions to the laboratory tasks — shared as study material and reference implementations, not for direct submission.
