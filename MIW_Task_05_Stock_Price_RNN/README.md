# Project 05 — Stock Price Prediction with RNNs

Three recurrent architectures benchmarked on **Apple Inc. (AAPL)** daily prices (2018–2024): a univariate **SimpleRNN** baseline, a univariate **LSTM**, and a **multivariate LSTM** fed with OHLCV data plus technical indicators. The goal is to test the common assumption that deeper, feature-richer models outperform simpler ones — and to show where that assumption breaks down on a strongly trending asset.

---

## Results

| Model | Input | MSE (USD²) | MAE (USD) |
|-------|:-----:|:----------:|:---------:|
| **SimpleRNN** — univariate | Close only | **10.22** | **2.46** |
| **LSTM** — univariate | Close only | 50.46 | 5.69 |
| **LSTM** — multivariate | 10 features | 80.03 | 6.95 |

Counter-intuitively, the **simplest model wins**. On an asset as strongly and smoothly trending as AAPL, the SimpleRNN's short memory is enough to track the price, while the added capacity and features of the LSTM models introduce more variance than signal.

---

## Models

### SimpleRNN — baseline

A single recurrent layer feeding a dense output:

- `SimpleRNN(units, tanh) → Dense(1)`
- Hyperparameters selected by **grid search** over window ∈ {20, 40} and units ∈ {32, 64}
- Best config: **window = 40, units = 32** (lowest validation MSE)

### LSTM — univariate

Two stacked LSTM layers with dropout regularisation:

- `LSTM(50, return_sequences) → Dropout(0.2) → LSTM(25) → Dropout(0.2) → Dense(1)`
- Gating mechanism is designed to capture longer-range dependencies than a SimpleRNN

### LSTM — multivariate

Same architecture, widened to ingest **10 features** per timestep:

- Inputs: `Open, High, Low, Close, LogVol, EMA_12, EMA_26, MACD, RSI, BB_width`
- A separate scaler is kept for `Close` so predictions can be inverse-transformed back to USD

All models use **Adam (lr = 1e-3)**, **MSE loss**, and **EarlyStopping** (restore best weights).

---

## Dataset & Pipeline

- **AAPL daily prices** — 1 739 trading days, 2018-01-02 → 2024-11-27, pulled via `yfinance` (local CSV fallback)
- **Chronological 60 / 20 / 20 split** — no shuffling, to prevent look-ahead **data leakage** from future to past
- **MinMax normalisation** fit on the training set only, then applied to val/test
- **Sliding window** sequence generation; the trailing `window` rows of each split are prepended to the next so no samples are lost at split boundaries

### Exploratory Analysis

<img width="1090" height="690" alt="1" src="https://github.com/user-attachments/assets/83de49fc-3295-4c14-abb3-c0e6dcc62fad" />

The price rises ~5× over the period with a strong, persistent uptrend. The 30-day rolling standard deviation shows volatility spiking sharply during the 2020 COVID crash and staying elevated through 2022–2024.

<img width="1089" height="790" alt="2" src="https://github.com/user-attachments/assets/7b13c442-6da2-45a7-8820-fcca47cf6ce5" />

Additive decomposition (period = 252 trading days) separates a smooth upward **trend**, a small but regular **seasonal** component (±5–10 USD), and **residuals** that grow noticeably during 2022–2023 — exactly the periods the models find hardest to predict.

An **ADF stationarity test** confirms the price series is non-stationary (p ≈ 0.93) while daily returns are stationary (p ≈ 0.00).

---

## Training

<img width="1489" height="340" alt="3" src="https://github.com/user-attachments/assets/1faa1d4b-c0c0-4439-86d7-b27a8093e1cc" />

The SimpleRNN converges quickly to a low, stable loss with train and validation curves staying tightly coupled — a clean, healthy fit. Both LSTM variants start from a much higher loss and show noisier validation curves through the early epochs, reflecting the harder optimisation landscape of deeper, wider models.

---

## Predictions vs. Actual

<img width="1189" height="889" alt="4" src="https://github.com/user-attachments/assets/7035779b-69ad-4364-9615-37d7049d3791" />

On the test set (Jul 2023 → Nov 2024), the **SimpleRNN** tracks the actual price almost like a shadow (MAE = 2.46 USD). The **univariate LSTM** lags the mid-2024 rally and drifts below the actual price (MAE = 5.69 USD). The **multivariate LSTM** diverges the most after the rally despite having ten features (MAE = 6.95 USD).

---

## Summary

**Why the simplest model wins here:**

- **Strong directional trend** — AAPL's price is dominated by a smooth uptrend, so a short 40-day window already carries most of the predictive signal
- **Capacity vs. data** — 10 features × 40-step windows over only ~1 000 training sequences pushes the multivariate LSTM toward overfitting
- **Feature noise** — on a trend-driven stock, indicators like RSI and MACD add more noise than usable signal
- **Right tool for the series** — model complexity only pays off when the data's structure (volatility, non-linear dependencies) actually demands it

A more volatile, less trend-driven asset would likely reverse this ranking — a natural direction for future work.

---

## Stack

Python · TensorFlow / Keras · NumPy · pandas · scikit-learn · statsmodels · Matplotlib · yfinance

---

## Usage

```bash
pip install tensorflow numpy pandas scikit-learn statsmodels matplotlib yfinance
python stock_prediction.py
```
