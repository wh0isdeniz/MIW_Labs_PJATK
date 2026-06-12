"""
Stock Price Prediction with Recurrent Neural Networks
=====================================================
Dataset: Apple Inc. (AAPL) daily prices, 2018-01-01 to 2024-11-29
Source:  Yahoo Finance via `yfinance` (local CSV fallback)

Six steps:
  1. Data loading, split, normalization, exploratory time-series analysis
  2. SimpleRNN baseline with hyperparameter tuning
  3. LSTM with dropout and early stopping
  4. Multivariate model using OHLCV + technical indicators
  5. Evaluation (MSE, MAE) and visualization
  6. Analysis and conclusions

Run:  python stock_prediction.py
"""

# ============================================================================
# 0. SETUP
# ============================================================================
import os, random, warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, SimpleRNN, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

SEED = 42
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)

plt.rcParams['figure.figsize'] = (11, 4)
plt.rcParams['axes.grid'] = True
print('TF', tf.__version__)


# ============================================================================
# 1. DATA LOADING
# ============================================================================
TICKER = 'AAPL'
START, END = '2018-01-01', '2024-11-29'

def load_data():
    try:
        import yfinance as yf
        df = yf.download(TICKER, start=START, end=END, progress=False, auto_adjust=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if len(df) > 0:
            print(f'Loaded from yfinance: {df.shape}')
            return df[['Open','High','Low','Close','Volume']]
        raise RuntimeError('empty yfinance df')
    except Exception as e:
        print('yfinance failed:', e)
        print('Falling back to local AAPL.csv')
        df = pd.read_csv('AAPL.csv')
        df['Date'] = pd.to_datetime(df['Date'], utc=True).dt.tz_localize(None)
        df = df.set_index('Date').sort_index().loc[START:END]
        return df[['Open','High','Low','Close','Volume']]

df = load_data()
print(df.head())
print(f'Shape: {df.shape}')
print(f'Range: {df.index.min().date()} -> {df.index.max().date()}')
print(f'Missing values:\n{df.isna().sum().to_string()}')
print(df.describe().round(2))


# ============================================================================
# 1b. TIME-SERIES ANALYSIS: trend, variability, seasonality
# ============================================================================
fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
axes[0].plot(df['Close'], label='Close')
axes[0].plot(df['Close'].rolling(30).mean(), label='30-day MA', linestyle='--')
axes[0].plot(df['Close'].rolling(90).mean(), label='90-day MA', linestyle=':')
axes[0].set_title(f'{TICKER} Closing Price + Moving Averages'); axes[0].legend()
axes[1].plot(df['Close'].rolling(30).std(), color='C3')
axes[1].set_title('30-day Rolling Standard Deviation (variability)')
plt.tight_layout(); plt.show()

# Seasonal decomposition (additive). Period 252 = trading days per year.
from statsmodels.tsa.seasonal import seasonal_decompose
decomp = seasonal_decompose(df['Close'], model='additive', period=252)
fig = decomp.plot(); fig.set_size_inches(11, 8); plt.tight_layout(); plt.show()

# ADF stationarity test
from statsmodels.tsa.stattools import adfuller
returns = df['Close'].pct_change().dropna()
print('ADF test on Close:   ', f'stat={adfuller(df.Close)[0]:.3f}, p={adfuller(df.Close)[1]:.4f}')
print('ADF test on Returns: ', f'stat={adfuller(returns)[0]:.3f}, p={adfuller(returns)[1]:.4f}')
# Low p on returns => returns are stationary; high p on price => not stationary.


# ============================================================================
# 2. PREPROCESSING: chronological split, normalization, sliding window
# ============================================================================
n = len(df)
train_end = int(n * 0.60)
val_end   = int(n * 0.80)

train_df = df.iloc[:train_end].copy()
val_df   = df.iloc[train_end:val_end].copy()
test_df  = df.iloc[val_end:].copy()
#We need to be doing chronological split, in order to not have "Data Leakeage" from future to past

print(f'Train: {len(train_df):4d}  {train_df.index.min().date()} -> {train_df.index.max().date()}')
print(f'Val:   {len(val_df):4d}  {val_df.index.min().date()} -> {val_df.index.max().date()}')
print(f'Test:  {len(test_df):4d}  {test_df.index.min().date()} -> {test_df.index.max().date()}')

close_scaler = MinMaxScaler() #0-1 normalization
train_close = close_scaler.fit_transform(train_df[['Close']].values)
val_close   = close_scaler.transform(val_df[['Close']].values)
test_close  = close_scaler.transform(test_df[['Close']].values)


def make_sequences(arr, window, target_idx=0):
    """Sliding window. arr shape (n, n_feat); returns X (n-w, w, n_feat), y (n-w,)."""
    X, y = [], []
    for i in range(len(arr) - window):
        X.append(arr[i:i+window])
        y.append(arr[i+window, target_idx])
    return np.array(X), np.array(y)


# Prepend last `window` rows of preceding split so first val/test prediction
# has full context, and we don't lose `window` samples at every boundary.
def build_xy(window):
    Xtr, ytr = make_sequences(train_close, window)
    Xv , yv  = make_sequences(np.vstack([train_close[-window:], val_close]),  window)
    Xte, yte = make_sequences(np.vstack([val_close[-window:],  test_close]),  window)
    return Xtr, ytr, Xv, yv, Xte, yte


# ============================================================================
# 3. SimpleRNN: baseline + hyperparameter tuning
# ============================================================================
def build_simple_rnn(window, units):
    m = Sequential([
        Input(shape=(window, 1)),
        SimpleRNN(units, activation='tanh'),
        Dense(1)
    ])
    m.compile(optimizer=Adam(1e-3), loss='mse', metrics=['mae'])
    return m

grid_results = []
for window in [20, 40]:
    for units in [32, 64]:
        Xtr, ytr, Xv, yv, _, _ = build_xy(window)
        tf.random.set_seed(SEED); np.random.seed(SEED)
        model = build_simple_rnn(window, units)
        h = model.fit(Xtr, ytr, validation_data=(Xv, yv),
                      epochs=25, batch_size=32, verbose=0)
        best_val = min(h.history['val_loss'])
        grid_results.append({'window': window, 'units': units, 'val_mse': best_val})
        print(f'window={window:>2}, units={units:>2}  ->  val MSE={best_val:.5f}')

grid = pd.DataFrame(grid_results).sort_values('val_mse')
print('\nBest config:'); print(grid.iloc[0].to_string())
BEST_WINDOW = int(grid.iloc[0]['window'])
BEST_UNITS  = int(grid.iloc[0]['units'])

# Re-train the best SimpleRNN with early stopping for the final comparison
Xtr, ytr, Xv, yv, Xte, yte = build_xy(BEST_WINDOW)
tf.random.set_seed(SEED); np.random.seed(SEED)
rnn = build_simple_rnn(BEST_WINDOW, BEST_UNITS)
es = EarlyStopping(patience=10, restore_best_weights=True)
rnn_hist = rnn.fit(Xtr, ytr, validation_data=(Xv, yv),
                   epochs=60, batch_size=32, verbose=0, callbacks=[es])
print(f'Final SimpleRNN: epochs={len(rnn_hist.history["loss"])}, '
      f'best val MSE={min(rnn_hist.history["val_loss"]):.5f}')


# ============================================================================
# 4. LSTM with dropout and early stopping
# ============================================================================
#Drop-OUT
def build_lstm(window, units, n_features=1, dropout=0.2):
    m = Sequential([
        Input(shape=(window, n_features)),
        LSTM(units, return_sequences=True),
        Dropout(dropout),
        LSTM(units // 2),
        Dropout(dropout),
        Dense(1)
    ])
    m.compile(optimizer=Adam(1e-3), loss='mse', metrics=['mae'])
    return m

tf.random.set_seed(SEED); np.random.seed(SEED)
lstm = build_lstm(BEST_WINDOW, 50)
es = EarlyStopping(patience=15, restore_best_weights=True)
lstm_hist = lstm.fit(Xtr, ytr, validation_data=(Xv, yv),
                     epochs=80, batch_size=32, verbose=0, callbacks=[es])
print(f'LSTM: epochs={len(lstm_hist.history["loss"])}, '
      f'best val MSE={min(lstm_hist.history["val_loss"]):.5f}')


# ============================================================================
# 5. MULTIVARIATE LSTM with additional features
# ============================================================================
def add_indicators(d):
    d = d.copy()
    d['EMA_12'] = d['Close'].ewm(span=12, adjust=False).mean() #Short-term trend
    d['EMA_26'] = d['Close'].ewm(span=26, adjust=False).mean() #long-term trend
    d['MACD']   = d['EMA_12'] - d['EMA_26'] #The difference between the two trends (momentum)
    delta = d['Close'].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    d['RSI'] = 100 - 100 / (1 + gain / loss.replace(0, np.nan)) #Overbought/oversold conditions (0–100)
    ma20, sd20 = d['Close'].rolling(20).mean(), d['Close'].rolling(20).std()
    d['BB_width'] = (4 * sd20) / ma20 #Price squeeze/expansion
    d['LogVol']   = np.log1p(d['Volume']) #Trading volume (loged)
    return d

feat_df = add_indicators(df).dropna()
FEATURES = ['Open','High','Low','Close','LogVol','EMA_12','EMA_26','MACD','RSI','BB_width']
print(f'Multivariate features ({len(FEATURES)}):', FEATURES)
print(f'After dropping NaNs: {feat_df.shape}')

n2 = len(feat_df); te2 = int(n2*0.6); ve2 = int(n2*0.8)
tr_m, va_m, ts_m = feat_df.iloc[:te2], feat_df.iloc[te2:ve2], feat_df.iloc[ve2:]

scaler_m = MinMaxScaler(); scaler_m.fit(tr_m[FEATURES].values)
tr_s = scaler_m.transform(tr_m[FEATURES].values)
va_s = scaler_m.transform(va_m[FEATURES].values)
ts_s = scaler_m.transform(ts_m[FEATURES].values)

# Separate scaler for Close so we can inverse-transform predictions
close_scaler_m = MinMaxScaler(); close_scaler_m.fit(tr_m[['Close']].values)
CI = FEATURES.index('Close')

W = BEST_WINDOW
Xtr_m, ytr_m = make_sequences(tr_s, W, CI)
Xv_m , yv_m  = make_sequences(np.vstack([tr_s[-W:], va_s]), W, CI)
Xte_m, yte_m = make_sequences(np.vstack([va_s[-W:], ts_s]), W, CI)
print(f'Multivariate seqs: train{Xtr_m.shape}, val{Xv_m.shape}, test{Xte_m.shape}')

tf.random.set_seed(SEED); np.random.seed(SEED)
lstm_m = build_lstm(W, 64, n_features=len(FEATURES))
es = EarlyStopping(patience=15, restore_best_weights=True)
lstm_m_hist = lstm_m.fit(Xtr_m, ytr_m, validation_data=(Xv_m, yv_m),
                         epochs=80, batch_size=32, verbose=0, callbacks=[es])
print(f'Multivariate LSTM: epochs={len(lstm_m_hist.history["loss"])}, '
      f'best val MSE={min(lstm_m_hist.history["val_loss"]):.5f}')


# ============================================================================
# 6. EVALUATION: predictions vs. actual on the test set
# ============================================================================
def eval_on_test(model, X, y, scaler, name):
    yp = model.predict(X, verbose=0).flatten()
    yp = scaler.inverse_transform(yp.reshape(-1, 1)).flatten()
    yt = scaler.inverse_transform(y.reshape(-1, 1)).flatten()
    return {'Model': name,
            'MSE': mean_squared_error(yt, yp), #It punishes major mistakes more severely.
            'MAE': mean_absolute_error(yt, yp), # Average error in dollars
            '_ytrue': yt, '_ypred': yp}

results = [
    eval_on_test(rnn,    Xte,   yte,   close_scaler,   'SimpleRNN (univariate)'),
    eval_on_test(lstm,   Xte,   yte,   close_scaler,   'LSTM (univariate)'),
    eval_on_test(lstm_m, Xte_m, yte_m, close_scaler_m, 'LSTM (multivariate)'),
]
metrics = pd.DataFrame([{k:v for k,v in r.items() if not k.startswith('_')}
                        for r in results]).round(4)
print('Test-set metrics (in USD):')
print(metrics.to_string(index=False))

# Training-loss curves
fig, axes = plt.subplots(1, 3, figsize=(15, 3.5))
for ax, (h, t) in zip(axes, [(rnn_hist, 'SimpleRNN'),
                              (lstm_hist, 'LSTM'),
                              (lstm_m_hist, 'LSTM multivariate')]):
    ax.plot(h.history['loss'], label='train')
    ax.plot(h.history['val_loss'], label='val')
    ax.set_title(t); ax.set_xlabel('epoch'); ax.set_ylabel('MSE (scaled)')
    ax.legend()
plt.tight_layout(); plt.show()

# Predictions vs actual on the test set
fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
test_dates_uni = test_df.index[-len(results[0]['_ytrue']):]
test_dates_mul = ts_m.index[-len(results[2]['_ytrue']):]
for ax, r in zip(axes, results):
    dates = test_dates_mul if 'multivariate' in r['Model'] else test_dates_uni
    ax.plot(dates, r['_ytrue'], label='Actual',    linewidth=1.6)
    ax.plot(dates, r['_ypred'], label='Predicted', linewidth=1.4, alpha=0.85)
    ax.set_title(f"{r['Model']}    MSE={r['MSE']:.2f}   MAE={r['MAE']:.2f}")
    ax.legend(loc='upper left')
axes[-1].set_xlabel('Date'); plt.tight_layout(); plt.show()

