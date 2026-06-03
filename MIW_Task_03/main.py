import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

np.random.seed(42)

DATA_FILE  = 'dane2.txt'
TEST_RATIO = 0.2       # 80% train, 20% test
EPOCHS     = 10000


# 1. Load data and split into train / test sets
d = np.loadtxt(DATA_FILE)
x, y = d[:, 0:1], d[:, 1:2]

# random shuffle then slice
idx  = np.random.permutation(len(x))
n_te = int(len(x) * TEST_RATIO)
x_te, y_te = x[idx[:n_te]],  y[idx[:n_te]]
x_tr, y_tr = x[idx[n_te:]], y[idx[n_te:]]

# Z-score normalisation — fit on training set, apply to both
mu, s  = x_tr.mean(), x_tr.std() + 1e-8
x_tr_n = (x_tr - mu) / s
x_te_n = (x_te - mu) / s

print(f"dane2.txt  ->  Train: {len(x_tr)}  |  Test: {len(x_te)}")
print(f"Underlying function: y ~ x^3 + x + noise  (x in [-3, 3])")


# 2. Neural network — architecture: Input(1) -> Hidden(H) -> Output(1)
class NN:
    def __init__(self, H=15, act='tanh', lr=0.005):
        self.H, self.act, self.lr = H, act, lr
        # weight initialisation: He for ReLU, Xavier for tanh/sigmoid
        sc = np.sqrt(2 / 1) if act == 'relu' else np.sqrt(2 / (1 + H))
        self.W1 = np.random.randn(1, H) * sc   # input -> hidden weights
        self.b1 = np.zeros((1, H))             # hidden biases
        self.W2 = np.random.randn(H, 1) * np.sqrt(2 / (H + 1))  # hidden -> output
        self.b2 = np.zeros((1, 1))             # output bias

    # non-linear activation function (tanh / sigmoid / ReLU)
    def _activate(self, z):
        if self.act == 'tanh':    return np.tanh(z)
        if self.act == 'sigmoid': return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
        return np.maximum(0, z)   # ReLU

    # derivative of the activation — needed for backprop
    def _activate_deriv(self, z):
        if self.act == 'tanh':
            return 1 - np.tanh(z) ** 2
        if self.act == 'sigmoid':
            s = 1 / (1 + np.exp(-np.clip(z, -500, 500)))
            return s * (1 - s)
        return (z > 0).astype(float)  # ReLU derivative (0 or 1)

    def forward(self, x):
        self.x  = x
        self.z1 = x @ self.W1 + self.b1        # pre-activation hidden layer
        self.a1 = self._activate(self.z1)       # hidden layer activations
        self.z2 = self.a1 @ self.W2 + self.b2   # linear output (no activation)
        return self.z2

    def predict(self, x):
        return self.forward(x)

    @staticmethod
    def mse(y_hat, y):
        return float(np.mean((y_hat - y) ** 2))

    def _backward(self, y):
        # compute gradients via chain rule
        N   = len(y)
        dz2 = 2 * (self.z2 - y) / N            # dL/dz2  (MSE gradient)
        dW2 = self.a1.T @ dz2                   # dL/dW2
        db2 = dz2.sum(0, keepdims=True)         # dL/db2
        da1 = dz2 @ self.W2.T                   # dL/da1
        dz1 = da1 * self._activate_deriv(self.z1)  # dL/dz1
        dW1 = self.x.T @ dz1                    # dL/dW1
        db1 = dz1.sum(0, keepdims=True)         # dL/db1
        return dW1, db1, dW2, db2

    def _update(self, dW1, db1, dW2, db2):
        # gradient descent weight update
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2

    # ── Batch training: all samples used per gradient step ───
    def train_batch(self, x_tr, y_tr, x_te, y_te, epochs):
        train_losses, val_losses = [], []
        for epoch in range(epochs):
            self.forward(x_tr)
            self._update(*self._backward(y_tr))
            if epoch % 100 == 0:
                train_losses.append(self.mse(self.predict(x_tr), y_tr))
                val_losses.append(self.mse(self.predict(x_te), y_te))
        return train_losses, val_losses

    # ── Online training: one random sample per gradient step ─
    def train_online(self, x_tr, y_tr, x_te, y_te, epochs):
        train_losses, val_losses = [], []
        N = len(x_tr)
        for epoch in range(epochs):
            for i in np.random.permutation(N):  # shuffle order each epoch
                self.forward(x_tr[i:i+1])
                self._update(*self._backward(y_tr[i:i+1]))
            if epoch % 100 == 0:
                train_losses.append(self.mse(self.predict(x_tr), y_tr))
                val_losses.append(self.mse(self.predict(x_te), y_te))
        return train_losses, val_losses

# 3. Fit assessment — detect underfitting / overfitting
def assess_fit(train_mse, test_mse):
    if np.isnan(train_mse) or np.isnan(test_mse):
        return "ERROR (NaN)"
    ratio = test_mse / (train_mse + 1e-12)
    if train_mse > 2.0 and test_mse > 2.0:
        return "UNDERFITTING (both losses high)"   # model too simple
    if ratio > 2.5:
        return f"OVERFITTING  (test/train={ratio:.2f})"  # gap too large
    return     f"GOOD FIT     (test/train={ratio:.2f})"


# ── Plotting helpers ──────────────────────────────────────────
def plot_fit(x_tr, y_tr, x_te, y_te, model, title, ax):
    xs = np.linspace(-2.2, 2.2, 300).reshape(-1, 1)
    ax.scatter(x_tr, y_tr, s=20, alpha=0.6, label='Train', color='steelblue')
    ax.scatter(x_te, y_te, s=25, alpha=0.8, label='Test',  color='darkorange',
               marker='x', linewidths=1.8)
    ax.plot(xs, model.predict(xs), 'r-', lw=2.2, label='Network output')
    ax.set_title(title, fontsize=10, pad=6)
    ax.legend(fontsize=8)
    ax.set_xlabel('x (normalised)')
    ax.set_ylabel('y')
    ax.grid(True, alpha=0.25)

def plot_loss(train_losses, val_losses, title, ax):
    ax.plot(train_losses, label='Train MSE', color='steelblue')
    ax.plot(val_losses,   label='Val MSE',   color='darkorange', ls='--')
    ax.set_title(title, fontsize=10)
    ax.set_xlabel('Epoch (x100)')
    ax.set_ylabel('MSE (log scale)')
    ax.legend(fontsize=8)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.25)

# Task 2 & 3 — Batch backpropagation with tanh activation

# optimal network: H=15 neurons, enough capacity without overfitting
print("\n[Task 2] Batch + tanh, H=15, lr=0.01 ...")
net = NN(H=15, act='tanh', lr=0.01)
tl, vl = net.train_batch(x_tr_n, y_tr, x_te_n, y_te, EPOCHS)
tm = NN.mse(net.predict(x_tr_n), y_tr)
vm = NN.mse(net.predict(x_te_n), y_te)
print(f"  -> Train: {tm:.5f}  Test: {vm:.5f}  {assess_fit(tm, vm)}")

# underfitting: H=1 — single neuron cannot approximate x^3+x
print("[Task 3a] Underfitting demo — H=1 ...")
net_u = NN(H=1, act='tanh', lr=0.01)
tl_u, vl_u = net_u.train_batch(x_tr_n, y_tr, x_te_n, y_te, EPOCHS)
tm_u = NN.mse(net_u.predict(x_tr_n), y_tr)
vm_u = NN.mse(net_u.predict(x_te_n), y_te)
print(f"  -> Train: {tm_u:.5f}  Test: {vm_u:.5f}  {assess_fit(tm_u, vm_u)}")

# overfitting: H=60 + 4x more epochs — model memorises training data
print("[Task 3b] Overfitting demo — H=60, 40k epochs ...")
net_o = NN(H=60, act='tanh', lr=0.01)
tl_o, vl_o = net_o.train_batch(x_tr_n, y_tr, x_te_n, y_te, EPOCHS * 4)
tm_o = NN.mse(net_o.predict(x_tr_n), y_tr)
vm_o = NN.mse(net_o.predict(x_te_n), y_te)
print(f"  -> Train: {tm_o:.5f}  Test: {vm_o:.5f}  {assess_fit(tm_o, vm_o)}")


# Task 4 — Switch to online (stochastic) training
# lower lr needed — each step uses only 1 sample, noisier updates
print("\n[Task 4] Online (stochastic) + tanh, H=15, lr=0.003 ...")
net_s = NN(H=15, act='tanh', lr=0.003)
tl_s, vl_s = net_s.train_online(x_tr_n, y_tr, x_te_n, y_te, EPOCHS)
tm_s = NN.mse(net_s.predict(x_tr_n), y_tr)
vm_s = NN.mse(net_s.predict(x_te_n), y_te)
print(f"  -> Train: {tm_s:.5f}  Test: {vm_s:.5f}  {assess_fit(tm_s, vm_s)}")


# ─────────────────────────────────────────────────────────────
# Task 5 — Replace tanh with ReLU in the hidden layer
# ─────────────────────────────────────────────────────────────
# ReLU: f(z) = max(0, z) — piecewise linear, no vanishing gradient
print("\n[Task 5] Batch + ReLU, H=15, lr=0.01 ...")
net_r = NN(H=15, act='relu', lr=0.01)
tl_r, vl_r = net_r.train_batch(x_tr_n, y_tr, x_te_n, y_te, EPOCHS)
tm_r = NN.mse(net_r.predict(x_tr_n), y_tr)
vm_r = NN.mse(net_r.predict(x_te_n), y_te)
print(f"  -> Train: {tm_r:.5f}  Test: {vm_r:.5f}  {assess_fit(tm_r, vm_r)}")


# Plots — Tasks 2, 3, 4
fig, axes = plt.subplots(3, 2, figsize=(13, 15))
fig.suptitle("MIW Lab 07 — dane2.txt  (y ~ x^3 + x + noise)\nNumPy Neural Network from Scratch",
             fontsize=13, fontweight='bold', y=1.01)

plot_fit(x_tr_n, y_tr, x_te_n, y_te, net,
    f"Task 2: Optimal fit  —  Batch+tanh  H=15\nTrain MSE={tm:.4f}   Test MSE={vm:.4f}   {assess_fit(tm, vm)}",
    axes[0, 0])
plot_loss(tl, vl, "Task 2: Loss curve  —  Batch+tanh", axes[0, 1])

plot_fit(x_tr_n, y_tr, x_te_n, y_te, net_u,
    f"Task 3a: UNDERFITTING  —  H=1\nTrain={tm_u:.4f}   Test={vm_u:.4f}   {assess_fit(tm_u, vm_u)}",
    axes[1, 0])
plot_fit(x_tr_n, y_tr, x_te_n, y_te, net_o,
    f"Task 3b: OVERFITTING  —  H=60, 40k epochs\nTrain={tm_o:.4f}   Test={vm_o:.4f}   {assess_fit(tm_o, vm_o)}",
    axes[1, 1])

plot_fit(x_tr_n, y_tr, x_te_n, y_te, net_s,
    f"Task 4: Online (stochastic)+tanh  H=15\nTrain={tm_s:.4f}   Test={tm_s:.4f}   {assess_fit(tm_s, vm_s)}",
    axes[2, 0])

ax = axes[2, 1]
ax.plot(tl,   label='Batch  — Train', color='steelblue',  lw=1.5)
ax.plot(vl,   label='Batch  — Val',   color='steelblue',  ls='--', lw=1.5)
ax.plot(tl_s, label='Online — Train', color='darkorange', lw=1.5)
ax.plot(vl_s, label='Online — Val',   color='darkorange', ls='--', lw=1.5)
ax.set_title("Task 4: Batch vs Online — loss comparison", fontsize=10)
ax.set_xlabel('Epoch (x100)')
ax.set_ylabel('MSE (log scale)')
ax.legend(fontsize=8)
ax.set_yscale('log')
ax.grid(True, alpha=0.25)

plt.tight_layout()
plt.savefig('results_tasks_2_3_4.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nSaved: results_tasks_2_3_4.png")

# Plots — Task 5: ReLU
fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))
fig2.suptitle("Task 5: ReLU Network — dane2.txt", fontsize=12, fontweight='bold')
plot_fit(x_tr_n, y_tr, x_te_n, y_te, net_r,
    f"Batch+ReLU  H=15\nTrain={tm_r:.4f}   Test={vm_r:.4f}   {assess_fit(tm_r, vm_r)}",
    axes2[0])
plot_loss(tl_r, vl_r, "ReLU loss curve", axes2[1])
plt.tight_layout()
plt.savefig('results_task5_relu.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: results_task5_relu.png")

# ── Summary table
print("\n" + "=" * 62)
print(f"{'Model':<34} {'Train MSE':>10} {'Test MSE':>10}")
print("=" * 62)
for name, tr, te in [
    ("Batch+tanh optimal  (H=15)", tm,   vm),
    ("Batch+tanh underfit (H=1)",  tm_u, vm_u),
    ("Batch+tanh overfit  (H=60)", tm_o, vm_o),
    ("Online+tanh         (H=15)", tm_s, vm_s),
    ("Batch+ReLU          (H=15)", tm_r, vm_r),
]:
    print(f"  {name:<32} {tr:>10.5f} {te:>10.5f}")
print("=" * 62)