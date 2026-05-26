"""
main.py — ML pipeline: make_moons dataset
Steps: LogReg (custom) | DecisionTree | RandomForest | SVM | Ensemble
"""

import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend — no display window needed
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression  # only for VotingClassifier
from sklearn.preprocessing import StandardScaler

from reglog import LogisticRegressionCustom  # local gradient-descent implementation

os.makedirs("outputs", exist_ok=True)

# ── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)

# ═══════════════════════════════════════════════════════════════════════════════
# 1. DATA GENERATION
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("1. DATA GENERATION")
print("=" * 60)

# make_moons produces two interleaved half-circles — a non-linearly separable dataset
X, y = make_moons(n_samples=1000, noise=0.25, random_state=SEED)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y  # stratify preserves class ratio in both splits
)

# Scaling is required for SVM and custom logistic regression (gradient-descent sensitive to scale)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)  # fit on train only to avoid data leakage
X_test_s = scaler.transform(X_test)

print(f"Dataset : {X.shape[0]:,} samples, {X.shape[1]} features")
print(f"Train   : {X_train.shape[0]:,}  |  Test: {X_test.shape[0]:,}")
print(f"Class balance (train): {np.bincount(y_train)}\n")

results = {}   # model_name -> accuracy

# ═══════════════════════════════════════════════════════════════════════════════
# 2. LOGISTIC REGRESSION (CUSTOM)
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("2. LOGISTIC REGRESSION (custom)")
print("=" * 60)

# Custom implementation trained with gradient descent (lr=learning rate, n_iter=epochs)
lr_custom = LogisticRegressionCustom(lr=0.5, n_iter=2000, random_state=SEED)
lr_custom.fit(X_train_s, y_train)

train_acc = lr_custom.score(X_train_s, y_train)
test_acc = lr_custom.score(X_test_s, y_test)

proba_sample = lr_custom.predict_proba(X_test_s[:5])
print(f"Train accuracy : {train_acc:.4f}")
print(f"Test  accuracy : {test_acc:.4f}")
print(f"Iterations ran : {len(lr_custom.loss_history_)}")
print(f"Sample probabilities (first 5 test rows):\n{proba_sample.round(4)}\n")

results["Logistic Regression\n(Custom)"] = test_acc

# Short-hand keys for later plot title lookups
LR_KEY = "Logistic Regression\n(Custom)"
DT_KEY = "Decision Tree\n(best)"
RF_KEY = "Random Forest\n(best)"
SVM_KEY = "SVM\n(RBF)"
ENS_KEY = "Ensemble\n(Voting)"

# ═══════════════════════════════════════════════════════════════════════════════
# 3. DECISION TREE
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("3. DECISION TREE")
print("=" * 60)

depths = [2, 3, 5, 7, 10, None]  # None = grow until all leaves are pure (highest risk of overfitting)
criteria = ["gini", "entropy"]

dt_results = {}  # (depth, criterion) -> (train_acc, test_acc)

print(f"{'Depth':<8} {'Criterion':<10} {'Train Acc':<12} {'Test Acc':<12}")
print("-" * 45)

best_dt, best_dt_acc = None, 0

for depth in depths:
    for crit in criteria:
        dt = DecisionTreeClassifier(
            criterion=crit,
            max_depth=depth,
            random_state=SEED
        )
        dt.fit(X_train, y_train)
        tr = accuracy_score(y_train, dt.predict(X_train))
        te = accuracy_score(y_test, dt.predict(X_test))
        dt_results[(depth, crit)] = (tr, te)
        depth_str = str(depth) if depth is not None else "None"
        print(f"{depth_str:<8} {crit:<10} {tr:<12.4f} {te:<12.4f}")
        if te > best_dt_acc:
            best_dt_acc = te
            best_dt = dt  # keep the tree with the highest test accuracy

results["Decision Tree\n(best)"] = best_dt_acc
print(f"\nBest DT test accuracy : {best_dt_acc:.4f}\n")

# ═══════════════════════════════════════════════════════════════════════════════
# 4. RANDOM FOREST
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("4. RANDOM FOREST")
print("=" * 60)

n_estimators_list = [1, 5, 10, 25, 50, 100, 200]  # sweep from a single tree up to a large ensemble
rf_results = {}  # n_trees -> (train_acc, test_acc)

print(f"{'n_trees':<10} {'Train Acc':<12} {'Test Acc':<12}")
print("-" * 36)

best_rf, best_rf_acc = None, 0

for n in n_estimators_list:
    rf = RandomForestClassifier(n_estimators=n, max_depth=5, random_state=SEED, n_jobs=-1)  # n_jobs=-1 uses all CPU cores
    rf.fit(X_train, y_train)
    tr = accuracy_score(y_train, rf.predict(X_train))
    te = accuracy_score(y_test, rf.predict(X_test))
    rf_results[n] = (tr, te)
    print(f"{n:<10} {tr:<12.4f} {te:<12.4f}")
    if te > best_rf_acc:
        best_rf_acc = te
        best_rf = rf

results["Random Forest\n(best)"] = best_rf_acc
print(f"\nBest RF test accuracy : {best_rf_acc:.4f}\n")

# ═══════════════════════════════════════════════════════════════════════════════
# 5. SVM
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("5. SVM")
print("=" * 60)

# RBF kernel maps data into infinite-dimensional space, making it effective on non-linear boundaries.
# probability=True enables predict_proba (required by the soft-voting ensemble below).
svm = SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=SEED)
svm.fit(X_train_s, y_train)

svm_train = accuracy_score(y_train, svm.predict(X_train_s))
svm_test = accuracy_score(y_test, svm.predict(X_test_s))

print(f"Train accuracy : {svm_train:.4f}")
print(f"Test  accuracy : {svm_test:.4f}\n")

results["SVM\n(RBF)"] = svm_test

# ═══════════════════════════════════════════════════════════════════════════════
# 6. ENSEMBLE — VotingClassifier
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("6. ENSEMBLE (VotingClassifier)")
print("=" * 60)

# Soft voting averages predicted probabilities across classifiers, which typically outperforms hard voting.
# All three base learners must support predict_proba for soft voting to work.
lr_sklearn = LogisticRegression(max_iter=1000, random_state=SEED)
rf_ens = RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1, max_depth= 3)
svm_ens = SVC(kernel="rbf", probability=True, random_state=SEED)

voting_clf = VotingClassifier(
    estimators=[("lr", lr_sklearn), ("rf", rf_ens), ("svm", svm_ens)],
    voting="soft",
)
voting_clf.fit(X_train_s, y_train)

ens_train = accuracy_score(y_train, voting_clf.predict(X_train_s))
ens_test = accuracy_score(y_test, voting_clf.predict(X_test_s))

print(f"Train accuracy : {ens_train:.4f}")
print(f"Test  accuracy : {ens_test:.4f}\n")

results["Ensemble\n(Voting)"] = ens_test

# ═══════════════════════════════════════════════════════════════════════════════
# 7. EVALUATION & PLOTS
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("7. EVALUATION SUMMARY")
print("=" * 60)

print(f"\n{'Model':<30} {'Test Accuracy':>14}")
print("-" * 46)
for name, acc in results.items():
    clean = name.replace("\n", " ")
    print(f"{clean:<30} {acc:>14.4f}")

# ── Helper: plot decision boundary ───────────────────────────────────────────
def plot_decision_boundary(ax, clf, X, y, title, scaled=False, h=0.05):
    """
    Shade the feature space by predicted class and overlay the test points.

    scaled: if True, applies the global StandardScaler to the mesh grid before
            predicting (required for models trained on scaled data).
    h:      mesh resolution in feature units — smaller = finer but slower.
    """
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, h),
        np.arange(y_min, y_max, h),
    )

    if scaled:
        grid_scaled = scaler.transform(np.c_[xx.ravel(), yy.ravel()])
        Z = clf.predict(grid_scaled).reshape(xx.shape)
    else:
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    cmap_bg = ListedColormap(["#FFDDC1", "#C1D4FF"])   # light orange / blue background regions
    cmap_pts = ListedColormap(["#FF6B35", "#1A6FBF"])  # darker orange / blue for scatter points

    ax.contourf(xx, yy, Z, alpha=0.35, cmap=cmap_bg)
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_pts, s=4, alpha=0.5, linewidths=0)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")


# ── Figure 1: Decision boundaries ────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle(
    "Decision Boundaries — make_moons (n=10,000, noise=0.25)",
    fontsize=15,
    fontweight="bold",
    y=1.01
)

plot_decision_boundary(
    axes[0, 0], lr_custom, X_test, y_test,
    f"Custom Logistic Reg\nacc={results[LR_KEY]:.4f}",
    scaled=True,  # trained on scaled data
)
plot_decision_boundary(
    axes[0, 1], best_dt, X_test, y_test,
    f"Decision Tree (best)\nacc={results[DT_KEY]:.4f}",
    scaled=False,  # decision trees are scale-invariant
)
plot_decision_boundary(
    axes[0, 2], best_rf, X_test, y_test,
    f"Random Forest (best)\nacc={results[RF_KEY]:.4f}",
    scaled=False,
)
plot_decision_boundary(
    axes[1, 0], svm, X_test, y_test,
    f"SVM (RBF)\nacc={results[SVM_KEY]:.4f}",
    scaled=True,
)
plot_decision_boundary(
    axes[1, 1], voting_clf, X_test, y_test,
    f"Ensemble (Voting)\nacc={results[ENS_KEY]:.4f}",
    scaled=True,
)

# Accuracy bar chart in the last panel
ax_bar = axes[1, 2]
labels = list(results.keys())
values = list(results.values())
colors = ["#FF6B35", "#2ECC71", "#3498DB", "#9B59B6", "#E74C3C"]

bars = ax_bar.bar(range(len(labels)), values, color=colors, width=0.6)
ax_bar.set_xticks(range(len(labels)))
ax_bar.set_xticklabels(labels, fontsize=8)
ax_bar.set_ylim(0.8, 1.0)  # zoom in on the relevant accuracy range
ax_bar.set_title("Test Accuracy Comparison", fontsize=11, fontweight="bold")
ax_bar.set_ylabel("Accuracy")

for bar, val in zip(bars, values):
    ax_bar.text(
        bar.get_x() + bar.get_width() / 2,
        val + 0.002,
        f"{val:.4f}",
        ha="center",
        va="bottom",
        fontsize=8
    )

ax_bar.grid(axis="y", alpha=0.4)

plt.tight_layout()
plt.savefig("outputs/decision_boundaries.png", dpi=150, bbox_inches="tight")
plt.close()
print("\nSaved: outputs/decision_boundaries.png")

# ── Figure 2: Decision Tree — depth & criterion analysis ─────────────────────
# Shows how test accuracy and overfitting gap change as the tree is allowed to grow deeper.
fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
fig2.suptitle("Decision Tree Analysis", fontsize=14, fontweight="bold")

depth_labels = [str(d) if d is not None else "None" for d in depths]

for crit, color, marker in [("gini", "#3498DB", "o"), ("entropy", "#E74C3C", "s")]:
    test_accs = [dt_results[(d, crit)][1] for d in depths]
    ax1.plot(
        depth_labels, test_accs,
        color=color, marker=marker,
        label=crit.capitalize(),
        linewidth=2, markersize=7
    )

ax1.set_title("Test Accuracy vs Tree Depth")
ax1.set_xlabel("Max Depth")
ax1.set_ylabel("Test Accuracy")
ax1.legend()
ax1.grid(alpha=0.4)

for crit, color, marker in [("gini", "#3498DB", "o"), ("entropy", "#E74C3C", "s")]:
    train_accs = [dt_results[(d, crit)][0] for d in depths]
    test_accs = [dt_results[(d, crit)][1] for d in depths]
    ax2.plot(
        depth_labels, train_accs,
        color=color, marker=marker,
        linestyle="--", alpha=0.5,
        label=f"{crit} (train)"
    )
    ax2.plot(
        depth_labels, test_accs,
        color=color, marker=marker,
        linestyle="-",
        label=f"{crit} (test)"
    )

ax2.set_title("Train vs Test Accuracy (Overfitting Analysis)")
ax2.set_xlabel("Max Depth")
ax2.set_ylabel("Accuracy")
ax2.legend(fontsize=8)
ax2.grid(alpha=0.4)

plt.tight_layout()
plt.savefig("outputs/decision_tree_analysis.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: outputs/decision_tree_analysis.png")

# ── Figure 3: Random Forest — n_trees analysis ───────────────────────────────
# Demonstrates the bias-variance improvement as more trees are averaged together.
fig3, ax = plt.subplots(figsize=(9, 5))
n_trees = list(rf_results.keys())
train_rf = [rf_results[n][0] for n in n_trees]
test_rf = [rf_results[n][1] for n in n_trees]

ax.plot(n_trees, train_rf, "o--", color="#3498DB", label="Train", linewidth=2)
ax.plot(n_trees, test_rf, "s-", color="#E74C3C", label="Test", linewidth=2)
ax.set_title("Random Forest — Effect of Number of Trees", fontsize=13, fontweight="bold")
ax.set_xlabel("Number of Trees")
ax.set_ylabel("Accuracy")
ax.legend()
ax.grid(alpha=0.4)
ax.set_xscale("log")  # log scale makes the low-tree region easier to read

plt.tight_layout()
plt.savefig("outputs/random_forest_analysis.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: outputs/random_forest_analysis.png")

# ── Figure 4: Custom LR — learning curve ─────────────────────────────────────
# A healthy curve should decrease steeply early and plateau — confirms convergence.
fig4, ax = plt.subplots(figsize=(9, 5))
ax.plot(lr_custom.loss_history_, color="#FF6B35", linewidth=2)
ax.set_title("Custom Logistic Regression — Training Loss", fontsize=13, fontweight="bold")
ax.set_xlabel("Iteration")
ax.set_ylabel("Binary Cross-Entropy Loss")
ax.grid(alpha=0.4)

plt.tight_layout()
plt.savefig("outputs/logistic_regression_loss.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: outputs/logistic_regression_loss.png")

print("\n✅ All done!")