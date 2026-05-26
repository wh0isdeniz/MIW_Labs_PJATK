"""
reglog.py — Custom Logistic Regression (no sklearn LogisticRegression)
Implements gradient descent with sigmoid activation.
"""

import numpy as np


class LogisticRegressionCustom:
    """
    Binary logistic regression trained via batch gradient descent.

    Parameters
    ----------
    lr : float
        Learning rate (default 0.1).
    n_iter : int
        Number of gradient-descent iterations (default 1000).
    tol : float
        Stop early if |loss change| < tol (default 1e-6).
    random_state : int or None
        Seed for reproducibility.
    """

    def __init__(self, lr: float = 0.1, n_iter: int = 1000,
                 tol: float = 1e-6, random_state=42):
        self.lr = lr
        self.n_iter = n_iter
        self.tol = tol
        self.random_state = random_state
        self.weights_ = None   # shape (n_features,) — learned after fit()
        self.bias_ = None      # scalar intercept term
        self.loss_history_ = []

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        """
        Numerically stable sigmoid.

        For z >= 0 uses the standard form 1/(1+e^-z).
        For z < 0 uses the equivalent e^z/(1+e^z) to avoid overflow in exp(-z).
        """
        return np.where(
            z >= 0,
            1.0 / (1.0 + np.exp(-z)),
            np.exp(z) / (1.0 + np.exp(z)),
        )

    def _binary_cross_entropy(self, y: np.ndarray, p: np.ndarray) -> float:
        """
        Compute mean binary cross-entropy loss: -mean(y*log(p) + (1-y)*log(1-p)).
        Probabilities are clipped to [eps, 1-eps] to prevent log(0).
        """
        eps = 1e-15
        p = np.clip(p, eps, 1 - eps)
        return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegressionCustom":
        """Train the model using batch gradient descent."""
        rng = np.random.default_rng(self.random_state)
        n_samples, n_features = X.shape

        # Xavier initialisation: keeps gradient magnitudes stable across layers
        self.weights_ = rng.normal(0, 1 / np.sqrt(n_features), size=n_features)
        self.bias_ = 0.0
        self.loss_history_ = []

        prev_loss = float("inf")

        for _ in range(self.n_iter):
            # Forward pass: linear combination → sigmoid → probability
            z = X @ self.weights_ + self.bias_
            p = self._sigmoid(z)

            loss = self._binary_cross_entropy(y, p)
            self.loss_history_.append(loss)

            # Gradients of BCE loss w.r.t. weights and bias
            error = p - y           # dL/dz = p - y (combined sigmoid + BCE gradient)
            dw = X.T @ error / n_samples
            db = error.mean()

            # Gradient descent parameter update
            self.weights_ -= self.lr * dw
            self.bias_ -= self.lr * db

            # Early stopping: halt if loss improvement falls below tolerance
            if abs(prev_loss - loss) < self.tol:
                break
            prev_loss = loss

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Return class-membership probabilities.

        Returns
        -------
        proba : ndarray of shape (n_samples, 2)
            Column 0 → P(class=0), Column 1 → P(class=1).
        """
        p1 = self._sigmoid(X @ self.weights_ + self.bias_)
        return np.column_stack([1 - p1, p1])

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return hard class labels (0 or 1) using a 0.5 decision threshold."""
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute classification accuracy on (X, y)."""
        return np.mean(self.predict(X) == y)