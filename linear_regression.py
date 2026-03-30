import numpy as np
from abc import ABC, abstractmethod
from scipy.sparse.linalg import svds


# ===== Analytic Solution Optimizer =====
# Defined here so the notebook can do:
#   from linear_regression import MSELoss, CustomLinearRegression, AnalyticSolutionOptimizer
class AnalyticSolutionOptimizer:
    """
    Non-iterative optimizer that delegates weight computation to the loss
    function's analytic_solution method.
    """
    def __init__(self):
        self.model = None

    def set_model(self, model):
        self.model = model

    def optimize(self):
        self.model.w = self.model.loss_function.analytic_solution(
            self.model.X_train, self.model.y_train
        )


# ===== Loss Function Interface =====
class LossFunction(ABC):
    """
    Abstract base class for all loss functions.
    Subclasses must implement `loss`, `gradient`, and optionally `analytic_solution`.
    """

    @abstractmethod
    def loss(self, X: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
        """Compute scalar loss value."""
        pass

    @abstractmethod
    def gradient(self, X: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
        """Compute gradient of the loss w.r.t. w."""
        pass

    def analytic_solution(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        raise NotImplementedError(
            f"{type(self).__name__} does not provide an analytic solution."
        )


# ===== MSE Loss =====
class MSELoss(LossFunction):
    """
    Mean Squared Error loss:
        Q(w) = (1/l) * ||X w - y||^2

    Gradient in matrix form:
        dQ/dw = (2/l) * X^T (X w - y)

    Analytic (closed-form) solution:
        w* = (X^T X)^{-1} X^T y   (plain)
        w* = V Sigma^{+} U^T y      (SVD-based, handles rank-deficient X)
    """

    def __init__(self, analytic_solution_func=None):
        """
        Parameters
        ----------
        analytic_solution_func : callable, optional
            If provided, overrides the default _plain_analytic_solution.
            Signature: func(self, X, y) -> w
        """
        self._analytic_solution_func = analytic_solution_func

    def loss(self, X: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
        residuals = X @ w - y
        return float(np.dot(residuals, residuals) / len(y))

    def gradient(self, X: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
        l = len(y)
        residuals = X @ w - y
        return (2.0 / l) * (X.T @ residuals)

    def analytic_solution(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        if self._analytic_solution_func is not None:
            return self._analytic_solution_func(self, X, y)
        return self._plain_analytic_solution(X, y)

    def _plain_analytic_solution(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        w* = (X^T X)^{-1} X^T y
        """
        return np.linalg.solve(X.T @ X, X.T @ y)

    def _svd_analytic_solution(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        SVD-based pseudo-inverse solution (handles rank-deficient X):
            X = U Sigma V^T
            w* = V Sigma^{+} U^T y

        Uses scipy.sparse.linalg.svds with the maximum available number of
        singular values (min(n, d) - 1) and highest numerical accuracy.

        Note: svds does not support computing ALL singular values, so we
        compute min(n, d) - 1 and rely on the fact that for our data the
        rank is at most min(n, d) - 1 (with probability 1 for random X,
        even with one linear-dependency column). This is the truncated SVD /
        economy SVD approach. The minimum number of singular values needed
        to recover the exact solution with probability 1 is rank(X), which
        equals min(n, d) - 1 when X has one linearly dependent column.
        """
        n, d = X.shape
        k = min(n, d) - 1  # max singular values svds can compute
        U, s, Vt = svds(X, k=k, tol=0)          # tol=0 → highest accuracy
        # Build pseudo-inverse: Sigma^+ applied only to non-zero singular values
        threshold = np.finfo(float).eps * max(n, d) * s.max()
        s_inv = np.where(s > threshold, 1.0 / s, 0.0)
        # w* = V @ diag(s_inv) @ U^T @ y
        return Vt.T @ (s_inv * (U.T @ y))


# ===== L2 Regularization Mixin =====
class L2Regularization(LossFunction):
    """
    Wraps any LossFunction and adds L2 regularization:
        Q_reg(w) = Q(w) + (mu/2) * ||w||^2

    Gradient:
        dQ_reg/dw = dQ/dw + mu * w
    """

    def __init__(self, base_loss: LossFunction, mu: float = 1e-3):
        self.base_loss = base_loss
        self.mu = mu

    def loss(self, X: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
        return self.base_loss.loss(X, y, w) + 0.5 * self.mu * np.dot(w, w)

    def gradient(self, X: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
        return self.base_loss.gradient(X, y, w) + self.mu * w

    def analytic_solution(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        # Ridge regression closed form: w* = (X^T X + mu * I)^{-1} X^T y
        d = X.shape[1]
        return np.linalg.solve(X.T @ X + self.mu * np.eye(d), X.T @ y)


# ===== LogCosh Loss =====
class LogCoshLoss(LossFunction):
    """
    Log-Cosh loss:
        L(y, a) = (1/n) * sum log(cosh(a_i - y_i))

    Gradient:
        dL/dw = (1/n) * X^T tanh(X w - y)
    """

    def loss(self, X: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
        residuals = X @ w - y
        return float(np.mean(np.log(np.cosh(residuals))))

    def gradient(self, X: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
        residuals = X @ w - y
        return (X.T @ np.tanh(residuals)) / len(y)


# ===== Huber Loss =====
class HuberLoss(LossFunction):
    """
    Huber loss with threshold delta:
        L(y, a) = (1/n) * sum l_i
        where l_i = 0.5 * (a_i - y_i)^2          if |a_i - y_i| < delta
                  = delta * |a_i - y_i| - 0.5 * delta^2  otherwise

    Gradient:
        dL/dw = (1/n) * X^T g
        where g_i = (a_i - y_i)   if |a_i - y_i| < delta
                  = delta * sign(a_i - y_i) otherwise
    """

    def __init__(self, delta: float = 1.0):
        self.delta = delta

    def loss(self, X: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
        r = X @ w - y
        mask = np.abs(r) < self.delta
        values = np.where(mask, 0.5 * r**2, self.delta * np.abs(r) - 0.5 * self.delta**2)
        return float(np.mean(values))

    def gradient(self, X: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
        r = X @ w - y
        g = np.where(np.abs(r) < self.delta, r, self.delta * np.sign(r))
        return (X.T @ g) / len(y)


# ===== Linear Regression Model =====
class CustomLinearRegression:
    """
    Linear regression model that combines a LossFunction and an optimizer
    through dependency injection.

    Usage
    -----
    lr = CustomLinearRegression(optimizer=VanillaGradientDescent(), loss_function=MSELoss())
    lr.fit(X, y)
    y_pred = lr.predict(X)
    """

    def __init__(self, optimizer=None, loss_function: LossFunction = None,
                 tolerance: float = 1e-6, max_iter: int = 1000):
        self.optimizer = optimizer
        self.loss_function = loss_function if loss_function is not None else MSELoss()
        self.tolerance = tolerance
        self.max_iter = max_iter
        self.w = None
        self.X_train = None
        self.y_train = None
        self.loss_history = []

        # Give the optimizer a back-reference to this model
        if optimizer is not None and hasattr(optimizer, 'set_model'):
            optimizer.set_model(self)

        # Allow tolerance / max_iter overrides on the optimizer
        if hasattr(optimizer, 'tolerance') and optimizer.tolerance == 1e-6:
            optimizer.tolerance = tolerance
        if hasattr(optimizer, 'max_iter') and optimizer.max_iter == 1000:
            optimizer.max_iter = max_iter

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predicted values X @ w."""
        return X @ self.w

    def compute_gradients(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Delegate gradient computation to the loss function."""
        return self.loss_function.gradient(X, y, self.w)

    def compute_loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """Delegate loss computation to the loss function."""
        return self.loss_function.loss(X, y, self.w)

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Train the model.
        - Initializes weights to zeros.
        - Delegates the optimization loop to self.optimizer.optimize().
        """
        self.X_train = X
        self.y_train = y
        self.w = np.zeros(X.shape[1])
        self.loss_history = []

        if self.optimizer is not None:
            self.optimizer.optimize()
        else:
            raise ValueError("No optimizer provided.")
