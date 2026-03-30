import numpy as np
from abc import ABC, abstractmethod


# ===== Learning Rate Schedules =====
class LearningRateSchedule(ABC):
    @abstractmethod
    def get_lr(self, iteration: int) -> float:
        pass


class ConstantLR(LearningRateSchedule):
    def __init__(self, lr: float = 0.1):
        self.lr = lr

    def get_lr(self, iteration: int) -> float:
        return self.lr


class TimeDecayLR(LearningRateSchedule):
    """
    Learning rate schedule:
        eta_k = lambda_ * (s0 / (s0 + k))^p
    with defaults s0 = 1, p = 0.5.
    """
    def __init__(self, lambda_: float = 1.0):
        self.s0 = 1
        self.p = 0.5
        self.lambda_ = lambda_

    def get_lr(self, iteration: int) -> float:
        return self.lambda_ * (self.s0 / (self.s0 + iteration)) ** self.p


# ===== Base Optimizer =====
class BaseDescent(ABC):
    def __init__(
        self,
        lr_schedule: LearningRateSchedule = TimeDecayLR,
        tolerance: float = 1e-6,
        max_iter: int = 1000,
    ):
        # lr_schedule may be passed as a class or an instance
        if isinstance(lr_schedule, type):
            self.lr_schedule = lr_schedule()
        else:
            self.lr_schedule = lr_schedule
        self.tolerance = tolerance
        self.max_iter = max_iter
        self.iteration = 0
        self.model = None

    def set_model(self, model):
        self.model = model

    @abstractmethod
    def _update_weights(self) -> np.ndarray:
        """Perform one weight update; return the weight difference w_new - w_old."""
        pass

    def optimize(self):
        """
        Main optimization loop used by all subclasses.

        Logs loss before each update (starting at iter 0) and once after the
        loop ends, giving max_iter+1 values when all iterations run.

        Stopping criteria:
          - ||w_{k+1} - w_k||^2 < tolerance
          - weight difference contains NaN
          - iteration count reaches max_iter
        """
        for _ in range(self.max_iter):
            # Record loss BEFORE this update
            self.model.loss_history.append(
                self.model.compute_loss(self.model.X_train, self.model.y_train)
            )
            delta = self._update_weights()
            self.iteration += 1

            # Stop if NaN or small enough update
            if np.any(np.isnan(delta)) or np.dot(delta, delta) < self.tolerance:
                break

        # Record final loss AFTER optimization finishes
        self.model.loss_history.append(
            self.model.compute_loss(self.model.X_train, self.model.y_train)
        )


# ===== Analytic Solution Optimizer =====
class AnalyticSolutionOptimizer:
    """
    Not an iterative optimizer — delegates weight computation to the loss
    function's analytic solution method.
    """
    def __init__(self):
        self.model = None

    def set_model(self, model):
        self.model = model

    def optimize(self):
        self.model.w = self.model.loss_function.analytic_solution(
            self.model.X_train, self.model.y_train
        )


# ===== Specific Optimizers =====
class VanillaGradientDescent(BaseDescent):
    """
    Full-batch gradient descent:
        w_{k+1} = w_k - eta_k * grad Q(w_k)
    """
    def _update_weights(self) -> np.ndarray:
        eta = self.lr_schedule.get_lr(self.iteration)
        gradient = self.model.compute_gradients(
            self.model.X_train, self.model.y_train
        )
        delta = -eta * gradient
        self.model.w += delta
        return delta


class StochasticGradientDescent(BaseDescent):
    """
    Mini-batch stochastic gradient descent:
        w_{k+1} = w_k - eta_k * (1/|B|) * sum_{i in B} grad q_i(w_k)
    """
    def __init__(
        self,
        lr_schedule: LearningRateSchedule = TimeDecayLR,
        batch_size: int = 1,
        tolerance: float = 1e-6,
        max_iter: int = 1000,
    ):
        super().__init__(lr_schedule, tolerance, max_iter)
        self.batch_size = batch_size

    def _update_weights(self) -> np.ndarray:
        n = self.model.X_train.shape[0]
        idx = np.random.randint(0, n, size=self.batch_size)
        X_batch = self.model.X_train[idx]
        y_batch = self.model.y_train[idx]

        eta = self.lr_schedule.get_lr(self.iteration)
        gradient = self.model.compute_gradients(X_batch, y_batch)
        delta = -eta * gradient
        self.model.w += delta
        return delta


class SAGDescent(BaseDescent):
    """
    Stochastic Average Gradient (SAG):
        g_bar <- g_bar + (1/n) * (g_j_new - g_j_old)
        w_{k+1} = w_k - eta_k * g_bar
    """
    def __init__(
        self,
        lr_schedule: LearningRateSchedule = TimeDecayLR,
        batch_size: int = 1,
        tolerance: float = 1e-6,
        max_iter: int = 1000,
    ):
        super().__init__(lr_schedule, tolerance, max_iter)
        self.batch_size = batch_size
        self.grad_memory = None
        self.avg_grad = None

    def _update_weights(self) -> np.ndarray:
        X_train = self.model.X_train
        y_train = self.model.y_train
        n, d = X_train.shape

        # Initialize grad memory on first call
        if self.grad_memory is None:
            self.grad_memory = np.zeros((n, d))
            self.avg_grad = np.zeros(d)

        # Pick a random batch
        idx = np.random.randint(0, n, size=self.batch_size)

        for j in idx:
            g_new = self.model.compute_gradients(
                X_train[j : j + 1], y_train[j : j + 1]
            )
            self.avg_grad += (g_new - self.grad_memory[j]) / n
            self.grad_memory[j] = g_new

        eta = self.lr_schedule.get_lr(self.iteration)
        delta = -eta * self.avg_grad
        self.model.w += delta
        return delta


class MomentumDescent(BaseDescent):
    """
    Gradient descent with momentum:
        h_0 = 0
        h_{k+1} = alpha * h_k + eta_k * grad Q(w_k)
        w_{k+1} = w_k - h_{k+1}
    """
    def __init__(
        self,
        lr_schedule: LearningRateSchedule = TimeDecayLR,
        beta: float = 0.9,
        tolerance: float = 1e-6,
        max_iter: int = 1000,
    ):
        super().__init__(lr_schedule, tolerance, max_iter)
        self.beta = beta
        self.velocity = None

    def _update_weights(self) -> np.ndarray:
        if self.velocity is None:
            self.velocity = np.zeros_like(self.model.w)

        eta = self.lr_schedule.get_lr(self.iteration)
        gradient = self.model.compute_gradients(
            self.model.X_train, self.model.y_train
        )
        self.velocity = self.beta * self.velocity + eta * gradient
        delta = -self.velocity
        self.model.w += delta
        return delta


class Adam(BaseDescent):
    """
    Adam (Adaptive Moment Estimation):
        m_0 = 0,  v_0 = 0
        m_{k+1} = beta1 * m_k + (1 - beta1) * grad
        v_{k+1} = beta2 * v_k + (1 - beta2) * grad^2
        m_hat = m_{k+1} / (1 - beta1^{k+1})
        v_hat = v_{k+1} / (1 - beta2^{k+1})
        w_{k+1} = w_k - eta_k / (sqrt(v_hat) + eps) * m_hat
    """
    def __init__(
        self,
        lr_schedule: LearningRateSchedule = TimeDecayLR,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        tolerance: float = 1e-6,
        max_iter: int = 1000,
    ):
        super().__init__(lr_schedule, tolerance, max_iter)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = None
        self.v = None

    def _update_weights(self) -> np.ndarray:
        if self.m is None:
            self.m = np.zeros_like(self.model.w)
            self.v = np.zeros_like(self.model.w)

        gradient = self.model.compute_gradients(
            self.model.X_train, self.model.y_train
        )
        # k+1 because iteration is 0-indexed; bias-correction uses the step number
        k = self.iteration + 1

        self.m = self.beta1 * self.m + (1 - self.beta1) * gradient
        self.v = self.beta2 * self.v + (1 - self.beta2) * gradient ** 2

        m_hat = self.m / (1 - self.beta1 ** k)
        v_hat = self.v / (1 - self.beta2 ** k)

        eta = self.lr_schedule.get_lr(self.iteration)
        delta = -eta / (np.sqrt(v_hat) + self.eps) * m_hat
        self.model.w += delta
        return delta
