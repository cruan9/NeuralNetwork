from abc import ABC, abstractmethod

import numpy as np

class Activation(ABC):

    @abstractmethod
    def value(self, x: np.ndarray) -> np.ndarray:
        return x

    @abstractmethod
    def derivative(self, x: np.ndarray) -> np.ndarray:
        return x

class Identity(Activation):

    def value(self, x: np.ndarray) -> np.ndarray:
        return x

    def derivative(self, x: np.ndarray) -> np.ndarray:
        return np.ones(x.shape)

class Sigmoid(Activation):

    def __init__(self, k: float=1.):
        self.k = k
        super(Sigmoid, self).__init__()

    def value(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-1 * self.k * x))

    def derivative(self, x: np.ndarray) -> np.ndarray:
        return (1- ((1/(1+np.exp((-1 * self.k) * x))))) * (1/(1+np.exp((-1 * self.k) * x)))


class ReLU(Activation):

    def __init__(self):
        super(ReLU, self).__init__()

    def value(self, x: np.ndarray) -> np.ndarray:
        def f(x):
            return max(0,x)
        return np.vectorize(f)(x)

    def derivative(self, x: np.ndarray) -> np.ndarray:
        def f(x):
            if x <= 0:
                return 0
            return 1
        return np.vectorize(f)(x)

class Loss(ABC):

    @abstractmethod
    def value(self, y_hat: np.ndarray, y: np.ndarray) -> float:
        return 0

    @abstractmethod
    def derivative(self, y_hat: np.ndarray, y: np.ndarray) -> np.ndarray:
        return y_hat


class MeanSquaredError(Loss):

    def value(self, y_hat: np.ndarray, y: np.ndarray) -> float:
        return np.square(np.subtract(y, y_hat)).mean()

    def derivative(self, y_hat: np.ndarray, y: np.ndarray) -> np.ndarray:
        return -2 * (y - y_hat)/y_hat.shape[0]

def mean_absolute_error(y_hat: np.ndarray, y: np.ndarray) -> float:
    sum = 0
    n = len(y)
    for i in range (n):
        sum += abs(y[i] - y_hat[i])
    return sum/n