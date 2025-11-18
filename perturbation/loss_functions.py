from abc import ABC, abstractmethod
import numpy as np

from rainnet.utils import invScaler


class LossFunction(ABC):

    @abstractmethod
    def calculate(self, predicted, gt):
        pass


class LogCosh(LossFunction):

    def calculate(self, pred, target):
        pred = np.asarray(pred, dtype=np.float32)
        target = np.asarray(target, dtype=np.float32)

        diff = pred - target
        log_cosh = np.log(np.cosh(diff))

        return float(np.mean(log_cosh))


class MSE(LossFunction):

    def calculate(self, pred, target):
        pred = np.asarray(pred, dtype=np.float32)
        target = np.asarray(target, dtype=np.float32)

        diff = pred - target
        mse = np.mean(diff ** 2)

        return float(mse)


class BMSE(LossFunction):
    # Balanced Mean Squared Error
    # Penalize more wrong values due to class imbalance
    def calculate(self, pred, target):
        weight = 5.0
        threshold = 0.01

        pred = np.asarray(invScaler(pred), dtype=np.float32)
        target = np.asarray(invScaler(target), dtype=np.float32)
        rain_mask = target > threshold
        weights = np.where(rain_mask, weight, 1.0)

        diff = pred - target
        return float(np.mean((diff ** 2) * weights))


class RainAccuracy(LossFunction):
    # accuracy of binary images after thresholding. Ignoring 0 == 0 matches due to class imbalance
    def calculate(self, pred, target):
        pred = invScaler(np.asarray(pred, dtype=np.float32))
        target = invScaler(np.asarray(target, dtype=np.float32))

        threshold = 0.01
        pred_mask = pred > threshold
        target_mask = target > threshold
        correct_sum = np.sum((pred_mask == 1) & (target_mask == 1))
        wrong_sum = np.sum((pred_mask == 1) & (target_mask == 0)) + np.sum((pred_mask == 0) & (target_mask == 1))
        return float(correct_sum / (correct_sum + wrong_sum))


def get_function(function):
    if function == "logcosh":
        return LogCosh()
    elif function == "MSE":
        return MSE()
    elif function == "BMSE":
        return BMSE()
    elif function == "accuracy":
        return RainAccuracy()
