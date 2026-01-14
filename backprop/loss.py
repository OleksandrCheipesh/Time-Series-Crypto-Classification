import numpy as np


class MSELoss:

    def __init__(self):
        self.y_pred = None
        self.y_true = None

    def forward(self, y_pred, y_true):

        self.y_pred = y_pred
        self.y_true = y_true

        diff = y_pred - y_true
        loss = np.mean(diff ** 2)

        return loss

    def backward(self):

        batch_size = self.y_pred.shape[0]
        grad = 2.0 * (self.y_pred - self.y_true) / batch_size

        return grad

    def __call__(self, y_pred, y_true):
        return self.forward(y_pred, y_true)