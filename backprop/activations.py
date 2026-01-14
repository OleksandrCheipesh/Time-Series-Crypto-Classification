import numpy as np
from backprop.module import Module


class Sigmoid(Module):
    def __init__(self):
        super().__init__()
        self.output = None

    def forward(self, x):
        self.output = 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
        return self.output

    def backward(self, grad_output):
        return grad_output * self.output * (1.0 - self.output)


class Tanh(Module):
    def __init__(self):
        super().__init__()
        self.output = None

    def forward(self, x):
        self.output = np.tanh(x)
        return self.output

    def backward(self, grad_output):
        return grad_output * (1.0 - self.output ** 2)


class ReLU(Module):
    def __init__(self):
        super().__init__()
        self.mask = None

    def forward(self, x):
        self.mask = (x > 0)  # Store mask for backward pass
        return x * self.mask

    def backward(self, grad_output):
        return grad_output * self.mask