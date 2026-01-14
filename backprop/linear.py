import numpy as np
from backprop.module import Module


class Linear(Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size


        std = np.sqrt(2.0 / input_size)
        self.weights = np.random.randn(input_size, output_size) * std

        self.bias = np.ones((1, output_size)) * 0.01

        self.x = None

        # Gradients
        self.grad_weights = None
        self.grad_bias = None

        # Momentum terms
        self.velocity_w = np.zeros_like(self.weights)
        self.velocity_b = np.zeros_like(self.bias)

    def forward(self, x):
        self.x = x
        return np.dot(x, self.weights) + self.bias

    def backward(self, grad_output):
        grad_input = np.dot(grad_output, self.weights.T)

        self.grad_weights = np.dot(self.x.T, grad_output)

        self.grad_bias = np.sum(grad_output, axis=0, keepdims=True)

        return grad_input

    def update(self, learning_rate, momentum=0.0):
        if momentum > 0:
            self.velocity_w = momentum * self.velocity_w - learning_rate * self.grad_weights
            self.velocity_b = momentum * self.velocity_b - learning_rate * self.grad_bias

            self.weights += self.velocity_w
            self.bias += self.velocity_b
        else:
            self.weights -= learning_rate * self.grad_weights
            self.bias -= learning_rate * self.grad_bias