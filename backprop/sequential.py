import numpy as np
from backprop.module import Module


class Sequential:

    def __init__(self, modules):

        self.modules = modules

    def forward(self, x):

        for module in self.modules:
            x = module.forward(x)
        return x

    def backward(self, grad_output):

        for module in reversed(self.modules):
            grad_output = module.backward(grad_output)
        return grad_output

    def update(self, learning_rate, momentum=0.0):

        for module in self.modules:
            module.update(learning_rate, momentum)

    def train(self):
        for module in self.modules:
            module.train()

    def eval(self):
        for module in self.modules:
            module.eval()

    def __call__(self, x):
        return self.forward(x)