import numpy as np


class Module:

    def __init__(self):
        self.training = True

    def forward(self, x):

        raise NotImplementedError("Subclass must implement forward()")

    def backward(self, grad_output):

        raise NotImplementedError("Subclass must implement backward()")

    def update(self, learning_rate, momentum=0.0):

        pass

    def train(self):
        self.training = True

    def eval(self):
        self.training = False