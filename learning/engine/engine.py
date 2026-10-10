import numpy as np
class Engine:
    def __init__(self):
        pass

    def predict(self, x, w, b):
        return x * w + b

    def loss(self, predict, target):
        return np.mean((predict - target)**2)

    def gradient(self, x, predict, target):
        d_loss = 2 * (predict - target)

        grad_w = np.mean(d_loss * x)
        grad_b = np.mean(d_loss)

        return grad_w, grad_b


    def update(self, w, b, grad_w, grad_b,  learning_rate: float):
        w = w - learning_rate * grad_w
        b = b - learning_rate * grad_b

        return w, b