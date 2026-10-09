class Engine:
    def __init__(self):
        pass

    def predict(self, x, w, b):
        return self.x * self.w + self.b

    def loss(self, predict, target):
        return (predict - target)**2

    def gradient(self, x, w, b, predict, target):
        d_loss = 2 * (predict - target)

        w.grad = d_loss * x.data
        b.grad = d_loss

        return x, b

