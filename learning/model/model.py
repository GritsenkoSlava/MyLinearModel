from learning.core.tensor import Tensor
from learning.engine.engine import Engine

class Model:
    def __init__(self):
        self.engine = Engine()
        self.x, self.y = Tensor().process_data()
        self.w = 0
        self.b = 0
        

    def learn(self, epochs):
        for epoch in range(epochs):
            prediction = self.engine.predict(self.x, self.w, self.b)
            loss = self.engine.loss(prediction, self.y)
            grad = self.engine.gradient(self.x, self.w, self.b, prediction, self.y)
            return loss


        if epochs % 100 == 0:
            for _ in range(epoch):
                print(f'Predict: {prediction} || x/Bias: {grad}')