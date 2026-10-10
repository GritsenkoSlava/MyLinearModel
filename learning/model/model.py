import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from learning.core.tensor import Tensor
from learning.engine.engine import Engine

class Model:
    def __init__(self):
        self.engine = Engine()
        self.x, self.y = Tensor().process_data()
        self.w = 0.0
        self.b = 0.0
        

    def learn(self, epochs):
        for epoch in range(epochs):
            prediction = self.engine.predict(self.x, self.w, self.b)
            loss = self.engine.loss(prediction, self.y)
            grad_w, grad_b = self.engine.gradient(self.x, prediction, self.y)
            self.w, self.b = self.engine.update(self.w, self.b, grad_w, grad_b, 0.0001)

            if (epoch + 1) % 100 == 0:
                print(
                    f'Loss: {loss:.6f}  |'
                    f'  Bias: {self.b:.6f}  '
                )

        return self.w, self.b

