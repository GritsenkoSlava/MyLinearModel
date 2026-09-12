import torch
import torch.nn as nn

class Linear:

    def __init__(self):
        self.x = torch.tensor(
            [[1], [2], [3], [4], [5]],
            dtype=torch.float32
        )

        self.y = torch.tensor(
            [[2], [4], [6], [8], [10]],
            dtype=torch.float32
        )

        self.model = nn.Linear(1, 1) 
        self.loss = nn.MSELoss()      
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)

    def prediction(self):
        predict = self.model(self.x)
        return predict

    def learn(self, epoch):
         for epoch in range(epoch):
            predict = self.prediction()
            loss = self.loss(predict, self.y)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()


            if epoch % 100 == 0:    # вывод каждых 100-а эпох и их результаты (при параметре epoch=1000)
                print(f'Epoch: {epoch}, Error: {loss.item():.6f}')

    def check(self):
        num = torch.tensor([[10.0]], dtype=torch.float32)
        result = self.model(num)
        return result

model = Linear()
model.learn(1000)
result = model.check()
print(f'Result: {result.item()}')