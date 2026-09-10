import numpy as np
import random

def train_model():
    x = np.array([1, 2, 3, 4, 5], dtype=float)  # Взодные данные
    y = np.array([2, 4, 6, 8, 10], dtype=float) # Выходные данные

    w = random.random() # начинаем со случайного веса
    lr = 0.01           # скорость обучения
    epoch = 10000       # кол-во эпох (циклов)

    for epoch in range(epoch):
        prediction = x * w

        error = prediction - y
        loss  = np.mean(error**2)

        gradient = np.mean(2*x*error)
        w = w-lr*gradient

        return w
        return prediction
    
w = train_model()
number = 20
result = number*w

print(f'Вес:         ', w)
print(f'Модель:      ', result)