import torch
import torch.nn as nn 

# Данные для обучения
x = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0]])  # Входные данные
y = torch.tensor([[2.0], [4.0], [6.0], [8.0], [10.0]]) # Выходные данные

# Создаем модель
model = nn.Linear(1, 1)  

# Функция ошибки
loss_func = nn.MSELoss()

# Оптимизатор
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)  

# Процесс обучения (1000 раз)
for epoch in range(1000):
    prediction = model(x)           # Предсказание модели
    loss = loss_func(prediction, y) # Считаем ошибку

    optimizer.zero_grad()           # Обнуляем старые градиенты
    loss.backward()                 # Вычисляем новые градиенты
    optimizer.step()                # Изменяем веса модели

# Проверяем модель
number = torch.tensor([[10.0]])  # Проверяем модель на новом числе
result = model(number)           # Получаем предсказание

print(result.item())             # Выводим результат