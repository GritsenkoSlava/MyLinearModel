import numpy as np

def train_model():
    x = np.array([1, 2, 3, 4, 5], dtype=float)
    y = np.array([2, 4, 6, 8, 10], dtype=float)

    lr   = 0.01
    w    = 1
    #b = 0

    epoch = 10000

    for i in range(epoch):
        #predict = w*x + b
        predict = w*x

        error = predict - y
        loss  = np.mean(error**2)

        gradient = np.mean(2*x*loss)
        w = w-lr*gradient
        return w

        #gradient_w = np.mean(w*x+b-y)*x
        #gradient_b = np.mean(w*x+b-y)

        #w = w-lr*gradient_w
        #b = b-lr*gradient_b

        #return w
        #return b

#b = train_model()
w = train_model()

number = 5
result = w*number

#print(f'Смещение:     ', b)
print(f'Вес:     ', w)
print(f'Модель:  ', result)