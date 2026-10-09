from numpy import array, mean

def train_model():
    x = array([1, 2, 3, 4, 5], dtype=float)
    y = array([2, 4, 6, 8, 10], dtype=float)

    lr   = 0.01
    w    = 1
    #b = 0

    epoch = 10000

    for i in range(epoch):
        #predict = w*x + b
        predict = w*x

        error = predict - y
        loss  = mean(error**2)

        gradient = mean(2*x*loss)
        w = w-lr*gradient
        yield w

        #gradient_w = np.mean(w*x+b-y)*x
        #gradient_b = np.mean(w*x+b-y)

        #w = w-lr*gradient_w
        #b = b-lr*gradient_b

        #yield w
        #yield b

#b = train_model()
w = next(train_model())

number = 5
result = w*number

#print(f'Смещение:     ', b)
print('Вес:     ', w)
print('Модель:  ', result)