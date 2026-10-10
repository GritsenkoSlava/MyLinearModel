import pandas as pd
import numpy as np

class Tensor:
    def __init__(self):
        self.data = pd.read_csv("/home/koturo/train_data/train_xy_2.csv") 

    def __repr__(self):
        return f"Tensor: {self.data}"

    def process_data(self):
        data_pd = pd.DataFrame(self.data)

        data_x_pd = data_pd.iloc[:, 0]
        data_y_pd = data_pd.iloc[:, 1]

        data_x = data_x_pd.to_numpy(dtype=float)
        data_y = data_y_pd.to_numpy(dtype=float)

        print("X:", data_x)
        print("Y:", data_y)

        return data_x, data_y

print(Tensor().process_data())