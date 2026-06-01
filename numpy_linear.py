import numpy as np
import pandas as pd

data = pd.read_csv('data.csv')
cols = ['LongestShell', 'Diameter', 'Height', 'WholeWeight', 'ShuckedWeight', 'VisceraWeight', 'ShellWeight'] 

x = data[cols].values
y = data['Rings'].values

split = int(0.8 * len(x))
x_train, y_train = x[:split], y[:split]

ones = np.ones((len(x_train), 1))
x_train_b = np.c_[ones, x_train]

w = np.linalg.pinv(x_train_b.T.dot(x_train_b)).dot(x_train_b.T).dot(y_train)

print(f"bias: {w[0]:.4f}")
print(f"weights: {[round(weight, 4) for weight in w[1:]]}")