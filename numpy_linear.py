import numpy as np
import pandas as pd

data=pd.read_csv('data.csv')
columns=['LongestShell','Diameter','Height','WholeWeight','ShuckedWeight','VisceraWeight','ShellWeight']
x=data[columns].values
y=data['Rings'].values
np.random.seed(42)
idx=np.random.permutation(len(x))
x,y=x[idx],y[idx]
split=int(0.8*len(x))
x_t,y_t=x[:split],y[:split]
x_test,y_test=x[split:],y[split:]
ones=np.ones((len(x_t),1))
x_t_b=np.c_[ones,x_t]
w=np.linalg.pinv(x_t_b.T.dot(x_t_b)).dot(x_t_b.T).dot(y_t)
test_ones = np.ones((len(x_test),1))
x_test_b=np.c_[test_ones,x_test]
preds=x_test_b.dot(w)
mse=np.mean((preds-y_test)**2)

print(f"bias: {w[0]:.4f}")
print(f"weights: {[round(weight,4) for weight in w[1:]]}")
print(f"mse: {mse:.4f}")
