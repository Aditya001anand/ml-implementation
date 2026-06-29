import pandas as pd
import torch as t
import torch.nn as nn
from torch.utils.data import TensorDataset,DataLoader

data=pd.read_csv('data.csv')
columns=['LongestShell','Diameter','Height','WholeWeight','ShuckedWeight','VisceraWeight','ShellWeight']
x=t.tensor(data[columns].values,dtype=t.float32)
y=t.tensor(data['Rings'].values, dtype=t.float32).view(-1,1)
split=int(0.8*len(x))
x_t,y_t=x[:split],y[:split]
x_test,y_test=x[split:],y[split:]
model=nn.Linear(7,1)
loss_fn=nn.MSELoss()
opt=t.optim.SGD(model.parameters(),lr=0.01)
loader=DataLoader(TensorDataset(x_t,y_t),batch_size=32,shuffle=True)
for epoch in range(50):
    for x_batch,y_batch in loader:
        preds=model(x_batch)
        loss=loss_fn(preds,y_batch)
        opt.zero_grad()
        loss.backward()
        opt.step()
test_preds=model(x_test)
mse=loss_fn(test_preds,y_test).item()
ss_res=t.sum((y_test-test_preds)**2)
ss_tot=t.sum((y_test-t.mean(y_test))**2)
r2=1-(ss_res/ss_tot)
print(f"bias:{model.bias.item():.4f}")
print(f"weights:{[round(w,4) for w in model.weight[0].tolist()]}") 
print(f"mse: {mse:.4f}")
print(f"r2:{r2.item():.4f}")
