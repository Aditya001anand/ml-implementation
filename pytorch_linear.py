import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

data = pd.read_csv('data.csv')
cols = ['LongestShell', 'Diameter', 'Height', 'WholeWeight', 'ShuckedWeight', 'VisceraWeight', 'ShellWeight']

x = torch.tensor(data[cols].values, dtype=torch.float32)
y = torch.tensor(data['Rings'].values, dtype=torch.float32).view(-1, 1)

split = int(0.8 * len(x))
x_train, y_train = x[:split], y[:split]
x_test, y_test = x[split:], y[split:]

model = nn.Linear(7, 1)
loss_fn = nn.MSELoss()
opt = torch.optim.SGD(model.parameters(), lr=0.01)

loader = DataLoader(TensorDataset(x_train, y_train), batch_size=32, shuffle=True)

for epoch in range(50):
    for x_batch, y_batch in loader:
        preds = model(x_batch)
        loss = loss_fn(preds, y_batch)
        opt.zero_grad()
        loss.backward()
        opt.step()

test_preds = model(x_test)
mse = loss_fn(test_preds, y_test).item()

print(f"bias: {model.bias.item():.4f}")
print(f"weights: {[round(w, 4) for w in model.weight[0].tolist()]}")
print(f"mse: {mse:.4f}")