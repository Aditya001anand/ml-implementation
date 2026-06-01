import csv

x, y = [], []
with open('data.csv', 'r') as f:
    reader = csv.DictReader(f)
    cols = ['LongestShell', 'Diameter', 'Height', 'WholeWeight', 'ShuckedWeight', 'VisceraWeight', 'ShellWeight']
    for row in reader:
        x.append([float(row[c]) for c in cols])
        y.append(float(row['Rings']))

rows = len(x)
cols_cnt = len(x[0])

for j in range(cols_cnt):
    col_vals = [x[i][j] for i in range(rows)]
    mean = sum(col_vals) / rows
    std = (sum((v - mean)**2 for v in col_vals) / rows) ** 0.5
    for i in range(rows):
        if std != 0:
            x[i][j] = (x[i][j] - mean) / std

for i in range(rows):
    x[i] = [1.0] + x[i]

cols_cnt += 1 
split = int(0.8 * rows)
x_train, y_train = x[:split], y[:split]

w = [0.0] * cols_cnt
lr = 0.1

for epoch in range(500):
    grads = [0.0] * cols_cnt
    for i in range(len(x_train)):
        pred = sum(w[j] * x_train[i][j] for j in range(cols_cnt))
        err = pred - y_train[i]
        for j in range(cols_cnt):
            grads[j] += err * x_train[i][j]
    for j in range(cols_cnt):
        w[j] -= lr * (grads[j] / len(x_train))

print(f"bias: {w[0]:.4f}")
print(f"weights: {[round(wt, 4) for wt in w[1:]]}")