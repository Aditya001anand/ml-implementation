import csv
x,y=[],[]
with open('data.csv','r') as f:
    reader=csv.DictReader(f)
    col=['LongestShell', 'Diameter', 'Height', 'WholeWeight', 'ShuckedWeight', 'VisceraWeight', 'ShellWeight']
    for row in reader:
        x.append([float(row[c]) for c in col])
        y.append(float(row['Rings']))
row_count=len(x)
column_count=len(x[0])
for i in range(column_count):
    column_values=[x[k][i] for k in range(row_count)]
    mean=sum(column_values)/row_count
    std=(sum((v-mean)**2 for v in column_values)/row_count)**0.5
    for j in range(row_count):
        if std!=0:
            x[j][i]=(x[j][i]-mean)/std
for i in range(row_count):
    x[i]=[1.0]+x[i]
column_count+=1
split=int(0.8*row_count)
x_t,y_t=x[:split],y[:split]
x_test,y_test=x[split:],y[split:]
w=[0.0]*column_count
lr=0.1
for epoch in range(500):
    grads=[0.0]*column_count
    for i in range(len(x_t)):
        pred=sum(w[j]*x_t[i][j] for j in range(column_count))
        err=pred-y_t[i]
        for j in range(column_count):
            grads[j]+=err*x_t[i][j]
    for j in range(column_count):
        w[j]-=lr*(grads[j]/len(x_t))
error_sum=0
for i in range(len(x_test)):
    test_pred=sum(w[j]*x_test[i][j] for j in range(column_count))
    error_sum+=(test_pred-y_test[i])**2
mse=error_sum/len(x_test)
print(f"bias:{w[0]:.4f}")
print(f"weights:{[round(wt,4) for wt in w[1:]]}")                                
print(f"mse:{mse:.4f}")
