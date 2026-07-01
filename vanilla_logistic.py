import csv
import math

def sigmoid(z):
    if z<-20: 
        return 0.0
    if z>20: 
        return 1.0
    return 1.0/(1.0+math.exp(-z))
x,y=[],[]
with open('IRIS.csv','r') as f:
    reader=csv.DictReader(f)
    col=['sepal_length','sepal_width','petal_length','petal_width']
    target=None
    for row in reader:
        x.append([float(row[c]) for c in col])
        if target is None:
            target=row['species']
        if row['species']==target:
            y.append(1.0)
        else:
            y.append(0.0)
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
        linear_math=sum(w[j]*x_t[i][j] for j in range(column_count))
        pred=sigmoid(linear_math)
        err=pred-y_t[i]
        for j in range(column_count):
            grads[j]+=err*x_t[i][j]
    for j in range(column_count):
        w[j]-=lr*(grads[j]/len(x_t))
tp=fp=tn=fn=0
for i in range(len(x_test)):
    linear_math=sum(w[j]*x_test[i][j] for j in range(column_count))
    prob=sigmoid(linear_math)
    if prob>=0.5:
        final_guess=1.0
    else:
        final_guess=0.0
    if final_guess==1.0 and y_test[i]==1.0:
        tp+=1
    elif final_guess==1.0 and y_test[i]==0.0:
        fp+=1
    elif final_guess==0.0 and y_test[i]==0.0:
        tn+=1
    elif final_guess==0.0 and y_test[i]==1.0:
        fn+=1
accuracy=((tp+tn)/len(x_test))*100
precision=tp/(tp+fp) if (tp+fp)>0 else 0
recall=tp/(tp+fn) if (tp+fn)>0 else 0
f1_score=2*(precision*recall)/(precision+recall) if (precision+recall)>0 else 0
print(f"bias:{w[0]:.4f}")
print(f"weights:{[round(wt,4) for wt in w[1:]]}")
print(f"Accuracy:{accuracy:.2f}%")
print(f"Precision:{precision:.4f}")
print(f"Recall:{recall:.4f}")
print(f"F1-Score:{f1_score:.4f}")
print("\nConfusion Matrix (Rows = Actual, Columns = Predicted):")
print(f"[[{tn}  {fp}]")
print(f" [{fn}  {tp}]]")
