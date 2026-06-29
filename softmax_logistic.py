import csv
import math

def softmax(z_list):
    max_z=max(z_list)
    exps=[math.exp(z-max_z) for z in z_list]
    sum_exps=sum(exps)
    return [e/sum_exps for e in exps]
x,y_raw=[],[]
with open('data.csv','r') as f:
    reader=csv.DictReader(f)
    col=['LongestShell','Diameter','Height','WholeWeight','ShuckedWeight','VisceraWeight','ShellWeight']
    for row in reader:
        x.append([float(row[c]) for c in col])
        rings=float(row['Rings'])
        if rings<=8:
            y_raw.append(0)
        elif rings<=10:
            y_raw.append(1)
        else:
            y_raw.append(2)
num_classes=3
y=[]
for val in y_raw:
    one_hot=[0.0]*num_classes
    one_hot[val]=1.0
    y.append(one_hot)
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
x_test,y_test=x[split:],y_raw[split:]
w=[[0.0]*column_count for _ in range(num_classes)]
lr=0.1
for epoch in range(500):
    grads=[[0.0]*column_count for _ in range(num_classes)]
    for i in range(len(x_t)):
        scores=[]
        for c in range(num_classes):
            score=sum(w[c][j]*x_t[i][j] for j in range(column_count))
            scores.append(score)
        probs=softmax(scores)
        for c in range(num_classes):
            err=probs[c]-y_t[i][c]
            for j in range(column_count):
                grads[c][j]+=err*x_t[i][j]
    for c in range(num_classes):
        for j in range(column_count):
            w[c][j]-=lr*(grads[c][j]/len(x_t))
correct=0
tp=[0]*num_classes
fp=[0]*num_classes
fn=[0]*num_classes
for i in range(len(x_test)):
    scores=[]
    for c in range(num_classes):
        score=sum(w[c][j]*x_test[i][j] for j in range(column_count))
        scores.append(score)
    probs=softmax(scores)
    pred=probs.index(max(probs))
    actual=y_test[i]
    if pred==actual:
        correct+=1
        tp[actual]+=1
    else:
        fp[pred]+=1
        fn[actual]+=1
accuracy=(correct/len(x_test))*100
precisions,recalls,f1s=[],[],[]
for c in range(num_classes):
    p=tp[c]/(tp[c]+fp[c]) if (tp[c]+fp[c])>0 else 0
    r=tp[c]/(tp[c]+fn[c]) if (tp[c]+fn[c])>0 else 0
    f=2*(p*r)/(p+r) if (p+r)>0 else 0
    precisions.append(p)
    recalls.append(r)
    f1s.append(f)
macro_precision=sum(precisions)/num_classes
macro_recall=sum(recalls)/num_classes
macro_f1=sum(f1s)/num_classes
for c in range(num_classes):
    print(f"Class {c} Bias:{w[c][0]:.4f}")
    print(f"Class {c} Weights:{[round(wt,4) for wt in w[c][1:]]}")
print(f"Accuracy:{accuracy:.2f}%")
print(f"Macro Precision:{macro_precision:.4f}")
print(f"Macro Recall:{macro_recall:.4f}")
print(f"Macro F1-Score:{macro_f1:.4f}")