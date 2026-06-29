import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv('data.csv')
cols = ['LongestShell', 'Diameter', 'Height', 'WholeWeight', 'ShuckedWeight', 'VisceraWeight', 'ShellWeight']
x = df[cols].values

def categorize_rings(r):
    if r <= 8: 
        return 0
    elif r <= 10: 
        return 1
    else: 
        return 2

y = df['Rings'].apply(categorize_rings).values
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)
model = LogisticRegression(solver='lbfgs', max_iter=1000)
model.fit(x_train_scaled, y_train)
preds = model.predict(x_test_scaled)
print(f"Library Softmax Accuracy: {accuracy_score(y_test, preds) * 100:.2f}%\n")
print("Detailed Classification Report (Precision, Recall, F1):")
print(classification_report(y_test, preds, target_names=["Class 0 (<=8)", "Class 1 (9-10)", "Class 2 (>10)"]))