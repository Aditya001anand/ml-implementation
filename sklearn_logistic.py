import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df = pd.read_csv('IRIS.csv')
cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
x = df[cols].values
y = df['species'].astype('category').cat.codes.values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

model = LogisticRegression(solver='lbfgs', max_iter=1000)
model.fit(x_train_scaled, y_train)
preds = model.predict(x_test_scaled)

print(f"Accuracy: {accuracy_score(y_test, preds) * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, preds))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, preds))
