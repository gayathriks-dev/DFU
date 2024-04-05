import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Step 1: Load the dataset
dataset = pd.read_csv("/Users/gayathriks/Desktop/DFU/DFU.csv")

# Step 2: Verify column names and correct if necessary
dataset.columns = dataset.columns.str.strip()
print(dataset.columns)

# Step 3: Split the dataset into X and y
X = dataset[["Thermistor_Value", "Foot_Pressure_Value"]]
y = dataset["Condition"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 4: Data cleaning (if required)

# Step 5: Feature scaling (if required)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 6: Model training
model = LogisticRegression()
model.fit(X_train_scaled, y_train)
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
import seaborn as sns

# Step 7: Model evaluation
y_pred = model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, pos_label="Normal")
recall = recall_score(y_test, y_pred, pos_label="Normal")
f1 = f1_score(y_test, y_pred, pos_label="Normal")

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-Score:", f1)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred, labels=["Normal", "Abnormal"])
cm_df = pd.DataFrame(
    cm,
    index=["Actual Normal", "Actual Abnormal"],
    columns=["Predicted Normal", "Predicted Abnormal"],
)

sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()
