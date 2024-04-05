import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
import pickle

# Step 1: Load and preprocess data
data = pd.read_csv("/Users/gayathriks/Desktop/DFU/DFU.csv")

# Verify column names and correct if necessary
data.columns = data.columns.str.strip()
print(data.columns)

# Step 2: Split data into training and testing sets
X = data[["Thermistor_Value", "Foot_Pressure_Value"]].copy()
y = data["Condition"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 3: Train the logistic regression model
logistic_model = LogisticRegression()
logistic_model.fit(X_train, y_train)

# Step 4: Serialize the logistic regression model using pickle
with open("logistic_model.pkl", "wb") as file:
    pickle.dump(logistic_model, file)

# Step 5: Load the serialized logistic regression model
with open("logistic_model.pkl", "rb") as file:
    loaded_logistic_model = pickle.load(file)

# Step 6: Use the loaded logistic regression model for predictions
logistic_predictions = loaded_logistic_model.predict(X_test)

# Step 7: Split the dataset into X and y
X = data[["Thermistor_Value", "Foot_Pressure_Value"]].copy()
y = data["Condition"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 8: Data cleaning (if required)

# Step 9: Feature scaling (if required)
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 10: Train the SVM model
svm_model = SVC()
svm_model.fit(X_train_scaled, y_train)

# Step 11: Serialize the SVM model using pickle
with open("svm_model.pkl", "wb") as file:
    pickle.dump(svm_model, file)

# Step 12: Load the serialized SVM model
with open("svm_model.pkl", "rb") as file:
    loaded_svm_model = pickle.load(file)

# Step 13: Model evaluation
svm_accuracy = loaded_svm_model.score(X_test_scaled, y_test)
print(f"Accuracy of SVM model: {svm_accuracy}")

# Step 14: Serialize the scaler using pickle
with open("scaler.pkl", "wb") as file:
    pickle.dump(scaler, file)

# Step 15: Load the serialized scaler
with open("scaler.pkl", "rb") as file:
    loaded_scaler = pickle.load(file)

# Step 16: Predict on new data (deployment and application)
new_data = [[38.2, 1.5]]  # Include values for all features used during training
new_data_scaled = loaded_scaler.transform(new_data)
svm_predictions = loaded_svm_model.predict(new_data_scaled)
print("SVM Predictions:", svm_predictions)

# Step 17: Use the logistic regression model for predictions
logistic_X = data[["Thermistor_Value", "Foot_Pressure_Value"]].copy()
logistic_predictions = loaded_logistic_model.predict(logistic_X)
print("Logistic Regression Predictions:", logistic_predictions)
