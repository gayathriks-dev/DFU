# Step 1: Import the required libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
import pickle

# Step 2: Load and preprocess data (similar to previous code)
data = pd.read_csv("/Users/gayathriks/Desktop/DFU/DFU.csv")
data.columns = data.columns.str.strip()

# Step 3: Split data into training and testing sets (similar to previous code)
X = data[["Thermistor_Value", "Foot_Pressure_Value"]].copy()
y = data["Condition"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 4: Data cleaning (if required)

# Step 5: Feature scaling (if required)
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 6: Load the serialized logistic regression model
with open("logistic_model.pkl", "rb") as file:
    loaded_logistic_model = pickle.load(file)

# Step 7: Use the loaded logistic regression model for predictions
logistic_predictions = loaded_logistic_model.predict(X_test)

# Step 8: Load the serialized SVM model
with open("svm_model.pkl", "rb") as file:
    loaded_svm_model = pickle.load(file)


# Step 9: Use the loaded SVM model for predictions
svm_predictions = loaded_svm_model.predict(X_test_scaled)

# Step 10: Load the serialized scaler
with open("scaler.pkl", "rb") as file:
    loaded_scaler = pickle.load(file)

# Step 11: Scale new data using the loaded scaler
new_data = [[38.2, 1.5]]  # Include values for all features used during training
new_data_scaled = loaded_scaler.transform(new_data)

# Step 12: Use the loaded SVM model for predictions on new data
svm_new_predictions = loaded_svm_model.predict(new_data_scaled)

print("Logistic Regression Predictions:", logistic_predictions)
print("SVM Predictions:", svm_predictions)
print("SVM Predictions on new data:", svm_new_predictions)
