import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

# Step 1: Load the dataset
dataset = pd.read_csv("/Users/gayathriks/Desktop/DFU/DFU.csv")


# Step 2: Verify column names and correct if necessary
def verify_column_names(dataset):
    dataset.columns = dataset.columns.str.strip()
    print(dataset.columns)


verify_column_names(dataset)


# Step 3: Split the dataset into X and y
def split_dataset(dataset):
    X = dataset[["Thermistor_Value", "Foot_Pressure_Value"]]
    y = dataset["Condition"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test


X_train, X_test, y_train, y_test = split_dataset(dataset)

# Step 4: Data cleaning (if required)


# Step 5: Feature scaling (if required)
def scale_data(X_train, X_test):
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled


X_train_scaled, X_test_scaled = scale_data(X_train, X_test)


# Step 6: Model training
def train_model(X_train_scaled, y_train):
    model = SVC()
    model.fit(X_train_scaled, y_train)
    return model


model = train_model(X_train_scaled, y_train)


# Step 7: Model evaluation
def evaluate_model(model, X_test_scaled, y_test):
    accuracy = model.score(X_test_scaled, y_test)
    print(f"Accuracy: {accuracy}")


evaluate_model(model, X_test_scaled, y_test)

# Step 8: Predict on new data (deployment and application)
new_data = [[38.2, 100], [40.5, 90]]
scaler = MinMaxScaler()
scaler.fit(X_train)
new_data_scaled = scaler.transform(new_data)
predictions = model.predict(new_data_scaled)
print("Predictions:", predictions)
