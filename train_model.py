import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import pickle

# -----------------------------------------------------------
# Load dataset
# -----------------------------------------------------------
df = pd.read_csv("data.csv")  # Change to your dataset filename

# Basic cleaning
df = df.dropna()

# -----------------------------------------------------------
# Feature selection
# -----------------------------------------------------------
# Example: Customize these based on your CSV columns
feature_cols = ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3", "Temperature", "Humidity"]
target_col = "AQI"

X = df[feature_cols]
y = df[target_col]

# -----------------------------------------------------------
# Train-test split
# -----------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------------------------------------
# Train model
# -----------------------------------------------------------
model = RandomForestRegressor(
    n_estimators=300,
    max_depth=12,
    random_state=42
)

model.fit(X_train, y_train)

# -----------------------------------------------------------
# Evaluate
# -----------------------------------------------------------
train_pred = model.predict(X_train)
test_pred = model.predict(X_test)

print("Train R2:", r2_score(y_train, train_pred))
print("Test R2:", r2_score(y_test, test_pred))
print("MAE:", mean_absolute_error(y_test, test_pred))

# -----------------------------------------------------------
# Save model
# -----------------------------------------------------------
model_dir = "model"
os.makedirs(model_dir, exist_ok=True)

model_path = os.path.join(model_dir, "air_model.pkl")

with open(model_path, "wb") as f:
    pickle.dump(model, f)

print("Model saved at:", model_path)
