# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import MinMaxScaler
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
# import numpy as np

# # Step 1: Load the dataset
# data = pd.read_csv("community_consumption.csv")

# # Step 2: Preprocess the data
# features = ["Winter", "Spring", "Summer", "Fall", "Outdoor Temp (°C)", "Humidity (%)", "Cloud Cover (%)",
#             "Occupancy", "Special Equipment [kW]", "Lighting [kW]", "HVAC [kW]", "Hour", "DayOfWeek", "Month", "IsWeekend", "IsHoliday"]
# target = "Use [kW]"

# X = data[features]
# y = data[target]

# # Normalize the features
# scaler = MinMaxScaler()
# X = scaler.fit_transform(X)

# # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Step 3: Define and train the model
# def create_model(input_dim):
#     model = Sequential([
#         Dense(64, activation="relu", input_shape=(input_dim,)),
#         Dense(32, activation="relu"),
#         Dense(1)  # Predict a single value (energy consumption)
#     ])
#     model.compile(optimizer="adam", loss="mse")
#     return model

# model = create_model(input_dim=X.shape[1])
# model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)

# # Step 4: Save the model and scaler
# model.export("energy_consumption_model")  # Save in SavedModel format
# import joblib
# joblib.dump(scaler, "scaler.pkl")

# print("Model and scaler saved successfully!")


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import joblib

# Step 1: Load the dataset
data = pd.read_csv("community_consumption.csv")

# Step 2: Feature Engineering
# Create lagged features
data['Use_lag1'] = data['Use [kW]'].shift(1)  # Lag by 1 hour
data['Use_lag24'] = data['Use [kW]'].shift(24)  # Lag by 24 hours

# Create rolling features
data['Use_rolling_mean_24'] = data['Use [kW]'].rolling(window=24).mean()
data['Use_rolling_std_24'] = data['Use [kW]'].rolling(window=24).std()

# Drop rows with NaN values (created by lagging and rolling)
data = data.dropna()

# Define features and target
features = ["Winter", "Spring", "Summer", "Fall", "Outdoor Temp (°C)", "Humidity (%)", "Cloud Cover (%)",
            "Occupancy", "Special Equipment [kW]", "Lighting [kW]", "HVAC [kW]", "Hour", "DayOfWeek", "Month", 
            "IsWeekend", "IsHoliday", "Use_lag1", "Use_lag24", "Use_rolling_mean_24", "Use_rolling_std_24"]
target = "Use [kW]"

X = data[features]
y = data[target]

# Normalize the features
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# Save the scaler for future use
joblib.dump(scaler, "scaler_lstm.pkl")

# Step 3: Create sequences for LSTM
def create_sequences(X, y, timesteps):
    X_seq, y_seq = [], []
    for i in range(len(X) - timesteps):
        X_seq.append(X[i:i+timesteps])
        y_seq.append(y[i+timesteps])
    return np.array(X_seq), np.array(y_seq)

timesteps = 24  # Use the past 24 hours to predict the next hour
X_seq, y_seq = create_sequences(X, y, timesteps)

# Step 4: Split the data into training and testing sets based on time
train_size = int(0.8 * len(X_seq))  # Train on 80% of the data
X_train, X_test = X_seq[:train_size], X_seq[train_size:]
y_train, y_test = y_seq[:train_size], y_seq[train_size:]

# Step 5: Define and train the LSTM model
def create_lstm_model(input_shape):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        LSTM(32),
        Dense(1)  # Predict a single value (energy consumption)
    ])
    model.compile(optimizer="adam", loss="mse")
    return model

model = create_lstm_model(input_shape=(X_train.shape[1], X_train.shape[2]))
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)

# Save the model with a valid extension
model.save("energy_consumption_lstm_model.keras")  # Use .keras for the native Keras format
# OR
# model.save("energy_consumption_lstm_model.h5")  # Use .h5 for the HDF5 format

print("Model and scaler saved successfully!")