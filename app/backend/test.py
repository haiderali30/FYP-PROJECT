# import pandas as pd
# import numpy as np
# from tensorflow.keras.layers import TFSMLayer
# from tensorflow.keras.models import Sequential
# import joblib
# from datetime import datetime, timedelta

# # Step 1: Load the trained model using TFSMLayer
# model_path = "energy_consumption_model"
# model_layer = TFSMLayer(model_path, call_endpoint="serving_default")  # Load the SavedModel

# # Wrap the TFSMLayer in a Sequential model for easier use
# model = Sequential([model_layer])

# # Load the scaler
# scaler = joblib.load("scaler.pkl")

# # Step 2: Define a function to calculate future time features
# def get_future_time_features(hours_ahead):
#     current_time = datetime.now()
#     future_time = current_time + timedelta(hours=hours_ahead)
#     future_features = {
#         "Hour": future_time.hour,
#         "DayOfWeek": future_time.weekday(),
#         "Month": future_time.month,
#         "IsWeekend": int(future_time.weekday() in [5, 6]),
#         "IsHoliday": 0,
#     }
#     return future_features

# # Step 3: Collect user inputs for other features
# def get_user_inputs():
#     outdoor_temp = float(input("Enter the expected outdoor temperature (°C): "))
#     humidity = float(input("Enter the expected humidity (%): "))
#     cloud_cover = float(input("Enter the expected cloud cover (%): "))
#     occupancy = float(input("Enter the expected occupancy: "))
#     special_equipment = float(input("Enter the expected special equipment usage [kW]: "))
#     lighting = float(input("Enter the expected lighting usage [kW]: "))
#     hvac = float(input("Enter the expected HVAC usage [kW]: "))
#     season = input("Enter the season (Winter, Spring, Summer, Fall): ").lower()

#     season_features = {
#         "Winter": 1 if season == "winter" else 0,
#         "Spring": 1 if season == "spring" else 0,
#         "Summer": 1 if season == "summer" else 0,
#         "Fall": 1 if season == "fall" else 0,
#     }

#     return {
#         "Outdoor Temp (°C)": outdoor_temp,
#         "Humidity (%)": humidity,
#         "Cloud Cover (%)": cloud_cover,
#         "Occupancy": occupancy,
#         "Special Equipment [kW]": special_equipment,
#         "Lighting [kW]": lighting,
#         "HVAC [kW]": hvac,
#         **season_features,
#     }

# # Step 4: Prepare input data for prediction
# def prepare_input_data(hours_ahead, user_inputs):
#     future_time_features = get_future_time_features(hours_ahead)
#     input_data = {**future_time_features, **user_inputs}
#     input_df = pd.DataFrame([input_data])
#     features = ["Winter", "Spring", "Summer", "Fall", "Outdoor Temp (°C)", "Humidity (%)", "Cloud Cover (%)",
#                 "Occupancy", "Special Equipment [kW]", "Lighting [kW]", "HVAC [kW]", "Hour", "DayOfWeek", "Month", "IsWeekend", "IsHoliday"]
#     input_df = input_df[features]
#     input_df = scaler.transform(input_df)
#     return input_df

# # Step 5: Make the prediction
# def predict_consumption(model, input_df):
#     # Get the prediction (output is a dictionary)
#     prediction_dict = model.predict(input_df)
    
#     # Extract the prediction value from the dictionary
#     # The key is usually 'output_0' or similar; check the SavedModel's output names
#     prediction_key = list(prediction_dict.keys())[0]  # Get the first key
#     predicted_consumption = prediction_dict[prediction_key][0][0]  # Extract the value
#     return predicted_consumption

# # Step 6: Main workflow for testing
# if __name__ == "__main__":
#     hours_ahead = int(input("Enter the number of hours ahead to predict (e.g., 12): "))
#     user_inputs = get_user_inputs()
#     input_df = prepare_input_data(hours_ahead, user_inputs)
#     predicted_consumption = predict_consumption(model, input_df)
#     print(f"Predicted energy consumption {hours_ahead} hours from now: {predicted_consumption:.2f} kW")

import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import joblib
from datetime import datetime, timedelta

# Step 1: Load the trained LSTM model and scaler
model = load_model("energy_consumption_lstm_model.keras")
scaler = joblib.load("scaler_lstm.pkl")

# Step 2: Define a function to calculate future time features
def get_future_time_features(hours_ahead):
    current_time = datetime.now()
    future_time = current_time + timedelta(hours=hours_ahead)
    future_features = {
        "Hour": future_time.hour,
        "DayOfWeek": future_time.weekday(),
        "Month": future_time.month,
        "IsWeekend": int(future_time.weekday() in [5, 6]),
        "IsHoliday": 0,
    }
    return future_features

# Step 3: Collect user inputs for other features
def get_user_inputs():
    outdoor_temp = float(input("Enter the expected outdoor temperature (°C): "))
    humidity = float(input("Enter the expected humidity (%): "))
    cloud_cover = float(input("Enter the expected cloud cover (%): "))
    occupancy = float(input("Enter the expected occupancy: "))
    special_equipment = float(input("Enter the expected special equipment usage [kW]: "))
    lighting = float(input("Enter the expected lighting usage [kW]: "))
    hvac = float(input("Enter the expected HVAC usage [kW]: "))
    season = input("Enter the season (Winter, Spring, Summer, Fall): ").lower()

    season_features = {
        "Winter": 1 if season == "winter" else 0,
        "Spring": 1 if season == "spring" else 0,
        "Summer": 1 if season == "summer" else 0,
        "Fall": 1 if season == "fall" else 0,
    }

    return {
        "Outdoor Temp (°C)": outdoor_temp,
        "Humidity (%)": humidity,
        "Cloud Cover (%)": cloud_cover,
        "Occupancy": occupancy,
        "Special Equipment [kW]": special_equipment,
        "Lighting [kW]": lighting,
        "HVAC [kW]": hvac,
        **season_features,
    }

# Step 4: Prepare input data for prediction
def prepare_input_data(hours_ahead, user_inputs):
    future_time_features = get_future_time_features(hours_ahead)
    input_data = {**future_time_features, **user_inputs}

    # Create a DataFrame with the same feature names and order as during training
    features = ["Winter", "Spring", "Summer", "Fall", "Outdoor Temp (°C)", "Humidity (%)", "Cloud Cover (%)",
                "Occupancy", "Special Equipment [kW]", "Lighting [kW]", "HVAC [kW]", "Hour", "DayOfWeek", "Month", 
                "IsWeekend", "IsHoliday", "Use_lag1", "Use_lag24", "Use_rolling_mean_24", "Use_rolling_std_24"]
    input_df = pd.DataFrame([input_data], columns=features)

    # Add lagged and rolling features (use dummy values for testing)
    input_df['Use_lag1'] = 0  # Replace with actual lagged value if available
    input_df['Use_lag24'] = 0  # Replace with actual lagged value if available
    input_df['Use_rolling_mean_24'] = 0  # Replace with actual rolling mean if available
    input_df['Use_rolling_std_24'] = 0  # Replace with actual rolling std if available

    # Normalize the input data
    input_df = scaler.transform(input_df)

    # Reshape for LSTM (samples, timesteps, features)
    input_df = input_df.reshape(1, 24, input_df.shape[1] // 24)
    return input_df

# Step 5: Make the prediction
def predict_consumption(model, input_df):
    predicted_consumption = model.predict(input_df)[0][0]
    return predicted_consumption

# Step 6: Main workflow for testing
if __name__ == "__main__":
    hours_ahead = int(input("Enter the number of hours ahead to predict (e.g., 12): "))
    user_inputs = get_user_inputs()
    input_df = prepare_input_data(hours_ahead, user_inputs)
    predicted_consumption = predict_consumption(model, input_df)
    print(f"Predicted energy consumption {hours_ahead} hours from now: {predicted_consumption:.2f} kW")