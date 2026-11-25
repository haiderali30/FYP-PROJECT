"""Flask API for the energy dashboard.

Endpoints:
  POST /predict      forecast consumption N hours ahead with the latest
                     federated global model
  GET  /consumption  historical consumption for a building and date range
  GET  /metrics      summary statistics (total, peak, peak hour, average)
"""

from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
from datetime import datetime, timedelta
from flask_cors import CORS
import logging
import os
logging.basicConfig(level=logging.DEBUG)
app = Flask(__name__)

# Configure CORS to allow all necessary headers and methods
CORS(app, resources={
    r"/*": {
        "origins": ["http://localhost:3000", "http://localhost:5173", "http://127.0.0.1:5173"],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization", "Accept"],
        "supports_credentials": True,
        "expose_headers": ["Content-Range", "X-Content-Range"]
    }
})

# Add CORS headers to all responses
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

SEQUENCE_LENGTH = 72  # must match trainer/config.py

# Find the directory this file lives in
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# The federated server writes the latest global model and scalers here
PREDICTION_DIR = os.path.join(BASE_DIR, "prediction_model_files_docker")

MODEL_PATH          = os.path.join(PREDICTION_DIR, "latest_model.h5")
FEATURE_SCALER_PATH = os.path.join(PREDICTION_DIR, "feature_scaler.save")
TARGET_SCALER_PATH  = os.path.join(PREDICTION_DIR, "target_scaler.save")
HISTORICAL_CSV_PATH = os.path.join(PREDICTION_DIR, "community_data.csv")
DATASETS_DIR        = os.path.join(BASE_DIR, "datasets")


def get_historical_sequence(csv_path, sequence_length, target_scaler):
    """
    Load historical consumption data from a CSV file, extract the "Use [kW]" column,
    take the last 'sequence_length' values, scale them using target_scaler,
    and reshape to (1, sequence_length, 1).
    """
    df = pd.read_csv(csv_path)
    if "Use [kW]" not in df.columns:
        raise ValueError("CSV file must contain 'Use [kW]' column.")
    consumption = df["Use [kW]"].values  # shape: (n,)
    if len(consumption) < sequence_length:
        raise ValueError(f"Not enough historical data. Required: {sequence_length}, Found: {len(consumption)}")
    seq = consumption[-sequence_length:]
    seq = seq.reshape(-1, 1)
    seq_scaled = target_scaler.transform(seq)
    return np.expand_dims(seq_scaled, axis=0)  # shape: (1, sequence_length, 1)

def iterative_forecast_backend(model, initial_sequence, exo_input, forecast_horizon):
    """
    Iteratively forecast consumption for 'forecast_horizon' steps ahead.
    At each step, predict the next consumption value using the current historical sequence and exogenous input,
    update the sequence by dropping the oldest value and appending the new prediction,
    and finally return the prediction of the final step.
    """
    current_sequence = initial_sequence.copy()  # shape: (1, SEQUENCE_LENGTH, 1)
    final_prediction = None
    for _ in range(forecast_horizon):
        # Hybrid model expects two inputs: [historical sequence, exogenous features]
        pred_scaled = model.predict([current_sequence, exo_input])
        final_prediction = pred_scaled  # save current prediction
        # Update sequence: remove oldest entry and append new prediction
        # pred_scaled is shape (1, 1); reshape it to (1, 1, 1) for concatenation
        pred_reshaped = np.expand_dims(pred_scaled, axis=1)  # shape: (1, 1, 1)
        current_sequence = np.concatenate([current_sequence[:, 1:, :], pred_reshaped], axis=1)
    return final_prediction

def get_exogenous_input_from_request(user_inputs):
    """
    Expecting user_inputs as a dict with keys (in the same order as during training):
      "Winter", "Spring", "Summer", "Fall",
      "Outdoor Temp (°C)", "Humidity (%)", "Cloud Cover (%)",
      "Occupancy", "Special Equipment [kW]", "Lighting [kW]", "HVAC [kW]"
    """
    feature_order = ["Winter", "Spring", "Summer", "Fall", 
                     "Outdoor Temp (°C)", "Humidity (%)", "Cloud Cover (%)",
                     "Occupancy", "Special Equipment [kW]", "Lighting [kW]", "HVAC [kW]"]
    try:
        values = [user_inputs[key] for key in feature_order]
    except KeyError as e:
        raise ValueError(f"Missing exogenous feature: {e}")
    return np.array(values).reshape(1, -1)  # shape: (1, 11)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # reload newest model+scalers every request
        model = load_model(MODEL_PATH)
        feature_scaler = joblib.load(FEATURE_SCALER_PATH)
        target_scaler = joblib.load(TARGET_SCALER_PATH)
        data = request.json
        forecast_horizon = int(data["hours_ahead"])  # e.g., 5 hours ahead
        user_inputs = data["user_inputs"]  # exogenous features provided by the operator
        
        # Prepare exogenous input vector (shape: (1, 11)) and scale it
        exo_input = get_exogenous_input_from_request(user_inputs)
        exo_input_scaled = feature_scaler.transform(exo_input)
        
        # Retrieve the historical consumption sequence from CSV (shape: (1, SEQUENCE_LENGTH, 1))
        initial_sequence = get_historical_sequence(HISTORICAL_CSV_PATH, SEQUENCE_LENGTH, target_scaler)
        
        # Use iterative forecasting to get the prediction for the final forecast step.
        final_prediction_scaled = iterative_forecast_backend(model, initial_sequence, exo_input_scaled, forecast_horizon)
        
        # Inverse-transform the prediction to get the actual consumption value.
        final_prediction = target_scaler.inverse_transform(final_prediction_scaled)
        
        return jsonify({"predicted_consumption": float(final_prediction[0][0])})
    except Exception as e:
        app.logger.error(f"Error in /predict endpoint: {e}")
        return jsonify({"error": str(e)}), 400

@app.route("/consumption", methods=["GET"])
def get_consumption_data():
    try:
        building = request.args.get('building')
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        view_type = request.args.get('view_type', 'hourly')  # Default to hourly if not specified

        app.logger.info(f"Received consumption data request for building: {building}, date range: {start_date} to {end_date}, view type: {view_type}")

        if not all([building, start_date, end_date]):
            return jsonify({"error": "Missing required parameters"}), 400

        # Normalize building name (remove spaces and handle special cases)
        building_name_map = {
            "House 1": "House1",
            "House 2": "House2",
        }
        normalized_building = building_name_map.get(building, building.replace(" ", ""))
        
        # Read the building-specific dataset
        file_path = os.path.join(DATASETS_DIR, f"{normalized_building}_data.csv")
        app.logger.info(f"Reading data from: {file_path}")
        
        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            app.logger.error(f"Error reading CSV file: {str(e)}")
            return jsonify({"error": f"Error reading data for building: {building}"}), 500

        # Identify the date/time column
        time_columns = ['Time', 'timestamp', 'date', 'DateTime']
        date_column = next((col for col in time_columns if col in df.columns), None)
        
        if not date_column:
            app.logger.error(f"No valid date column found. Available columns: {df.columns.tolist()}")
            return jsonify({"error": "No valid date/time column found in the dataset"}), 500

        # Identify the energy consumption column
        energy_columns = ['Use [kW]', 'Energy [kW]', 'Consumption [kW]', 'Power [kW]']
        energy_column = next((col for col in energy_columns if col in df.columns), None)
        
        if not energy_column:
            app.logger.error(f"No valid energy column found. Available columns: {df.columns.tolist()}")
            return jsonify({"error": "No valid energy consumption column found in the dataset"}), 500

        # Convert date column and filter by date range
        df[date_column] = pd.to_datetime(df[date_column])
        mask = (df[date_column].dt.date >= pd.to_datetime(start_date).date()) & \
               (df[date_column].dt.date <= pd.to_datetime(end_date).date())
        df_filtered = df[mask].copy()

        if df_filtered.empty:
            app.logger.warning(f"No data found for date range: {start_date} to {end_date}")
            return jsonify({"error": "No data available for the selected date range"}), 404

        # Aggregate data based on view type
        if view_type == 'daily':
            # For daily view, group by date and calculate mean consumption
            df_filtered['date'] = df_filtered[date_column].dt.date
            df_agg = df_filtered.groupby('date', as_index=False).agg({
                energy_column: 'mean'
            })
            df_agg[date_column] = pd.to_datetime(df_agg['date'])
        else:  # hourly view
            # For hourly view, use the original timestamps
            df_agg = df_filtered

        # Format data for the chart
        consumption_data = df_agg.apply(
            lambda row: {
                "timestamp": row[date_column].isoformat(),
                "consumption": float(row[energy_column])
            },
            axis=1
        ).tolist()

        return jsonify(consumption_data)

    except Exception as e:
        app.logger.error(f"Error in /consumption endpoint: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/")
def home():
    return "Energy Consumption Prediction API is running!"

@app.route("/metrics", methods=["GET"])
def get_metrics():
    try:
        building = request.args.get('building')
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')

        app.logger.info(f"Received request for building: {building}, date range: {start_date} to {end_date}")

        if not all([building, start_date, end_date]):
            return jsonify({"error": "Missing required parameters"}), 400

        # Normalize building name (remove spaces and handle special cases)
        building_name_map = {
            "House 1": "House1",
            "House 2": "House2",
        }
        normalized_building = building_name_map.get(building, building.replace(" ", ""))
        
        # Read the building-specific dataset
        file_path = os.path.join(DATASETS_DIR, f"{normalized_building}_data.csv")
        app.logger.info(f"Reading data from: {file_path}")
        
        try:
            df = pd.read_csv(file_path)
            app.logger.info(f"Available columns: {df.columns.tolist()}")
        except Exception as e:
            app.logger.error(f"Error reading CSV file: {str(e)}")
            return jsonify({"error": f"Error reading data for building: {building}"}), 500

        # Identify the date/time column (it might be 'timestamp', 'date', or 'Time')
        time_columns = ['Time', 'timestamp', 'date', 'DateTime']  # Prioritize 'Time' as it's in our data
        date_column = next((col for col in time_columns if col in df.columns), None)
        
        if not date_column:
            app.logger.error(f"No valid date column found. Available columns: {df.columns.tolist()}")
            return jsonify({"error": "No valid date/time column found in the dataset"}), 500
            
        app.logger.info(f"Using date column: {date_column}")

        # Convert date column and filter by date range
        df[date_column] = pd.to_datetime(df[date_column])
        mask = (df[date_column].dt.date >= pd.to_datetime(start_date).date()) & \
               (df[date_column].dt.date <= pd.to_datetime(end_date).date())
        df_filtered = df[mask].copy()  # Use copy to avoid SettingWithCopyWarning

        if df_filtered.empty:
            app.logger.warning(f"No data found for date range: {start_date} to {end_date}")
            return jsonify({"error": "No data available for the selected date range"}), 404

        # Identify the energy consumption column
        energy_columns = ['Use [kW]', 'Energy [kW]', 'Consumption [kW]', 'Power [kW]']
        energy_column = next((col for col in energy_columns if col in df.columns), None)
        
        if not energy_column:
            app.logger.error(f"No valid energy column found. Available columns: {df.columns.tolist()}")
            return jsonify({"error": "No valid energy consumption column found in the dataset"}), 500

        app.logger.info(f"Using energy column: {energy_column}")

        # Calculate metrics
        total_consumption = df_filtered[energy_column].sum()
        peak_demand = df_filtered[energy_column].max()
        
        # Find peak hours (hours with highest average consumption)
        df_filtered.loc[:, 'hour'] = df_filtered[date_column].dt.hour  # Use loc to avoid warning
        hourly_avg = df_filtered.groupby('hour')[energy_column].mean()
        peak_hour = hourly_avg.idxmax()
        peak_hour_str = f"{peak_hour:02d}:00 - {(peak_hour + 1):02d}:00"

        # Calculate average consumption
        avg_consumption = df_filtered[energy_column].mean()

        # Prepare response
        response = {
            "total_consumption": float(total_consumption),
            "peak_demand": float(peak_demand),
            "peak_hour": peak_hour_str,
            "average_consumption": float(avg_consumption)
        }

        app.logger.info(f"Successfully calculated metrics: {response}")
        return jsonify(response)

    except FileNotFoundError:
        app.logger.error(f"File not found for building: {normalized_building}")
        return jsonify({"error": f"Data not found for building: {building}"}), 404
    except Exception as e:
        app.logger.error(f"Error processing metrics: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5001)
