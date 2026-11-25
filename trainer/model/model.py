"""Model definitions, preprocessing, training and evaluation utilities.

Shared by the federated learning server and clients.
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import load_model, Sequential, Model
from tensorflow.keras.layers import (
    LSTM, Dense, Dropout, Input,
    MultiHeadAttention, LayerNormalization,
    Bidirectional, Concatenate
)
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import Adam
from config import SEQUENCE_LENGTH, MODEL_TYPE, LEARNING_RATE


def create_model(input_shape, model_path=None):
    """Create and compile (or load) a model based on MODEL_TYPE.

    The default "Hybrid" model has two branches: an LSTM over the last
    SEQUENCE_LENGTH consumption values and a dense network over the
    exogenous features (season, weather, occupancy, HVAC/lighting load).
    """

    # If an existing model file is provided and exists, load it
    if model_path and os.path.exists(model_path):
        return load_model(model_path)

    # Otherwise, build from scratch
    if MODEL_TYPE == "LSTM":
        model = Sequential([
            Input(shape=input_shape),
            Bidirectional(LSTM(100, return_sequences=True)),
            Dropout(0.3),
            Bidirectional(LSTM(100)),
            Dropout(0.3),
            Dense(50, activation='relu'),
            Dropout(0.3),
            Dense(1)
        ])
    elif MODEL_TYPE == "XGBoost":
        from xgboost import XGBRegressor
        model = XGBRegressor(
            objective='reg:squarederror',
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        return model
    elif MODEL_TYPE == "Transformer":
        inputs = Input(shape=input_shape)
        x = MultiHeadAttention(num_heads=4, key_dim=64)(inputs, inputs)
        x = LayerNormalization(epsilon=1e-6)(x + inputs)
        x = Dense(64, activation='relu')(x)
        x = Dropout(0.2)(x)
        outputs = Dense(1)(x)
        model = Model(inputs, outputs)
    elif MODEL_TYPE == "Hybrid":
        # Branch for sequence
        lstm_in = Input(shape=(SEQUENCE_LENGTH, 1), name="lstm_input")
        x1 = LSTM(64, name="lstm_layer")(lstm_in)
        x1 = Dropout(0.2, name="lstm_dropout")(x1)
        # Branch for exogenous
        exo_in = Input(shape=(11,), name="exo_input")
        x2 = Dense(32, activation='relu', name="exo_dense1")(exo_in)
        x2 = Dropout(0.2, name="exo_dropout")(x2)
        x2 = Dense(16, activation='relu', name="exo_dense2")(x2)
        # Merge
        merged = Concatenate(name="concatenate")([x1, x2])
        x = Dense(32, activation='relu', name="merged_dense")(merged)
        x = Dropout(0.2, name="merged_dropout")(x)
        output = Dense(1, name="output")(x)
        model = Model(inputs=[lstm_in, exo_in], outputs=output)
    elif MODEL_TYPE == "MLP":
        model = Sequential([
            Input(shape=(11,)),
            Dense(64, activation='relu'),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(16, activation='relu'),
            Dense(1)
        ])
    else:
        raise ValueError(f"Unsupported model type: {MODEL_TYPE}")

    # Compile if Keras model
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss=MeanSquaredError(),
        metrics=['mae'],
    )
    return model


def load_and_preprocess_data(filepath, validation=False):
    """Load a full CSV, scale features and target, and build sequence windows.

    Returns ([X_seq, X_exo], y, feature_scaler, target_scaler). With
    validation=True only the last 20% of the windows is returned.
    """
    df = pd.read_csv(filepath)
    df = df.drop(columns=[c for c in ("Time",) if c in df.columns])

    exo = df.drop(columns=["Use [kW]"])
    target = df["Use [kW]"].values.reshape(-1, 1)

    feature_scaler = MinMaxScaler().fit(exo)
    target_scaler  = MinMaxScaler().fit(target)

    scaled_exo   = feature_scaler.transform(exo)
    scaled_target= target_scaler.transform(target)

    X_seq, X_exo, y = [], [], []
    for i in range(SEQUENCE_LENGTH, len(scaled_target)):
        X_seq.append(scaled_target[i-SEQUENCE_LENGTH:i])
        X_exo.append(scaled_exo[i])
        y.append(scaled_target[i, 0])
    X_seq = np.array(X_seq)
    X_exo = np.array(X_exo)
    y     = np.array(y)

    if validation:
        idx = int(0.8 * len(y))
        return [X_seq[idx:], X_exo[idx:]], y[idx:], feature_scaler, target_scaler
    return [X_seq, X_exo], y, feature_scaler, target_scaler


def train_model(model, X_train, y_train, epochs=10, batch_size=32):
    """Train the model locally and return (model, history).

    A 20% validation split is used when there is more than one sample;
    a single streaming slice is trained on directly.
    """
    if MODEL_TYPE == "XGBoost":
        n, sl, f = X_train.shape
        model.fit(X_train.reshape(n, sl * f), y_train)
        return model, None

    fit_kwargs = dict(epochs=epochs, batch_size=batch_size, verbose=1)
    if len(y_train) > 1:
        fit_kwargs["validation_split"] = 0.2
    history = model.fit(X_train, y_train, **fit_kwargs)
    return model, history


def calculate_metrics(y_true, y_pred, target_scaler=None):
    """Compute MSE, RMSE, MAE, MAPE and error std-dev in original units (kW)."""
    if target_scaler:
        y_t = target_scaler.inverse_transform(y_true.reshape(-1,1)).flatten()
        y_p = target_scaler.inverse_transform(y_pred.reshape(-1,1)).flatten()
    else:
        y_t, y_p = y_true, y_pred
    mse   = mean_squared_error(y_t, y_p)
    rmse  = np.sqrt(mse)
    mae   = mean_absolute_error(y_t, y_p)
    mape  = np.mean(np.abs((y_t - y_p) / np.where(y_t==0,1,y_t))) * 100
    std   = np.std(y_t - y_p)
    return {"mse": mse, "rmse": rmse, "mae": mae, "ma_percentage_error": mape, "std_dev": std}


def preprocess_one_row(row_df: pd.DataFrame, sequence_length: int, model_type: str):
    """Build model inputs for a single streaming reading (one DataFrame row).

    Returns ([X_seq, X_exo], y, feature_scaler, target_scaler). The history
    window X_seq is filled with the current reading, because clients do not
    keep a consumption history in the streaming simulation.
    """
    df = row_df.copy()
    if "Time" in df.columns:
        df = df.drop(columns=["Time"])
    exo = df.drop(columns=["Use [kW]"]).values.reshape(1, -1)
    y   = df["Use [kW]"].values  # shape (1,)
    fs  = MinMaxScaler().fit(exo)
    ts  = MinMaxScaler().fit(y.reshape(-1,1))
    last= y[-1]
    seq = np.full((sequence_length,1), last)
    X_seq = seq.reshape(1, sequence_length,1)
    return [X_seq, exo], y, fs, ts
