# # model.py
# import tensorflow as tf
# from tensorflow.keras.models import Sequential, Model
# from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, Input, MultiHeadAttention, LayerNormalization, Bidirectional, Conv1D, MaxPooling1D, Flatten, GlobalAveragePooling1D, BatchNormalization, Attention
# from sklearn.preprocessing import MinMaxScaler
# import numpy as np
# import pandas as pd
# from config import SEQUENCE_LENGTH, MODEL_TYPE, LEARNING_RATE
# from sklearn.metrics import mean_squared_error, mean_absolute_error
# from tensorflow.keras.losses import MeanSquaredError
# from tensorflow.keras.optimizers import Adam, SGD
# from tensorflow.keras.layers import Input, Dropout
# from tensorflow.keras.losses import Huber

# def create_model(input_shape):
#     """
#     Create and compile a model based on the specified MODEL_TYPE.
#     """
#     print("Input shape for global model:", input_shape)
#     if MODEL_TYPE == "LSTM":
#         # LSTM-based model using Bidirectional LSTM layers and Dropout for regularization.
#         model = Sequential([
#             Input(shape=input_shape),
#             Bidirectional(LSTM(100, return_sequences=True)),
#             Dropout(0.3),
#             Bidirectional(LSTM(100, return_sequences=False)),
#             Dropout(0.3),
#             Dense(50, activation='relu'),
#             Dropout(0.3),
#             Dense(1)
#         ])
#         optimizer = Adam(learning_rate=LEARNING_RATE)
#         model.compile(optimizer=optimizer, loss=MeanSquaredError(), metrics=['mae'])
#         return model

#     elif MODEL_TYPE == "GRU":
#         # GRU-based model: Define optimizer locally.
#         optimizer = Adam(learning_rate=LEARNING_RATE)
#         model = Sequential([
#             Input(shape=input_shape),
#             GRU(50, return_sequences=True),
#             Dropout(0.2),
#             GRU(50, return_sequences=False),
#             Dropout(0.2),
#             Dense(25, activation='relu'),
#             Dense(1)
#         ])
#         model.compile(optimizer=optimizer, loss=MeanSquaredError(), metrics=['mae'])
#         return model

#     elif MODEL_TYPE == "Transformer":
#         # A simple Transformer-based model.
#         inputs = Input(shape=input_shape)
#         x = MultiHeadAttention(num_heads=4, key_dim=64)(inputs, inputs)
#         x = LayerNormalization(epsilon=1e-6)(x + inputs)
#         x = Dense(64, activation='relu')(x)
#         x = Dropout(0.2)(x)
#         outputs = Dense(1)(x)
#         model = Model(inputs, outputs)
#         optimizer = Adam(learning_rate=LEARNING_RATE)
#         model.compile(optimizer=optimizer, loss=MeanSquaredError(), metrics=['mae'])
#         return model

#     else:
#         raise ValueError(f"Unsupported model type: {MODEL_TYPE}")

# def analyze_training_dataset(filepath):
#     """
#     Analyze and print statistics of the training dataset.
#     """
#     df = pd.read_csv(filepath)
    
#     print("\nTraining Dataset Analysis:")
#     print("-" * 50)
    
#     print("\nFeature Statistics:")
#     print(df.describe())
    
#     print("\nTarget Variable (Use [kW]) Statistics:")
#     print(df['Use [kW]'].describe())
    
#     print("\nValue Ranges:")
#     for column in df.columns:
#         print(f"{column}: [{df[column].min()}, {df[column].max()}]")
    
#     return df.describe()

# def load_and_preprocess_data(filepath, validation=False):
#     """
#     Load data from CSV, drop the 'Time' column, normalize features and target,
#     and create sequences for training.
    
#     If validation=True, return the last 20% of the sequences as a validation split.
#     """
#     df = pd.read_csv(filepath)
#     df = df.drop(columns=["Time"])  # Remove 'Time' column

#     # Separate features and target
#     features = df.drop(columns=["Use [kW]"])
#     print("Feature columns after dropping target:", features.columns.tolist())
#     target = df["Use [kW]"].values.reshape(-1, 1)

#     # Normalize using MinMaxScaler for both features and target
#     feature_scaler = MinMaxScaler(feature_range=(0, 1))
#     target_scaler = MinMaxScaler(feature_range=(0, 1))
#     scaled_features = feature_scaler.fit_transform(features)
#     scaled_target = target_scaler.fit_transform(target)
    
#     print("Scaled features shape:", scaled_features.shape)

#     # Create sequences using the defined SEQUENCE_LENGTH
#     X, y = [], []
#     for i in range(SEQUENCE_LENGTH, len(scaled_features)):
#         X.append(scaled_features[i-SEQUENCE_LENGTH:i, :])
#         y.append(scaled_target[i, 0])
    
#     X, y = np.array(X), np.array(y)
    
#     # If validation=True, split off the last 20% of the sequences
#     if validation:
#         split_index = int(0.8 * len(X))
#         X_val = X[split_index:]
#         y_val = y[split_index:]
#         return X_val, y_val, feature_scaler, target_scaler
#     else:
#         return X, y, feature_scaler, target_scaler

# def train_model(model, X_train, y_train, epochs=10, batch_size=32):
#     """
#     Train the model and return the trained model along with the training history.
#     """
#     history = model.fit(
#         X_train, y_train,
#         epochs=epochs,
#         batch_size=batch_size,
#         verbose=1,
#         validation_split=0.2  # Use 20% of data for validation during training
#     )
#     return model, history

# def calculate_metrics(y_true, y_pred, target_scaler=None):
#     """
#     Calculate evaluation metrics (MSE, RMSE, MAE, MA Percentage Error, Std Dev)
#     on the original scale (after inverse transforming).
#     """
#     min_length = min(len(y_true), len(y_pred))
#     y_true = y_true[:min_length]
#     y_pred = y_pred[:min_length]

#     if target_scaler is not None:
#         y_true_reshaped = y_true.reshape(-1, 1)
#         y_pred_reshaped = y_pred.reshape(-1, 1)
#         y_true_original = target_scaler.inverse_transform(y_true_reshaped).flatten()
#         y_pred_original = target_scaler.inverse_transform(y_pred_reshaped).flatten()
#     else:
#         y_true_original = y_true
#         y_pred_original = y_pred

#     mse = mean_squared_error(y_true_original, y_pred_original)
#     rmse = np.sqrt(mse)
#     mae = mean_absolute_error(y_true_original, y_pred_original)

#     with np.errstate(divide='ignore', invalid='ignore'):
#         ma_percentage_error = np.mean(np.abs((y_true_original - y_pred_original) /
#                                               np.where(y_true_original == 0, 1, y_true_original))) * 100
#         ma_percentage_error = np.nan_to_num(ma_percentage_error, nan=0.0, posinf=0.0, neginf=0.0)

#     std_dev = np.std(y_true_original - y_pred_original)
        
#     return {
#         "mse": mse,
#         "rmse": rmse,
#         "mae": mae,
#         "ma_percentage_error": ma_percentage_error,
#         "std_dev": std_dev
#     }


# model/model.py
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
    """
    Create and compile (or load) a model based on MODEL_TYPE.
    Hybrid uses two inputs: sequence and exogenous features.
    """
    print("Input shape for global model:", input_shape)

    # If an existing model file is provided and exists, load it
    if model_path and os.path.exists(model_path):
        print(f"🔁 Loading existing model from {model_path}")
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
    """
    Load a full CSV, scale features & target, build sequence windows.
    Returns X (or [X_seq, X_exo]), y, feature_scaler, target_scaler.
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
    if MODEL_TYPE == "XGBoost":
        n, sl, f = X_train.shape
        flat = X_train.reshape(n, sl*f)
        model.fit(flat, y_train)
        return model, None
    hist = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        verbose=1,
    )
    return model, hist


def train_model(model, X_train, y_train, epochs=10, batch_size=32):
    if MODEL_TYPE == "XGBoost":
        n, sl, f = X_train.shape
        flat = X_train.reshape(n, sl*f)
        model.fit(flat, y_train)
        return model, None
    else:
            # If we have more than 1 sample, do Keras validation split,
            # otherwise train directly on that single slice.
            if len(y_train) > 1:
                history = model.fit(
                    X_train,
                    y_train,
                    epochs=epochs,
                    batch_size=batch_size,
                    verbose=1,
                    validation_split=0.2,
                )
            else:
                history = model.fit(
                    X_train,
                    y_train,
                    epochs=epochs,
                    batch_size=batch_size,
                    verbose=1,
                )
            return model, history



def calculate_metrics(y_true, y_pred, target_scaler=None):
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
    """
    Build X_seq, X_exo, y, scalers for a **single** DataFrame row.
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