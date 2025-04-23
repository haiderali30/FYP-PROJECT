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
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, MultiHeadAttention, LayerNormalization, Bidirectional, Concatenate, Flatten
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
from config import SEQUENCE_LENGTH, MODEL_TYPE, LEARNING_RATE
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import Adam

def create_model(input_shape):
    """
    Create and compile a model based on the specified MODEL_TYPE.
    For the Hybrid model, we assume two inputs:
      1. A historical sequence of consumption values with shape (SEQUENCE_LENGTH, 1)
      2. A snapshot of exogenous features with shape (11,)
    """
    print("Input shape for global model:", input_shape)
    
    if MODEL_TYPE == "LSTM":
        model = Sequential([
            Input(shape=input_shape),
            Bidirectional(LSTM(100, return_sequences=True)),
            Dropout(0.3),
            Bidirectional(LSTM(100, return_sequences=False)),
            Dropout(0.3),
            Dense(50, activation='relu'),
            Dropout(0.3),
            Dense(1)
        ])
        optimizer = Adam(learning_rate=LEARNING_RATE)
        model.compile(optimizer=optimizer, loss=MeanSquaredError(), metrics=['mae'])
        return model

    elif MODEL_TYPE == "XGBoost":
        # Create the XGBRegressor without using Keras's compile method.
        from xgboost import XGBRegressor
        model = XGBRegressor(
            objective='reg:squarederror',
            n_estimators=100,        # Number of trees (boosting rounds)
            max_depth=5,             # Tree depth (controls complexity)
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
        optimizer = Adam(learning_rate=LEARNING_RATE)
        model.compile(optimizer=optimizer, loss=MeanSquaredError(), metrics=['mae'])
        return model

    elif MODEL_TYPE == "Hybrid":
        # Hybrid model: two inputs
        # 1. Historical consumption (sequence) input: shape (SEQUENCE_LENGTH, 1)
        lstm_input = Input(shape=(SEQUENCE_LENGTH, 1), name="lstm_input")
        x_seq = LSTM(64, return_sequences=False, name="lstm_layer")(lstm_input)
        x_seq = Dropout(0.2, name="lstm_dropout")(x_seq)
        
        # 2. Exogenous features input: shape (11,)
        exo_input = Input(shape=(11,), name="exo_input")
        x_exo = Dense(32, activation='relu', name="exo_dense1")(exo_input)
        x_exo = Dropout(0.2, name="exo_dropout")(x_exo)
        x_exo = Dense(16, activation='relu', name="exo_dense2")(x_exo)
        
        # Merge both branches
        merged = Concatenate(name="concatenate")([x_seq, x_exo])
        x = Dense(32, activation='relu', name="merged_dense")(merged)
        x = Dropout(0.2, name="merged_dropout")(x)
        output = Dense(1, name="output")(x)
        
        model = Model(inputs=[lstm_input, exo_input], outputs=output)
        optimizer = Adam(learning_rate=LEARNING_RATE)
        model.compile(optimizer=optimizer, loss=MeanSquaredError(), metrics=['mae'])
        return model

    elif MODEL_TYPE == "MLP":
        model = Sequential([
            Dense(64, activation='relu', input_shape=(11,)),  # Expects a vector of 11 features
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(16, activation='relu'),
            Dense(1)  # Single prediction output per sample
        ])
        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
        return model

    else:
        raise ValueError(f"Unsupported model type: {MODEL_TYPE}")

def analyze_training_dataset(filepath):
    df = pd.read_csv(filepath)
    print("\nTraining Dataset Analysis:")
    print("-" * 50)
    print("\nFeature Statistics:")
    print(df.describe())
    print("\nTarget Variable (Use [kW]) Statistics:")
    print(df['Use [kW]'].describe())
    print("\nValue Ranges:")
    for column in df.columns:
        print(f"{column}: [{df[column].min()}, {df[column].max()}]")
    return df.describe()

def load_and_preprocess_data(filepath, validation=False):
    """
    This function must be adjusted for the hybrid model.
    For the hybrid model, you need to prepare two inputs:
      1. Historical consumption sequences: e.g., using a sliding window on the 'Use [kW]' column.
      2. Exogenous features: the remaining features (11 columns) for the current time-step.
    
    For illustration, below is one possible approach:
    """
    import pandas as pd
    from sklearn.preprocessing import MinMaxScaler
    import numpy as np

    df = pd.read_csv(filepath)
    df = df.drop(columns=["Time"])
    
    # Separate exogenous features and target consumption.
    exo_features = df.drop(columns=["Use [kW]"])
    print("Feature columns after dropping target:", exo_features.columns.tolist())
    target = df["Use [kW]"].values.reshape(-1, 1)
    
    # Scale exogenous features and target separately.
    feature_scaler = MinMaxScaler(feature_range=(0, 1))
    target_scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_exo = feature_scaler.fit_transform(exo_features)  # shape: (num_samples, 11)
    scaled_target = target_scaler.fit_transform(target)       # shape: (num_samples, 1)
    
    # Build sequences for historical consumption.
    # For each sample, take the previous SEQUENCE_LENGTH consumption values as the sequence input.
    X_seq, X_exo, y = [], [], []
    for i in range(SEQUENCE_LENGTH, len(scaled_target)):
        # Historical consumption sequence: shape (SEQUENCE_LENGTH, 1)
        seq = scaled_target[i-SEQUENCE_LENGTH:i, :]
        X_seq.append(seq)
        # Exogenous features for the current time-step: from scaled_exo
        X_exo.append(scaled_exo[i, :])
        # Target: consumption at time i
        y.append(scaled_target[i, 0])
    X_seq = np.array(X_seq)  # shape: (num_samples - SEQUENCE_LENGTH, SEQUENCE_LENGTH, 1)
    X_exo = np.array(X_exo)  # shape: (num_samples - SEQUENCE_LENGTH, 11)
    y = np.array(y)         # shape: (num_samples - SEQUENCE_LENGTH,)
    
    if validation:
        split_index = int(0.8 * len(X_seq))
        X_seq_val = X_seq[split_index:]
        X_exo_val = X_exo[split_index:]
        y_val = y[split_index:]
        return [X_seq_val, X_exo_val], y_val, feature_scaler, target_scaler
    else:
        return [X_seq, X_exo], y, feature_scaler, target_scaler

def train_model(model, X_train, y_train, epochs=10, batch_size=32):
    if MODEL_TYPE == "XGBoost":
        # For XGBoost, flatten the time-series data.
        n_samples, seq_len, n_features = X_train.shape
        X_train_flat = X_train.reshape(n_samples, seq_len * n_features)
        model.fit(X_train_flat, y_train)
        history = None
        return model, history
    else:
        # For LSTM, Transformer, Hybrid, and MLP, use Keras fit method.
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            verbose=1,
            validation_split=0.2
        )
        return model, history

def calculate_metrics(y_true, y_pred, target_scaler=None):
    min_length = min(len(y_true), len(y_pred))
    y_true = y_true[:min_length]
    y_pred = y_pred[:min_length]
    if target_scaler is not None:
        y_true_reshaped = y_true.reshape(-1, 1)
        y_pred_reshaped = y_pred.reshape(-1, 1)
        y_true_original = target_scaler.inverse_transform(y_true_reshaped).flatten()
        y_pred_original = target_scaler.inverse_transform(y_pred_reshaped).flatten()
    else:
        y_true_original = y_true
        y_pred_original = y_pred
    mse = mean_squared_error(y_true_original, y_pred_original)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true_original, y_pred_original)
    with np.errstate(divide='ignore', invalid='ignore'):
        ma_percentage_error = np.mean(np.abs((y_true_original - y_pred_original) / 
                                               np.where(y_true_original == 0, 1, y_true_original))) * 100
        ma_percentage_error = np.nan_to_num(ma_percentage_error, nan=0.0, posinf=0.0, neginf=0.0)
    std_dev = np.std(y_true_original - y_pred_original)
    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "ma_percentage_error": ma_percentage_error,
        "std_dev": std_dev
    }
