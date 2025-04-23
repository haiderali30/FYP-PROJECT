# # client.py
# import tensorflow as tf
# from model import create_model, load_and_preprocess_data, train_model, calculate_metrics
# import flwr as fl
# from config import EPOCHS, BATCH_SIZE, SERVER_ADDRESS
# import logging
# import numpy as np

# logging.basicConfig(level=logging.INFO)

# class EnergyClient(fl.client.NumPyClient):
#     def __init__(self, dataset_path, client_id):
#         self.dataset_path = dataset_path
#         self.client_id = client_id
#         # Load the training data (default: full training set)
#         self.X_train, self.y_train, self.feature_scaler, self.target_scaler = load_and_preprocess_data(dataset_path)
        
#         input_shape = (self.X_train.shape[1], self.X_train.shape[2])
#         logging.info(f"Client {self.client_id}: Input shape for model: {input_shape}")
#         self.model = create_model(input_shape)
        
#         # Load validation data (using the new "validation" parameter)
#         self.X_val, self.y_val, _, _ = load_and_preprocess_data(dataset_path, validation=True)

#     def get_parameters(self, config=None):
#         weights = self.model.get_weights()
#         logging.info(f"Client {self.client_id}: Model has {len(weights)} weight tensors")
#         return weights

#     def fit(self, parameters, config):
#         logging.info(f"Client {self.client_id}: Received global model, starting training...")
#         self.model.set_weights(parameters)

#         logging.info(f"Client {self.client_id}: X_train shape: {self.X_train.shape}, y_train shape: {self.y_train.shape}")
#         self.model, history = train_model(
#             self.model, self.X_train, self.y_train,
#             epochs=EPOCHS, batch_size=BATCH_SIZE
#         )

#         # Calculate validation metrics on the local validation set
#         y_val_pred = self.model.predict(self.X_val)
#         metrics = calculate_metrics(self.y_val, y_val_pred.flatten(), self.target_scaler)
#         logging.info(f"Client {self.client_id}: Validation Metrics: {metrics}")

#         # Return the updated weights, number of training examples, and validation metrics
#         return self.model.get_weights(), len(self.X_train), metrics

#     def evaluate(self, parameters, config):
#         logging.info(f"Client {self.client_id}: Received global model, starting evaluation...")
#         self.model.set_weights(parameters)
#         y_pred = self.model.predict(self.X_train)
#         metrics = calculate_metrics(self.y_train, y_pred.flatten(), self.target_scaler)
#         logging.info(f"Client {self.client_id}: Evaluation completed. Metrics: {metrics}")
#         return metrics["mse"], len(self.X_train), {"mae": metrics["mae"]}

# def start_client(dataset_path, client_id):
#     client = EnergyClient(dataset_path, client_id)
#     fl.client.start_numpy_client(server_address=SERVER_ADDRESS, client=client)

# if __name__ == "__main__":
#     import sys
#     dataset_path = sys.argv[1]  # Provide dataset path as first argument
#     client_id = sys.argv[2]     # Provide client ID as second argument
#     start_client(dataset_path, client_id)

import pandas as pd
import numpy as np
import flwr as fl
import logging
# from model import create_model, train_model, calculate_metrics, preprocess_one_row
from model.model import create_model, train_model, calculate_metrics, preprocess_one_row

from config import (
    EPOCHS, BATCH_SIZE,
    SERVER_ADDRESS, MODEL_TYPE,
    ROUNDS_PER_SLICE, SEQUENCE_LENGTH,
)

logging.basicConfig(level=logging.INFO)

class EnergyClient(fl.client.NumPyClient):
    def __init__(self, csv_path: str, client_id: str):
        self.client_id = client_id
        self.df = pd.read_csv(csv_path)
        # Dummy input shape for get_parameters
        dummy_shape = [(SEQUENCE_LENGTH,1), (len(self.df.columns)-2,)]
        self.model = create_model(dummy_shape)

    def _get_slice(self, slice_idx: int):
        row = self.df.iloc[[slice_idx]]
        return preprocess_one_row(row, SEQUENCE_LENGTH, MODEL_TYPE)

    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        rnd = config.get("round", 1)
        slice_idx = (rnd-1) // ROUNDS_PER_SLICE
        logging.info(f"[Client {self.client_id}] Round {rnd}: slice {slice_idx}")
        # 1) load weights
        self.model.set_weights(parameters)
        # 2) get one-row training data
        X, y, _, _ = self._get_slice(slice_idx)
        # 3) train
        self.model, _ = train_model(self.model, X, y, epochs=EPOCHS, batch_size=BATCH_SIZE)
        # 4) return update
        return self.model.get_weights(), len(y), {}

    def evaluate(self, parameters, config):
        # Optional: evaluate on last row
        self.model.set_weights(parameters)
        X, y, _, _ = self._get_slice(len(self.df)-1)
        loss, mae = self.model.evaluate(X, y, verbose=0)
        return loss, len(y), {"mae": float(mae)}

if __name__ == "__main__":
    import sys
    fl.client.start_numpy_client(
        server_address=SERVER_ADDRESS,
        client=EnergyClient(sys.argv[1], sys.argv[2])
    )