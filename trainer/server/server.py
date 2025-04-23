# # server.py
# import joblib
# from tabulate import tabulate
# import flwr as fl
# from config import NUM_CLIENTS, SEQUENCE_LENGTH, EPOCHS, BATCH_SIZE, NUM_ROUNDS, SERVER_ADDRESS
# import logging
# import numpy as np
# import pandas as pd
# from model import create_model, load_and_preprocess_data, calculate_metrics
# import tensorflow as tf
# from typing import Dict, List, Optional, Tuple
# from flwr.common import Parameters, Scalar

# logging.basicConfig(level=logging.INFO)

# class SaveModelStrategy(fl.server.strategy.FedAvg):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
        
#         # Load validation datasets from files (adjust paths as needed)
#         self.dataset_paths = [
#             "datasets/House1_data.csv",
#             "datasets/House2_data.csv",
#             "datasets/Hospital_data.csv",
#             "datasets/Industry_data.csv",
#             "datasets/Office_data.csv",
#             "datasets/School_data.csv"
#         ]
#         self.validation_data = [load_and_preprocess_data(path) for path in self.dataset_paths]

#         # Determine the input shape from the first validation dataset
#         X_val, _, feature_scaler, target_scaler = self.validation_data[0]
#         logging.info(f"Validation dataset 0: X_val shape: {X_val.shape}")

#         input_shape = (X_val.shape[1], X_val.shape[2])  # (SEQUENCE_LENGTH, num_features)
#         logging.info(f"Input shape for global model: {input_shape}")
        
#         # Initialize the global model with the correct input shape
#         self.global_model = create_model(input_shape)
#         logging.info("Global model structure:")
#         self.global_model.summary()
        
#         # Initialize metrics history to track performance across rounds
#         self.metrics_history = []
#         # Early stopping settings (optional)
#         self.patience = 5
#         self.best_mse = float("inf")
#         self.no_improvement_count = 0

#         # Log shapes of each validation dataset
#         for i, (X, y, _, _) in enumerate(self.validation_data):
#             logging.info(f"Validation dataset {i} shapes - X: {X.shape}, y: {y.shape}")

#     def save_metrics_history(self, metrics_history):
#         """Save the metrics history to a CSV file."""
#         df = pd.DataFrame(metrics_history)
#         df.to_csv("Federated_Load_Forecasting_Metrics_History.csv", index=False)
#         logging.info("Metrics history saved to Federated_Load_Forecasting_Metrics_History.csv")
        
#     def aggregate_fit(
#         self,
#         rnd: int,
#         results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
#         failures: List[BaseException],
#     ) -> Optional[Tuple[Parameters, Dict[str, Scalar]]]:
#         """Aggregate model weights and metrics from clients."""
#         if not results:
#             return None

#         # Get weights from all clients
#         weights_results = []
#         num_samples = []
#         for client_proxy, fit_res in results:
#             if fit_res.parameters is not None:
#                 weights = fl.common.parameters_to_ndarrays(fit_res.parameters)
#                 weights_results.append(weights)
#                 num_samples.append(fit_res.num_examples)

#         # Weighted averaging of model weights
#         aggregated_weights = [
#             np.zeros_like(weights_results[0][i]) for i in range(len(weights_results[0]))
#         ]
#         for client_weights, num in zip(weights_results, num_samples):
#             for i, layer_weights in enumerate(client_weights):
#                 aggregated_weights[i] += layer_weights * num
#         total_samples = sum(num_samples)
#         aggregated_weights = [w / total_samples for w in aggregated_weights]

#         # Update global model with aggregated weights
#         self.global_model.set_weights(aggregated_weights)

#         # Evaluate global model on each validation dataset and aggregate metrics
#         aggregated_metrics = {"mse": 0, "rmse": 0, "mae": 0, "ma_percentage_error": 0, "std_dev": 0}
#         total_val_samples = 0
#         for i, (X_val, y_val, _, target_scaler) in enumerate(self.validation_data):
#             num_samples_val = len(y_val)
#             total_val_samples += num_samples_val
#             y_pred = self.global_model.predict(X_val)
#             metrics = calculate_metrics(y_val, y_pred.flatten(), target_scaler)
#             logging.info(f"Validation Dataset {i} Metrics (Original Scale): {metrics}")
#             for key in aggregated_metrics.keys():
#                 aggregated_metrics[key] += metrics[key] * num_samples_val

#         for key in aggregated_metrics.keys():
#             aggregated_metrics[key] /= total_val_samples

#         logging.info(f"Global Model Metrics (Round {rnd}): {aggregated_metrics}")
#         self.metrics_history.append({"Round": rnd, **aggregated_metrics})
#         self.save_metrics_history(self.metrics_history)

#         # Early stopping logic based on MSE improvement
#         if aggregated_metrics["mse"] < self.best_mse:
#             self.best_mse = aggregated_metrics["mse"]
#             self.no_improvement_count = 0
#         else:
#             self.no_improvement_count += 1
#             logging.info(f"No improvement in MSE for {self.no_improvement_count} rounds. Best MSE: {self.best_mse}")
#         if self.no_improvement_count >= self.patience:
#             logging.info(f"Early stopping triggered after {self.no_improvement_count} rounds.")
#             return None

#         # Optionally, print final metrics table at the last round
#         if rnd == NUM_ROUNDS:
#             headers = ["Round", "MSE (kW²)", "RMSE (kW)", "MAE (kW)", "MA Percentage Error (%)", "Std Dev (kW)"]
#             table_data = [
#                 [metrics["Round"], metrics["mse"], metrics["rmse"], metrics["mae"], metrics["ma_percentage_error"], metrics["std_dev"]]
#                 for metrics in self.metrics_history
#             ]
#             print("\nAggregated Global Model Metrics (Original Scale):")
#             print(tabulate(table_data, headers=headers, tablefmt="pretty", floatfmt=".6f"))

#             # Save final global model and scalers
#             self.global_model.save("Federated_Load_Forecasting_Global_Model.h5")
#             logging.info("Global model saved to Federated_Load_Forecasting_Global_Model.h5")
#             # Save feature names and scalers (if available)
#             _, _, feature_scaler, target_scaler = self.validation_data[0]
#             feature_names = feature_scaler.feature_names_in_ if hasattr(feature_scaler, 'feature_names_in_') else list(range(self.global_model.input_shape[-1]))
#             joblib.dump(feature_names, "Federated_Load_Forecasting_Feature_Names.pkl")
#             joblib.dump(feature_scaler, "Federated_Load_Forecasting_Feature_Scaler.pkl")
#             joblib.dump(target_scaler, "Federated_Load_Forecasting_Target_Scaler.pkl")
#             logging.info("Feature names and scalers saved.")

#         aggregated_parameters = fl.common.ndarrays_to_parameters(aggregated_weights)
#         return aggregated_parameters, aggregated_metrics

# def main():
#     """Start the Flower server with the custom strategy."""
#     strategy = SaveModelStrategy(
#         fraction_fit=1.0,
#         min_fit_clients=NUM_CLIENTS,
#         min_available_clients=NUM_CLIENTS,
#         on_fit_config_fn=lambda rnd: {"epochs": EPOCHS, "batch_size": BATCH_SIZE},
#     )

#     fl.server.start_server(
#         server_address="0.0.0.0:8080",
#         config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
#         strategy=strategy,
#     )

# if __name__ == "__main__":
#     main()




# import joblib
# from tabulate import tabulate
# import flwr as fl
# from config import NUM_CLIENTS, SEQUENCE_LENGTH, EPOCHS, BATCH_SIZE, NUM_ROUNDS, SERVER_ADDRESS, MODEL_TYPE
# import logging
# import numpy as np
# import pandas as pd
# from model import create_model, load_and_preprocess_data, calculate_metrics
# import tensorflow as tf
# from typing import Dict, List, Optional, Tuple
# from flwr.common import Parameters, Scalar
# import joblib

# logging.basicConfig(level=logging.INFO)

# class SaveModelStrategy(fl.server.strategy.FedAvg):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)

#         # Load validation data with validation=True so that it returns split data.
#         self.validation_data = [load_and_preprocess_data("datasets/community_data.csv", validation=True)]

#         # Determine the input shape(s) from the first validation dataset
#         val_data = self.validation_data[0]
#         # For Hybrid, load_and_preprocess_data returns ([X_seq, X_exo], y, feature_scaler, target_scaler)
#         if isinstance(val_data[0], list):  # Hybrid mode
#             X_val_seq, X_val_exo = val_data[0]
#             seq_shape = X_val_seq.shape[1:]  # e.g., (SEQUENCE_LENGTH, 1)
#             exo_shape = X_val_exo.shape[1:]  # e.g., (11,)
#             input_shape = [seq_shape, exo_shape]
#         else:
#             X_val = val_data[0]  # Non-hybrid mode: X_val is a NumPy array
#             input_shape = (X_val.shape[1],)  # e.g., (11,)

#         print("Input shape for global model:", input_shape)
#         logging.info(f"Input shape for global model: {input_shape}")

#         # Initialize the global model with the correct input shape.
#         # For Hybrid, create_model() should accept a list of shapes.
#         self.global_model = create_model(input_shape)
#         logging.info("Global model structure:")
#         self.global_model.summary()

#         # Initialize metrics history
#         self.metrics_history = []

#         # Log validation dataset shapes
#         for i, (X, y, _, _) in enumerate(self.validation_data):
#             if isinstance(X, list):
#                 shape_info = [a.shape for a in X]
#             else:
#                 shape_info = X.shape
#             logging.info(f"Validation dataset {i} shapes - X: {shape_info}, y: {y.shape}")

#     def save_metrics_history(self, metrics_history):
#         """Save the metrics history to a CSV file."""
#         df = pd.DataFrame(metrics_history)
#         df.to_csv("Using Federated Learning for Short-term Residential Load Forecasting.csv", index=False)
#         logging.info("Metrics history saved to Using Federated Learning for Short-term Residential Load Forecasting.csv")
        
#     def aggregate_fit(
#         self,
#         rnd: int,
#         results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
#         failures: List[BaseException],
#     ) -> Optional[Tuple[Parameters, Dict[str, Scalar]]]:
#         """Aggregate model weights and metrics."""
#         if not results:
#             return None

#         # Get weights from all clients
#         weights_results = []
#         num_samples = []
#         for client_proxy, fit_res in results:
#             if fit_res.parameters is not None:
#                 weights = fl.common.parameters_to_ndarrays(fit_res.parameters)
#                 weights_results.append(weights)
#                 num_samples.append(fit_res.num_examples)  # Collect number of samples

#         # Calculate weighted average of weights
#         aggregated_weights = [
#             np.zeros_like(weights_results[0][i]) 
#             for i in range(len(weights_results[0]))
#         ]

#         for client_weights, num in zip(weights_results, num_samples):
#             for i, layer_weights in enumerate(client_weights):
#                 aggregated_weights[i] += layer_weights * num  # Weighted by number of samples

#         total_samples = sum(num_samples)
#         aggregated_weights = [w / total_samples for w in aggregated_weights]  # Normalize by total samples

#         # Update global model
#         self.global_model.set_weights(aggregated_weights)

#         # Evaluate on validation datasets
#         aggregated_metrics = {"mse": 0, "rmse": 0, "mae": 0, "ma_percentage_error": 0, "std_dev": 0}
#         total_samples = 0
#         for i, (X_val, y_val, _, target_scaler) in enumerate(self.validation_data):
#             num_samples_val = len(y_val)
#             total_samples += num_samples_val
#             y_pred = self.global_model.predict(X_val)
#             metrics = calculate_metrics(y_val, y_pred.flatten(), target_scaler)
#             logging.info(f"Validation Dataset {i} Metrics (Original Scale): {metrics}")
#             for key in aggregated_metrics.keys():
#                 aggregated_metrics[key] += metrics[key] * num_samples_val  # Weighted by dataset size

#         # Normalize metrics by total samples
#         for key in aggregated_metrics.keys():
#             aggregated_metrics[key] /= total_samples

#         # Log and store metrics
#         logging.info(f"Global Model Metrics (Round {rnd}): {aggregated_metrics}")
#         self.metrics_history.append({"Round": rnd, **aggregated_metrics})
#         self.save_metrics_history(self.metrics_history)

#        # Early Stopping with Patience
#         self.patience = 5
#         self.best_mse = float("inf")
#         self.no_improvement_count = 0

#         if aggregated_metrics["mse"] < self.best_mse:
#             self.best_mse = aggregated_metrics["mse"]
#             self.no_improvement_count = 0
#         else:
#             self.no_improvement_count += 1
#             logging.info(f"No improvement in MSE for {self.no_improvement_count} rounds. Best MSE: {self.best_mse}")

#         if self.no_improvement_count >= self.patience:
#             logging.info(f"Early stopping triggered after {self.no_improvement_count} rounds: No improvement in validation MSE.")
#             return None

#         if rnd == 10:
#             headers = ["Round", "MSE (kW²)", "RMSE (kW)", "MAE (kW)", "MA Percentage Error (%)", "Std Dev (kW)"]
#             table_data = [
#                 [
#                     metrics["Round"],
#                     metrics["mse"],
#                     metrics["rmse"],
#                     metrics["mae"],
#                     metrics["ma_percentage_error"],
#                     metrics["std_dev"]
#                 ]
#                 for metrics in self.metrics_history
#             ]
#             print("\nAggregated Global Model Metrics (Original Scale):")
#             print(tabulate(table_data, headers=headers, tablefmt="pretty", floatfmt=".6f"))

#         if rnd == 10:
#             self.global_model.save("Using Federated Learning for Short-term Residential Load Forecasting.h5")
#             logging.info("Global model saved to Using Federated Learning for Short-term Residential Load Forecasting.h5")
#             _, _, feature_scaler, target_scaler = self.validation_data[0]
#             feature_names = feature_scaler.feature_names_in_
#             joblib.dump(feature_names, "Using Federated Learning for Short-term Residential Load Forecasting.save")
#             logging.info("Feature names saved.")
#             feature_scaler_path = "Using Federated Learning for Short-term Residential Load Forecasting_feature.save"
#             target_scaler_path = "Using Federated Learning for Short-term Residential Load Forecasting_target.save"
#             joblib.dump(feature_scaler, feature_scaler_path)
#             joblib.dump(target_scaler, target_scaler_path)
#             logging.info(f"Feature scaler saved to {feature_scaler_path}")
#             logging.info(f"Target scaler saved to {target_scaler_path}")

#         aggregated_parameters = fl.common.ndarrays_to_parameters(aggregated_weights)
#         return aggregated_parameters, aggregated_metrics

# def main():
#     """Start the Flower server."""
#     strategy = SaveModelStrategy(
#         fraction_fit=1.0,
#         min_fit_clients=NUM_CLIENTS,
#         min_available_clients=NUM_CLIENTS,
#         on_fit_config_fn=lambda rnd: {"epochs": EPOCHS, "batch_size": BATCH_SIZE},
#     )

#     fl.server.start_server(
#         server_address="0.0.0.0:8080",
#         config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
#         strategy=strategy,
#     )

# if __name__ == "__main__":
#     main()


# import os
# import joblib
# from tabulate import tabulate
# import flwr as fl
# from config import NUM_CLIENTS, SEQUENCE_LENGTH, EPOCHS, BATCH_SIZE, NUM_ROUNDS, SERVER_ADDRESS, MODEL_TYPE
# import logging
# import numpy as np
# import pandas as pd
# from model import create_model, load_and_preprocess_data, calculate_metrics
# import tensorflow as tf
# from typing import Dict, List, Optional, Tuple
# from flwr.common import Parameters, Scalar
# import joblib

# logging.basicConfig(level=logging.INFO)

# class SaveModelStrategy(fl.server.strategy.FedAvg):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)

#         # Load validation data with validation=True so that it returns split data.
#         self.validation_data = [load_and_preprocess_data("datasets/community_data.csv", validation=True)]

#         # Determine the input shape(s) from the first validation dataset
#         val_data = self.validation_data[0]
#         # For Hybrid, load_and_preprocess_data returns ([X_seq, X_exo], y, feature_scaler, target_scaler)
#         if isinstance(val_data[0], list):  # Hybrid mode
#             X_val_seq, X_val_exo = val_data[0]
#             seq_shape = X_val_seq.shape[1:]  # e.g., (SEQUENCE_LENGTH, 1)
#             exo_shape = X_val_exo.shape[1:]  # e.g., (11,)
#             input_shape = [seq_shape, exo_shape]
#         else:
#             X_val = val_data[0]  # Non-hybrid mode: X_val is a NumPy array
#             input_shape = (X_val.shape[1],)  # e.g., (11,)

#         print("Input shape for global model:", input_shape)
#         logging.info(f"Input shape for global model: {input_shape}")

#         # Initialize the global model with the correct input shape.
#         # For Hybrid, create_model() should accept a list of shapes.
#         self.global_model = create_model(input_shape)
#         logging.info("Global model structure:")
#         self.global_model.summary()

#         # Initialize metrics history
#         self.metrics_history = []

#         # Log validation dataset shapes
#         for i, (X, y, _, _) in enumerate(self.validation_data):
#             if isinstance(X, list):
#                 shape_info = [a.shape for a in X]
#             else:
#                 shape_info = X.shape
#             logging.info(f"Validation dataset {i} shapes - X: {shape_info}, y: {y.shape}")

#     def save_metrics_history(self, metrics_history):
#         """Save the metrics history to a CSV file."""
#         df = pd.DataFrame(metrics_history)
#         df.to_csv("Using Federated Learning for Short-term Residential Load Forecasting.csv", index=False)
#         logging.info("Metrics history saved to Using Federated Learning for Short-term Residential Load Forecasting.csv")
        
#     def aggregate_fit(
#         self,
#         rnd: int,
#         results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
#         failures: List[BaseException],
#     ) -> Optional[Tuple[Parameters, Dict[str, Scalar]]]:
#         """Aggregate model weights and metrics."""
#         if not results:
#             return None

#         # Get weights from all clients
#         weights_results = []
#         num_samples = []
#         for client_proxy, fit_res in results:
#             if fit_res.parameters is not None:
#                 weights = fl.common.parameters_to_ndarrays(fit_res.parameters)
#                 weights_results.append(weights)
#                 num_samples.append(fit_res.num_examples)  # Collect number of samples

#         # Calculate weighted average of weights
#         aggregated_weights = [
#             np.zeros_like(weights_results[0][i]) 
#             for i in range(len(weights_results[0]))
#         ]

#         for client_weights, num in zip(weights_results, num_samples):
#             for i, layer_weights in enumerate(client_weights):
#                 aggregated_weights[i] += layer_weights * num  # Weighted by number of samples

#         total_samples = sum(num_samples)
#         aggregated_weights = [w / total_samples for w in aggregated_weights]  # Normalize by total samples

#         # Update global model
#         self.global_model.set_weights(aggregated_weights)

#         # Evaluate on validation datasets
#         aggregated_metrics = {"mse": 0, "rmse": 0, "mae": 0, "ma_percentage_error": 0, "std_dev": 0}
#         total_samples_val = 0
#         for i, (X_val, y_val, _, target_scaler) in enumerate(self.validation_data):
#             num_samples_val = len(y_val)
#             total_samples_val += num_samples_val
#             y_pred = self.global_model.predict(X_val)
#             metrics = calculate_metrics(y_val, y_pred.flatten(), target_scaler)
#             logging.info(f"Validation Dataset {i} Metrics (Original Scale): {metrics}")
#             for key in aggregated_metrics.keys():
#                 aggregated_metrics[key] += metrics[key] * num_samples_val  # Weighted by dataset size

#         # Normalize metrics by total samples
#         for key in aggregated_metrics.keys():
#             aggregated_metrics[key] /= total_samples_val

#         # Log and store metrics
#         logging.info(f"Global Model Metrics (Round {rnd}): {aggregated_metrics}")
#         self.metrics_history.append({"Round": rnd, **aggregated_metrics})
#         self.save_metrics_history(self.metrics_history)

#         # Early Stopping with Patience
#         self.patience = 5
#         self.best_mse = float("inf")
#         self.no_improvement_count = 0

#         if aggregated_metrics["mse"] < self.best_mse:
#             self.best_mse = aggregated_metrics["mse"]
#             self.no_improvement_count = 0
#         else:
#             self.no_improvement_count += 1
#             logging.info(f"No improvement in MSE for {self.no_improvement_count} rounds. Best MSE: {self.best_mse}")

#         if self.no_improvement_count >= self.patience:
#             logging.info(f"Early stopping triggered after {self.no_improvement_count} rounds: No improvement in validation MSE.")
#             return None

#         # Print aggregated metrics in a clear table format after each round
#         headers = ["Round", "MSE (kW²)", "RMSE (kW)", "MAE (kW)", "MA Percentage Error (%)", "Std Dev (kW)"]
#         table_data = [[
#             rnd,
#             aggregated_metrics["mse"],
#             aggregated_metrics["rmse"],
#             aggregated_metrics["mae"],
#             aggregated_metrics["ma_percentage_error"],
#             aggregated_metrics["std_dev"]
#         ]]
#         print("\nAggregated Global Model Metrics (Round {}):".format(rnd))
#         print(tabulate(table_data, headers=headers, tablefmt="pretty", floatfmt=".6f"))

#         # Final round: Save the model and scalers
#         # if rnd == 10:
#         #     self.global_model.save("Using Federated Learning for Short-term Residential Load Forecasting.h5")
#         # logging.info("Global model saved to Using Federated Learning for Short-term Residential Load Forecasting.h5")
#         # _, _, feature_scaler, target_scaler = self.validation_data[0]
#         # feature_names = feature_scaler.feature_names_in_
#         # joblib.dump(feature_names, "Using Federated Learning for Short-term Residential Load Forecasting.save")
#         # logging.info("Feature names saved.")
#         # feature_scaler_path = "Using Federated Learning for Short-term Residential Load Forecasting_feature.save"
#         # target_scaler_path = "Using Federated Learning for Short-term Residential Load Forecasting_target.save"
#         # joblib.dump(feature_scaler, feature_scaler_path)
#         # joblib.dump(target_scaler, target_scaler_path)
#         # logging.info(f"Feature scaler saved to {feature_scaler_path}")
#         # logging.info(f"Target scaler saved to {target_scaler_path}")



#         # === save model + scalers every round ===
#         model_dir = "/app/trained_model"
#         os.makedirs(model_dir, exist_ok=True)

#         # 1) overwrite the H5
#         h5_path = os.path.join(model_dir, "latest_model.h5")
#         self.global_model.save(h5_path)
#         logging.info(f"[Round {rnd}] Saved global model to {h5_path}")

#         # 2) overwrite your scalers too
#         _, _, feature_scaler, target_scaler = self.validation_data[0]
#         joblib.dump(feature_scaler, os.path.join(model_dir, "feature_scaler.save"))
#         joblib.dump(target_scaler, os.path.join(model_dir, "target_scaler.save"))
#         logging.info(f"[Round {rnd}] Saved scalers to {model_dir}")

#         aggregated_parameters = fl.common.ndarrays_to_parameters(aggregated_weights)
#         return aggregated_parameters, aggregated_metrics

# def main():
#     """Start the Flower server."""
#     strategy = SaveModelStrategy(
#         fraction_fit=1.0,
#         min_fit_clients=NUM_CLIENTS,
#         min_available_clients=NUM_CLIENTS,
#         on_fit_config_fn=lambda rnd: {"epochs": EPOCHS, "batch_size": BATCH_SIZE},
#     )

#     fl.server.start_server(
#         server_address="0.0.0.0:8080",
#         config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
#         strategy=strategy,
#     )

# if __name__ == "__main__":
#     main()





# import os
# import logging
# import joblib
# import numpy as np
# import pandas as pd
# from tabulate import tabulate
# import flwr as fl
# from typing import List, Optional, Tuple, Dict
# from flwr.common import Parameters, Scalar
# from config import NUM_CLIENTS, EPOCHS, BATCH_SIZE, NUM_ROUNDS
# from model import create_model, load_and_preprocess_data, calculate_metrics

# # Configure logging
# logging.basicConfig(level=logging.INFO)

# class SaveModelStrategy(fl.server.strategy.FedAvg):
#     def __init__(
#         self,
#         fraction_fit: float,
#         min_fit_clients: int,
#         min_available_clients: int,
#         on_fit_config_fn,
#     ):
#         super().__init__(
#             fraction_fit=fraction_fit,
#             min_fit_clients=min_fit_clients,
#             min_available_clients=min_available_clients,
#             on_fit_config_fn=on_fit_config_fn,
#         )

#         # Load validation data (returns ([X_seq, X_exo], y, feature_scaler, target_scaler))
#         self.validation_data = [
#             load_and_preprocess_data("datasets/community_data.csv", validation=True)
#         ]

#         # Determine input shapes
#         val_X, _, _, _ = self.validation_data[0]
#         if isinstance(val_X, list):
#             seq_shape = val_X[0].shape[1:]
#             exo_shape = val_X[1].shape[1:]
#             input_shape = [seq_shape, exo_shape]
#         else:
#             input_shape = (val_X.shape[1],)

#         logging.info(f"Input shape for global model: {input_shape}")

#         # Initialize the global model
#         self.global_model = create_model(input_shape)
#         logging.info("Global model structure:")
#         self.global_model.summary()

#         # Metrics history and early-stopping state
#         self.metrics_history: List[Dict[str, float]] = []
#         self.best_mse: float = float("inf")
#         self.no_improvement_count: int = 0
#         self.patience: int = 5

#     def save_metrics_history(self) -> None:
#         df = pd.DataFrame(self.metrics_history)
#         df.to_csv("trained_model/metrics_history.csv", index=False)
#         logging.info("Metrics history saved to trained_model/metrics_history.csv")

#     # def save_model_and_scalers(self, rnd: int) -> None:
#     #     model_dir = "/app/trained_model"
#     #     os.makedirs(model_dir, exist_ok=True)

#     #     # Save model
#     #     h5_path = os.path.join(model_dir, "latest_model.h5")
#     #     self.global_model.save(h5_path)
#     #     logging.info(f"[Round {rnd}] Saved model to {h5_path}")

#     #     # Save scalers
#     #     _, _, feature_scaler, target_scaler = self.validation_data[0]
#     #     joblib.dump(feature_scaler, os.path.join(model_dir, "feature_scaler.save"))
#     #     joblib.dump(target_scaler, os.path.join(model_dir, "target_scaler.save"))
#     #     logging.info(f"[Round {rnd}] Saved scalers to {model_dir}")


#     def save_model_and_scalers(self, rnd: int):
#         # Because we bind-mounted ../app/backend/prediction_model_files_docker → /app/prediction_model_files_docker
#         model_dir = os.path.join(os.getcwd(), "prediction_model_files_docker")
#         os.makedirs(model_dir, exist_ok=True)

#         # 1) Save the latest model weights
#         h5_path = os.path.join(model_dir, "latest_model.h5")
#         self.global_model.save(h5_path)
#         logging.info(f"[Round {rnd}] Saved global model to {h5_path}")

#         # 2) Save the feature and target scalers
#         _, _, feature_scaler, target_scaler = self.validation_data[0]
#         joblib.dump(feature_scaler, os.path.join(model_dir, "feature_scaler.save"))
#         joblib.dump(target_scaler, os.path.join(model_dir, "target_scaler.save"))
#         logging.info(f"[Round {rnd}] Saved scalers to {model_dir}")

#     def aggregate_fit(
#         self,
#         rnd: int,
#         results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
#         failures: List[BaseException],
#     ) -> Optional[Tuple[Parameters, Dict[str, Scalar]]]:
#         if not results:
#             return None

#         # Weighted aggregation of parameters
#         weights_results = []
#         num_samples = []
#         for _, fit_res in results:
#             if fit_res.parameters is None:
#                 continue
#             weights_results.append(
#                 fl.common.parameters_to_ndarrays(fit_res.parameters)
#             )
#             num_samples.append(fit_res.num_examples)

#         total_samples = sum(num_samples)
#         aggregated_weights = [
#             np.zeros_like(layer) for layer in weights_results[0]
#         ]
#         for weights, n in zip(weights_results, num_samples):
#             for i, layer in enumerate(weights):
#                 aggregated_weights[i] += layer * n
#         aggregated_weights = [w / total_samples for w in aggregated_weights]

#         # Update global model weights
#         self.global_model.set_weights(aggregated_weights)

#         # Validation
#         metrics_sum = {k: 0.0 for k in ["mse", "rmse", "mae", "ma_percentage_error", "std_dev"]}
#         val_count = 0
#         X_val, y_val, _, target_scaler = self.validation_data[0]
#         y_pred = self.global_model.predict(X_val)
#         metrics = calculate_metrics(y_val, y_pred.flatten(), target_scaler)
#         logging.info(f"Round {rnd} validation metrics: {metrics}")
#         for k, v in metrics.items():
#             metrics_sum[k] += v
#         val_count += 1

#         # Average metrics
#         averaged_metrics = {k: metrics_sum[k] / val_count for k in metrics_sum}
#         self.metrics_history.append({"round": rnd, **averaged_metrics})
#         self.save_metrics_history()

#         # Early stopping
#         if averaged_metrics["mse"] < self.best_mse:
#             self.best_mse = averaged_metrics["mse"]
#             self.no_improvement_count = 0
#         else:
#             self.no_improvement_count += 1
#             logging.info(f"No improvement count: {self.no_improvement_count}/{self.patience}")
#         if self.no_improvement_count >= self.patience:
#             logging.info("Early stopping triggered")
#             return None

#         # Print table
#         headers = ["Round", "MSE", "RMSE", "MAE", "MAPE", "StdDev"]
#         row = [
#             rnd,
#             averaged_metrics["mse"],
#             averaged_metrics["rmse"],
#             averaged_metrics["mae"],
#             averaged_metrics["ma_percentage_error"],
#             averaged_metrics["std_dev"],
#         ]
#         print(tabulate([row], headers=headers, floatfmt=".6f"))

#         # Save model/scalers every round
#         self.save_model_and_scalers(rnd)

#         return fl.common.ndarrays_to_parameters(aggregated_weights), averaged_metrics


# def main() -> None:
#     strategy = SaveModelStrategy(
#         fraction_fit=1.0,
#         min_fit_clients=NUM_CLIENTS,
#         min_available_clients=NUM_CLIENTS,
#         on_fit_config_fn=lambda rnd: {"epochs": EPOCHS, "batch_size": BATCH_SIZE},
#     )

#     fl.server.start_server(
#         server_address="0.0.0.0:8080",
#         config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
#         strategy=strategy,
#     )

# if __name__ == "__main__":
#     main()



import os
import time
import logging
import joblib
import numpy as np
import pandas as pd
import flwr as fl
from tabulate import tabulate
from typing import List, Tuple, Dict, Optional
from flwr.common import Parameters, Scalar
from config import NUM_CLIENTS, EPOCHS, BATCH_SIZE, NUM_ROUNDS, ROUNDS_PER_SLICE
from model import create_model, load_and_preprocess_data, calculate_metrics

MODEL_DIR = os.environ.get("MODEL_DIR", "/app/trained_model")
logging.basicConfig(level=logging.INFO)

class SaveModelStrategy(fl.server.strategy.FedAvg):
    def __init__(self, fraction_fit, min_fit_clients, min_available_clients, on_fit_config_fn):
        super().__init__(
            fraction_fit=fraction_fit,
            min_fit_clients=min_fit_clients,
            min_available_clients=min_available_clients,
            on_fit_config_fn=on_fit_config_fn,
        )
        # Load validation split once
        self.validation_data = [
            load_and_preprocess_data("datasets/community_data.csv", validation=True)
        ]
        Xv, _, _, _ = self.validation_data[0]
        if isinstance(Xv, list):
            shapes = [Xv[0].shape[1:], Xv[1].shape[1:]]
        else:
            shapes = (Xv.shape[1],)
        logging.info(f"Global model input: {shapes}")
        self.global_model = create_model(shapes)
        self.global_model.summary()
        self.metrics_history = []
        self.best_mse = float('inf')
        self.no_improvement = 0
        self.patience = 5

    def save_metrics_history(self):
        df = pd.DataFrame(self.metrics_history)
        os.makedirs(MODEL_DIR, exist_ok=True)
        df.to_csv(os.path.join(MODEL_DIR, "metrics_history.csv"), index=False)
        logging.info("Saved metrics_history.csv")

    def save_model_and_scalers(self, rnd: int):
        os.makedirs(MODEL_DIR, exist_ok=True)
        # model
        path = os.path.join(MODEL_DIR, "latest_model.h5")
        self.global_model.save(path)
        logging.info(f"[Round {rnd}] Model → {path}")
        # scalers
        _, _, fs, ts = self.validation_data[0]
        joblib.dump(fs, os.path.join(MODEL_DIR, "feature_scaler.save"))
        joblib.dump(ts, os.path.join(MODEL_DIR, "target_scaler.save"))
        logging.info(f"[Round {rnd}] Scalers → {MODEL_DIR}")

    def aggregate_fit(
        self, rnd: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[BaseException],
    ) -> Optional[Tuple[Parameters, Dict[str, Scalar]]]:
        if not results:
            return None
        # Weighted avg of client weights
        weights_list, samples = [], []
        for _, fit in results:
            if fit.parameters is None:
                continue
            weights_list.append(fl.common.parameters_to_ndarrays(fit.parameters))
            samples.append(fit.num_examples)
        total = sum(samples)
        new_weights = [np.zeros_like(w) for w in weights_list[0]]
        for w, n in zip(weights_list, samples):
            for i, lw in enumerate(w): new_weights[i] += lw * n
        new_weights = [w/total for w in new_weights]
        self.global_model.set_weights(new_weights)

        # Validate on hold‑out
        Xv, yv, _, tsc = self.validation_data[0]
        ypred = self.global_model.predict(Xv)
        mets = calculate_metrics(yv, ypred.flatten(), tsc)
        logging.info(f"Round {rnd} val metrics: {mets}")
        self.metrics_history.append({"round": rnd, **mets})
        self.save_metrics_history()

        # Early stop
        if mets['mse'] < self.best_mse:
            self.best_mse = mets['mse']
            self.no_improvement = 0
        else:
            self.no_improvement += 1
        if self.no_improvement >= self.patience:
            logging.info("Early stopping")
            return None

        # Print table
        print(tabulate(
            [[rnd, mets['mse'], mets['rmse'], mets['mae'], mets['ma_percentage_error'], mets['std_dev']]],
            headers=["Rnd","MSE","RMSE","MAE","MAPE","StdDev"], floatfmt=".6f"
        ))

        # Persist model & scalers
        self.save_model_and_scalers(rnd)
        return fl.common.ndarrays_to_parameters(new_weights), mets


def main():
    def on_fit(rnd: int):
        if rnd>1 and (rnd-1)%ROUNDS_PER_SLICE==0:
            time.sleep(5*60)
        return {"epochs": EPOCHS, "batch_size": BATCH_SIZE, "round": rnd}

    strat = SaveModelStrategy(
        fraction_fit=1.0,
        min_fit_clients=NUM_CLIENTS,
        min_available_clients=NUM_CLIENTS,
        on_fit_config_fn=on_fit,
    )
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strat,
    )

if __name__ == '__main__':
    main()