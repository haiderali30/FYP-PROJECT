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




import joblib
from tabulate import tabulate
import flwr as fl
from config import NUM_CLIENTS, SEQUENCE_LENGTH, EPOCHS, BATCH_SIZE, NUM_ROUNDS, SERVER_ADDRESS, MODEL_TYPE
import logging
import numpy as np
import pandas as pd
from model import create_model, load_and_preprocess_data, calculate_metrics
import tensorflow as tf
from typing import Dict, List, Optional, Tuple
from flwr.common import Parameters, Scalar
import joblib

logging.basicConfig(level=logging.INFO)

class SaveModelStrategy(fl.server.strategy.FedAvg):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Load validation data with validation=True so that it returns split data.
        self.validation_data = [load_and_preprocess_data("datasets/community_data.csv", validation=True)]

        # Determine the input shape(s) from the first validation dataset
        val_data = self.validation_data[0]
        # For Hybrid, load_and_preprocess_data returns ([X_seq, X_exo], y, feature_scaler, target_scaler)
        if isinstance(val_data[0], list):  # Hybrid mode
            X_val_seq, X_val_exo = val_data[0]
            seq_shape = X_val_seq.shape[1:]  # e.g., (SEQUENCE_LENGTH, 1)
            exo_shape = X_val_exo.shape[1:]  # e.g., (11,)
            input_shape = [seq_shape, exo_shape]
        else:
            X_val = val_data[0]  # Non-hybrid mode: X_val is a NumPy array
            input_shape = (X_val.shape[1],)  # e.g., (11,)

        print("Input shape for global model:", input_shape)
        logging.info(f"Input shape for global model: {input_shape}")

        # Initialize the global model with the correct input shape.
        # For Hybrid, create_model() should accept a list of shapes.
        self.global_model = create_model(input_shape)
        logging.info("Global model structure:")
        self.global_model.summary()

        # Initialize metrics history
        self.metrics_history = []

        # Log validation dataset shapes
        for i, (X, y, _, _) in enumerate(self.validation_data):
            if isinstance(X, list):
                shape_info = [a.shape for a in X]
            else:
                shape_info = X.shape
            logging.info(f"Validation dataset {i} shapes - X: {shape_info}, y: {y.shape}")

    def save_metrics_history(self, metrics_history):
        """Save the metrics history to a CSV file."""
        df = pd.DataFrame(metrics_history)
        df.to_csv("Using Federated Learning for Short-term Residential Load Forecasting.csv", index=False)
        logging.info("Metrics history saved to Using Federated Learning for Short-term Residential Load Forecasting.csv")
        
    def aggregate_fit(
        self,
        rnd: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[BaseException],
    ) -> Optional[Tuple[Parameters, Dict[str, Scalar]]]:
        """Aggregate model weights and metrics."""
        if not results:
            return None

        # Get weights from all clients
        weights_results = []
        num_samples = []
        for client_proxy, fit_res in results:
            if fit_res.parameters is not None:
                weights = fl.common.parameters_to_ndarrays(fit_res.parameters)
                weights_results.append(weights)
                num_samples.append(fit_res.num_examples)  # Collect number of samples

        # Calculate weighted average of weights
        aggregated_weights = [
            np.zeros_like(weights_results[0][i]) 
            for i in range(len(weights_results[0]))
        ]

        for client_weights, num in zip(weights_results, num_samples):
            for i, layer_weights in enumerate(client_weights):
                aggregated_weights[i] += layer_weights * num  # Weighted by number of samples

        total_samples = sum(num_samples)
        aggregated_weights = [w / total_samples for w in aggregated_weights]  # Normalize by total samples

        # Update global model
        self.global_model.set_weights(aggregated_weights)

        # Evaluate on validation datasets
        aggregated_metrics = {"mse": 0, "rmse": 0, "mae": 0, "ma_percentage_error": 0, "std_dev": 0}
        total_samples = 0
        for i, (X_val, y_val, _, target_scaler) in enumerate(self.validation_data):
            num_samples_val = len(y_val)
            total_samples += num_samples_val
            y_pred = self.global_model.predict(X_val)
            metrics = calculate_metrics(y_val, y_pred.flatten(), target_scaler)
            logging.info(f"Validation Dataset {i} Metrics (Original Scale): {metrics}")
            for key in aggregated_metrics.keys():
                aggregated_metrics[key] += metrics[key] * num_samples_val  # Weighted by dataset size

        # Normalize metrics by total samples
        for key in aggregated_metrics.keys():
            aggregated_metrics[key] /= total_samples

        # Log and store metrics
        logging.info(f"Global Model Metrics (Round {rnd}): {aggregated_metrics}")
        self.metrics_history.append({"Round": rnd, **aggregated_metrics})
        self.save_metrics_history(self.metrics_history)

       # Early Stopping with Patience
        self.patience = 5
        self.best_mse = float("inf")
        self.no_improvement_count = 0

        if aggregated_metrics["mse"] < self.best_mse:
            self.best_mse = aggregated_metrics["mse"]
            self.no_improvement_count = 0
        else:
            self.no_improvement_count += 1
            logging.info(f"No improvement in MSE for {self.no_improvement_count} rounds. Best MSE: {self.best_mse}")

        if self.no_improvement_count >= self.patience:
            logging.info(f"Early stopping triggered after {self.no_improvement_count} rounds: No improvement in validation MSE.")
            return None

        if rnd == 10:
            headers = ["Round", "MSE (kW²)", "RMSE (kW)", "MAE (kW)", "MA Percentage Error (%)", "Std Dev (kW)"]
            table_data = [
                [
                    metrics["Round"],
                    metrics["mse"],
                    metrics["rmse"],
                    metrics["mae"],
                    metrics["ma_percentage_error"],
                    metrics["std_dev"]
                ]
                for metrics in self.metrics_history
            ]
            print("\nAggregated Global Model Metrics (Original Scale):")
            print(tabulate(table_data, headers=headers, tablefmt="pretty", floatfmt=".6f"))

        if rnd == 10:
            self.global_model.save("Using Federated Learning for Short-term Residential Load Forecasting.h5")
            logging.info("Global model saved to Using Federated Learning for Short-term Residential Load Forecasting.h5")
            _, _, feature_scaler, target_scaler = self.validation_data[0]
            feature_names = feature_scaler.feature_names_in_
            joblib.dump(feature_names, "Using Federated Learning for Short-term Residential Load Forecasting.save")
            logging.info("Feature names saved.")
            feature_scaler_path = "Using Federated Learning for Short-term Residential Load Forecasting_feature.save"
            target_scaler_path = "Using Federated Learning for Short-term Residential Load Forecasting_target.save"
            joblib.dump(feature_scaler, feature_scaler_path)
            joblib.dump(target_scaler, target_scaler_path)
            logging.info(f"Feature scaler saved to {feature_scaler_path}")
            logging.info(f"Target scaler saved to {target_scaler_path}")

        aggregated_parameters = fl.common.ndarrays_to_parameters(aggregated_weights)
        return aggregated_parameters, aggregated_metrics

def main():
    """Start the Flower server."""
    strategy = SaveModelStrategy(
        fraction_fit=1.0,
        min_fit_clients=NUM_CLIENTS,
        min_available_clients=NUM_CLIENTS,
        on_fit_config_fn=lambda rnd: {"epochs": EPOCHS, "batch_size": BATCH_SIZE},
    )

    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
    )

if __name__ == "__main__":
    main()
