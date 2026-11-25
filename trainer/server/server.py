"""Flower server that aggregates building-level energy forecasting models.

The server runs weighted Federated Averaging (FedAvg) over the clients'
model weights. After every round it evaluates the new global model on a
held-out community dataset, logs the metrics, and saves the model and
scalers so the dashboard backend can serve the latest version.
"""

import logging
import os
import time
from typing import Dict, List, Optional, Tuple

import flwr as fl
import joblib
import numpy as np
import pandas as pd
from flwr.common import Parameters, Scalar
from tabulate import tabulate

from config import BATCH_SIZE, EPOCHS, NUM_CLIENTS, NUM_ROUNDS, ROUNDS_PER_SLICE
from model import calculate_metrics, create_model, load_and_preprocess_data

MODEL_DIR = os.environ.get("MODEL_DIR", "/app/trained_model")
VALIDATION_CSV = "datasets/community_data.csv"
SLICE_WAIT_SECONDS = 5 * 60  # wait for the next 5-minute data slice

logging.basicConfig(level=logging.INFO)


class SaveModelStrategy(fl.server.strategy.FedAvg):
    """FedAvg with server-side validation, early stopping, and checkpointing."""

    def __init__(self, fraction_fit, min_fit_clients, min_available_clients, on_fit_config_fn):
        super().__init__(
            fraction_fit=fraction_fit,
            min_fit_clients=min_fit_clients,
            min_available_clients=min_available_clients,
            on_fit_config_fn=on_fit_config_fn,
        )

        # Held-out validation data: last 20% of the community dataset.
        self.validation_data = load_and_preprocess_data(VALIDATION_CSV, validation=True)
        x_val = self.validation_data[0]
        if isinstance(x_val, list):  # Hybrid model: [sequence input, exogenous input]
            shapes = [x_val[0].shape[1:], x_val[1].shape[1:]]
        else:
            shapes = (x_val.shape[1],)
        logging.info(f"Global model input shapes: {shapes}")

        self.global_model = create_model(shapes)
        self.global_model.summary()

        self.metrics_history: List[Dict[str, float]] = []
        self.best_mse = float("inf")
        self.rounds_without_improvement = 0
        self.patience = 5

    def save_metrics_history(self) -> None:
        """Write the per-round validation metrics to metrics_history.csv."""
        os.makedirs(MODEL_DIR, exist_ok=True)
        pd.DataFrame(self.metrics_history).to_csv(
            os.path.join(MODEL_DIR, "metrics_history.csv"), index=False
        )

    def save_model_and_scalers(self, rnd: int) -> None:
        """Save the global model and the scalers used to normalise its inputs."""
        os.makedirs(MODEL_DIR, exist_ok=True)
        model_path = os.path.join(MODEL_DIR, "latest_model.h5")
        self.global_model.save(model_path)

        _, _, feature_scaler, target_scaler = self.validation_data
        joblib.dump(feature_scaler, os.path.join(MODEL_DIR, "feature_scaler.save"))
        joblib.dump(target_scaler, os.path.join(MODEL_DIR, "target_scaler.save"))
        logging.info(f"[Round {rnd}] Saved model and scalers to {MODEL_DIR}")

    def aggregate_fit(
        self,
        rnd: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[BaseException],
    ) -> Optional[Tuple[Parameters, Dict[str, Scalar]]]:
        """Aggregate client weights with FedAvg, then validate and checkpoint."""
        if not results:
            return None

        # 1) Weighted average of client weights (weight = number of local samples).
        client_weights, num_samples = [], []
        for _, fit_res in results:
            if fit_res.parameters is None:
                continue
            client_weights.append(fl.common.parameters_to_ndarrays(fit_res.parameters))
            num_samples.append(fit_res.num_examples)

        total_samples = sum(num_samples)
        new_weights = [np.zeros_like(layer) for layer in client_weights[0]]
        for weights, n in zip(client_weights, num_samples):
            for i, layer in enumerate(weights):
                new_weights[i] += layer * n
        new_weights = [layer / total_samples for layer in new_weights]
        self.global_model.set_weights(new_weights)

        # 2) Evaluate the new global model on the held-out community data.
        x_val, y_val, _, target_scaler = self.validation_data
        y_pred = self.global_model.predict(x_val)
        metrics = calculate_metrics(y_val, y_pred.flatten(), target_scaler)
        self.metrics_history.append({"round": rnd, **metrics})
        self.save_metrics_history()
        print(tabulate(
            [[rnd, metrics["mse"], metrics["rmse"], metrics["mae"],
              metrics["ma_percentage_error"], metrics["std_dev"]]],
            headers=["Round", "MSE", "RMSE", "MAE", "MAPE (%)", "Std dev"],
            floatfmt=".6f",
        ))

        # 3) Early stopping: once validation MSE has not improved for `patience`
        #    rounds, stop accepting new aggregates (the global model is frozen).
        if metrics["mse"] < self.best_mse:
            self.best_mse = metrics["mse"]
            self.rounds_without_improvement = 0
        else:
            self.rounds_without_improvement += 1
        if self.rounds_without_improvement >= self.patience:
            logging.info("Early stopping: validation MSE stopped improving")
            return None

        # 4) Checkpoint the model for the dashboard backend.
        self.save_model_and_scalers(rnd)
        return fl.common.ndarrays_to_parameters(new_weights), metrics


def fit_config(rnd: int) -> Dict[str, Scalar]:
    """Training config sent to clients; waits when a new data slice starts."""
    if rnd > 1 and (rnd - 1) % ROUNDS_PER_SLICE == 0:
        time.sleep(SLICE_WAIT_SECONDS)
    return {"epochs": EPOCHS, "batch_size": BATCH_SIZE, "round": rnd}


def main() -> None:
    strategy = SaveModelStrategy(
        fraction_fit=1.0,
        min_fit_clients=NUM_CLIENTS,
        min_available_clients=NUM_CLIENTS,
        on_fit_config_fn=fit_config,
    )
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
    )


if __name__ == "__main__":
    main()
