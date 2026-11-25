"""Flower client representing one building in the federation.

Each client keeps its building's data locally and only shares model
weights with the server. To simulate streaming smart-meter data, the
client trains on a new data slice every ROUNDS_PER_SLICE rounds.

Usage:
    python client/client.py <path/to/building.csv> <client_id>
"""

import logging
import sys

import flwr as fl
import pandas as pd

from config import BATCH_SIZE, EPOCHS, MODEL_TYPE, ROUNDS_PER_SLICE, SEQUENCE_LENGTH, SERVER_ADDRESS
from model.model import create_model, preprocess_one_row, train_model

logging.basicConfig(level=logging.INFO)


class EnergyClient(fl.client.NumPyClient):
    """A single building that trains the shared model on its private data."""

    def __init__(self, csv_path: str, client_id: str):
        self.client_id = client_id
        self.df = pd.read_csv(csv_path)
        # Input shapes: consumption history window + exogenous features
        # (all columns except "Time" and the target "Use [kW]").
        input_shapes = [(SEQUENCE_LENGTH, 1), (len(self.df.columns) - 2,)]
        self.model = create_model(input_shapes)

    def _get_slice(self, slice_idx: int):
        """Return the training data for one streaming time slice."""
        row = self.df.iloc[[slice_idx]]
        return preprocess_one_row(row, SEQUENCE_LENGTH, MODEL_TYPE)

    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        """Load the global weights, train locally, and return the update."""
        rnd = config.get("round", 1)
        slice_idx = (rnd - 1) // ROUNDS_PER_SLICE
        logging.info(f"[Client {self.client_id}] Round {rnd}: training on slice {slice_idx}")

        self.model.set_weights(parameters)
        x, y, _, _ = self._get_slice(slice_idx)
        self.model, _ = train_model(self.model, x, y, epochs=EPOCHS, batch_size=BATCH_SIZE)
        return self.model.get_weights(), len(y), {}

    def evaluate(self, parameters, config):
        """Evaluate the global model on this building's most recent reading."""
        self.model.set_weights(parameters)
        x, y, _, _ = self._get_slice(len(self.df) - 1)
        loss, mae = self.model.evaluate(x, y, verbose=0)
        return loss, len(y), {"mae": float(mae)}


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("Usage: python client/client.py <path/to/building.csv> <client_id>")
    fl.client.start_numpy_client(
        server_address=SERVER_ADDRESS,
        client=EnergyClient(sys.argv[1], sys.argv[2]),
    )
