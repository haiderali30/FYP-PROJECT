"""Shared configuration for the federated learning server and clients."""

# Length of the consumption history window (hours) fed to the LSTM branch.
SEQUENCE_LENGTH = 72

# Local training settings used by every client in each round.
EPOCHS = 3
BATCH_SIZE = 32
LEARNING_RATE = 0.001

# Model architecture: "Hybrid" (LSTM + exogenous features), "LSTM",
# "Transformer", "MLP", or "XGBoost". See model/model.py.
MODEL_TYPE = "Hybrid"

# Federation settings.
NUM_CLIENTS = 6                 # one client per building
SERVER_ADDRESS = "server:8080"  # docker-compose service name of the server

# Streaming simulation: each new 5-minute data slice is used for
# ROUNDS_PER_SLICE federated rounds before the clients move to the next slice.
ROUNDS_PER_SLICE = 2
NUM_SLICES = 10
NUM_ROUNDS = ROUNDS_PER_SLICE * NUM_SLICES
