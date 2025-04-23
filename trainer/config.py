# # config.py
# SEQUENCE_LENGTH = 72
# EPOCHS = 3
# BATCH_SIZE = 32
# MODEL_TYPE = "Hybrid"
# NUM_CLIENTS = 6
# SERVER_ADDRESS = "server:8080"  # Use the service name from docker-compose.yaml
# LEARNING_RATE = 0.001
# NUM_ROUNDS = 2
# config.py
SEQUENCE_LENGTH   = 72
EPOCHS            = 3
BATCH_SIZE        = 32
MODEL_TYPE        = "Hybrid"
NUM_CLIENTS       = 6
SERVER_ADDRESS    = "server:8080"
LEARNING_RATE     = 0.001

# “How many federated rounds per 5-min slice”
ROUNDS_PER_SLICE  = 2
# Federated rounds per 5-min slice\ROUNDS_PER_SLICE = 2
# If each future_*.csv has N rows, total = 2 * N
NUM_ROUNDS        = ROUNDS_PER_SLICE * 10