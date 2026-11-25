#!/usr/bin/env python3
"""Stream "future" smart-meter readings to Kafka, one row per building every 5 minutes.

Reads datasets/future_<Building>_data.csv and publishes each row as JSON to
the KAFKA_TOPIC topic, simulating live data arriving from each building.
"""

import os
import time
import json
import pandas as pd
from kafka import KafkaProducer

KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP", "kafka:9092")
TOPIC = os.getenv("KAFKA_TOPIC", "building-data")
DATA_DIR = "./datasets"
SLEEP_SECONDS = 5 * 60  # 5 minutes

producer = KafkaProducer(
    bootstrap_servers=KAFKA_BOOTSTRAP,
    value_serializer=lambda v: json.dumps(v).encode("utf-8"),
)

# Load every future_*.csv file, keyed by building name
building_data = {}
for fn in os.listdir(DATA_DIR):
    if fn.startswith("future_") and fn.endswith(".csv"):
        path = os.path.join(DATA_DIR, fn)
        df = pd.read_csv(path, parse_dates=["Time"])
        # normalize name: future_Hospital_data.csv → Hospital
        building = fn.replace("future_", "").replace("_data.csv", "")
        building_data[building] = df.to_dict("records")

# Publish one row per building per interval until all data is sent
pointers = {b: 0 for b in building_data}

while True:
    all_done = True
    for bldg, records in building_data.items():
        idx = pointers[bldg]
        if idx < len(records):
            rec = records[idx]
            rec["building"] = bldg
            producer.send(TOPIC, rec)
            print(f"[{time.strftime('%H:%M:%S')}] → {bldg}  {rec['Time']}")
            pointers[bldg] += 1
            all_done = False
    producer.flush()
    if all_done:
        print("All future data exhausted, exiting.")
        break
    time.sleep(SLEEP_SECONDS)
