#!/usr/bin/env python3
# trainer/realtime_producer.py

import os
import time
import json
import pandas as pd
from kafka import KafkaProducer

# 1) configure
KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP", "kafka:9092")
TOPIC = os.getenv("KAFKA_TOPIC", "building-data")
DATA_DIR = "./datasets"
SLEEP_SECONDS = 5 * 60  # 5 minutes

# 2) start producer
producer = KafkaProducer(
    bootstrap_servers=KAFKA_BOOTSTRAP,
    value_serializer=lambda v: json.dumps(v).encode("utf-8"),
)

# 3) load all the future_* CSVs
building_data = {}
for fn in os.listdir(DATA_DIR):
    if fn.startswith("future_") and fn.endswith(".csv"):
        path = os.path.join(DATA_DIR, fn)
        df = pd.read_csv(path, parse_dates=["Time"])
        # normalize name: future_Hospital_data.csv → Hospital
        building = fn.replace("future_", "").replace("_data.csv", "")
        building_data[building] = df.to_dict("records")

# 4) stream one row per building every interval
pointers = {b: 0 for b in building_data}

while True:
    all_done = True
    for bldg, records in building_data.items():
        idx = pointers[bldg]
        if idx < len(records):
            rec = records[idx]
            # attach building name
            rec["building"] = bldg
            # send
            producer.send(TOPIC, rec)
            print(f"[{time.strftime('%H:%M:%S')}] → {bldg}  {rec['Time']}")
            pointers[bldg] += 1
            all_done = False
    producer.flush()
    if all_done:
        print("All future data exhausted, exiting.")
        break
    time.sleep(SLEEP_SECONDS)
