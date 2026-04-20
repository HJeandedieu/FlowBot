"""
FlowPay - subscriber.py
Subscribes to "flowpay/sensor", prints every message, and saves all
readings to data/readings.csv using pandas.

Run alongside simulator.py for 10-15 minutes to collect training data.
"""

import json
import os
import time
import pandas as pd
import paho.mqtt.client as mqtt

# CONFIG

BROKER = "localhost"
PORT = 1883
TOPIC = "flowpay/sensor"
CSV_PATH = os.path.join("data", "readings.csv")

# Check if path exists

os.makedirs("data",exist_ok=True)

# Write CSV header once if file doesn't exist yet

if not os.path.exists(CSV_PATH):
    pd.DataFrame(columns=["timestamp", "flow_lpm"]).to_csv(CSV_PATH, index=False)
    print(f"[Subscriber] Created {CSV_PATH}")

# Buffer - flush to CSV every N rows to reduce I/O

BUFFER: list[dict] = []
FLUSH_EVERY = 10    # write to disk after every 10 readings

def flush_buffer():
    if BUFFER:
        pd.DataFrame(BUFFER).to_csv(CSV_PATH, mode="a", header=False, index=False)
        print(f"[Subscriber] Saved {len(BUFFER)} rows to {CSV_PATH}")
        BUFFER.clear()


# callbacks

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print(f"[Subscriber] Connected to {BROKER}:{PORT}")
        client.subscribe(TOPIC)
        print(f"[Subscriber] Subscribed to '{TOPIC}' - waiting for messages ...\n")

    else:
        print(f"[Subscriber] connection failed (rc={rc})\n")

def on_message(client, userdata, msg):
    try:
        data = json.loads(msg.payload.decode())
        flow = data["flow_lpm"]
        ts = data['timestamp']

        # Label spikes for easy inspection in the csv

        label = "spike" if flow >= 3.5 else "normal"
        print(f"  flow={flow:.4f} L/min ts={ts:.2f} [{label}]")

        BUFFER.append({"timestamp": ts, "flow_lpm": flow, "label": label})

        if len(BUFFER) >= FLUSH_EVERY:
            flush_buffer()
    except (json.JSONDecodeError, KeyError) as e:
        print(f"[Subscriber] Bad message: {e}")


#  MQTT setup
client = mqtt.Client(client_id="flowpay-subscriber")
client.on_connect = on_connect
client.on_message = on_message
client.connect(BROKER, PORT, keepalive=60)

try:
    client.loop_forever()
except KeyboardInterrupt:
    print("\n [Subscriber] Stopping....")
finally:
    flush_buffer()
    client.disconnect()
    print(f"[Subscriber] All data saved to {CSV_PATH}")
    print(f"[Subscriber Total rows: {sum(1 for _ in open(CSV_PATH)) - 1}")