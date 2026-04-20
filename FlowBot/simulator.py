"""
FlowPay - simulator.py

Publishes simulated water flow readings to MQTT topic "flowpay/sensor"
    -90% normal flow: 0.4 - 0.8 L/min
    -80% spike (leak): 3.5 - 8.0 L/min
One reading per second, payload: {"flow_lpm": float, "timestamp": float}
"""

import json
import random
import time
import paho.mqtt.client as mqtt

# CONFIG

BROKER = "localhost"
PORT = 1883
TOPIC = "flowpay/sensor"
INTERVAL = 1 # seconds between readings

# MQTT SETUP

client = mqtt.Client(client_id="flowpay-simulator")
client.connect(BROKER, PORT, keepalive=60)
client.loop_start()

print(f"[Simulator] Connected to {BROKER}:{PORT} - publishing to '{TOPIC}' ...")
print(f"[Simulator] Press Ctrl+C to stop.\n")

try:
    while True:
        # 10% chance of a spike (leak/ burst pipe)
        if random.random() < 0.10:
            flow = round(random.uniform(3.5, 8.0),4)
            kind = "SPIKE"
        else:
            flow = round(random.uniform(0.4, 0.8), 4)
            kind = "normal"

        payload = json.dumps({
            "flow_lpm": flow,
            "timestamp": time.time()
        })

        client.publish(TOPIC, payload)
        print(f"[{kind:6s}] Published: {payload}")
        time.sleep(INTERVAL)
except KeyboardInterrupt:
    print("\n[Simulator] Stopped.")

finally:
    client.loop_stop()
    client.disconnect()