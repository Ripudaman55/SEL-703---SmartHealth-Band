import streamlit as st
import serial
import pandas as pd
import numpy as np
import time

# --- CONFIGURATION ---
PORT = "COM3" # Change this to your ESP32 port (e.g., COM4 or /dev/ttyUSB0)
BAUD = 115200

st.set_page_config(page_title="IoT Heart Monitor", layout="wide")
st.title("🫀 Live ECG Dashboard & ML Classifier")

# Placeholder for the chart and metrics
chart_placeholder = st.empty()
status_placeholder = st.empty()

# Initialize Serial
try:
    ser = serial.Serial(PORT, BAUD, timeout=1)
except:
    st.error(f"Could not open {PORT}. Check your connection!")
    st.stop()

data_buffer = []

# --- LIVE LOOP ---
while True:
    line = ser.readline().decode('utf-8').strip()
    if line.isdigit():
        val = int(line)
        data_buffer.append(val)
        
        # Keep only the last 100 points for the graph
        if len(data_buffer) > 100:
            data_buffer.pop(0)

        # 1. Update Graph
        chart_placeholder.line_chart(data_buffer)

        # 2. Simple ML Logic (Placeholder for your trained model)
        # For now, let's use a logic-based classifier
        if val > 3500:
            status_placeholder.warning("⚠️ High Spike Detected! (Stress/Movement)")
        elif val < 500:
            status_placeholder.error("🚨 Sensor Disconnected / Lead Off")
        else:
            status_placeholder.success("✅ Normal Heart Rhythm")

    time.sleep(0.01)