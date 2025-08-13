import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# ---------------------
# Load Data
# ---------------------
@st.cache_data
def load_data():
    df = pd.read_excel("Jen_Pier_Wind_Data_Energy.xlsx", sheet_name=0)
    return df

df = load_data()

# ---------------------
# Page Setup
# ---------------------
st.set_page_config(page_title="Wind Energy Game", layout="wide")
st.title("💨 Wind Energy Game")
st.write("Enter a wind speed to see how much energy you can produce and how much CO₂ you can prevent!")

# ---------------------
# Wind Speed Input
# ---------------------
wind_speed = st.number_input("Enter wind speed (m/s)", min_value=0.0, max_value=30.0, value=5.0, step=0.1)

# ---------------------
# Energy Calculation
# ---------------------
# Example formula: Power = k * v³ (scaled to your dataset)
k = 0.5
instantaneous_power = k * (wind_speed ** 3)  # Watts

# Convert to kWh for a day, week, month
energy_day = instantaneous_power * 24 / 1000
energy_week = energy_day * 7
energy_month = energy_day * 30

# ---------------------
# Cool Speedometer (Plotly)
# ---------------------
fig_speedometer = go.Figure(go.Indicator(
    mode="gauge+number",
    value=instantaneous_power,
    title={"text": "Instantaneous Power (W)"},
    gauge={"axis": {"range": [0, max(5000, instantaneous_power * 1.2)]}}
))
st.plotly_chart(fig_speedometer, use_container_width=True)

# ---------------------
# Odometer for Energy Produced
# ---------------------
fig_odometer = go.Figure(go.Indicator(
    mode="number",
    value=energy_day,
    title={"text": "Energy Produced Today (kWh)"},
    number={"suffix": " kWh"}
))
st.plotly_chart(fig_odometer, use_container_width=True)

# ---------------------
# Fun Comparisons
# ---------------------
comparisons = [
    ("🍕 Pizza Oven", 12),
    ("🚲 Electric Bike", 0.5),
    ("🚗 Electric Car (100 km)", 15),
    ("🏠 Average Home (Day)", 30),
]
st.subheader("What could you power?")
for item, usage in comparisons:
    st.write(f"- {item}: **{energy_day / usage:.1f}** times")

# ---------------------
# CO2 Prevention
# ---------------------
co2_per_kwh = 0.92  # kg CO₂ per kWh (approx US grid)
co2_saved = energy_day * co2_per_kwh
st.subheader("🌍 CO₂ Emissions Prevented")
st.write(f"**{co2_saved:.2f} kg** of CO₂ avoided today.")

