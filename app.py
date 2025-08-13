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
# Wind Speed Input (mph instead of m/s)
# ---------------------
wind_speed_mph = st.number_input("Enter wind speed (mph)", min_value=0.0, max_value=67.0, value=11.2, step=0.1)  
# Convert mph to m/s for formula
wind_speed_ms = wind_speed_mph / 2.23694

# ---------------------
# Energy Calculation
# ---------------------
k = 0.5
instantaneous_power = k * (wind_speed_ms ** 3)  # Watts

# Convert to kWh for a day, week, month
energy_day = instantaneous_power * 24 / 1000
energy_week = energy_day * 7
energy_month = energy_day * 30

# ---------------------
# Cool Speedometer (Plotly) with W label
# ---------------------
fig_speedometer = go.Figure(go.Indicator(
    mode="gauge+number",
    value=instantaneous_power,
    number={"suffix": " W"},  # add watts label
    title={"text": "Instantaneous Power"},
    gauge={"axis": {"range": [0, max(5000, instantaneous_power * 1.2)]}}
))
st.plotly_chart(fig_speedometer, use_container_width=True)

# ---------------------
# Odometer for Energy Produced
# ---------------------
fig_odometer = go.Figure(go.Indicator(
    mode="number",
    value=energy_day,
    title={"text": "Energy Produced Today"},
    number={"suffix": " kWh"}
))
st.plotly_chart(fig_odometer, use_container_width=True)

# ---------------------
# Fun Comparisons
# ---------------------
comparisons = [
    ("🍕 Pizza Oven", 12),
    ("🚲 Electric Bike", 0.5),
    ("🚗 Electric Car (100 miles)", 24.14),  # ~15 kWh per 100 km → converted to miles
    ("🏠 Average Home (Day)", 30),
]
st.subheader("What could you power?")
for item, usage in comparisons:
    st.write(f"- {item}: **{energy_day / usage:.1f}** times")

# ---------------------
# CO2 Prevention (lbs instead of kg)
# ---------------------
co2_per_kwh_kg = 0.92  # kg CO₂ per kWh
co2_per_kwh_lbs = co2_per_kwh_kg * 2.20462
co2_saved_lbs = energy_day * co2_per_kwh_lbs
st.subheader("🌍 CO₂ Emissions Prevented")
st.write(f"**{co2_saved_lbs:.2f} lbs** of CO₂ avoided today.")


