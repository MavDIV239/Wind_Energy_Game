# app.py — Wind Energy Game (kiosk-friendly; CO₂ in lbs; EV in miles; kWh fixed)
from typing import Dict
import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# -----------------------------
# Page & styling (kiosk-friendly)
# -----------------------------
st.set_page_config(page_title="Wind Energy Game", page_icon="💨", layout="wide")
st.markdown(
    """
<style>
:root { --base: 1.15rem; }
html, body, [class*="css"] { font-size: var(--base); }
h1 { font-size: 2.2rem; }
.block-container { padding-top: 1rem; }
footer {visibility: hidden;}
</style>
""",
    unsafe_allow_html=True,
)

# -----------------------------
# Defaults / factors (editable by staff)
# -----------------------------
DEFAULTS = dict(
    rated_kw=10.0,          # sets gauge max (visual only)
    home_kwh_per_day=30.0,  # avg US home usage
    ev_kwh_per_mile=0.27,   # kWh per mile (EV efficiency) -> miles = kWh / 0.27
    ebike_batt_kwh=0.5,     # e-bike full charge
    pizza_oven_kw=6.0,      # pizza oven draw (kW)
    co2_lbs_per_kwh=0.88,   # CO₂ avoided per kWh, in pounds (≈ 0.4 kg/kWh)
)

# “What you could power” catalog (kWh per unit) — qty = energy_kwh / kwh_per_unit
POWER_ITEMS = [
    # (key, nice_name, kwh_per_unit, unit_label, image_path, emoji_fallback)
    ("pizza",  "Pizza oven (hours)",    6.0,   "hours",      "images/pizza_oven.png",   "🍕"),
    ("ebike",  "E-bike full charges",   0.5,   "charges",    "images/ebike.png",        "🚲"),
    ("console","Game console (hours)",  0.15,  "hours",      "images/game_console.png", "🎮"),
    ("ev",     "EV driving (miles)",    0.27,  "miles",      "images/ev.png",           "🚗"),
    ("house",  "Average home (days)",   30.0,  "days",       "images/house.png",        "🏠"),
    ("bulb",   "LED bulbs (10 hrs)",    0.10,  "bulbs×10h",  "images/lightbulb.png",    "💡"),
]

DURATIONS = {"Day": 1, "Week": 7, "Month": 30}

# -----------------------------
# Data loading — combine all sheets
# -----------------------------
@st.cache_data(show_spinner=False)
def load_all_sheets(file) -> pd.DataFrame:
    xl = pd.ExcelFile(file)
    frames = []
    for name in xl.sheet_names:
        df = xl.parse(name)
        df.columns = [c.strip().lower() for c in df.columns]
        rename_map = {}
        for c in list(df.columns):
            cl = c.lower()
            if cl == "date" or cl.startswith("date ") or "utc" in cl:
                rename_map[c] = "date"
            if "energy" in cl and "kwh" in cl:
                rename_map[c] = "energy_kwh"
            if cl.startswith("min"):
                rename_map[c] = "min_w"
            if cl.startswith("max"):
                rename_map[c] = "max_w"
            if cl.startswith("avg") and "watt" in cl:
                rename_map[c] = "avg_w"
            if ("average wind speed" in cl or "avg wind speed" in cl) and "mph" in cl:
                rename_map[c] = "wind_mph"
        df = df.rename(columns=rename_map)
        keep = [c for c in ["date","energy_kwh","min_w","max_w","avg_w","wind_mph"] if c in df.columns]
        if not keep:
            continue
        df = df[keep].copy()
        # need wind + avg power to build power curve
        need = [col for col in ["wind_mph","avg_w"] if col in df.columns]
        if need:
            df = df.dropna(subset=need)
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    all_df = pd.concat(frames, ignore_index=True)
    if "wind_mph" in all_df.columns:
        all_df = all_df.sort_values("wind_mph").reset_index(drop=True)
    return all_df

# -----------------------------
# Interpolators (power & energy)
# -----------------------------
def build_interpolators(df: pd.DataFrame):
    """
    Returns:
      f_avg(v_mph)    -> Avg Power (W)
      f_min(v_mph)    -> Min Power (W)    (fallback to avg)
      f_max(v_mph)    -> Max Power (W)    (fallback to avg)
      f_energy(v_mph) -> Daily Energy (kWh) from data, fallback to 24 * avg_w/1000
    """
    if df.empty or "wind_mph" not in df.columns:
        return (lambda v: 0.0,)*4

    x = df["wind_mph"].to_numpy(float)
    # ensure strictly increasing for np.interp
    for i in range(1, len(x)):
        if x[i] <= x[i-1]:
            x[i] = x[i-1] + 1e-6

    def _interp(col: str, fallback: np.ndarray | None = None):
        if col in df.columns:
            y = df[col].to_numpy(float)
        else:
            y = fallback if fallback is not None else np.zeros_like(x, dtype=float)
        def f(v_mph: float) -> float:
            v = float(v_mph)
            return float(np.interp(v, x, y, left=y[0], right=y[-1]))
        return f

    y_avg = df["avg_w"].to_numpy(float) if "avg_w" in df.columns else np.zeros_like(x, dtype=float)

    f_avg = _interp("avg_w")
    f_min = _interp("min_w", fallback=y_avg)
    f_max = _interp("max_w", fallback=y_avg)

    if "energy_kwh" in df.columns:
        f_energy = _interp("energy_kwh")
    else:
        # fallback daily energy = 24 * (avg W / 1000)
        f_energy = lambda v_mph: 24.0 * (f_avg(v_mph) / 1000.0)

    return f_avg, f_min, f_max, f_energy

# -----------------------------
# Kiosk keypad (no physical keyboard needed)
# -----------------------------
def keypad_input(
    label: str,
    unit: str,
    *,
    min_val: float,
    max_val: float,
    default: float,
    step: float = 0.5,
    state_key: str = "wind_keypad",
):
    """
    Touch-friendly keypad + presets + +/- nudges. Returns a float.
    IMPORTANT: Stores its value in st.session_state[val_key] ONLY.
               The visible field uses a different key and is read-only.
    """
    import math

    # Internal storage (NOT a widget key)
    val_key = f"{state_key}_val"         # stores the editable text (string)
    disp_key = f"{state_key}_display"    # read-only text_input key

    # Initialize storage
    if val_key not in st.session_state:
        st.session_state[val_key] = f"{default:.1f}"

    def get_txt() -> str:
        return st.session_state.get(val_key, f"{default:.1f}")

    def set_txt(txt: str):
        st.session_state[val_key] = txt

    def clamp_num(v: float) -> float:
        try:
            v = float(v)
            if math.isnan(v) or math.isinf(v):
                return default
        except Exception:
            return default
        return min(max(v, min_val), max_val)

    # Presets
    presets = [5, 12, 20, 30, 40, 50] if unit == "mph" else [2, 5, 9, 13, 18, 22]

    st.markdown(f"**{label}**")
    ctop = st.columns(len(presets))
    for i, p in enumerate(presets):
        with ctop[i]:
            if st.button(f"{p:g} {unit}", key=f"preset_{unit}_{p}"):
                set_txt(f"{p:g}")

    # Display + +/- nudges (display is read-only so we don't fight Streamlit)
    row = st.columns([3, 1, 1])
    with row[0]:
        st.text_input(
            "Tap numbers or use presets",
            value=get_txt(),
            key=disp_key,
            disabled=True,
            label_visibility="collapsed",
        )
    with row[1]:
        if st.button(f"+{step:g}", key=f"plus_{state_key}"):
            try:
                v = float(get_txt()) + step
            except Exception:
                v = default
            set_txt(f"{clamp_num(v):.1f}")
    with row[2]:
        if st.button(f"−{step:g}", key=f"minus_{state_key}"):
            try:
                v = float(get_txt()) - step
            except Exception:
                v = default
            set_txt(f"{clamp_num(v):.1f}")

    # Keypad grid
    r1 = st.columns(3); r2 = st.columns(3); r3 = st.columns(3); r4 = st.columns(3)
    for lbl, col in zip(("7","8","9"), r1):
        if col.button(lbl, key=f"{state_key}_b{lbl}"):
            txt = get_txt()
            if txt == "0": txt = ""
            set_txt(txt + lbl)
    for lbl, col in zip(("4","5","6"), r2):
        if col.button(lbl, key=f"{state_key}_b{lbl}"):
            txt = get_txt()
            if txt == "0": txt = ""
            set_txt(txt + lbl)
    for lbl, col in zip(("1","2","3"), r3):
        if col.button(lbl, key=f"{state_key}_b{lbl}"):
            txt = get_txt()
            if txt == "0": txt = ""
            set_txt(txt + lbl)
    if r4[0].button("0", key=f"{state_key}_b0"):
        txt = get_txt()
        if txt == "0": txt = ""
        set_txt(txt + "0")
    if r4[1].button(".", key=f"{state_key}_bdot"):
        txt = get_txt()
        if "." not in txt:
            set_txt(txt + ".")
    if r4[2].button("⌫", key=f"{state_key}_bdel"):
        txt = get_txt()
        set_txt(txt[:-1] if len(txt) > 1 else "0")

    # Clear / Set
    b = st.columns(2)
    if b[0].button("Clear", key=f"{state_key}_clear"):
        set_txt("0")

    # Compute chosen numeric value (clamped), and normalize on Set
    try:
        chosen = clamp_num(float(get_txt()))
    except Exception:
        chosen = default

    if b[1].button(f"Set {unit.upper()}", key=f"{state_key}_set"):
        set_txt(f"{chosen:.1f}")

    return chosen

# -----------------------------
# Visual components (Plotly)
# -----------------------------
def gauge(power_kw: float, rated_kw: float) -> go.Figure:
    max_kw = max(rated_kw, power_kw * 1.25, 0.5)
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=power_kw,
        number={"valueformat": ",.2f", "font": {"size": 42}},
        title={"text": "Instant Power (kW)", "font": {"size": 20}},
        delta={"reference": max_kw/2, "increasing": {"color": "#16a34a"}},
        gauge={
            "axis": {"range": [0, max_kw]},
            "bar": {"thickness": 0.25},
            "steps": [
                {"range": [0, max_kw*0.33], "color": "#d1fae5"},
                {"range": [max_kw*0.33, max_kw*0.66], "color": "#a7f3d0"},
                {"range": [max_kw*0.66, max_kw], "color": "#6ee7b7"},
            ],
        },
    ))
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=30, b=10))
    return fig

def odometer(kwh: float, title: str = "Energy (kWh)") -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="number",
        value=kwh,
        number={"valueformat": ",.2f", "font": {"size": 46}},
        title={"text": title, "font": {"size": 20}},
    ))
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=30, b=10))
    return fig

def co2_bubbles(lbs: float) -> go.Figure:
    n = 20
    rng = np.random.default_rng(42)
    x = rng.uniform(0, 1, n); y = rng.uniform(0, 1, n)
    size = (np.cbrt(max(lbs, 0.001)) * 4.5) * rng.uniform(0.6, 1.4, n)
    fig = go.Figure(go.Scatter(
        x=x, y=y, mode="markers",
        marker=dict(size=size, opacity=0.55),
        hovertext=[f"CO₂ avoided: {lbs:,.1f} lbs"]*n, hoverinfo="text"
    ))
    fig.update_xaxes(visible=False); fig.update_yaxes(visible=False)
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=30, b=10),
                      title={"text": "CO₂ Avoided (lbs)", "x": 0.5, "font": {"size": 20}})
    return fig

# -----------------------------
# “What you could power” cards (with images)
# -----------------------------
def item_cards(energy_kwh: float):
    cols = st.columns(3)
    for i, (key, nice, kwh_per_unit, unit_label, img_path, emoji) in enumerate(POWER_ITEMS):
        qty = 0.0 if kwh_per_unit <= 0 else energy_kwh / kwh_per_unit
        with cols[i % 3]:
            if os.path.exists(img_path):
                st.image(img_path, use_container_width=True)
            else:
                st.markdown(f"<div style='font-size:64px;text-align:center;'>{emoji}</div>", unsafe_allow_html=True)
            st.markdown(f"**{nice}**")
            st.markdown(f"{qty:,.1f} {unit_label}")

# -----------------------------
# Sidebar (staff controls)
# -----------------------------
st.sidebar.header("Setup (staff)")
uploaded = st.sidebar.file_uploader("Upload Excel (.xlsx) with wind/energy", type=["xlsx"])
rated_kw = st.sidebar.number_input("Gauge rated power (kW)", 0.5, 500.0, DEFAULTS["rated_kw"], 0.5)
co2_lbs_per_kwh = st.sidebar.number_input("CO₂ avoided (lbs/kWh)", 0.0, 5.0, DEFAULTS["co2_lbs_per_kwh"], 0.05)
home_kwh_per_day = st.sidebar.number_input("Avg home use (kWh/day)", 5.0, 100.0, DEFAULTS["home_kwh_per_day"], 1.0)
ev_kwh_per_mile = st.sidebar.number_input("EV use (kWh/mile)", 0.10, 0.60, DEFAULTS["ev_kwh_per_mile"], 0.01)
ebike_batt_kwh = st.sidebar.number_input("E-bike battery (kWh)", 0.2, 2.0, DEFAULTS["ebike_batt_kwh"], 0.1)
pizza_oven_kw = st.sidebar.number_input("Pizza oven power (kW)", 1.0, 20.0, DEFAULTS["pizza_oven_kw"], 0.5)
cfg = dict(home_kwh_per_day=home_kwh_per_day, ev_kwh_per_mile=ev_kwh_per_mile,
           ebike_batt_kwh=ebike_batt_kwh, pizza_oven_kw=pizza_oven_kw,
           co2_lbs_per_kwh=co2_lbs_per_kwh)

# -----------------------------
# Main UI
# -----------------------------
st.title("💨 Wind Energy Game")
st.caption("Tap a wind speed — see instant power, data-driven energy (kWh), and what you could power!")

# Data (upload or demo)
if uploaded is not None:
    df = load_all_sheets(uploaded)
else:
    df = pd.DataFrame({
        "wind_mph": [5, 10, 15, 20, 25, 30],
        "avg_w":   [50, 200, 600, 1500, 3000, 4500],
        "min_w":   [0, 0,  50,  100,  200,  300],
        "max_w":   [200, 800, 2000, 5000, 9000, 11000],
        "energy_kwh": [2, 6, 14, 28, 55, 85],  # demo daily kWh so the app runs without a file
    })

f_avg, f_min, f_max, f_energy = build_interpolators(df)

# Controls (touch keypad + slider)
c1, c2, c3 = st.columns([1, 2, 1])
with c2:
    unit = st.toggle("Use m/s (off = mph)")
    if unit:
        wind_val = keypad_input("Wind speed (m/s)", "m/s", min_val=0.0, max_val=30.0, default=9.0, step=0.2, state_key="keypad_ms")
        wind_val = st.slider("or drag: m/s", 0.0, 30.0, float(wind_val), 0.1, key="slider_ms")
        wind_mph = wind_val * 2.236936
        speed_label = f"{wind_val:.1f} m/s"
    else:
        wind_val = keypad_input("Wind speed (mph)", "mph", min_val=0.0, max_val=60.0, default=20.0, step=0.5, state_key="keypad_mph")
        wind_val = st.slider("or drag: mph", 0.0, 60.0, float(wind_val), 0.5, key="slider_mph")
        wind_mph = wind_val
        speed_label = f"{wind_val:.1f} mph"

# Compute with dataset-driven interpolators
p_avg_w = max(0.0, f_avg(wind_mph))
p_min_w = max(0.0, f_min(wind_mph))
p_max_w = max(0.0, f_max(wind_mph))
p_avg_kw = p_avg_w / 1000.0
energy_day_kwh = max(0.0, f_energy(wind_mph))  # daily kWh from data (or fallback)

# Speedometer (kW) + caption with W values
st.plotly_chart(gauge(p_avg_kw, rated_kw), use_container_width=True)
st.caption(f"Instant power at **{speed_label}** — Min: {p_min_w:.0f} W · Avg: {p_avg_w:.0f} W · Max: {p_max_w:.0f} W")

# Tabs for Day / Week / Month — all in kWh, CO₂ in lbs
tabs = st.tabs(list(DURATIONS.keys()))
for tab_name, tab in zip(DURATIONS.keys(), tabs):
    with tab:
        multiplier = DURATIONS[tab_name]
        energy_kwh = energy_day_kwh * multiplier
        co2_lbs = energy_kwh * cfg["co2_lbs_per_kwh"]

        cA, cB, cC = st.columns([1, 1, 1])
        with cA:
            st.plotly_chart(odometer(energy_kwh, title=f"Energy ({tab_name}) — kWh"), use_container_width=True)
        with cB:
            st.plotly_chart(co2_bubbles(co2_lbs), use_container_width=True)
            st.caption(f"**CO₂ avoided** ≈ {co2_lbs:,.1f} lbs")
        with cC:
            st.markdown("**Quick facts**")
            st.write(f"- Daily energy at {speed_label}: **{energy_day_kwh:.2f} kWh/day**")
            st.write(f"- Period: **{tab_name}** × {multiplier} day(s)")

        st.subheader("What could you power? ✨")
        item_cards(energy_kwh)

# Celebration
if p_avg_kw >= max(1.0, rated_kw * 0.5):
    st.balloons()
