streamlit #-----------------------------
# streamlit_app.py
#-----------------------------

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from datetime import timedelta

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="Delhi Air Alert", page_icon="🫁", layout="wide")

# -----------------------------
# AQI conversion (PM2.5 -> AQI) - US EPA breakpoints
# -----------------------------
BP = [
    (0.0, 12.0, 0, 50),
    (12.1, 35.4, 51, 100),
    (35.5, 55.4, 101, 150),
    (55.5, 150.4, 151, 200),
    (150.5, 250.4, 201, 300),
    (250.5, 350.4, 301, 400),
    (350.5, 500.4, 401, 500),
]

def pm25_to_aqi(pm):
    if pm is None or (isinstance(pm, float) and np.isnan(pm)):
        return np.nan
    pm = float(pm)
    pm = max(0.0, min(pm, 500.4))
    for Cl, Ch, Il, Ih in BP:
        if Cl <= pm <= Ch:
            return int(round((Ih - Il) / (Ch - Cl) * (pm - Cl) + Il))
    return 500

def aqi_label(aqi):
    if np.isnan(aqi): return "Unknown"
    if aqi <= 50: return "Good ✅"
    if aqi <= 100: return "Moderate 🙂"
    if aqi <= 150: return "Unhealthy (Sensitive) ⚠️"
    if aqi <= 200: return "Unhealthy ❌"
    if aqi <= 300: return "Very Unhealthy 🚫"
    return "Hazardous ☠️"

def citizen_advice(pm_next, threshold=200):
    if pm_next is None or (isinstance(pm_next, float) and np.isnan(pm_next)):
        return False, ["Not enough info to forecast (missing features)."]

    unsafe = pm_next >= threshold
    if unsafe:
        advice = [
            "✅ Stay indoors if possible",
            "✅ If going outside: wear N95/N99 mask",
            "✅ Avoid outdoor exercise",
            "✅ Close windows / consider air purifier",
            "✅ Keep inhaler/meds ready (if applicable)"
        ]
    else:
        advice = [
            "✅ Normal activity is okay",
            "✅ Sensitive people: monitor symptoms",
            "✅ Carry a mask just in case"
        ]
    return unsafe, advice

# -----------------------------
# Load files (cached)
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("delhi_ml.csv")
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)

@st.cache_resource
def load_model():
    return joblib.load("rfr_model.pkl")

@st.cache_data
def load_features():
    with open("reg_features.txt", "r") as f:
        return [line.strip() for line in f if line.strip()]

# -----------------------------
# winter pattern: 
# -----------------------------
def seasonal_bias_delhi(d: pd.Timestamp) -> float:
    """
    Original simple seasonal bias.
    """
    d = pd.to_datetime(d)
    m = d.month

    winter = 16.0 if m in [11, 12, 1, 2] else 0.0
    monsoon = -12.0 if m in [7, 8] else 0.0
    shoulder = 12.0 if m in [3, 4, 9] else 0.0
    bump = 10.0 if m in [10, 11] else 0.0

    return float(winter + monsoon + shoulder + bump)

def estimate_daily_sigma(hist_df: pd.DataFrame, anchor_date: pd.Timestamp, lookback_days: int = 180) -> float:
    """
    Estimate typical daily variability from historical day-to-day differences.
    """
    recent = hist_df[hist_df["date"] <= anchor_date].copy().sort_values("date").tail(lookback_days)
    if len(recent) < 30:
        return 10.0
    diffs = recent["pm25"].astype(float).diff().dropna()
    s = float(diffs.std())
    if np.isnan(s) or s <= 0:
        return 10.0
    return s

# -----------------------------
# Recursive forecasting
# -----------------------------
def recompute_pm25_engineering(hist_pm25: pd.DataFrame) -> pd.Series:
    h = hist_pm25.copy().sort_values("date")
    h["pm25_lag1"]  = h["pm25"].shift(1)
    h["pm25_lag7"]  = h["pm25"].shift(7)
    h["pm25_lag14"] = h["pm25"].shift(14)

    h["pm25_roll3"]  = h["pm25"].rolling(3).mean()
    h["pm25_roll7"]  = h["pm25"].rolling(7).mean()
    h["pm25_roll14"] = h["pm25"].rolling(14).mean()
    h["pm25_std7"]   = h["pm25"].rolling(7).std()

    return h.tail(1).iloc[0]

def smoothstep01(x: float) -> float:
    """
    Smooth interpolation from 0→1.
    Used to avoid sudden seasonal jumps.
    """
    x = float(np.clip(x, 0.0, 1.0))
    return x * x * (3 - 2 * x)

def simulate_pm25_until(df_base: pd.DataFrame,
                        model,
                        reg_features: list,
                        start_from_date: pd.Timestamp,
                        end_date: pd.Timestamp,
                        max_days: int = 3650):
    if end_date <= start_from_date:
        return {}

    days_needed = (end_date.date() - start_from_date.date()).days
    if days_needed > max_days:
        return {}

    hist = df_base[["date", "pm25"]].copy().sort_values("date").reset_index(drop=True)
    template = df_base.tail(1).copy()

    preds = {}
    current = pd.to_datetime(start_from_date)

    # Keep your “physics” guardrail
    max_change_per_day = 15.0  # tighten if you want less sudden moves

    # If you are using wiggle, keep it deterministic
    rng = np.random.default_rng(42)
    sigma_d = estimate_daily_sigma(hist, hist["date"].max(), lookback_days=180)
    wiggle_sigma = 0.10 * sigma_d

    for _ in range(days_needed):
        nxt = current + pd.Timedelta(days=1)

        # calendar features if present
        if "date" in template.columns:
            template.loc[:, "date"] = nxt
        if "month" in template.columns:
            template.loc[:, "month"] = nxt.month
        if "dayofweek" in template.columns:
            template.loc[:, "dayofweek"] = nxt.dayofweek
        if "is_winter" in template.columns:
            template.loc[:, "is_winter"] = int(nxt.month in [11, 12, 1, 2])

        # recompute pm25 engineered features
        last_eng = recompute_pm25_engineering(hist)
        for c in ["pm25_lag1","pm25_lag7","pm25_lag14",
                  "pm25_roll3","pm25_roll7","pm25_roll14","pm25_std7"]:
            if c in template.columns:
                template.loc[:, c] = float(last_eng.get(c, np.nan))

        X = template[reg_features].copy()
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

        raw_pred = float(model.predict(X)[0])

        # 1) Rate limiter (prevents crazy jumps)
        last_pm = float(hist["pm25"].iloc[-1])
        delta = raw_pred - last_pm
        delta = float(np.clip(delta, -max_change_per_day, max_change_per_day))
        pm_pred = last_pm + delta

        # 2) Seasonal bias — SAME VALUES, but SMOOTH transition across month boundaries
        bias_today = seasonal_bias_delhi(nxt)

        # Smooth Feb→Mar (winter 18 → shoulder 6) over first 10 days of March
        if nxt.month == 3 and nxt.day <= 10:
            t = (nxt.day - 1) / 9.0  # 0..1
            w = smoothstep01(t)
            winter_bias = 18.0
            shoulder_bias = 6.0
            bias_today = (1 - w) * winter_bias + w * shoulder_bias

        # Smooth Oct→Nov (shoulder 6 → winter 18) over last 10 days of Oct
        if nxt.month == 10 and nxt.day >= 22:
            t = (nxt.day - 22) / 9.0  # 0..1
            w = smoothstep01(t)
            shoulder_bias = 6.0
            winter_bias = 18.0
            # (still allows your bump rule to work via seasonal_bias_delhi overall)
            bias_today = (1 - w) * shoulder_bias + w * winter_bias

        pm_pred = pm_pred + float(bias_today)

        # 3) Small wiggle (optional, realistic daily bumps)
        wiggle = float(rng.normal(0, wiggle_sigma))
        wiggle = float(np.clip(wiggle, -0.7 * max_change_per_day, 0.7 * max_change_per_day))
        pm_pred = pm_pred + wiggle

        # 4) Final limiter again after bias+wiggle
        last_pm2 = float(hist["pm25"].iloc[-1])
        delta2 = pm_pred - last_pm2
        delta2 = float(np.clip(delta2, -max_change_per_day, max_change_per_day))
        pm_pred = last_pm2 + delta2

        pm_pred = float(np.clip(pm_pred, 0.0, 500.4))

        preds[nxt.date()] = pm_pred
        hist = pd.concat([hist, pd.DataFrame({"date": [nxt], "pm25": [pm_pred]})], ignore_index=True)
        current = nxt

    return preds

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("⚙️ Settings")
threshold = st.sidebar.slider("Unsafe threshold PM2.5 (µg/m³)", 20, 250, 100, step=5)
timeline_window = st.sidebar.selectbox("Timeline window (days)", [60, 90, 120, 180, 240, 270, 300, 330, 365], index=1)
max_sim_days = st.sidebar.selectbox("Max future forecasting range (days)", [30, 90, 180, 365, 730], index=3)
st.sidebar.caption("Tip: Very far future forecasts may be slow or unreliable. Keep within the max range.")

# -----------------------------
# Load data/model/features
# -----------------------------
df = load_data()
model = load_model()
REG_FEATURES = load_features()
st.write(df["date"].min(), df["date"].max())

missing = [c for c in REG_FEATURES if c not in df.columns]
if missing:
    st.error(f"Your delhi_ml.csv is missing these regression features:\n{missing}")
    st.stop()

min_date = df["date"].min().date()
last_data_date = df["date"].max().date()
ui_max_date = last_data_date + timedelta(days=3650)

# -----------------------------
# Title
# -----------------------------
st.markdown("## 🫁 Delhi Air Alert — Citizen Prototype Dashboard")
st.caption("Select any date → see PM2.5 (actual if available, otherwise forecast) → next 2 days forecast → Safe/Unsafe + advice.")

# -----------------------------
# Date selection (calendar)
# -----------------------------
selected = st.date_input(
    "📅 Choose a day (past or future)",
    value=last_data_date,
    min_value=min_date,
    max_value=ui_max_date
)

selected_dt = pd.to_datetime(selected)
day0 = selected
day1 = (selected_dt + pd.Timedelta(days=1)).date()
day2 = (selected_dt + pd.Timedelta(days=2)).date()

# -----------------------------
# Selected day PM2.5 (actual or forecast)
# -----------------------------
selected_is_actual = False
pm0 = np.nan

if selected <= last_data_date:
    row0 = df[df["date"].dt.date == selected].tail(1)
    if not row0.empty:
        pm0 = float(row0["pm25"].iloc[0])
        selected_is_actual = True
    else:
        idx = (df["date"] - selected_dt).abs().idxmin()
        pm0 = float(df.loc[idx, "pm25"])
        selected_is_actual = True

# -----------------------------
# Forecast PM(t+1), PM(t+2) from selected day
# -----------------------------
forecast_needed_end = pd.to_datetime(day2)

if selected <= last_data_date:
    df_base = df[df["date"].dt.date <= selected].copy()
    start_from = pd.to_datetime(selected)
else:
    df_base = df.copy()
    start_from = pd.to_datetime(last_data_date)

sim_days_required = (forecast_needed_end.date() - start_from.date()).days
if sim_days_required > max_sim_days:
    st.warning(
        f"You selected {selected}, which requires simulating {sim_days_required} days. "
        f"Your max simulation is {max_sim_days} days. Increase it in sidebar OR choose a nearer date."
    )
    preds = {}
else:
    preds = simulate_pm25_until(
        df_base=df_base,
        model=model,
        reg_features=REG_FEATURES,
        start_from_date=start_from,
        end_date=forecast_needed_end,
        max_days=max_sim_days
    )

if selected > last_data_date:
    pm0 = float(preds.get(day0, np.nan))

pm1 = float(preds.get(day1, np.nan))
pm2 = float(preds.get(day2, np.nan))

# -----------------------------
# AQI + Decision
# -----------------------------
aqi0 = pm25_to_aqi(pm0)
aqi1 = pm25_to_aqi(pm1)
aqi2 = pm25_to_aqi(pm2)

unsafe, advice = citizen_advice(pm1, threshold)

badge = "🟩 SAFE ✅" if not unsafe else "🟥 UNSAFE ⚠️"
badge_color = "#10B981" if not unsafe else "#EF4444"

# -----------------------------
# Top Cards
# -----------------------------
st.markdown("---")
c1, c2, c3, c4 = st.columns(4)

with c1:
    st.subheader(f"Selected Day: {day0}")
    if selected_is_actual:
        st.metric("PM2.5 (Actual)", "N/A" if np.isnan(pm0) else f"{pm0:.1f}")
    else:
        st.metric("PM2.5 (Forecast)", "N/A" if np.isnan(pm0) else f"{pm0:.1f}")
        st.caption("Future date → forecasted value (not actual).")
    st.write(f"AQI: {aqi0} → {aqi_label(aqi0)}")

with c2:
    st.subheader(f"Forecast: {day1} (PM(t+1))")
    st.metric("PM2.5", "N/A" if np.isnan(pm1) else f"{pm1:.1f}")
    st.write(f"AQI: {aqi1} → {aqi_label(aqi1)}")

with c3:
    st.subheader(f"Forecast: {day2} (PM(t+2))")
    st.metric("PM2.5", "N/A" if np.isnan(pm2) else f"{pm2:.1f}")
    st.write(f"AQI: {aqi2} → {aqi_label(aqi2)}")

with c4:
    st.subheader("Citizen Decision")
    st.markdown(
        f"""
        <div style="padding:12px;border-radius:12px;background:{badge_color};color:white;font-size:18px;font-weight:700;text-align:center;">
            {badge}
        </div>
        """,
        unsafe_allow_html=True
    )
    st.write(f"Threshold = {threshold} µg/m³")
    st.write("What you should do:")
    for a in advice:
        st.write(a)

# -----------------------------
# Timeline Plot (FORECAST ONLY) — 
# -----------------------------
st.markdown("---")
st.subheader("📈 PM2.5 Timeline (Forecast Only)")

anchor_date = pd.to_datetime(selected) if selected <= last_data_date else pd.to_datetime(last_data_date)

forecast_days = int(timeline_window)
forecast_end_date = anchor_date + pd.Timedelta(days=forecast_days)

if selected <= last_data_date:
    df_base_timeline = df[df["date"].dt.date <= selected].copy()
else:
    df_base_timeline = df.copy()

preds_timeline = simulate_pm25_until(
    df_base=df_base_timeline,
    model=model,
    reg_features=REG_FEATURES,
    start_from_date=anchor_date,
    end_date=forecast_end_date,
    max_days=max_sim_days
)

if not preds_timeline:
    st.warning("No forecast produced (horizon too far or max_sim_days too low).")
    st.stop()

forecast_timeline = (
    pd.DataFrame({
        "date": [pd.to_datetime(d) for d in preds_timeline.keys()],
        "pm25": list(preds_timeline.values())
    })
    .sort_values("date")
    .reset_index(drop=True)
)

# Very light smoothing only (keeps realistic bumps)
forecast_timeline["pm25_smooth"] = (
    forecast_timeline["pm25"]
    .rolling(window=3, min_periods=1)
    .mean()
)

fig, ax = plt.subplots(figsize=(12, 4))

ax.plot(
    forecast_timeline["date"],
    forecast_timeline["pm25"],
    linestyle="--",
    linewidth=2.2,
    label=f"Forecast (next {forecast_days} days)"
)

ax.axhline(threshold, linestyle="--", linewidth=2, label=f"Threshold ({threshold})")

anchor_row = df[df["date"] <= anchor_date].tail(1)
if len(anchor_row) > 0:
    ax.scatter(anchor_row["date"], anchor_row["pm25"], s=70, label="Anchor (last actual)")

ax.set_xlabel("Date")
ax.set_ylabel("PM2.5 (µg/m³)")
ax.set_title("PM2.5: Forward Forecast Only ")
ax.legend()

st.pyplot(fig)

# -----------------------------
# Distribution Plot (FULL HISTORICAL ACTUAL DATA ONLY)
# -----------------------------
st.markdown("---")
st.subheader("📊 PM2.5 Distribution ")

view = df.copy()

fig2 = plt.figure(figsize=(12, 4))
plt.hist(view["pm25"].dropna(), bins=35)
plt.axvline(
    threshold,
    linestyle="--",
    color="red",
    linewidth=2,
    label=f"DANGER POINT = {threshold}"
)
plt.title("How often PM2.5 crosses unsafe level (full historical data)")
plt.xlabel("PM2.5 (µg/m³)")
plt.ylabel("Count")
plt.legend()
st.pyplot(fig2)

st.markdown("---")
st.caption(
    "Prototype: shows actual values when available; forecasts future dates using recursive PM(t+1) simulation from your trained RandomForest model. "
    "A fixed rate limiter prevents unrealistic day-to-day spikes/drops. A seasonal bias models Delhi winter rise and monsoon dip. "
    "Controlled wiggle is added using historical volatility for realistic daily bumps. Only light 3-day smoothing is used for display."
)