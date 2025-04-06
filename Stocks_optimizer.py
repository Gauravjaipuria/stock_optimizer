# streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(page_title="AI Stock Optimizer", layout="wide")
st.title("📈 AI-Powered Stock Portfolio Optimizer")

# Sidebar Inputs
st.sidebar.header("User Inputs")
market = st.sidebar.selectbox("Select Market", ["India", "Other"])
symbols_input = st.sidebar.text_input("Enter stock symbols (comma-separated)", "BPCL,ROSSELLIND,RITES")
years = st.sidebar.slider("Historical Years", 1, 10, 3)
days_forecast = st.sidebar.slider("Forecast Days", 1, 365, 30)
total_amount = st.sidebar.number_input("Investment Amount (₹)", value=50000.0)
risk_level = st.sidebar.selectbox("Risk Profile", ["Low", "Medium", "High"])

risk_map = {"Low": 1, "Medium": 2, "High": 3}
risk_code = risk_map[risk_level]

symbols = [s.strip().upper() + (".NS" if market == "India" else "") for s in symbols_input.split(",")]

forecasted_prices, volatilities, trend_signals, allocation = {}, {}, {}, {}
charts = {}

risk_allocation_map = {1: 0.7, 2: 0.5, 3: 0.3}

for symbol in symbols:
    df = yf.download(symbol, period=f"{years}y", interval="1d", auto_adjust=True)
    if df.empty:
        continue

    df = df[['Close']]
    df['MA_50'] = df['Close'].rolling(50).mean()
    df['MA_200'] = df['Close'].rolling(200).mean()
    df['Lag_1'] = df['Close'].shift(1)
    df.dropna(inplace=True)

    trend_signals[symbol] = "Bullish 🟢 (Buy)" if df['MA_50'].iloc[-1] > df['MA_200'].iloc[-1] else "Bearish 🔴 (Sell)"

    train = df.iloc[:int(len(df)*0.8)]
    
    xgb = XGBRegressor(n_estimators=100)
    xgb.fit(train[['Lag_1']], train['Close'])

    rf = RandomForestRegressor(n_estimators=100)
    rf.fit(train[['Lag_1']], train['Close'])

    forecast_xgb = [xgb.predict([[df['Lag_1'].iloc[-1]]])[0] for _ in range(days_forecast)]
    forecast_rf = [rf.predict([[df['Lag_1'].iloc[-1]]])[0] for _ in range(days_forecast)]

    vol = np.std(df['Close'].pct_change().dropna())
    volatilities[symbol] = vol
    forecasted_prices[symbol] = {"XGBoost": forecast_xgb[-1], "RandomForest": forecast_rf[-1]}

    # Plotting
    future_dates = pd.date_range(df.index[-1], periods=days_forecast+1, freq='B')[1:]
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.set_style("whitegrid")
    ax.plot(df.index, df['Close'], label="Historical", color="black")
    ax.plot(df.index, df['MA_50'], label="50-DMA", linestyle='--', color="blue")
    ax.plot(df.index, df['MA_200'], label="200-DMA", linestyle='--', color="purple")
    ax.plot(future_dates, forecast_xgb, label="XGBoost Forecast", color="red", linestyle='dashed', marker='o')
    ax.plot(future_dates, forecast_rf, label="RF Forecast", color="green", linestyle='dashed', marker='x')
    ax.set_title(f"{symbol} Forecast & Moving Averages")
    ax.legend()
    charts[symbol] = fig

# Allocation Logic
safe_stocks = [s for s, v in volatilities.items() if v <= 0.03]
risky_stocks = [s for s, v in volatilities.items() if v > 0.03]

risky_amt = total_amount * risk_allocation_map[risk_code]
safe_amt = total_amount - risky_amt

for s in risky_stocks:
    allocation[s] = risky_amt / len(risky_stocks) if risky_stocks else 0
for s in safe_stocks:
    allocation[s] = safe_amt / len(safe_stocks) if safe_stocks else 0

# Display Results
st.subheader("📊 Allocation Summary")
alloc_df = pd.DataFrame({
    "Investment Amount (₹)": allocation,
    "Percentage Allocation (%)": {k: round(v/total_amount*100, 2) for k, v in allocation.items()}
})
st.dataframe(alloc_df)

st.subheader("📢 AI Trend Predictions")
trend_df = pd.DataFrame.from_dict(trend_signals, orient="index", columns=["Trend Signal"])
st.dataframe(trend_df)

st.subheader("🧠 Forecasted Prices (Last Prediction)")
forecast_df = pd.DataFrame(forecasted_prices).T
st.dataframe(forecast_df)

st.subheader("⚠️ Risk Levels")
risk_df = pd.DataFrame.from_dict({
    s: "3 (High Risk)" if v > 0.03 else "2 (Medium Risk)" if v > 0.01 else "1 (Low Risk)"
    for s, v in volatilities.items()
}, orient='index', columns=["Risk Level"])
st.dataframe(risk_df)

st.subheader("📈 Forecast Charts with Moving Averages")
for symbol, fig in charts.items():
    st.pyplot(fig)
