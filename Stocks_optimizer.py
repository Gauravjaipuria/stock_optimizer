import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(layout="wide")
st.title("📈 AI-Powered Stock Portfolio Optimizer")

# --- User Inputs ---
country = st.radio("Select Market:", ["India", "Other"])
selected_stocks = st.text_input("Enter stock symbols (comma-separated):").strip().upper().split(',')
selected_stocks = [stock.strip() + ".NS" if country == "India" else stock.strip() for stock in selected_stocks if stock]

years_to_use = st.number_input("Enter number of years for historical data:", min_value=1, max_value=10, value=2)
forecast_days = st.number_input("Enter forecast period (in days):", min_value=1, max_value=365, value=30)
investment_amount = st.number_input("Enter total investment amount (₹):", min_value=1000.0, value=50000.0)
risk_profile = st.radio("Select your risk level:", [1, 2, 3], format_func=lambda x: {1: "Low", 2: "Medium", 3: "High"}[x])

# --- Storage Init ---
forecasted_prices, volatilities, trend_signals, latest_prices = {}, {}, {}, {}
forecast_table = []

for stock in selected_stocks:
    df = yf.download(stock, period=f"{years_to_use}y", interval="1d", auto_adjust=True)
    if df.empty:
        st.warning(f"Skipping {stock}: No valid data available.")
        continue

    try:
        today_df = yf.download(stock, period="1d", interval="1d", auto_adjust=True)
        latest_prices[stock] = today_df['Close'].iloc[-1]
    except:
        st.warning(f"Couldn't fetch latest price for {stock}")
        latest_prices[stock] = np.nan

    df = df[['Close']].dropna()
    future_dates = pd.date_range(df.index[-1], periods=forecast_days + 1, freq='B')[1:]

    df['MA_50'] = df['Close'].rolling(window=50).mean()
    df['MA_200'] = df['Close'].rolling(window=200).mean()
    trend_signals[stock] = "Bullish 🟢 (Buy)" if df['MA_50'].iloc[-1] > df['MA_200'].iloc[-1] else "Bearish 🔴 (Sell)"

    df['Lag_1'] = df['Close'].shift(1)
    df.dropna(inplace=True)

    train_size = int(len(df) * 0.8)
    train = df.iloc[:train_size]

    xgb_model = XGBRegressor(objective='reg:squarederror', n_estimators=100)
    xgb_model.fit(train[['Lag_1']], train['Close'])
    future_xgb = [xgb_model.predict(np.array([[df['Lag_1'].iloc[-1]]]).reshape(1, -1))[0] for _ in range(forecast_days)]

    rf_model = RandomForestRegressor(n_estimators=100)
    rf_model.fit(train[['Lag_1']], train['Close'])
    future_rf = [rf_model.predict(np.array([[df['Lag_1'].iloc[-1]]]).reshape(1, -1))[0] for _ in range(forecast_days)]

    volatilities[stock] = float(np.std(df['Close'].pct_change().dropna()))
    forecasted_prices[stock] = {"XGBoost": future_xgb[-1], "RandomForest": future_rf[-1]}

    forecast_table.append({
        "Stock": stock,
        "Latest Price": round(latest_prices[stock], 2),
        "XGBoost": round(future_xgb[-1], 2),
        "RandomForest": round(future_rf[-1], 2)
    })

    st.subheader(f"📊 Forecast for {stock}")
    plt.figure(figsize=(14, 7))
    sns.set_style("darkgrid")
    plt.plot(df.index, df['Close'], label=f'{stock} Historical', linewidth=2)
    plt.plot(df.index, df['MA_50'], label='50-Day MA', linestyle='dashed', color='blue')
    plt.plot(df.index, df['MA_200'], label='200-Day MA', linestyle='dashed', color='purple')
    plt.plot(future_dates, future_xgb, label='XGBoost Forecast', linestyle='dashed', color='red', marker='o')
    plt.plot(future_dates, future_rf, label='RandomForest Forecast', linestyle='dashed', color='green', marker='x')

    try:
        z = np.polyfit(range(len(df)), df['Close'].values.flatten(), 1)
        p = np.poly1d(z)
        plt.plot(df.index, p(range(len(df))), "--", label='Trend Line', color='orange')
    except:
        st.warning(f"Trend line could not be plotted for {stock}.")

    plt.legend()
    plt.title(f"Historical and Forecasted Prices for {stock}")
    plt.xticks(rotation=45)
    st.pyplot(plt)

# --- Forecast Table ---
if forecast_table:
    st.subheader("🧙 Forecasted Prices (Last Prediction)")
    forecast_df = pd.DataFrame(forecast_table).set_index("Stock")
    st.table(forecast_df)

# --- Risk Allocation ---
if forecasted_prices:
    risk_splits = {
        1: {"Low": 0.7, "Medium": 0.2, "High": 0.1},
        2: {"Low": 0.3, "Medium": 0.4, "High": 0.3},
        3: {"Low": 0.1, "Medium": 0.2, "High": 0.7},
    }

    risk_buckets = {"Low": [], "Medium": [], "High": []}
    risk_levels = {}

    for stock, vol in volatilities.items():
        if vol > 0.03:
            risk_buckets["High"].append(stock)
            risk_levels[stock] = "3 (High Risk)"
        elif vol > 0.01:
            risk_buckets["Medium"].append(stock)
            risk_levels[stock] = "2 (Medium Risk)"
        else:
            risk_buckets["Low"].append(stock)
            risk_levels[stock] = "1 (Low Risk)"

    # Normalize splits based on available buckets
    available = {k: v for k, v in risk_buckets.items() if v}
    total_ratio = sum([risk_splits[risk_profile][k] for k in available])
    adjusted_splits = {k: risk_splits[risk_profile][k] / total_ratio for k in available}

    allocation = {}
    for level, stocks in available.items():
        amt = investment_amount * adjusted_splits[level]
        if stocks:
            per_stock = amt / len(stocks)
            for stock in stocks:
                allocation[stock] = per_stock

    total_alloc = sum(allocation.values())
    allocation_percentage = {stock: round((amt / total_alloc) * 100, 4) for stock, amt in allocation.items()}

    st.subheader("💰 Diversified Allocation by Risk Profile")
    alloc_df = pd.DataFrame.from_dict(allocation, orient='index', columns=['Investment Amount (₹)'])
    alloc_df["Percentage Allocation (%)"] = alloc_df.index.map(lambda s: allocation_percentage[s])
    st.table(alloc_df)

    st.subheader("⚠️ Risk Levels in Investment")
    risk_df = pd.DataFrame.from_dict(risk_levels, orient='index', columns=['Risk Level'])
    st.table(risk_df)

    st.subheader("📢 AI Trend Predictions")
    trend_df = pd.DataFrame.from_dict(trend_signals, orient='index', columns=['Trend Signal'])
    st.table(trend_df)
