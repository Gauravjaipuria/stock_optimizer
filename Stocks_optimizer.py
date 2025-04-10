# Full Streamlit app with improved forecasting and Sharpe ratio
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from ta.momentum import RSIIndicator
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="AI-Powered Stock Portfolio Optimizer", layout="wide")
st.title("📊 AI-Powered Stock Portfolio Optimizer")

# Sidebar
country = st.sidebar.selectbox("Market", ["India", "other"])
stocks = st.sidebar.text_input("Enter Stock Symbols (comma-separated)", "BPCL, RITES")
years = st.sidebar.slider("Years of Historical Data", 1, 10, 3)
forecast_days = st.sidebar.slider("Forecast Period (Days)", 30, 365, 90)
investment = st.sidebar.number_input("Total Investment (₹)", value=50000.0)
risk_profile = st.sidebar.selectbox("Risk Profile", ["Low", "Medium", "High"])

# Risk mapping
risk_map = {"Low": 1, "Medium": 2, "High": 3}
risk_level = risk_map[risk_profile]

# Format stock list
stock_list = [s.strip().upper() + ".NS" if country == "India" else s.strip().upper() for s in stocks.split(",")]

# Containers
forecasted_prices = {}
volatilities = {}
trend_signals = {}
xgb_forecasts = {}
rf_forecasts = {}
actual_vs_predicted = {}

for stock in stock_list:
    df = yf.download(stock, period=f"{years}y", interval="1d", auto_adjust=True)
    if df.empty or df['Close'].isnull().all():
        st.warning(f"Skipping {stock} due to lack of data.")
        continue

    df = df[['Close']].dropna()
    df['Return'] = df['Close'].pct_change()
    df['MA7'] = df['Close'].rolling(window=7).mean()
    df['MA14'] = df['Close'].rolling(window=14).mean()
    df['Volatility'] = df['Return'].rolling(window=14).std()
    df.dropna(inplace=True)

    # Label (y) is Close, Features (X) are dynamic
    X = df[['Return', 'MA7', 'MA14', 'Volatility']]
    y = df['Close']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train/test split
    train_size = int(len(X_scaled) * 0.8)
    X_train, X_test = X_scaled[:train_size], X_scaled[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Models
    xgb = XGBRegressor(n_estimators=100, objective='reg:squarederror')
    xgb.fit(X_train, y_train)
    xgb_pred = xgb.predict(X_test)

    rf = RandomForestRegressor(n_estimators=100)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)

    # Forecast future by replicating last known values
    last_features = df[['Return', 'MA7', 'MA14', 'Volatility']].iloc[-1].values.reshape(1, -1)
    last_features_scaled = scaler.transform(last_features)
    future_xgb = [xgb.predict(last_features_scaled)[0] for _ in range(forecast_days)]
    future_rf = [rf.predict(last_features_scaled)[0] for _ in range(forecast_days)]

    # Volatility
    volatilities[stock] = df['Return'].std()

    # Store forecasts
    forecasted_prices[stock] = {'XGBoost': future_xgb[-1], 'RandomForest': future_rf[-1]}
    trend_signals[stock] = "Bullish (Buy)" if df['MA7'].iloc[-1] > df['MA14'].iloc[-1] else "Bearish (Sell)"
    actual_vs_predicted[stock] = (y_test, xgb_pred, rf_pred)

    # Plot
    future_dates = pd.date_range(df.index[-1], periods=forecast_days + 1, freq='B')[1:]
    plt.figure(figsize=(12, 6))
    plt.plot(df['Close'], label="Historical", color='black')
    plt.plot(future_dates, future_xgb, label="XGBoost Forecast", linestyle='--', color='red')
    plt.plot(future_dates, future_rf, label="RF Forecast", linestyle='--', color='green')
    plt.title(f"{stock} Forecast")
    plt.legend()
    st.pyplot(plt.gcf())
    plt.close()

# RSI
st.subheader("📈 RSI Analysis")
rsi_rows = []
for stock in stock_list:
    df_rsi = yf.download(stock, period=f"{years}y", interval="1d", auto_adjust=True)
    rsi = RSIIndicator(close=df_rsi['Close'], window=14).rsi().dropna()
    latest_rsi = rsi.iloc[-1]
    signal = "Oversold (Buy)" if latest_rsi < 30 else "Overbought (Sell)" if latest_rsi > 70 else "Neutral"
    rsi_rows.append({"Stock": stock, "RSI": round(latest_rsi, 2), "Signal": signal})

st.dataframe(pd.DataFrame(rsi_rows))

# Risk Classification
low, med, high = [], [], []
for s, v in volatilities.items():
    if v <= 0.01:
        low.append(s)
    elif v <= 0.03:
        med.append(s)
    else:
        high.append(s)

st.subheader("🔍 Risk Classification")
st.dataframe(pd.DataFrame({
    "Stock": low + med + high,
    "Risk Category": ["Low"] * len(low) + ["Medium"] * len(med) + ["High"] * len(high)
}))

# Allocation Strategy
allocation = {}
if len(low) == len(volatilities):
    per_stock = investment / len(low)
    for stock in low:
        allocation[stock] = per_stock
elif len(med) == len(volatilities):
    per_stock = investment / len(med)
    for stock in med:
        allocation[stock] = per_stock
elif len(high) == len(volatilities):
    per_stock = investment / len(high)
    for stock in high:
        allocation[stock] = per_stock
else:
    r_alloc = {1: 0.3, 2: 0.5, 3: 0.7}
    risky = investment * r_alloc[risk_level]
    safe = investment - risky
    safe_stocks = low + med if risk_level == 3 else low
    risky_stocks = high if risk_level == 3 else med + high
    if risky_stocks:
        per = risky / len(risky_stocks)
        for s in risky_stocks:
            allocation[s] = per
    if safe_stocks:
        per = safe / len(safe_stocks)
        for s in safe_stocks:
            allocation[s] = per

st.subheader("💰 Optimized Portfolio Allocation")
alloc_df = pd.DataFrame.from_dict(allocation, orient='index', columns=["Investment ₹"])
alloc_df['% Allocation'] = round((alloc_df["Investment ₹"] / investment) * 100, 2)
st.dataframe(alloc_df)

# AI Trend
st.subheader("📢 AI Trend Signals")
st.dataframe(pd.DataFrame.from_dict(trend_signals, orient='index', columns=["Trend"]))

# Forecast Table
st.subheader("🧐 Forecasted Prices")
st.dataframe(pd.DataFrame.from_dict(forecasted_prices, orient='index'))

# Sharpe Ratio
st.subheader("📉 Sharpe Ratio & Return Forecast")
sharpe_data = []
risk_free = 0.05
for stock in stock_list:
    df = yf.download(stock, period=f"{years}y", interval="1d", auto_adjust=True)
    df['Return'] = df['Close'].pct_change()
    annual_return = df['Return'].mean() * 252
    annual_vol = df['Return'].std() * np.sqrt(252)
    sharpe = (annual_return - risk_free) / annual_vol if annual_vol != 0 else 0
    sharpe_data.append([stock, round(annual_return, 4), round(annual_vol, 4), round(sharpe, 4)])

st.dataframe(pd.DataFrame(sharpe_data, columns=["Stock", "Annual Return", "Volatility", "Sharpe Ratio"]).set_index("Stock"))
