import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(page_title="AI-Powered Stock Portfolio Optimizer", layout="wide")
st.title("📊 AI-Powered Stock Portfolio Optimizer")

# Sidebar inputs
country = st.sidebar.selectbox("Market", ["India", "other"])
stocks = st.sidebar.text_input("Enter Stock Symbols (comma-separated)", "BPCL, RITES")
years = st.sidebar.slider("Years of Historical Data", 1, 10, 3)
forecast_days = st.sidebar.slider("Forecast Period (Days)", 30, 365, 90)
investment = st.sidebar.number_input("Total Investment (₹)", value=50000.0)
risk_profile = st.sidebar.selectbox("Risk Profile", ["Low", "Medium", "High"])

# Risk level mapping
risk_map = {"Low": 1, "Medium": 2, "High": 3}
risk_level = risk_map[risk_profile]

# Prepare stock list
stock_list = [s.strip().upper() + ".NS" if country == "India" else s.strip().upper() for s in stocks.split(",")]

# Initialize containers
forecasted_prices = {}
volatilities = {}
trend_signals = {}
rf_forecasts = {}
xgb_forecasts = {}
actual_vs_predicted = {}

# Process each stock
for stock in stock_list:
    df = yf.download(stock, period=f"{years}y", interval="1d", auto_adjust=True)
    if df.empty:
        st.warning(f"Skipping {stock} due to lack of data.")
        continue

    df = df[['Close']].dropna()
    df['MA_50'] = df['Close'].rolling(window=50).mean()
    df['MA_200'] = df['Close'].rolling(window=200).mean()
    trend_signals[stock] = "Bullish (Buy)" if df['MA_50'].iloc[-1] > df['MA_200'].iloc[-1] else "Bearish (Sell)"

    df['Lag_1'] = df['Close'].shift(1)
    df.dropna(inplace=True)

    train_size = int(len(df) * 0.8)
    train, test = df.iloc[:train_size], df.iloc[train_size:]

    # Models
    xgb_model = XGBRegressor(objective='reg:squarederror', n_estimators=100)
    xgb_model.fit(train[['Lag_1']], train['Close'])
    xgb_pred = xgb_model.predict(test[['Lag_1']])
    future_xgb = [xgb_model.predict([[df['Lag_1'].iloc[-1]]])[0] for _ in range(forecast_days)]

    rf_model = RandomForestRegressor(n_estimators=100)
    rf_model.fit(train[['Lag_1']], train['Close'])
    rf_pred = rf_model.predict(test[['Lag_1']])
    future_rf = [rf_model.predict([[df['Lag_1'].iloc[-1]]])[0] for _ in range(forecast_days)]

    # Store results
    xgb_forecasts[stock] = xgb_pred[-1]
    rf_forecasts[stock] = rf_pred[-1]
    volatilities[stock] = float(np.std(df['Close'].pct_change().dropna()))
    forecasted_prices[stock] = {'XGBoost': future_xgb[-1], 'RandomForest': future_rf[-1]}
    actual_vs_predicted[stock] = (test['Close'], xgb_pred, rf_pred)

    # Plot
    future_dates = pd.date_range(df.index[-1], periods=forecast_days + 1, freq='B')[1:]
    plt.figure(figsize=(12, 6))
    plt.plot(df['Close'], label="Historical", color='black')
    plt.plot(df['MA_50'], label="50-Day MA", linestyle='--', color='blue')
    plt.plot(df['MA_200'], label="200-Day MA", linestyle='--', color='purple')
    plt.plot(future_dates, future_xgb, label="XGBoost Forecast", linestyle='--', color='red')
    plt.plot(future_dates, future_rf, label="RF Forecast", linestyle='--', color='green')
    plt.legend()
    plt.title(f"{stock} Price Forecast with MAs")
    st.pyplot(plt.gcf())
    plt.close()

# Classify risk levels
low_risk = []
medium_risk = []
high_risk = []

for stock, vol in volatilities.items():
    if vol <= 0.01:
        low_risk.append(stock)
    elif vol <= 0.03:
        medium_risk.append(stock)
    else:
        high_risk.append(stock)

st.subheader("🔍 Stock Risk Classification")
risk_classification_df = pd.DataFrame({
    "Stock": low_risk + medium_risk + high_risk,
    "Risk Category": (["Low"] * len(low_risk)) + (["Medium"] * len(medium_risk)) + (["High"] * len(high_risk))
})
st.dataframe(risk_classification_df)

st.subheader("💸 Portfolio Allocation Based on Risk")

allocation = {}

# If all stocks fall under one risk category, allocate full investment there
if len(low_risk) == len(volatilities):
    per_stock = investment / len(low_risk)
    for stock in low_risk:
        allocation[stock] = per_stock
elif len(medium_risk) == len(volatilities):
    per_stock = investment / len(medium_risk)
    for stock in medium_risk:
        allocation[stock] = per_stock
elif len(high_risk) == len(volatilities):
    per_stock = investment / len(high_risk)
    for stock in high_risk:
        allocation[stock] = per_stock
else:
    if risk_profile == "Low":
        risk_allocation = {1: 0.7, 2: 0.5, 3: 0.3}
        risky_allocation = investment * risk_allocation[risk_level]
        safe_allocation = investment - risky_allocation
    elif risk_profile == "Medium":  
        risk_allocation = {1: 0.5, 2: 0.7, 3: 0.3}
        risky_allocation = investment * risk_allocation[risk_level]
        safe_allocation = investment - risky_allocation
    else:
        risk_allocation = {1: 0.3, 2: 0.5, 3: 0.7}
        risky_allocation = investment * risk_allocation[risk_level]
        safe_allocation = investment - risky_allocation

    safe_stocks = low_risk + medium_risk if risk_level == 3 else low_risk
    risky_stocks = high_risk if risk_level == 3 else medium_risk + high_risk

    if risky_stocks:
        per_risky_stock = risky_allocation / len(risky_stocks)
        for stock in risky_stocks:
            allocation[stock] = per_risky_stock

    if safe_stocks:
        per_safe_stock = safe_allocation / len(safe_stocks)
        for stock in safe_stocks:
            allocation[stock] = per_safe_stock

# Calculate total allocation and percentage
total_allocation = sum(allocation.values())
alloc_percent = {stock: round((amount / total_allocation) * 100, 2) for stock, amount in allocation.items()}

# Display Allocation
st.subheader("💰 Optimized Stock Allocation (100% Distributed)")
alloc_df = pd.DataFrame.from_dict(allocation, orient='index', columns=['Investment Amount (₹)'])
alloc_df['Percentage Allocation (%)'] = alloc_df.index.map(lambda s: alloc_percent[s])
st.dataframe(alloc_df)

# Trend Signals
st.subheader("📢 AI Trend Predictions")
trend_df = pd.DataFrame.from_dict(trend_signals, orient='index', columns=['Trend Signal'])
st.dataframe(trend_df)

# Forecast Table
st.subheader("🧐 Forecasted Prices (Last Prediction)")
forecast_df = pd.DataFrame.from_dict(forecasted_prices, orient='index')
st.dataframe(forecast_df)

# Risk Levels
st.subheader("⚠️ Risk Levels in Investment")
risk_tiers = {s: "3 (High Risk)" if vol > 0.03 else "2 (Medium Risk)" if vol > 0.01 else "1 (Low Risk)" for s, vol in volatilities.items()}
risk_df = pd.DataFrame.from_dict(risk_tiers, orient='index', columns=['Risk Level'])
st.dataframe(risk_df)

# Sharpe Ratio & Returns
st.subheader("📉 Sharpe Ratio & Return Forecast")
sharpe_rows = []
risk_free_rate = 0.05
for stock in stock_list:
    df = yf.download(stock, period=f"{years}y", interval="1d", auto_adjust=True)
    df['Returns'] = df['Close'].pct_change()
    annual_return = df['Returns'].mean() * 252
    annual_volatility = df['Returns'].std() * np.sqrt(252)
    sharpe = (annual_return - risk_free_rate) / annual_volatility
    sharpe_rows.append([stock, round(annual_return, 4), round(annual_volatility, 4), round(sharpe, 4)])

sharpe_df = pd.DataFrame(sharpe_rows, columns=['Stock', 'Annual Return', 'Annual Volatility', 'Sharpe Ratio'])
st.dataframe(sharpe_df.set_index('Stock'))
