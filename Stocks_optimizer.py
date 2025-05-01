import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor

# Streamlit Inputs
country = st.selectbox("Select Market", ("India", "America"))
selected_stocks = st.text_input("Enter stock symbols (comma-separated)").upper().split(',')
years_to_use = st.slider("Enter number of years for historical data (1-10)", 1, 10, 5)
forecast_days = st.slider("Enter forecast period in days (1-365)", 1, 365, 30)
investment_amount = st.number_input("Enter total investment amount (₹)", min_value=0.0)
risk_profile = st.radio("Select your risk level", ("Low", "Medium", "High"))

# Convert to Yahoo Finance format
selected_stocks = [stock.strip() + ".NS" if country.lower() == "india" else stock.strip() for stock in selected_stocks]

forecasted_prices = {}
volatilities = {}
trend_signals = {}

for stock in selected_stocks:
    df = yf.download(stock, period=f"{years_to_use}y", interval="1d", auto_adjust=True)
    if df.empty:
        st.write(f"Skipping {stock}: No valid data available.")
        continue

    df = df[['Close']]
    df.dropna(inplace=True)
    future_dates = pd.date_range(df.index[-1], periods=forecast_days + 1, freq='B')[1:]

    # Moving Averages
    df['MA_50'] = df['Close'].rolling(window=50).mean()
    df['MA_200'] = df['Close'].rolling(window=200).mean()
    trend_signals[stock] = "Bullish 🟢 (Buy)" if df['MA_50'].iloc[-1] > df['MA_200'].iloc[-1] else "Bearish 🔴 (Sell)"

    # Feature Engineering
    df['Lag_1'] = df['Close'].shift(1)
    df.dropna(inplace=True)
    train_size = int(len(df) * 0.8)
    train, test = df.iloc[:train_size], df.iloc[train_size:]

    # XGBoost Model
    xgb_model = XGBRegressor(objective='reg:squarederror', n_estimators=100)
    xgb_model.fit(train[['Lag_1']], train['Close'])
    future_xgb = [xgb_model.predict(np.array([[df['Lag_1'].iloc[-1]]]).reshape(1, -1))[0] for _ in range(forecast_days)]

    # Random Forest Model
    rf_model = RandomForestRegressor(n_estimators=100)
    rf_model.fit(train[['Lag_1']], train['Close'])
    future_rf = [rf_model.predict(np.array([[df['Lag_1'].iloc[-1]]]).reshape(1, -1))[0] for _ in range(forecast_days)]

    # Store Results
    volatilities[stock] = float(np.std(df['Close'].pct_change().dropna()))
    forecasted_prices[stock] = future_xgb[-1]

    # Plot each stock separately
    plt.figure(figsize=(14, 7))
    sns.set_style("darkgrid")
    plt.plot(df.index, df['Close'], label=f'{stock} Historical', linewidth=2, color='black')
    plt.plot(df.index, df['MA_50'], label='50-Day MA', linestyle='dashed', color='blue')
    plt.plot(df.index, df['MA_200'], label='200-Day MA', linestyle='dashed', color='purple')
    plt.plot(future_dates, future_xgb, label=f'{stock} Forecasted (XGBoost)', linestyle='dashed', color='red', marker='o')
    plt.plot(future_dates, future_rf, label=f'{stock} Forecasted (Random Forest)', linestyle='dashed', color='green', marker='x')
    plt.legend(fontsize=12, loc='upper left')
    plt.title(f"Historical and Forecasted Prices for {stock}", fontsize=16, fontweight='bold')
    plt.xlabel("Date", fontsize=14)
    plt.ylabel("Close Price", fontsize=14)
    plt.xticks(rotation=45, fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)

    # Show plot in Streamlit
    st.pyplot()

# Display AI-Based Trend Predictions
trend_df = pd.DataFrame.from_dict(trend_signals, orient='index', columns=['Trend Signal'])
st.write("📢 AI Trend Predictions")
st.write(trend_df)

# Portfolio Optimization
risk_allocation = {1: 0.7, 2: 0.5, 3: 0.3}
allocation = {}
safe_stocks = []
risky_stocks = []

for stock, vol in volatilities.items():
    (risky_stocks if vol > 0.03 else safe_stocks).append(stock)

# Handle empty safe/risky cases
if not risky_stocks:
    risky_allocation = 0
    safe_allocation = investment_amount
elif not safe_stocks:
    safe_allocation = 0
    risky_allocation = investment_amount
else:
    risky_allocation = investment_amount * risk_allocation[risk_profile]
    safe_allocation = investment_amount - risky_allocation

# Allocate to risky stocks
if risky_stocks:
    per_risky_stock = risky_allocation / len(risky_stocks)
    for stock in risky_stocks:
        allocation[stock] = per_risky_stock

# Allocate to safe stocks
if safe_stocks:
    per_safe_stock = safe_allocation / len(safe_stocks)
    for stock in safe_stocks:
        allocation[stock] = per_safe_stock

# Calculate total allocation and percentage
total_allocation = sum(allocation.values())
allocation_percentage = {stock: round((amount / total_allocation) * 100, 2) for stock, amount in allocation.items()}

st.write("💰 Optimized Stock Allocation")
allocation_df = pd.DataFrame.from_dict(allocation, orient='index', columns=['Investment Amount (₹)'])
allocation_df["Percentage (%)"] = allocation_df.index.map(lambda stock: allocation_percentage[stock])
st.write(allocation_df)

# Risk Levels
risk_levels = {stock: "3 (High Risk)" if vol > 0.03 else "2 (Medium Risk)" if 0.01 < vol <= 0.03 else "1 (Low Risk)" for stock, vol in volatilities.items()}
risk_df = pd.DataFrame.from_dict(risk_levels, orient='index', columns=['Risk Level'])
st.write("⚠️ Risk Levels in Investment")
st.write(risk_df)
