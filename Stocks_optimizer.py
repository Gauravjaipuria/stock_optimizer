import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor

# Streamlit UI
st.title("📈 AI-Powered Stock Portfolio Optimizer")

# User Inputs
selected_stocks = st.text_input("Enter stock symbols (comma-separated):").strip().upper().split(',')
selected_stocks = [stock.strip() for stock in selected_stocks if stock]

years_to_use = st.number_input("Enter number of years for historical data:", min_value=1, max_value=10, value=2)
forecast_days = st.number_input("Enter forecast period (in days):", min_value=1, max_value=365, value=30)
investment_amount = st.number_input("Enter total investment amount (₹):", min_value=1000.0, value=50000.0)
risk_profile = st.radio("Select your risk level:", [1, 2, 3], format_func=lambda x: {1: "Low", 2: "Medium", 3: "High"}[x])

# Initialize Storage
forecasted_prices = {}
volatilities = {}

# Process Each Stock
for stock in selected_stocks:
    df = yf.download(stock, period=f"{years_to_use}y", interval="1d", auto_adjust=True)

    if df.empty:
        st.warning(f"Skipping {stock}: No valid data available.")
        continue

    df = df[['Close']]
    df.dropna(inplace=True)
    
    future_dates = pd.date_range(df.index[-1], periods=forecast_days + 1, freq='B')[1:]

    # Calculate Moving Averages
    df['MA_50'] = df['Close'].rolling(window=50).mean()
    df['MA_200'] = df['Close'].rolling(window=200).mean()

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

    # Calculate Volatility
    volatilities[stock] = float(np.std(df['Close'].pct_change().dropna()))
    forecasted_prices[stock] = future_xgb[-1]

    # Plot Historical and Forecasted Prices
    st.subheader(f"📊 Forecast for {stock}")
    plt.figure(figsize=(14, 7))
    sns.set_style("darkgrid")
    plt.plot(df.index, df['Close'], label=f'{stock} Historical', linewidth=2, color='black')
    plt.plot(df.index, df['MA_50'], label='50-Day MA', linestyle='dashed', color='blue')
    plt.plot(df.index, df['MA_200'], label='200-Day MA', linestyle='dashed', color='purple')
    plt.plot(future_dates, future_xgb, label=f'{stock} Forecasted (XGBoost)', linestyle='dashed', color='red', marker='o')
    plt.plot(future_dates, future_rf, label=f'{stock} Forecasted (Random Forest)', linestyle='dashed', color='green', marker='x')
    plt.legend(fontsize=12, loc='upper left', frameon=True, shadow=True, fancybox=True)
    plt.title(f"Historical and Forecasted Prices for {stock}", fontsize=16, fontweight='bold')
    plt.xlabel("Date", fontsize=14)
    plt.ylabel("Close Price", fontsize=14)
    plt.xticks(rotation=45, fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    st.pyplot(plt)

# Portfolio Optimization
if forecasted_prices:
    risk_allocation = {1: 0.7, 2: 0.5, 3: 0.3}  # Low Risk: 70% Safe, Medium: 50%, High: 30%
    
    allocation = {}
    safe_stocks = []
    risky_stocks = []
    
    for stock, vol in volatilities.items():
        if vol > 0.03:
            risky_stocks.append(stock)
        else:
            safe_stocks.append(stock)

    # Investment split based on risk profile
    risky_allocation = investment_amount * risk_allocation[risk_profile]
    safe_allocation = investment_amount - risky_allocation

    # Distribute investment
    if risky_stocks:
        per_risky_stock = risky_allocation / len(risky_stocks)
        for stock in risky_stocks:
            allocation[stock] = per_risky_stock
    
    if safe_stocks:
        per_safe_stock = safe_allocation / len(safe_stocks)
        for stock in safe_stocks:
            allocation[stock] = per_safe_stock

    # Ensure total allocation sums to 100%
    total_allocation = sum(allocation.values())
    allocation_percentage = {stock: round((amount / total_allocation) * 100, 2) for stock, amount in allocation.items()}

    # Adjust first stock to correct rounding errors
    total_percentage = sum(allocation_percentage.values())
    if total_percentage != 100:
        first_stock = next(iter(allocation_percentage))
        allocation_percentage[first_stock] += 100 - total_percentage

    # Display Optimized Stock Allocation
    st.subheader("💰 Optimized Stock Allocation")
    allocation_df = pd.DataFrame.from_dict(allocation, orient='index', columns=['Investment Amount (₹)'])
    allocation_df["Percentage (%)"] = allocation_df.index.map(lambda stock: allocation_percentage[stock])
    st.table(allocation_df)

    # Risk Classification
    def classify_risk_level(volatility):
        volatility = float(volatility)
        if volatility > 0.03:
            return "3 (High Risk)"
        elif 0.01 < volatility <= 0.03:
            return "2 (Medium Risk)"
        else:
            return "1 (Low Risk)"

    risk_levels = {stock: classify_risk_level(vol) for stock, vol in volatilities.items()}
    risk_df = pd.DataFrame.from_dict(risk_levels, orient='index', columns=['Risk Level'])
    st.subheader("⚠️ Risk Levels in Investment")
    st.table(risk_df)
