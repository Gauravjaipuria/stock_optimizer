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
country = st.radio("Select Market:", ["India", "Other"])
selected_stocks = st.text_input("Enter stock symbols (comma-separated):").strip().upper().split(',')
selected_stocks = [stock.strip() + ".NS" if country == "India" else stock.strip() for stock in selected_stocks if stock]

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

    # Feature Engineering
    df['Lag_1'] = df['Close'].shift(1)
    df.dropna(inplace=True)

    # AI Model Training
    train_size = int(len(df) * 0.8)
    train, test = df.iloc[:train_size], df.iloc[train_size:]

    xgb_model = XGBRegressor(objective='reg:squarederror', n_estimators=100)
    xgb_model.fit(train[['Lag_1']], train['Close'])
    
    forecast_price = xgb_model.predict(np.array([[df['Lag_1'].iloc[-1]]]).reshape(1, -1))[0]

    # Calculate Volatility
    volatilities[stock] = float(np.std(df['Close'].pct_change().dropna()))
    forecasted_prices[stock] = forecast_price

# Portfolio Allocation (Ensuring 100% Distribution)
if forecasted_prices:
    weights = np.array([1 / len(forecasted_prices)] * len(forecasted_prices))  # Equal weight distribution
    weights = weights / np.sum(weights)  # Normalize to ensure sum is 100%
    
    allocation = {stock: investment_amount * weight for stock, weight in zip(forecasted_prices.keys(), weights)}
    
    # Convert allocation to percentage
    allocation_df = pd.DataFrame.from_dict(allocation, orient='index', columns=['Investment Amount (₹)'])
    allocation_df['Percentage Allocation (%)'] = (allocation_df['Investment Amount (₹)'] / investment_amount) * 100

    # Display Optimized Stock Allocation
    st.subheader("💰 Optimized Stock Allocation (100% Distributed)")
    st.table(allocation_df)
