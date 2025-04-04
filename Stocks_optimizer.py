import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor

# Streamlit UI
st.title("\ud83d\udcc8 AI-Powered Stock Portfolio Optimizer")

# User Inputs
country = st.radio("Select Market:", ["India", "Other"])
selected_stocks = st.text_input("Enter stock symbols (comma-separated):").strip().upper().split(',')
selected_stocks = [stock.strip() + ".NS" if country == "India" else stock.strip() for stock in selected_stocks if stock]

years_to_use = st.number_input("Enter number of years for historical data:", min_value=1, max_value=10, value=2)
forecast_days = st.number_input("Enter forecast period (in days):", min_value=1, max_value=365, value=30)
investment_amount = st.number_input("Enter total investment amount (\u20b9):", min_value=1000.0, value=50000.0)
risk_profile = st.radio("Select your risk level:", [1, 2, 3], format_func=lambda x: {1: "Low", 2: "Medium", 3: "High"}[x])

# Initialize Storage
forecasted_prices = {}
volatilities = {}
trend_signals = {}
rsi_values = {}
closing_prices = {}

# Process Each Stock
for stock in selected_stocks:
    df = yf.download(stock, period=f"{years_to_use}y", interval="1d", auto_adjust=True)
    
    if df.empty:
        st.warning(f"Skipping {stock}: No valid data available.")
        continue
    
    df = df[['Close']]
    df.dropna(inplace=True)
    
    closing_prices[stock] = df['Close'].iloc[-1]  # Store last closing price
    
    future_dates = pd.date_range(df.index[-1], periods=forecast_days + 1, freq='B')[1:]
    
    # Moving Averages
    df['MA_50'] = df['Close'].rolling(window=50).mean()
    df['MA_200'] = df['Close'].rolling(window=200).mean()
    
    # RSI Calculation
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    rsi_values[stock] = df['RSI'].iloc[-1]  # Store RSI value
    
    # Trend Prediction (Including RSI)
    if df['MA_50'].iloc[-1] > df['MA_200'].iloc[-1] and df['RSI'].iloc[-1] > 50:
        trend_signals[stock] = "Bullish \ud83d\udfe2 (Buy) - MA & RSI Indicate Uptrend"
    elif df['MA_50'].iloc[-1] < df['MA_200'].iloc[-1] and df['RSI'].iloc[-1] < 50:
        trend_signals[stock] = "Bearish \ud83d\udd34 (Sell) - MA & RSI Indicate Downtrend"
    else:
        trend_signals[stock] = "Neutral \u26aa (Hold) - Mixed Signals"
    
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
    forecasted_prices[stock] = {'XGBoost': future_xgb[-1], 'RandomForest': future_rf[-1]}
    
    # Plot Results
    st.subheader(f"\ud83d\udcca Forecast for {stock}")
    plt.figure(figsize=(14, 7))
    sns.set_style("darkgrid")
    plt.plot(df.index, df['Close'], label=f'{stock} Historical', linewidth=2, color='black')
    plt.plot(df.index, df['MA_50'], label='50-Day MA', linestyle='dashed', color='blue')
    plt.plot(df.index, df['MA_200'], label='200-Day MA', linestyle='dashed', color='purple')
    plt.plot(future_dates, future_xgb, label='XGBoost Forecast', linestyle='dashed', color='red', marker='o')
    plt.plot(future_dates, future_rf, label='Random Forest Forecast', linestyle='dashed', color='green', marker='x')
    
    plt.legend(fontsize=12, loc='upper left', frameon=True, shadow=True, fancybox=True)
    plt.title(f"Historical and Forecasted Prices for {stock}", fontsize=16, fontweight='bold')
    plt.xlabel("Date", fontsize=14)
    plt.ylabel("Close Price", fontsize=14)
    plt.xticks(rotation=45, fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    st.pyplot(plt)

# Display Trend Analysis
st.subheader("\ud83d\udce2 AI Trend Predictions")
st.table(pd.DataFrame.from_dict(trend_signals, orient='index', columns=['Trend Signal']))

# Display RSI Values
st.subheader("\ud83d\udd22 RSI Values")
st.table(pd.DataFrame.from_dict(rsi_values, orient='index', columns=['RSI Value']))

# Display Last Closing Price
st.subheader("\ud83d\udcc5 Last Closing Prices")
st.table(pd.DataFrame.from_dict(closing_prices, orient='index', columns=['Last Close (₹)']))
