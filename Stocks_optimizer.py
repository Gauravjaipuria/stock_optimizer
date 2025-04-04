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

# Function to calculate RSI
def compute_rsi(series, period=14):
    delta = series.diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Process Each Stock
for stock in selected_stocks:
    df = yf.download(stock, period=f"{years_to_use}y", interval="1d", auto_adjust=True)

    if df.empty:
        st.warning(f"Skipping {stock}: No valid data available.")
        continue

    df = df[['Close']]
    df.dropna(inplace=True)  # Ensure clean data
    
    future_dates = pd.date_range(df.index[-1], periods=forecast_days + 1, freq='B')[1:]

    # Moving Averages
    df['MA_50'] = df['Close'].rolling(window=50).mean()
    df['MA_200'] = df['Close'].rolling(window=200).mean()

    # AI Trend Prediction
    if df['MA_50'].iloc[-1] > df['MA_200'].iloc[-1]:
        trend_signal = "Bullish 🟢 (Buy)"
        trend_reason = "Short-term price momentum is stronger than long-term trend."
    else:
        trend_signal = "Bearish 🔴 (Sell)"
        trend_reason = "Short-term price momentum is weaker than the long-term trend."

    # Calculate RSI
    df['RSI'] = compute_rsi(df['Close'])
    latest_rsi = df['RSI'].iloc[-1]
    
    # RSI Recommendation
    if latest_rsi < 30:
        rsi_recommendation = "📈 Strong Buy (Oversold)"
    elif 30 <= latest_rsi <= 70:
        rsi_recommendation = "⏳ Hold (Neutral)"
    else:
        rsi_recommendation = "📉 Strong Sell (Overbought)"

    # Feature Engineering
    df['Lag_1'] = df['Close'].shift(1)
    df.dropna(inplace=True)

    train_size = int(len(df) * 0.8)
    train, test = df.iloc[:train_size], df.iloc[train_size:]

    # XGBoost Model
    xgb_model = XGBRegressor(objective='reg:squarederror', n_estimators=100)
    xgb_model.fit(train[['Lag_1']], train['Close'])
    future_xgb = xgb_model.predict(np.array([[df['Lag_1'].iloc[-1]] for _ in range(forecast_days)]))

    # Random Forest Model
    rf_model = RandomForestRegressor(n_estimators=100)
    rf_model.fit(train[['Lag_1']], train['Close'])
    future_rf = rf_model.predict(np.array([[df['Lag_1'].iloc[-1]] for _ in range(forecast_days)]))

    # Last Traded Price Handling (Fixes TypeError)
    last_traded_price = df['Close'].iloc[-1] if not df.empty else None
    last_traded_price_str = f"₹{last_traded_price:.2f}" if pd.notna(last_traded_price) else "Data Not Available"

    # Display Analysis
    st.subheader(f"📊 Forecast for {stock}")
    st.write(f"📉 **Last Traded Price**: {last_traded_price_str}")
    st.write(f"📊 **RSI (14-day)**: {latest_rsi:.2f}")
    st.write(f"📌 **Recommendation**: {rsi_recommendation}")
    st.write(f"📢 **Trend Signal**: {trend_signal}")
    st.write(f"💡 **Trend Reason**: {trend_reason}")

    # Plot Historical and Forecasted Prices
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
