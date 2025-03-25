import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

# Streamlit UI
st.title("📈 AI-Powered Stock Portfolio Optimizer")

# User Inputs
country = st.radio("Select Country:", ["India", "US"])
selected_stocks = st.text_input("Enter stock symbols (comma-separated):").strip().upper().split(',')
selected_stocks = [stock.strip() + ".NS" if country == "India" else stock.strip() for stock in selected_stocks if stock]

years_to_use = st.number_input("Enter number of years for historical data:", min_value=1, max_value=10, value=2)
forecast_days = st.number_input("Enter forecast period (in days):", min_value=1, max_value=365, value=30)
investment_amount = st.number_input("Enter total investment amount (₹):", min_value=1000.0, value=50000.0)
risk_profile = st.radio("Select your risk level:", [1, 2, 3], format_func=lambda x: {1: "Low", 2: "Medium", 3: "High"}[x])

# Initialize Storage
forecasted_prices = {}
volatilities = {}
sector_data = {}
returns = {}

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

    # Backtesting (Simple Buy & Hold Strategy)
    initial_price = df['Close'].iloc[0]
    final_price = df['Close'].iloc[-1]
    returns[stock] = ((final_price - initial_price) / initial_price) * 100
    
    # AI-Based Stock Screener (Linear Regression for Trend Analysis)
    lr_model = LinearRegression()
    lr_model.fit(np.arange(len(df)).reshape(-1, 1), df['Close'])
    trend_slope = lr_model.coef_[0]
    
    # Store Data
    forecasted_prices[stock] = future_xgb[-1]
    volatilities[stock] = float(np.std(df['Close'].pct_change().dropna()))
    sector_data[stock] = "Technology"  # Placeholder; real data should be fetched
    
    # Plot Historical and Forecasted Prices
    st.subheader(f"📊 Forecast for {stock}")
    plt.figure(figsize=(14, 7))
    sns.set_style("darkgrid")
    plt.plot(df.index, df['Close'], label=f'{stock} Historical', linewidth=2, color='black')
    plt.plot(df.index, df['MA_50'], label='50-Day MA', linestyle='dashed', color='blue')
    plt.plot(df.index, df['MA_200'], label='200-Day MA', linestyle='dashed', color='purple')
    plt.plot(future_dates, future_xgb, label=f'{stock} Forecasted (XGBoost)', linestyle='dashed', color='red', marker='o')
    plt.legend()
    plt.title(f"Historical and Forecasted Prices for {stock}")
    st.pyplot(plt)

# Display Backtesting Results
st.subheader("📊 Backtesting Performance")
returns_df = pd.DataFrame.from_dict(returns, orient='index', columns=['Return (%)'])
st.table(returns_df)

# Display Sector-wise Diversification
st.subheader("📌 Sector-wise Diversification")
sector_df = pd.DataFrame.from_dict(sector_data, orient='index', columns=['Sector'])
st.table(sector_df)

# AI Stock Screener (Simple Trend Analysis)
st.subheader("🤖 AI-Based Stock Screener")
trend_df = pd.DataFrame.from_dict({stock: trend_slope for stock in selected_stocks}, orient='index', columns=['Trend Slope'])
trend_df['Signal'] = trend_df['Trend Slope'].apply(lambda x: 'Bullish' if x > 0 else 'Bearish')
st.table(trend_df)
