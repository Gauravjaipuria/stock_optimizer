import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression

def calculate_indicators(df):
    df['RSI'] = compute_rsi(df['Close'])
    df['MACD'], df['Signal_Line'] = compute_macd(df['Close'])
    df['Upper_BB'], df['Lower_BB'] = compute_bollinger_bands(df['Close'])
    return df

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def compute_macd(series, short_window=12, long_window=26, signal_window=9):
    short_ema = series.ewm(span=short_window, adjust=False).mean()
    long_ema = series.ewm(span=long_window, adjust=False).mean()
    macd = short_ema - long_ema
    signal = macd.ewm(span=signal_window, adjust=False).mean()
    return macd, signal

def compute_bollinger_bands(series, window=20):
    sma = series.rolling(window=window).mean()
    std = series.rolling(window=window).std()
    upper_band = sma + (std * 2)
    lower_band = sma - (std * 2)
    return upper_band, lower_band

st.title("📈 AI-Powered Stock Portfolio Optimizer")

country = st.radio("Select Country:", ["India", "US"])
selected_stocks = st.text_input("Enter stock symbols (comma-separated):").strip().upper().split(',')
selected_stocks = [stock.strip() + ".NS" if country == "India" else stock.strip() for stock in selected_stocks if stock]

years_to_use = st.number_input("Enter number of years for historical data:", min_value=1, max_value=10, value=2)
forecast_days = st.number_input("Enter forecast period (in days):", min_value=1, max_value=365, value=30)
investment_amount = st.number_input("Enter total investment amount (₹):", min_value=1000.0, value=50000.0)
risk_profile = st.radio("Select your risk level:", [1, 2, 3], format_func=lambda x: {1: "Low", 2: "Medium", 3: "High"}[x])

forecasted_prices = {}
returns = {}

for stock in selected_stocks:
    df = yf.download(stock, period=f"{years_to_use}y", interval="1d", auto_adjust=True)
    
    if df.empty:
        st.warning(f"Skipping {stock}: No valid data available.")
        continue

    df = df[['Close']]
    df = calculate_indicators(df)
    df.dropna(inplace=True)
    
    train_size = int(len(df) * 0.8)
    train, test = df.iloc[:train_size], df.iloc[train_size:]
    
    xgb_model = XGBRegressor(objective='reg:squarederror', n_estimators=100)
    xgb_model.fit(train[['Close']], train['Close'])
    
    future_dates = pd.date_range(df.index[-1], periods=forecast_days + 1, freq='B')[1:]
    future_xgb = [xgb_model.predict(np.array([[df['Close'].iloc[-1]]]).reshape(1, -1))[0] for _ in range(forecast_days)]
    forecasted_prices[stock] = future_xgb[-1]
    
    initial_price = df['Close'].iloc[0]
    final_price = df['Close'].iloc[-1]
    returns[stock] = ((final_price - initial_price) / initial_price) * 100
    
    st.subheader(f"📊 Forecast for {stock}")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name=f'{stock} Historical'))
    fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], mode='lines', name='RSI'))
    fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], mode='lines', name='MACD'))
    fig.add_trace(go.Scatter(x=future_dates, y=future_xgb, mode='lines+markers', name=f'{stock} Forecasted (XGBoost)'))
    fig.update_layout(title=f"Historical and Forecasted Prices for {stock}")
    st.plotly_chart(fig)

st.subheader("📊 Backtesting Performance")
returns_df = pd.DataFrame.from_dict(returns, orient='index', columns=['Return (%)'])
st.table(returns_df)
