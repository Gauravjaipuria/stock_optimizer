import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from ta.momentum import RSIIndicator

st.set_page_config(page_title="AI-Powered Stock Portfolio Optimizer", layout="wide")
st.title("📊 AI-Powered Stock Portfolio Optimizer")

# Sidebar inputs
country = st.sidebar.selectbox("Market", ["India", "other"])
stocks = st.sidebar.text_input("Enter Stock Symbols (comma-separated)", "BPCL, RITES")
years = st.sidebar.slider("Years of Historical Data", 1, 10, 3)
forecast_days = st.sidebar.slider("Forecast Period (Days)", 30, 365, 90)
investment = st.sidebar.number_input("Total Investment (\u20b9)", value=50000.0)
risk_profile = st.sidebar.selectbox("Risk Profile", ["Low", "Medium", "High"])

risk_map = {"Low": 1, "Medium": 2, "High": 3}
risk_level = risk_map[risk_profile]

stock_list = [s.strip().upper() + ".NS" if country == "India" else s.strip().upper() for s in stocks.split(",")]

forecasted_prices = {}
volatilities = {}
trend_signals = {}
rf_forecasts = {}
xgb_forecasts = {}
actual_vs_predicted = {}

n_lags = 5

def create_lag_features(df, n_lags):
    for i in range(1, n_lags + 1):
        df[f'Lag_{i}'] = df['Close'].shift(i)
    return df.dropna()

def recursive_forecast_multifeature(model, last_values, days):
    preds = []
    seq = last_values[-n_lags:]
    for _ in range(days):
        x_input = np.array(seq[-n_lags:]).reshape(1, -1)
        pred = model.predict(x_input)[0]
        preds.append(pred)
        seq.append(pred)
    return preds

for stock in stock_list:
    df = yf.download(stock, period=f"{years}y", interval="1d", auto_adjust=True)
    if df.empty:
        st.warning(f"Skipping {stock} due to lack of data.")
        continue

    df = df[['Close']].dropna()
    df['MA_50'] = df['Close'].rolling(window=50).mean()
    df['MA_200'] = df['Close'].rolling(window=200).mean()
    trend_signals[stock] = "Bullish (Buy)" if df['MA_50'].iloc[-1] > df['MA_200'].iloc[-1] else "Bearish (Sell)"

    df_lagged = create_lag_features(df.copy(), n_lags)
    train_size = int(len(df_lagged) * 0.8)
    train, test = df_lagged.iloc[:train_size], df_lagged.iloc[train_size:]

    x_cols = [f'Lag_{i}' for i in range(1, n_lags + 1)]

    xgb_model = XGBRegressor(objective='reg:squarederror', n_estimators=100)
    xgb_model.fit(train[x_cols], train['Close'])
    xgb_pred = xgb_model.predict(test[x_cols])
    future_xgb = recursive_forecast_multifeature(xgb_model, df['Close'].tolist(), forecast_days)

    rf_model = RandomForestRegressor(n_estimators=100)
    rf_model.fit(train[x_cols], train['Close'])
    rf_pred = rf_model.predict(test[x_cols])
    future_rf = recursive_forecast_multifeature(rf_model, df['Close'].tolist(), forecast_days)

    xgb_forecasts[stock] = xgb_pred[-1]
    rf_forecasts[stock] = rf_pred[-1]
    volatilities[stock] = float(np.std(df['Close'].pct_change().dropna()))
    forecasted_prices[stock] = {'XGBoost': future_xgb[-1], 'RandomForest': future_rf[-1]}
    actual_vs_predicted[stock] = (test['Close'], xgb_pred, rf_pred)

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

# RSI Analysis
st.subheader("\ud83d\udcc8 RSI Analysis (Relative Strength Index)")
rsi_signals = []
for stock in stock_list:
    df_rsi = yf.download(stock, period=f"{years}y", interval="1d", auto_adjust=True)
    close_series = df_rsi['Close'].squeeze()
    rsi = RSIIndicator(close=close_series, window=14).rsi()
    latest_rsi = rsi.iloc[-1]
    signal = "Oversold (Buy)" if latest_rsi < 30 else "Overbought (Sell)" if latest_rsi > 70 else "Neutral"
    rsi_signals.append({"Stock": stock, "RSI": round(latest_rsi, 2), "Signal": signal})

st.dataframe(pd.DataFrame(rsi_signals))

# Risk classification
st.subheader("\ud83d\udd0d Stock Risk Classification")
low_risk, medium_risk, high_risk = [], [], []
for stock, vol in volatilities.items():
    if vol <= 0.01:
        low_risk.append(stock)
    elif vol <= 0.03:
        medium_risk.append(stock)
    else:
        high_risk.append(stock)

risk_classification_df = pd.DataFrame({
    "Stock": low_risk + medium_risk + high_risk,
    "Risk Category": (["Low"] * len(low_risk)) + (["Medium"] * len(medium_risk)) + (["High"] * len(high_risk))
})
st.dataframe(risk_classification_df)

# Portfolio Allocation
st.subheader("\ud83d\udcb8 Portfolio Allocation Based on Risk")
allocation = {}
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
    risk_allocation = {1: 0.3, 2: 0.5, 3: 0.7}
    risky_allocation = investment * risk_allocation[risk_level]
    safe_allocation = investment - risky_allocation
    safe_stocks = low_risk + medium_risk if risk_level == 3 else low_risk
    risky_stocks = high_risk if risk_level == 3 else medium_risk + high_risk
    if risky_stocks:
        per_risky = risky_allocation / len(risky_stocks)
        for stock in risky_stocks:
            allocation[stock] = per_risky
    if safe_stocks:
        per_safe = safe_allocation / len(safe_stocks)
        for stock in safe_stocks:
            allocation[stock] = per_safe

# Display Allocation
alloc_df = pd.DataFrame.from_dict(allocation, orient='index', columns=['Investment Amount (\u20b9)'])
total_alloc = sum(allocation.values())
alloc_df['Percentage Allocation (%)'] = alloc_df['Investment Amount (\u20b9)'].apply(lambda x: round((x / total_alloc) * 100, 2))
st.subheader("\ud83d\udcb0 Optimized Stock Allocation (100% Distributed)")
st.dataframe(alloc_df)

# Trend Signals
st.subheader("\ud83d\udce2 AI Trend Predictions")
trend_df = pd.DataFrame.from_dict(trend_signals, orient='index', columns=['Trend Signal'])
st.dataframe(trend_df)

# Forecast Table
st.subheader("\ud83d\ude42 Forecasted Prices (Last Prediction)")
forecast_df = pd.DataFrame.from_dict(forecasted_prices, orient='index')
st.dataframe(forecast_df)

# Risk Levels
st.subheader("\u26a0\ufe0f Risk Levels in Investment")
risk_tiers = {s: "3 (High Risk)" if vol > 0.03 else "2 (Medium Risk)" if vol > 0.01 else "1 (Low Risk)" for s, vol in volatilities.items()}
st.dataframe(pd.DataFrame.from_dict(risk_tiers, orient='index', columns=['Risk Level']))

# Sharpe Ratio
st.subheader("\ud83d\udcc9 Sharpe Ratio & Return Forecast")
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
