import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import base64
import io
from fpdf import FPDF

st.set_page_config(page_title="AI-Powered Stock Portfolio Optimizer", layout="wide")
st.title("📊 AI-Powered Stock Portfolio Optimizer")

# Sidebar inputs
country = st.sidebar.selectbox("Market", ["India", "America"])
stocks = st.sidebar.text_input("Enter Stock Symbols (comma-separated)", "BPCL, RITES")
years = st.sidebar.slider("Years of Historical Data", 1, 10, 3)
forecast_days = st.sidebar.slider("Forecast Period (Days)", 30, 365, 90)
investment = st.sidebar.number_input("Total Investment (₹)", value=50000.0)
risk_profile = st.sidebar.selectbox("Risk Profile", ["Low", "Medium", "High"])

risk_map = {"Low": 1, "Medium": 2, "High": 3}
risk_level = risk_map[risk_profile]

# Convert and clean stock symbols
stock_list = [s.strip().upper() + ".NS" if country == "India" else s.strip().upper() for s in stocks.split(",")]

forecasted_prices = {}
volatilities = {}
trend_signals = {}
rf_forecasts = {}
xgb_forecasts = {}
actual_vs_predicted = {}

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

    # Store
    xgb_forecasts[stock] = xgb_pred[-1]
    rf_forecasts[stock] = rf_pred[-1]
    volatilities[stock] = float(np.std(df['Close'].pct_change().dropna()))
    forecasted_prices[stock] = {'XGBoost': future_xgb[-1], 'RandomForest': future_rf[-1]}
    actual_vs_predicted[stock] = (test['Close'], xgb_pred, rf_pred)

    # Chart
    future_dates = pd.date_range(df.index[-1], periods=forecast_days+1, freq='B')[1:]
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

# Allocation logic
risky_stocks = [s for s in volatilities if volatilities[s] > 0.03]
safe_stocks = [s for s in volatilities if s not in risky_stocks]
risk_alloc_pct = {1: 0.7, 2: 0.5, 3: 0.3}[risk_level]

risky_amt = investment * risk_alloc_pct
safe_amt = investment - risky_amt
allocation = {}

if risky_stocks:
    for s in risky_stocks:
        allocation[s] = risky_amt / len(risky_stocks)
if safe_stocks:
    for s in safe_stocks:
        allocation[s] = safe_amt / len(safe_stocks)

total_alloc = sum(allocation.values())
alloc_percent = {s: round((amt / total_alloc) * 100, 4) for s, amt in allocation.items()}

# Display
st.subheader("💰 Optimized Stock Allocation (100% Distributed)")
alloc_df = pd.DataFrame.from_dict(allocation, orient='index', columns=['Investment Amount (₹)'])
alloc_df['Percentage Allocation (%)'] = alloc_df.index.map(lambda s: alloc_percent[s])
st.dataframe(alloc_df)

st.subheader("📢 AI Trend Predictions")
trend_df = pd.DataFrame.from_dict(trend_signals, orient='index', columns=['Trend Signal'])
st.dataframe(trend_df)

st.subheader("🧠 Forecasted Prices (Last Prediction)")
forecast_df = pd.DataFrame.from_dict(forecasted_prices, orient='index')
st.dataframe(forecast_df)

# Sharpe Ratio & Risk Level
st.subheader("⚠️ Risk Levels in Investment")
risk_tiers = {s: "3 (High Risk)" if vol > 0.03 else "2 (Medium Risk)" if vol > 0.01 else "1 (Low Risk)" for s, vol in volatilities.items()}
risk_df = pd.DataFrame.from_dict(risk_tiers, orient='index', columns=['Risk Level'])
st.dataframe(risk_df)

# Backtest Chart
st.subheader("📆 Backtest: Actual vs Predicted")
for stock in actual_vs_predicted:
    actual, xgb_pred, rf_pred = actual_vs_predicted[stock]
    plt.figure(figsize=(10, 5))
    plt.plot(actual.index, actual, label='Actual', color='black')
    plt.plot(actual.index, xgb_pred, label='XGBoost', color='red')
    plt.plot(actual.index, rf_pred, label='Random Forest', color='green')
    plt.title(f"{stock} - Backtest Forecast Accuracy")
    plt.legend()
    st.pyplot(plt.gcf())
    plt.close()

# Sharpe Ratio
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

# Downloadable PDF
if st.button("📥 Download Portfolio Report (PDF)"):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="AI-Powered Stock Portfolio Optimizer Report", ln=True, align='C')

    pdf.cell(200, 10, txt="Optimized Allocation:", ln=True)
    for index, row in alloc_df.iterrows():
        amt = f"{row['Investment Amount (₹)']:.2f}".replace("₹", "Rs.")
        pdf.cell(200, 10, txt=f"{index}: Rs. {amt} ({row['Percentage Allocation (%)']}%)", ln=True)

    pdf.cell(200, 10, txt="\nTrend Predictions:", ln=True)
    for index, row in trend_df.iterrows():
        pdf.cell(200, 10, txt=f"{index}: {row['Trend Signal']}", ln=True)

    pdf.cell(200, 10, txt="\nForecasted Prices:", ln=True)
    for index, row in forecast_df.iterrows():
        pdf.cell(200, 10, txt=f"{index}: XGBoost: {row['XGBoost']:.2f}, RF: {row['RandomForest']:.2f}", ln=True)

    pdf_output = io.BytesIO()
    pdf.output(pdf_output)
    st.download_button(label="Download PDF Report", data=pdf_output.getvalue(), file_name="portfolio_report.pdf", mime='application/pdf')
