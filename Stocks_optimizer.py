import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor

# Streamlit App Title
st.title("📈 AI-Powered Stock Forecast & Portfolio Optimizer")

# User Inputs
selected_stocks = st.text_input("Enter stock symbols separated by commas:", "AAPL, MSFT").strip().upper().split(',')
years_to_use = st.slider("Select the number of years of data to use:", 1, 5, 2)
forecast_days = st.slider("Select the number of days to forecast:", 1, 30, 7)
investment_amount = st.number_input("Enter total investment amount (₹):", min_value=1000.0, value=50000.0)
risk_profile = st.radio("Select your risk level:", [1, 2, 3], format_func=lambda x: {1: "Low", 2: "Medium", 3: "High"}[x])

# Initialize Dictionaries
forecasted_prices = {}
volatilities = {}

# Processing Each Stock
for selected_stock in selected_stocks:
    st.subheader(f"📊 Processing {selected_stock}")
    df = yf.download(selected_stock, period=f"{years_to_use}y", interval="1d", auto_adjust=True)

    if df.empty:
        st.warning(f"Skipping {selected_stock}: No valid data available.")
        continue

    df = df[['Close']]
    df.dropna(inplace=True)
    
    # Generate Future Dates
    future_dates = pd.date_range(df.index[-1], periods=forecast_days + 1, freq='B')[1:]

    # Moving Averages
    df['MA_50'] = df['Close'].rolling(window=50).mean()
    df['MA_200'] = df['Close'].rolling(window=200).mean()

    # Prepare Data for Models
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
    returns = df['Close'].pct_change().dropna()
    volatilities[selected_stock] = np.std(returns).item()  # Convert to float

    # Store Forecasted Price
    forecasted_prices[selected_stock] = future_xgb[-1]

    # Plotting
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.set_style("darkgrid")
    ax.plot(df.index, df['Close'], label=f'{selected_stock} Historical', color='black')
    ax.plot(df.index, df['MA_50'], label='50-Day MA', linestyle='dashed', color='blue')
    ax.plot(df.index, df['MA_200'], label='200-Day MA', linestyle='dashed', color='purple')
    ax.plot(future_dates, future_xgb, label=f'{selected_stock} Forecast (XGBoost)', linestyle='dashed', color='red', marker='o')
    ax.plot(future_dates, future_rf, label=f'{selected_stock} Forecast (Random Forest)', linestyle='dashed', color='green', marker='x')
    ax.legend()
    ax.set_title(f"Stock Price Forecast for {selected_stock}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (₹)")
    ax.tick_params(axis='x', rotation=45)
    st.pyplot(fig)

# Portfolio Optimization Based on Risk Profile
risk_allocation = {1: 0.7, 2: 0.5, 3: 0.3}
allocation = {
    stock: investment_amount * risk_allocation[risk_profile] if volatilities[stock] > 0.03
    else investment_amount * (1 - risk_allocation[risk_profile])
    for stock in volatilities
}
total_allocation = sum(allocation.values())
allocation_percentage = {stock: (amount / total_allocation) * 100 for stock, amount in allocation.items()}

# Display Optimized Portfolio Allocation
st.subheader("💰 Optimized Stock Allocation")
allocation_df = pd.DataFrame.from_dict(allocation, orient='index', columns=['Investment Amount (₹)'])
allocation_df["Percentage (%)"] = allocation_df["Investment Amount (₹)"] / investment_amount * 100
st.dataframe(allocation_df)

# Risk Classification
def classify_risk_level(volatility):
    if volatility > 0.03:
        return "3 (High Risk)"
    elif 0.01 < volatility <= 0.03:
        return "2 (Medium Risk)"
    else:
        return "1 (Low Risk)"

st.subheader("⚖️ Risk Classification of Selected Stocks")
risk_df = pd.DataFrame.from_dict({stock: classify_risk_level(vol) for stock, vol in volatilities.items()}, 
                                 orient='index', columns=['Risk Level'])
st.dataframe(risk_df)

# Summary
st.success("✅ Stock forecasting and portfolio optimization completed successfully!")

