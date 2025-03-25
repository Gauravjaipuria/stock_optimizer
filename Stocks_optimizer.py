import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor

# Streamlit UI
st.set_page_config(page_title="Stock Forecast & Allocation", layout="wide")

st.title("📈 AI-Powered Stock Forecast & Investment Optimizer")

# Get user inputs
selected_stocks = st.text_input("Enter stock symbols separated by commas (e.g., TCS.NS, INFY.NS):")
years_to_use = st.slider("Select number of years of data to use:", 1, 5, 2)
forecast_years = st.slider("Select forecast period in years:", 1, 5, 1)
investment_amount = st.number_input("Enter the total investment amount (₹):", min_value=1000, step=500)

# Get client risk profile
risk_profile = st.radio("Select your risk level:", ["Low", "Medium", "High"])

# Convert to numerical risk factor
risk_factor = {"Low": 1, "Medium": 2, "High": 3}[risk_profile]

# Process stock symbols
if selected_stocks:
    selected_stocks = [stock.strip().upper() for stock in selected_stocks.split(',')]

    forecast_days = forecast_years * 365  # Convert years to days
    forecasted_prices = {}
    volatilities = {}

    for selected_stock in selected_stocks:
        df = yf.download(selected_stock, period=f"{years_to_use}y", interval="1d", auto_adjust=True)

        if df.empty:
            st.warning(f"Skipping {selected_stock}: No valid data available.")
            continue

        df = df[['Close']]
        df.dropna(inplace=True)

        # Create moving averages
        df['MA_50'] = df['Close'].rolling(window=50).mean()
        df['MA_200'] = df['Close'].rolling(window=200).mean()

        # Feature Engineering
        df['Lag_1'] = df['Close'].shift(1)
        df.dropna(inplace=True)

        # Train-test split
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

        # Calculate Volatility (Standard Deviation of Returns)
        returns = df['Close'].pct_change().dropna()
        volatilities[selected_stock] = np.std(returns)
        forecasted_prices[selected_stock] = future_xgb[-1]

        # Plot Historical and Forecasted Prices
        plt.figure(figsize=(14, 7))
        sns.set_style("darkgrid")
        plt.plot(df.index, df['Close'], label=f'{selected_stock} Historical', linewidth=2, color='black')
        plt.plot(df.index, df['MA_50'], label='50-Day MA', linestyle='dashed', color='blue')
        plt.plot(df.index, df['MA_200'], label='200-Day MA', linestyle='dashed', color='purple')
        plt.plot(pd.date_range(df.index[-1], periods=forecast_days, freq='B'), future_xgb, 
                 label=f'{selected_stock} Forecasted (XGBoost)', linestyle='dashed', color='red', marker='o')
        plt.plot(pd.date_range(df.index[-1], periods=forecast_days, freq='B'), future_rf, 
                 label=f'{selected_stock} Forecasted (Random Forest)', linestyle='dashed', color='green', marker='x')
        plt.legend(fontsize=12, loc='upper left')
        plt.title(f"Historical and Forecasted Prices for {selected_stock}", fontsize=16, fontweight='bold')
        plt.xlabel("Date", fontsize=14)
        plt.ylabel("Close Price", fontsize=14)
        plt.xticks(rotation=45)
        plt.grid(True, linestyle='--', alpha=0.6)
        st.pyplot(plt)

    # Portfolio Optimization: Allocate based on risk profile
    risk_allocation = {1: 0.7, 2: 0.5, 3: 0.3}
    
    # Allocate investment based on volatility
    allocation = {
        stock: investment_amount * risk_allocation[risk_factor] if float(volatilities[stock]) > 0.03
        else investment_amount * (1 - risk_allocation[risk_factor]) 
        for stock in volatilities
    }

    # Normalize allocation percentages to sum to 100%
    total_allocation = sum(allocation.values())

    if total_allocation > 0:
        allocation_percentage = {stock: (amount / total_allocation) * 100 for stock, amount in allocation.items()}
    else:
        allocation_percentage = {stock: 0 for stock in allocation}  

    # Convert to DataFrame for display
    allocation_df = pd.DataFrame.from_dict(allocation, orient='index', columns=['Investment Amount (₹)'])
    allocation_df["Percentage (%)"] = allocation_df["Investment Amount (₹)"] / investment_amount * 100

    # Ensure percentages sum up to 100%
    allocation_df["Percentage (%)"] = (allocation_df["Percentage (%)"] / allocation_df["Percentage (%)"].sum()) * 100

    # Display Results
    st.subheader("💰 Optimized Stock Allocation")
    st.table(allocation_df)

    # Risk Classification
        def classify_risk_level(volatility):
            volatility = float(volatility)  # Ensure it's a float
            if volatility > 0.03:
                return "3 (High Risk)"
            elif 0.01 < volatility <= 0.03:
                return "2 (Medium Risk)"
            else:
                return "1 (Low Risk)"
    

    st.subheader("⚠️ Risk Levels in Investment")
    risk_df = pd.DataFrame.from_dict({stock: classify_risk_level(vol) for stock, vol in volatilities.items()}, 
                                     orient='index', columns=['Risk Level'])
    st.table(risk_df)
