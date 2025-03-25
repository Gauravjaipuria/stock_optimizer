import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor

st.title("Stock Forecasting & Portfolio Optimization")

# User Inputs
selected_stocks = st.text_input("Enter stock symbols separated by commas:").strip().upper().split(',')
years_to_use = st.number_input("Enter the number of years of data to use:", min_value=1, max_value=5, value=2)
forecast_days = st.number_input("Enter the number of days to forecast:", min_value=1, max_value=30, value=5)
investment_amount = st.number_input("Enter the total amount to invest:", min_value=1000.0, value=100000.0)
risk_profile = st.radio("Select your risk level:", [1, 2, 3], format_func=lambda x: {1: "Low", 2: "Medium", 3: "High"}[x])

if st.button("Run Forecast and Optimize Portfolio"):
    forecasted_prices = {}
    volatilities = {}
    
    for selected_stock in selected_stocks:
        df = yf.download(selected_stock, period=f"{years_to_use}y", interval="1d", auto_adjust=True)
        
        if df.empty:
            st.warning(f"Skipping {selected_stock}: No valid data available.")
            continue
        
        df = df[['Close']]
        df.dropna(inplace=True)
        
        future_dates = pd.date_range(df.index[-1], periods=forecast_days + 1, freq='B')[1:]
        
        # Add moving averages
        df['MA_50'] = df['Close'].rolling(window=50).mean()
        df['MA_200'] = df['Close'].rolling(window=200).mean()
        
        # XGBoost Model
        df['Lag_1'] = df['Close'].shift(1)
        df.dropna(inplace=True)
        train_size = int(len(df) * 0.8)
        train, test = df.iloc[:train_size], df.iloc[train_size:]
        xgb_model = XGBRegressor(objective='reg:squarederror', n_estimators=100)
        xgb_model.fit(train[['Lag_1']], train['Close'])
        future_xgb = [xgb_model.predict(np.array([[df['Lag_1'].iloc[-1]]]).reshape(1, -1))[0] for _ in range(forecast_days)]
        
        # Random Forest Model
        rf_model = RandomForestRegressor(n_estimators=100)
        rf_model.fit(train[['Lag_1']], train['Close'])
        future_rf = [rf_model.predict(np.array([[df['Lag_1'].iloc[-1]]]).reshape(1, -1))[0] for _ in range(forecast_days)]
        
        # Calculate Volatility
        returns = df['Close'].pct_change().dropna()
        volatilities[selected_stock] = np.std(returns)
        forecasted_prices[selected_stock] = future_xgb[-1]
        
        # Plot Results
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.set_style("darkgrid")
        ax.plot(df.index, df['Close'], label=f'{selected_stock} Historical', linewidth=2, color='black')
        ax.plot(df.index, df['MA_50'], label='50-Day MA', linestyle='dashed', color='blue')
        ax.plot(df.index, df['MA_200'], label='200-Day MA', linestyle='dashed', color='purple')
        ax.plot(future_dates, future_xgb, label=f'Forecasted Prices (XGBoost)', linestyle='dashed', color='red', marker='o')
        ax.plot(future_dates, future_rf, label=f'Forecasted Prices (Random Forest)', linestyle='dashed', color='green', marker='x')
        ax.legend(fontsize=12, loc='upper left')
        ax.set_title(f"Historical & Forecasted Prices for {selected_stock}", fontsize=16, fontweight='bold')
        ax.set_xlabel("Date")
        ax.set_ylabel("Close Price")
        plt.xticks(rotation=45)
        st.pyplot(fig)
    
    # Portfolio Optimization
    risk_allocation = {1: 0.7, 2: 0.5, 3: 0.3}
    allocation = {stock: investment_amount * risk_allocation[risk_profile] if volatilities[stock] > 0.03 else investment_amount * (1 - risk_allocation[risk_profile]) for stock in volatilities}
    total_allocation = sum(allocation.values())
    allocation_percentage = {stock: (amount / total_allocation) * 100 for stock, amount in allocation.items()}
    
    st.subheader("Optimized Stock Allocation")
    for stock, amount in allocation.items():
        st.write(f"{stock}: ₹{amount:.2f} ({allocation_percentage[stock]:.2f}%)")
    
    # Risk Level Classification
    def classify_risk_level(volatility):
        if volatility > 0.03:
            return "High Risk"
        elif 0.01 < volatility <= 0.03:
            return "Medium Risk"
        else:
            return "Low Risk"
    
    st.subheader("Risk Levels in Investment Optimization")
    for stock, vol in volatilities.items():
        st.write(f"{stock}: {classify_risk_level(vol)}")
