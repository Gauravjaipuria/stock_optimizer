import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression

# Streamlit UI
st.title("📈 AI-Powered Stock Portfolio Optimizer")

# User Inputs
country = st.radio("Select Country:", ["India", "US"])
selected_stocks = st.text_input("Enter stock symbols (comma-separated):").strip().upper().split(',')
selected_stocks = [stock.strip() + ".NS" if country == "India" else stock.strip() for stock in selected_stocks if stock]

years_to_use = st.number_input("Enter number of years for historical data:", min_value=1, max_value=10, value=2)
forecast_days = st.number_input("Enter forecast period (in days):", min_value=1, max_value=365, value=30)

# Initialize Storage
forecast_results = {}

# Process Each Stock
for stock in selected_stocks:
    df = yf.download(stock, period=f"{years_to_use}y", interval="1d", auto_adjust=True)
    
    if df.empty or len(df) < 50:
        st.warning(f"Skipping {stock}: Not enough valid data.")
        continue

    df = df[['Close']]
    df.dropna(inplace=True)
    
    future_dates = pd.date_range(df.index[-1], periods=forecast_days + 1, freq='B')[1:]

    # Feature Engineering
    df['Lag_1'] = df['Close'].shift(1)
    df['MA_50'] = df['Close'].rolling(window=50).mean()
    df['MA_200'] = df['Close'].rolling(window=200).mean()
    df.dropna(inplace=True)

    train_size = int(len(df) * 0.8)
    train, test = df.iloc[:train_size], df.iloc[train_size:]

    # Model Training and Forecasting
    models = {
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "XGBoost": XGBRegressor(objective='reg:squarederror', n_estimators=100),
        "ARIMA": None
    }

    forecasts = {}
    errors = {}

    # Train ML Models
    for model_name, model in models.items():
        if model_name == "ARIMA":
            try:
                arima_model = ARIMA(train['Close'], order=(5,1,0)).fit()
                forecast = arima_model.forecast(steps=len(test))
                forecasts[model_name] = forecast
                errors[model_name] = mean_squared_error(test['Close'], forecast, squared=False)
            except Exception as e:
                st.warning(f"Skipping ARIMA for {stock} due to error: {e}")
        else:
            model.fit(train[['Lag_1']], train['Close'])
            predictions = model.predict(test[['Lag_1']])
            forecasts[model_name] = model.predict(np.array(df['Lag_1'].iloc[-forecast_days:]).reshape(-1, 1))
            errors[model_name] = mean_squared_error(test['Close'], predictions, squared=False)

    # Select Best Model
    best_model = min(errors, key=errors.get)
    best_forecast = forecasts[best_model]
    forecast_results[stock] = {"Best Model": best_model, "Forecast": best_forecast[-1]}

    # Plot Historical and Forecasted Prices
    st.subheader(f"📊 Forecast for {stock} (Best Model: {best_model})")
    plt.figure(figsize=(14, 7))
    sns.set_style("darkgrid")
    plt.plot(df.index, df['Close'], label=f'{stock} Historical', linewidth=2, color='black')
    plt.plot(df.index, df['MA_50'], label='50-Day MA', linestyle='dashed', color='blue')
    plt.plot(df.index, df['MA_200'], label='200-Day MA', linestyle='dashed', color='purple')
    plt.plot(future_dates, best_forecast, label=f'{stock} Forecasted ({best_model})', linestyle='dashed', color='red', marker='o')
    plt.legend()
    plt.title(f"Historical and Forecasted Prices for {stock}")
    st.pyplot(plt)

# Display Forecast Results
st.subheader("📌 Forecast Results")
if forecast_results:
    forecast_df = pd.DataFrame.from_dict(forecast_results, orient='index')
    st.table(forecast_df)
else:
    st.warning("No forecast data available.")
