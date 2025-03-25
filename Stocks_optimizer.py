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
selected_stocks = st.text_input("Enter stock symbols (comma-separated):").strip().upper().split(',')
selected_stocks = [stock.strip() for stock in selected_stocks if stock]

years_to_use = st.number_input("Enter number of years for historical data:", min_value=1, max_value=10, value=2)
forecast_days = st.number_input("Enter forecast period (in days):", min_value=1, max_value=365, value=30)

# Storage for Forecast Results
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
    df.dropna(inplace=True)

    train_size = int(len(df) * 0.8)
    train, test = df.iloc[:train_size], df.iloc[train_size:]

    # Ensure test is not empty
    if test.empty:
        st.warning(f"Skipping {stock}: Not enough test data after split.")
        continue

    # Define Models
    models = {
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "XGBoost": XGBRegressor(objective='reg:squarederror', n_estimators=100),
        "ARIMA": None
    }

    forecasts = {}
    errors = {}

    # Train & Evaluate Models
    for model_name, model in models.items():
        if model_name == "ARIMA":
            try:
                arima_model = ARIMA(train['Close'], order=(5,1,0)).fit()
                forecast = arima_model.forecast(steps=len(test))
                forecast = np.array(forecast).reshape(-1)  # Ensure correct shape
                forecasts[model_name] = forecast

                # Fix shape mismatch
                test_values = test['Close'].values.reshape(-1)
                if len(test_values) == len(forecast):
                    errors[model_name] = mean_squared_error(test_values, forecast, squared=False)
            except Exception as e:
                st.warning(f"Skipping ARIMA for {stock} due to error: {e}")
        else:
            # Ensure input shape compatibility
            train_X, test_X = train[['Lag_1']], test[['Lag_1']]
            train_y, test_y = train['Close'], test['Close']

            # Convert to numpy arrays to ensure compatibility
            train_X, test_X = train_X.to_numpy(), test_X.to_numpy()
            train_y, test_y = train_y.to_numpy().reshape(-1), test_y.to_numpy().reshape(-1)

            if not train_X.size or not test_X.size:
                st.warning(f"Skipping {stock}: Model training data is empty.")
                continue

            model.fit(train_X, train_y)
            predictions = model.predict(test_X)  # Predictions should match test_y size

            # Ensure test_y and predictions have the same shape
            if len(test_y) == len(predictions):
                errors[model_name] = mean_squared_error(test_y, predictions, squared=False)

    # Select Best Model Based on RMSE
    if errors:
        best_model = min(errors, key=errors.get)
    else:
        st.warning(f"No valid model found for {stock}")
        continue

    best_forecast = forecasts.get(best_model, [])
    forecast_results[stock] = {"Best Model": best_model, "Forecast": best_forecast[-1] if len(best_forecast) else "N/A"}

    # Plot Historical and Forecasted Prices
    st.subheader(f"📊 Forecast for {stock} (Best Model: {best_model})")
    plt.figure(figsize=(14, 7))
    sns.set_style("darkgrid")
    plt.plot(df.index, df['Close'], label=f'{stock} Historical', linewidth=2, color='black')
    if len(best_forecast) == forecast_days:
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
