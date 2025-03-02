import streamlit as st
from nselib import capital_market
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from statsmodels.tsa.arima.model import ARIMA
import plotly.graph_objects as go

# Function to fetch stock data
def get_stock_data(symbol, from_date, to_date):
    try:
        data = capital_market.price_volume_and_deliverable_position_data(symbol=symbol, from_date=from_date, to_date=to_date)
        df = pd.DataFrame(data)
        
        if df.empty or "ClosePrice" not in df:
            return None  

        df["ClosePrice"] = pd.to_numeric(df["ClosePrice"], errors="coerce")
        df["Date"] = pd.to_datetime(df["Date"])
        df.dropna(subset=["ClosePrice"], inplace=True)
        df = df.sort_values("Date")

        return df
    except Exception as e:
        st.error(f"Error fetching data for {symbol}: {str(e)}")
        return None

# Function to optimize capital allocation
def optimize_allocation(stock_data, capital):
    closing_prices = {
        stock: df["ClosePrice"].iloc[-1] if df is not None and not df.empty else np.nan 
        for stock, df in stock_data.items()
    }

    prices = np.array([price for price in closing_prices.values() if not np.isnan(price)])
    stocks = [stock for stock, price in closing_prices.items() if not np.isnan(price)]

    if len(stocks) < 2:  
        return {stock: capital / len(stocks) for stock in stocks}, {stock: 100 / len(stocks) for stock in stocks} if stocks else ({}, {})

    def objective(weights):
        return np.sum(weights * prices)

    constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})
    bounds = [(0, 1)] * len(stocks)
    init_guess = np.ones(len(stocks)) / len(stocks)
    
    result = minimize(objective, init_guess, bounds=bounds, constraints=constraints)

    if not result.success:
        st.warning("Optimization failed, using equal allocation.")
        allocation = {stocks[i]: capital / len(stocks) for i in range(len(stocks))}
        allocation_percentage = {stocks[i]: 100 / len(stocks) for i in range(len(stocks))}
    else:
        optimized_weights = result.x
        allocation = {stocks[i]: optimized_weights[i] * capital for i in range(len(stocks))}
        allocation_percentage = {stocks[i]: optimized_weights[i] * 100 for i in range(len(stocks))}

    return allocation, allocation_percentage

# Function to forecast stock price using ARIMA
def forecast_price(df, days=730):
    if df is None or df.empty or "ClosePrice" not in df:
        return None  

    try:
        df["ClosePrice"] = pd.to_numeric(df["ClosePrice"], errors="coerce")
        df.dropna(subset=["ClosePrice"], inplace=True)

        if len(df) < 50:  # More data is better for long-term forecasting
            return None

        model = ARIMA(df["ClosePrice"], order=(2, 1, 2))  
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=days)
        future_dates = pd.date_range(start=df["Date"].iloc[-1], periods=days+1, freq='D')[1:]
        return pd.DataFrame({"Date": future_dates, "Forecasted Price": forecast})
    except Exception as e:
        return None  

# Streamlit UI
st.title("🚀 AI-Powered Stock Allocation & Forecasting")
st.markdown("### Optimize your stock investments and predict future trends with AI-powered analysis! 📊")

st.sidebar.header("Enter Stock Details")
num_stocks = st.sidebar.number_input("Number of Stocks", min_value=1, max_value=10, value=3)
stocks = [st.sidebar.text_input(f"Stock {i+1} Symbol", "").strip().upper() for i in range(num_stocks)]
from_date = st.sidebar.text_input("Start Date (DD-MM-YYYY)", "01-01-2022")
to_date = st.sidebar.text_input("End Date (DD-MM-YYYY)", "01-01-2025")
capital = st.sidebar.number_input("Total Capital (₹)", min_value=10000, value=100000)
forecast_days = st.sidebar.number_input("Forecast Days", min_value=365, max_value=1095, value=730)

if st.sidebar.button("Run Analysis"):
    stock_data = {stock: get_stock_data(stock, from_date, to_date) for stock in stocks if stock}
    stock_data = {k: v for k, v in stock_data.items() if v is not None}

    if len(stock_data) == 0:
        st.error("No valid stock data available. Please check stock symbols and try again.")
    else:
        allocation, allocation_percentage = optimize_allocation(stock_data, capital)
        forecasts = {}

        st.subheader("💰 Optimized Capital Allocation")
        allocation_df = pd.DataFrame({
            "Stock": allocation.keys(), 
            "Allocation (₹)": allocation.values(), 
            "Allocation (%)": allocation_percentage.values()
        })
        st.dataframe(allocation_df)

        st.subheader("📉 Historical & Forecasted Price Trends")
        fig = go.Figure()

        for stock, df in stock_data.items():
            future_prices = forecast_price(df, forecast_days)
            if future_prices is not None:
                forecasts[stock] = future_prices

                # Historical prices (Optional: Apply smoothing)
                df["Smoothed Price"] = df["ClosePrice"].rolling(window=7).mean()

                # Add historical prices to the chart
                fig.add_trace(go.Scatter(
                    x=df["Date"], y=df["Smoothed Price"],
                    mode='lines', name=f"{stock} Historical",
                    line=dict(color='royalblue', width=2),
                    hoverinfo='x+y'
                ))

                # Add forecasted prices to the chart
                fig.add_trace(go.Scatter(
                    x=future_prices["Date"], y=future_prices["Forecasted Price"],
                    mode='lines', name=f"{stock} Forecast",
                    line=dict(color='firebrick', width=2, dash='dot'),
                    hoverinfo='x+y'
                ))

        # Improve layout
        fig.update_layout(
            title="📈 Historical & Forecasted Stock Price Trends",
            xaxis=dict(title="Date", showgrid=True, tickangle=-45),
            yaxis=dict(title="Price (₹)", showgrid=True),
            template="plotly_white",
            legend=dict(title="Stock Data", x=0, y=1.1, orientation="h"),
            hovermode="x unified"
        )

        st.plotly_chart(fig)
        
        st.success("✔ Analysis Completed!")
