import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import plotly.graph_objects as go

# Function to fetch stock data from Yahoo Finance
def get_stock_data(symbol, from_date, to_date):
    try:
        df = yf.download(symbol, start=from_date, end=to_date)
        if df.empty:
            return None  

        df = df.reset_index()
        df.rename(columns={"Adj Close": "ClosePrice"}, inplace=True)
        df["Date"] = pd.to_datetime(df["Date"])
        
        return df
    except Exception as e:
        st.error(f"Error fetching data for {symbol}: {str(e)}")
        return None

# Function to calculate Sharpe ratio
def sharpe_ratio(weights, returns, risk_free_rate=0.03):
    portfolio_return = np.sum(weights * returns.mean()) * 252
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
    return -(portfolio_return - risk_free_rate) / portfolio_volatility  # Negative because we minimize

# Function to optimize capital allocation for max Sharpe Ratio
def optimize_allocation(stock_data, capital):
    try:
        returns = pd.DataFrame({
            stock: df["ClosePrice"].pct_change().dropna() for stock, df in stock_data.items() if "ClosePrice" in df
        })
        
        if returns.empty:
            st.error("No valid stock return data available.")
            return {}, {}

        stocks = list(returns.columns)

        if len(stocks) < 2:
            return {stock: capital / len(stocks) for stock in stocks}, {stock: 100 / len(stocks) for stock in stocks} if stocks else ({}, {})

        constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})
        bounds = [(0, 1)] * len(stocks)
        init_guess = np.ones(len(stocks)) / len(stocks)
        result = minimize(sharpe_ratio, init_guess, args=(returns,), bounds=bounds, constraints=constraints)

        if not result.success:
            st.warning("Optimization failed, using equal allocation.")
            allocation = {stocks[i]: capital / len(stocks) for i in range(len(stocks))}
            allocation_percentage = {stocks[i]: 100 / len(stocks) for i in range(len(stocks))}
        else:
            optimized_weights = result.x
            allocation = {stocks[i]: optimized_weights[i] * capital for i in range(len(stocks))}
            allocation_percentage = {stocks[i]: optimized_weights[i] * 100 for i in range(len(stocks))}

        return allocation, allocation_percentage

    except Exception as e:
        st.error(f"Error in allocation optimization: {str(e)}")
        return {}, {}

# Streamlit UI
st.title("🚀 AI-Powered Stock Allocation & Forecasting")
st.markdown("### Optimize your stock investments and predict future trends with AI-powered analysis! 📊")

st.sidebar.header("Enter Stock Details")
num_stocks = st.sidebar.number_input("Number of Stocks", min_value=1, max_value=10, value=3)
stocks = [st.sidebar.text_input(f"Stock {i+1} Symbol", "").strip().upper() for i in range(num_stocks)]
from_date = st.sidebar.text_input("Start Date (YYYY-MM-DD)", "2022-01-01")
to_date = st.sidebar.text_input("End Date (YYYY-MM-DD)", "2025-01-01")
capital = st.sidebar.number_input("Total Capital (₹)", min_value=10000, value=100000)

if st.sidebar.button("Run Analysis"):
    stock_data = {stock: get_stock_data(stock, from_date, to_date) for stock in stocks if stock}
    stock_data = {k: v for k, v in stock_data.items() if v is not None}

    if len(stock_data) == 0:
        st.error("No valid stock data available. Please check stock symbols and try again.")
    else:
        allocation, allocation_percentage = optimize_allocation(stock_data, capital)

        st.subheader("💰 Optimized Capital Allocation")
        allocation_df = pd.DataFrame({
            "Stock": allocation.keys(), 
            "Allocation (₹)": allocation.values(), 
            "Allocation (%)": allocation_percentage.values()
        })
        allocation_df["Allocation Ratio"] = allocation_df["Allocation (%)"] / 100  # Allocation ratio column
        st.dataframe(allocation_df)
        
        # Plot stock data
        st.subheader("📉 Historical & Forecasted Price Trends")
        fig = go.Figure()
        for stock, df in stock_data.items():
            fig.add_trace(go.Scatter(
                x=df["Date"], y=df["ClosePrice"],
                mode='lines', name=f"{stock} Historical",
                line=dict(width=2),
                hoverinfo='x+y'
            ))
        
        fig.update_layout(
            title="📈 Historical Stock Price Trends",
            xaxis=dict(title="Date", showgrid=True, tickangle=-45),
            yaxis=dict(title="Price (₹)", showgrid=True),
            template="plotly_white",
            legend=dict(title="Stock Data", x=0, y=1.1, orientation="h"),
            hovermode="x unified"
        )
        st.plotly_chart(fig)
        
        st.success("✔ Analysis Completed!")

