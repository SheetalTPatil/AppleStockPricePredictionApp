import streamlit as st
import pandas as pd
import numpy as np
import datetime
from statsmodels.tsa.arima.model import ARIMA

# Configure page layout
st.set_page_config(page_title="Apple Stock Prediction Dashboard", layout="wide")
st.title("🍏 Apple Stock (AAPL) 30-Day Predictive Forecasting Dashboard")
st.markdown("This dashboard leverages an optimized ARIMA(1,1,3) statistical time series architecture to predict stock prices.")

@st.cache_data
def load_historical_data():
    df = pd.read_csv('preprocessed_apple_stock.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    return df

try:
    df = load_historical_data()
    
    # Sidebar control panel
    st.sidebar.header("🛠️ Forecast Configuration")
    forecast_steps = st.sidebar.slider("Forecast Horizon (Days ahead)", min_value=1, max_value=60, value=30)
    
    # Summary Cards
    last_date = df['Date'].max().strftime('%Y-%m-%d')
    last_price = df['Close'].iloc[-1]
    
    col1, col2 = st.columns(2)
    col1.metric("Last Historic Trading Date", last_date)
    col2.metric("Last Observed Close Price", f"${last_price:.2f}")
    
    # Train forecasting engine on latest data
    if st.sidebar.button("Generate Forecast"):
        with st.spinner("Refitting ARIMA model weights to latest data..."):
            model = ARIMA(df['Close'], order=(1, 1, 3))
            results = model.fit()
            forecast_values = results.forecast(steps=forecast_steps)
            
            # Formulate future index dates
            future_dates = pd.date_range(start=df['Date'].max() + datetime.timedelta(days=1), periods=forecast_steps, freq='B')
            forecast_df = pd.DataFrame({'Date': future_dates, 'Predicted Close': forecast_values})
            
            # Visual presentation
            st.subheader(f"📈 Estimated Price Path for Next {forecast_steps} Business Days")
            chart_df = pd.DataFrame({
                'Historical': df['Close'].iloc[-90:].values
            }, index=df['Date'].iloc[-90:])
            
            # Combine historical and forecasted data for plotting
            fut_df = pd.DataFrame({'Predicted Forecast': forecast_values.values}, index=future_dates)
            combined_plot_df = pd.concat([chart_df, fut_df], axis=1)
            
            st.line_chart(combined_plot_df)
            
            # Tabular results
            st.subheader("📋 Forecasted Prices Table")
            st.dataframe(forecast_df.style.format({'Predicted Close': '${:.2f}'}), use_container_width=True)
            
except Exception as e:
    st.error(f"Initialization Error: Ensure 'preprocessed_apple_stock.csv' is present. Details: {e}")