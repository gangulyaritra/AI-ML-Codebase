"""
======================================================================================
References:
    1. FB Prophet: https://facebook.github.io/prophet/
    2. Streamlit: https://streamlit.io/
    3. Yahoo Finance: https://finance.yahoo.com/
    4. YouTube Explanation Video: https://www.youtube.com/watch?v=0E_31WqVzCY

Requirements:
    1. Python 3.7.0
    2. pip install pystan==2.19.1.1 fbprophet streamlit yfinance plotly
======================================================================================
"""

# Import Library.
import streamlit as st
from datetime import date
import yfinance as yf
from fbprophet import Prophet
from fbprophet.plot import plot_plotly
from plotly import graph_objs as go

START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title("Stock Forecasting App")

stocks = ("GOOG", "AAPL", "MSFT", "FB", "AMZN", "NFLX")
selected_stock = st.selectbox("Select SYMBOL to Forecast Pricing Trends", stocks)

n_years = st.slider("Years of Prediction", 1, 4)
period = n_years * 365


@st.cache
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data


data_load_state = st.text("Loading Data.....")
data = load_data(selected_stock)
data_load_state.text("Loading Data..... DONE!")

st.subheader("Historical Dataset")
st.write(data.tail())

# Plot Historical Dataset.
def plot_historical_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data["Date"], y=data["Open"], name="Opening Price"))
    fig.add_trace(go.Scatter(x=data["Date"], y=data["Close"], name="Closing Price"))
    fig.layout.update(
        title_text="Time Series Data with Rangeslider", xaxis_rangeslider_visible=True
    )
    st.plotly_chart(fig)


plot_historical_data()

# Forecast Prices with FB Prophet.
df_train = data[["Date", "Close"]]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

# Plot Forecast Trends.
st.subheader("Forecast Dataset")
st.write(forecast.tail())

st.write(f"Forecast Plot for {n_years} year(s)")
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

st.write("Forecast Components")
fig2 = m.plot_components(forecast)
st.write(fig2)

# """ Run this on Terminal: `streamlit run FBProphetApp.py` """
