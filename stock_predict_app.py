import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model

# Load trained model (you must save this earlier as 'stock_model.h5')
model = load_model('stock_model.h5')

st.title("📈 AI Stock Direction Predictor")
stock_symbol = st.text_input("Enter Stock Ticker (e.g., AAPL, RELIANCE.NS)", 'AAPL')

if st.button("Predict"):
    data = yf.download(stock_symbol, period="1y")
    
    # Indicators
    data['MA10'] = data['Close'].rolling(10).mean()
    data['MA50'] = data['Close'].rolling(50).mean()
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    
    data.dropna(inplace=True)
    X = data[['MA10', 'MA50', 'RSI']]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    latest_data = X_scaled[-1].reshape(1, -1)
    prediction = model.predict(latest_data)[0][0]

    st.subheader("🔮 Prediction:")
    if prediction > 0.5:
        st.success("Tomorrow: Stock will go **UP** 📈")
    else:
        st.error("Tomorrow: Stock will go **DOWN** 📉")

    st.line_chart(data['Close'])
model.save("stock_model.h5")
