import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Load model
model = load_model("Stock Prediction Model.keras")

st.header('📈 Stock Price Prediction')

# Input
stock = st.text_input('Enter Stock Symbol', 'GOOG')
start = '2012-01-01'
end = '2024-12-31'

# Download stock data
data = yf.download(stock, start, end)
st.subheader('Stock Data')
st.write(data.tail(5))  # show last 5 rows

# Close price
close_data = data['Close']

# Scaling
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(np.array(close_data).reshape(-1,1))

# Get last 100 days for prediction
last_100_days = scaled_data[-100:]

future_days = 30
predictions = []

current_batch = last_100_days.reshape(1, 100, 1)

for i in range(future_days):
    predicted_price = model.predict(current_batch)[0,0]
    predictions.append(predicted_price)
    
    # Update current batch by appending the new prediction and removing first value
    current_batch = np.append(current_batch[:,1:,:], predicted_price.reshape(1,1,1), axis=1)

# Reverse scaling
predicted_prices = scaler.inverse_transform(np.array(predictions).reshape(-1,1))

# Prepare future dates
last_date = data.index[-1]
future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=future_days)

# Display predicted prices
st.subheader(f'📅 Predicted Prices for Next {future_days} Days')
pred_df = pd.DataFrame({'Date': future_dates, 'Predicted Close': predicted_prices.flatten()})
st.write(pred_df)

# Plot historical + predicted prices
st.subheader('📈 Historical Close vs Predicted Future Close')
plt.figure(figsize=(12,6))
plt.plot(close_data, label='Historical Close')
plt.plot(future_dates, predicted_prices, label='Predicted Future Close', color='red')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
st.pyplot(plt)
