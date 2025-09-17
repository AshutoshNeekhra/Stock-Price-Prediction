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
st.subheader('Stock Data (last 5 rows)')
st.write(data.tail(5))

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

# Predict future prices
for i in range(future_days):
    predicted_price = model.predict(current_batch)[0,0]
    predictions.append(predicted_price)
    # Maintain proper shape for next iteration
    current_batch = np.append(current_batch[:,1:,:], predicted_price.reshape(1,1,1), axis=1)

# Reverse scaling
predicted_prices = scaler.inverse_transform(np.array(predictions).reshape(-1,1)).flatten()

# Prepare future dates
future_prices_for_plot = np.concatenate((np.array([close_data.iloc[-1]]), predicted_prices))
future_dates = pd.date_range(close_data.index[-1], periods=future_days+1)
# Calculate moving averages
ma_50 = close_data.rolling(50).mean()
ma_100 = close_data.rolling(100).mean()

# Extend moving averages to future (flat extension)
ma_50_future = np.concatenate((ma_50.values, [ma_50.values[-1]]*future_days))
ma_100_future = np.concatenate((ma_100.values, [ma_100.values[-1]]*future_days))
all_dates = pd.date_range(close_data.index[0], periods=len(ma_50_future))

# Display predicted prices
st.subheader(f'📅 Predicted Prices for Next {future_days} Days')
pred_df = pd.DataFrame({'Date': future_dates[1:], 'Predicted Close': predicted_prices})
st.write(pred_df)

# Plot everything
st.subheader('📈 Stock Price Prediction with Moving Averages')
fig, ax = plt.subplots(figsize=(12,6))

# Historical close
ax.plot(close_data.index, close_data, label='Historical Close', color='green')

# Moving averages
ax.plot(all_dates, ma_50_future, label='MA50', color='orange', linestyle='--')
ax.plot(all_dates, ma_100_future, label='MA100', color='blue', linestyle='--')

# Predicted future close
ax.plot(future_dates, future_prices_for_plot, 'r-o', label='Predicted Future Close')

ax.set_xlabel('Date')
ax.set_ylabel('Price')
ax.legend()
plt.xticks(rotation=30)
st.pyplot(fig)
