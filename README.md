WEBSITE LINK - https://stock-price-prediction-net8spwqkenfskfdkx7wae.streamlit.app/



Stock Price Prediction Using LSTM

A web-based application to predict stock prices using LSTM (Long Short-Term Memory) deep learning models and visualize historical vs predicted data. The app also provides a 30-day future price forecast table for interactive stock analysis.

Features

Predicts stock prices based on historical data using an LSTM neural network.

Displays historical vs predicted prices in a clear plot.

Provides a 30-day future stock price prediction table.

Interactive input: enter any stock symbol to get predictions.

Web-based interface built with Streamlit for easy access.

Technologies Used

Python

TensorFlow / Keras – LSTM model for sequence prediction

Streamlit – Web app interface

Yahoo Finance API (yfinance) – Fetch historical stock data

NumPy & Pandas – Data manipulation

Matplotlib – Data visualization

Scikit-learn – Data preprocessing (MinMaxScaler)

Installation

Clone the repository:

git clone <your-repo-url>
cd stock-price-prediction


Create a virtual environment (optional but recommended):

python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows


Install dependencies:

pip install -r requirements.txt


Run the Streamlit app:

streamlit run app.py

Usage

Open the app in your browser.

Enter a stock symbol (e.g., GOOG, AAPL) in the input box.

View:

Historical stock data

Plots of original vs predicted prices

30-day future price prediction table

Project Structure
stock-price-prediction/
│
├── app.py               # Main Streamlit app
├── Stock Prediction Model.keras  # Trained LSTM model
├── requirements.txt     # Python dependencies
└── README.md            # Project documentation

Future Improvements

Add technical indicators like MA50, RSI, MACD for better prediction accuracy.

Enable multi-stock predictions in a single view.

Deploy as a public web app with continuous updates from live stock data.

Author

Ashutosh Neekhra

Email: neekhraashutosh@gmail.com

GitHub: [your-github-link]
