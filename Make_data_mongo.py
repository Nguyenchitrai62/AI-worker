import ccxt
import pandas as pd
import numpy as np
from datetime import datetime
import ta
import tensorflow as tf
import os
from pymongo import MongoClient
from pymongo.server_api import ServerApi
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
password = os.getenv('MONGO_PASSWORD')

# Connect to MongoDB Atlas
uri = f"mongodb+srv://trainguyenchi30:{password}@cryptodata.t2i1je2.mongodb.net/?retryWrites=true&w=majority&appName=CryptoData"
client = MongoClient(uri, server_api=ServerApi('1'))
db = client['my_database']
collection = db['AI_prediction']

# Load LSTM model
try:
    model = tf.keras.models.load_model('lstm_model.keras')
    print("✅ Model loaded")
except Exception as e:
    print(f"❌ Model loading failed: {e}")
    model = None

# Fetch data from Binance
def fetch_data(limit=130):
    try:
        binance = ccxt.binance()
        ohlcv = binance.fetch_ohlcv('BTC/USDT', '1h', limit=limit)
        df = pd.DataFrame(ohlcv, columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])
        df['Date'] = pd.to_datetime(df['Date'], unit='ms')
        return df
    except Exception as e:
        print(f"❌ Fetch data error: {e}")
        return pd.DataFrame()

# Add indicators
def add_indicators(df):
    df['close/open'] = df['Close'] / df['Open'] - 1
    df['high-low'] = (df['High'] - df['Low']) / df['Close']
    df['ema12'] = ta.trend.ema_indicator(df['Close'], window=12) / df['Close'] - 1
    df['ema26'] = ta.trend.ema_indicator(df['Close'], window=26) / df['Close'] - 1
    df['macd'] = df['ema12'] - df['ema26']
    df['rsi14'] = ta.momentum.rsi(df['Close'], window=14) / 50 - 1
    df['stoch_rsi'] = ta.momentum.stochrsi(df['Close'], window=14) * 2 - 1
    bb = ta.volatility.BollingerBands(df['Close'], window=14)
    df['bb_width'] = (bb.bollinger_hband() - bb.bollinger_lband()) / bb.bollinger_mavg()
    return df.dropna(subset=['ema26']).reset_index(drop=True)

# Predict
def predict(df):
    features = ['close/open', 'high-low', 'ema12', 'ema26', 'macd', 'rsi14', 'stoch_rsi', 'bb_width']
    X = df[features].values

    def make_sequences(X, timesteps=100):
        return np.array([X[i:i+timesteps] for i in range(len(X) - timesteps)])

    X_seq = make_sequences(X)
    preds = model.predict(X_seq, verbose=0)
    confidence = preds[:, 1]
    labels = (confidence > 0.5).astype(int)

    df['predict'] = [np.nan]*100 + labels.tolist()
    df['confidence'] = [np.nan]*100 + confidence.tolist()
    return df.dropna()

# Run full pipeline
def run_pipeline():
    df = fetch_data(1000)
    if df.empty or model is None:
        print("❌ No data or model not loaded")
        return

    df = add_indicators(df)
    df = predict(df)

    # Convert datetime to ISO format for MongoDB
    df['Date'] = df['Date'].astype(str)
    docs = df.to_dict(orient='records')

    # Fastest MongoDB refresh: drop then insert
    collection.delete_many({})
    collection.insert_many(docs)
    print(f"✅ Inserted {len(docs)} rows into MongoDB")

# Trigger
if __name__ == "__main__":
    run_pipeline()
