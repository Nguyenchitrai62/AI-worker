import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import ta
import tensorflow as tf
import time
import os
from pymongo import MongoClient, UpdateOne
from pymongo.server_api import ServerApi
from dotenv import load_dotenv
from fastapi import FastAPI
import threading
import gc

# Load environment variables
load_dotenv()
password = os.getenv('MONGO_PASSWORD')

# Connect to MongoDB Atlas
uri = f"mongodb+srv://trainguyenchi30:{password}@cryptodata.t2i1je2.mongodb.net/?retryWrites=true&w=majority&appName=CryptoData"
client = MongoClient(uri, server_api=ServerApi('1'))
db = client['my_database']
collection = db['my_collection']

# Connect to Binance
binance = ccxt.binance({
    'apiKey': '',
    'secret': '',
})

def fetch_data(count=5):
    try:
        ohlcv = binance.fetch_ohlcv('BTC/USDT', timeframe='1h', limit=count)
        df = pd.DataFrame(ohlcv, columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])
        df['Date'] = pd.to_datetime(df['Date'], unit='ms')
        return df
    except Exception as e:
        print(f"❌ Error fetching data: {e}")
        return pd.DataFrame()

def add_technical_indicators(df):
    df['close/open'] = df['Close'] / df['Open'] - 1
    df['high-low'] = (df['High'] - df['Low']) / df['Close']
    df['ema12'] = ta.trend.ema_indicator(close=df['Close'], window=12) / df['Close'] - 1
    df['ema26'] = ta.trend.ema_indicator(close=df['Close'], window=26) / df['Close'] - 1
    df['macd'] = df['ema12'] - df['ema26']
    df['rsi14'] = ta.momentum.rsi(close=df['Close'], window=14) / 50 - 1
    df['stoch_rsi'] = ta.momentum.stochrsi(close=df['Close'], window=14) * 2 - 1
    bb = ta.volatility.BollingerBands(close=df['Close'], window=14)
    df['bb_width'] = (bb.bollinger_hband() - bb.bollinger_lband()) / bb.bollinger_mavg()
    
    df.dropna(subset=['ema26'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    df = df.tail(105)
    return df

def predict_with_model(df):
    features = ['close/open', 'high-low', 'ema12', 'ema26', 'macd', 'rsi14', 'stoch_rsi', 'bb_width']
    X_raw = df[features].values

    def create_input_sequences(X, timesteps=100):
        n_samples = len(X) - timesteps
        Xs = np.empty((n_samples, timesteps, X.shape[1]))
        for i in range(n_samples):
            Xs[i] = X[i:i+timesteps]
        return Xs

    timesteps = 100
    X_seq = create_input_sequences(X_raw, timesteps)
    pred_probs = model.predict(X_seq, batch_size=32, verbose=0)
    tf.keras.backend.clear_session()  # Clear TensorFlow session
    confidence = pred_probs[:, 1]
    pred_labels = (confidence > 0.5).astype(int)

    padding = [np.nan] * timesteps
    df['predict'] = padding + pred_labels.tolist()
    df['confidence'] = padding + confidence.tolist()
    return df.dropna()

def update_mongo(df):
    try:
        data_dict = df.to_dict(orient='records')
        operations = [
            UpdateOne({'Date': doc['Date']}, {'$set': doc}, upsert=True)
            for doc in data_dict
        ]
        if operations:
            result = collection.bulk_write(operations, ordered=False)
            print(f"✅ Updated {result.modified_count} rows, inserted {result.upserts} rows into MongoDB")
    except Exception as e:
        print(f"❌ Error updating MongoDB: {e}")

def fetch_mongo_data(count=130):
    try:
        cursor = collection.find({}, {'_id': 0}).sort('Date', -1).limit(count)
        df = pd.DataFrame(list(cursor)).sort_values('Date')
        cursor.close()
        return df
    except Exception as e:
        print(f"❌ Error fetching MongoDB data: {e}")
        return pd.DataFrame()

model = tf.keras.models.load_model('transformer_model_balanced.keras')

def main_loop():
    while True:
        start_time = time.time()
        df = fetch_data(5)
        if not df.empty:
            update_mongo(df)
        df_mongo = fetch_mongo_data(300)
        if not df_mongo.empty:
            df_mongo = add_technical_indicators(df_mongo)
            df_mongo = predict_with_model(df_mongo)
            update_mongo(df_mongo)
        else:
            print("❌ Skipping predictions due to empty MongoDB data")
        elapsed = time.time() - start_time
        print(f"✅ Loop completed in {elapsed:.2f} seconds")
        gc.collect()  # Force garbage collection
        if elapsed < 2:
            time.sleep(2 - elapsed)

app = FastAPI()
main_loop_thread = None

def start_main_loop():
    global main_loop_thread
    if main_loop_thread is None or not main_loop_thread.is_alive():
        print("⚠️ main_loop thread is not alive. Restarting...")
        main_loop_thread = None
        gc.collect()
        main_loop_thread = threading.Thread(target=main_loop, daemon=True)
        main_loop_thread.start()
    else:
        print("✅ main_loop thread is still alive.")

@app.get("/ping")
async def ping():
    start_main_loop()
    return {"message": "Server alive"}

if __name__ == "__main__":
    start_main_loop()
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)