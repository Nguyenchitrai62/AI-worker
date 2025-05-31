import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import ta
import tensorflow as tf
import time
import os
from pymongo import MongoClient
from pymongo.server_api import ServerApi
from dotenv import load_dotenv
from fastapi import FastAPI
import threading
import gc;

# Load environment variables
load_dotenv()
password = os.getenv('MONGO_PASSWORD')

# Connect to MongoDB Atlas
uri = f"mongodb+srv://trainguyenchi30:{password}@cryptodata.t2i1je2.mongodb.net/?retryWrites=true&w=majority&appName=CryptoData"
client = MongoClient(uri, server_api=ServerApi('1'))

# Access DB and Collection
db = client['my_database']
collection = db['AI_prediction']

# Connect to Binance
binance = ccxt.binance({
    'apiKey': '',
    'secret': '',
})

def fetch_data(count = 5):
    try:
        ohlcv = binance.fetch_ohlcv('BTC/USDT', timeframe='1h', limit=count)
        df = pd.DataFrame(ohlcv, columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])
        df['Date'] = pd.to_datetime(df['Date'], unit='ms')
        # print(f"✅ Fetched {len(df)} recent sessions")
        return df
    except Exception as e:
        print(f"❌ Error fetching data: {e}")
        return pd.DataFrame()

# Add technical indicators to DataFrame
def add_technical_indicators(df):
    df['close/open'] = df['Close'] / df['Open'] -1 
    df['high-low'] = (df['High'] - df['Low']) / df['Close']
    df['ema12'] = ta.trend.ema_indicator(close=df['Close'], window=12) /df['Close'] -1
    df['ema26'] = ta.trend.ema_indicator(close=df['Close'], window=26) /df['Close'] -1
    df['macd'] = (df['ema12'] - df['ema26'])
    df['rsi14'] = ta.momentum.rsi(close=df['Close'], window=14) / 50 - 1
    df['stoch_rsi'] = ta.momentum.stochrsi(close=df['Close'], window=14) * 2 -1 
    bb = ta.volatility.BollingerBands(close=df['Close'], window=14)
    df['bb_width'] = (bb.bollinger_hband() - bb.bollinger_lband()) / bb.bollinger_mavg()
    
    df = df.dropna(subset=['ema26']).tail(102).reset_index(drop=True)

    return df


# Predict with Transformer model
def predict_with_model(df):
    features = ['close/open', 'high-low', 'ema12', 'ema26', 'macd', 'rsi14', 'stoch_rsi', 'bb_width']
    X_raw = df[features].values

    def create_input_sequences(X, timesteps=100):
        Xs = []
        for i in range(len(X) - timesteps):
            Xs.append(X[i:i+timesteps])
        return np.array(Xs)

    timesteps = 100
    X_seq = create_input_sequences(X_raw, timesteps)

    pred_probs = model.predict(X_seq, verbose=0)
    confidence = pred_probs[:, 1]
    pred_labels = (confidence > 0.5).astype(int)

    padding = [np.nan] * timesteps
    df['predict'] = padding + pred_labels.tolist()
    df['confidence'] = padding + confidence.tolist()

    return df.dropna()

# Update MongoDB with new data
def update_mongo(df):
    try:
        data_dict = df.to_dict(orient='records')
        updated, inserted = 0, 0
        for doc in data_dict:
            existing_doc = collection.find_one({'Date': doc['Date']})
            if existing_doc is None:
                collection.insert_one(doc)
                inserted += 1
            else:
                different = any(existing_doc.get(k) != doc.get(k) for k in doc.keys() if k != '_id')
                if different:
                    collection.update_one({'Date': doc['Date']}, {'$set': doc})
                    updated += 1
        # print(f"✅ Updated {updated} rows, inserted {inserted} rows into MongoDB")
    except Exception as e:
        print(f"❌ Error updating MongoDB: {e}")

def fetch_mongo_data(count = 130):
    try:
        cursor = collection.find({}, {'_id': 0}).sort('Date', -1).limit(count)
        df = pd.DataFrame(cursor).sort_values('Date')
        return df
    except Exception as e:
        print(f"❌ Error fetching MongoDB data: {e}")
        return pd.DataFrame()
    
# ===== Linear Positional Encoding =====
class LinearPositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(LinearPositionalEncoding, self).__init__(**kwargs)
    
    def call(self, inputs):
        seq_len = tf.shape(inputs)[1]
        feature_dim = tf.shape(inputs)[2]
        
        # Tạo position weights: càng gần hiện tại thì weight càng cao
        positions = tf.range(seq_len, dtype=tf.float32)
        position_weights = 0.05 + positions / tf.cast(seq_len - 1, tf.float32)
        
        # Reshape để broadcast
        position_weights = tf.reshape(position_weights, [1, seq_len, 1])
        position_weights = tf.tile(position_weights, [1, 1, feature_dim])
        
        return inputs * position_weights
    
class ExponentialPositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, decay_rate=0.03, **kwargs):
        super(ExponentialPositionalEncoding, self).__init__(**kwargs)
        self.decay_rate = decay_rate
    
    def call(self, inputs):
        seq_len = tf.shape(inputs)[1]
        feature_dim = tf.shape(inputs)[2]
        
        positions = tf.range(seq_len, dtype=tf.float32)
        position_weights = tf.exp(-self.decay_rate * (tf.cast(seq_len - 1, tf.float32) - positions))
        
        position_weights = tf.reshape(position_weights, [1, seq_len, 1])
        position_weights = tf.tile(position_weights, [1, 1, feature_dim])
        
        return inputs * position_weights
    
    def get_config(self):
        config = super().get_config()
        config.update({"decay_rate": self.decay_rate})
        return config
    
model = tf.keras.models.load_model('model_e.keras', custom_objects={'LinearPositionalEncoding': LinearPositionalEncoding})
# model = tf.keras.models.load_model('transformer_model_balanced.keras')

def main_loop():
    while True:
        start_time = time.time()

        df = fetch_data(5)
        if not df.empty:
            update_mongo(df)

        df_mongo = fetch_mongo_data(200)
        if not df_mongo.empty:
            df_mongo = add_technical_indicators(df_mongo)
            df_mongo = predict_with_model(df_mongo)
            
            # print(df_mongo['confidence'])
            
            update_mongo(df_mongo)
            
        else:
            print("❌ Skipping predictions due to empty MongoDB data")


        
        elapsed = time.time() - start_time
        print(f"✅ Loop completed in {elapsed:.2f} seconds")

        # Wait for next iteration (minimum 2 seconds)
        if elapsed < 2:
            time.sleep(2 - elapsed)

# Create FastAPI app
app = FastAPI()

# Global thread holder
main_loop_thread = None

def start_main_loop():
    global main_loop_thread
    if main_loop_thread is None or not main_loop_thread.is_alive():
        print("⚠️ main_loop thread is not alive. Restarting...")
        main_loop_thread = None 
        tf.keras.backend.clear_session()
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
