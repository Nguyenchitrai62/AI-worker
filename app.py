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

# Load environment variables
load_dotenv()
password = os.getenv('MONGO_PASSWORD')

# Connect to MongoDB Atlas
uri = f"mongodb+srv://trainguyenchi30:{password}@cryptodata.t2i1je2.mongodb.net/?retryWrites=true&w=majority&appName=CryptoData"
client = MongoClient(uri, server_api=ServerApi('1'))

# Access DB and Collection
db = client['my_database']
collection = db['my_collection']

# Connect to Binance
binance = ccxt.binance({
    'apiKey': '',
    'secret': '',
})

def fetch_data():
    try:
        ohlcv = binance.fetch_ohlcv('BTC/USDT', timeframe='1h', limit=5)
        df = pd.DataFrame(ohlcv, columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])
        df['Date'] = pd.to_datetime(df['Date'], unit='ms')
        # print(f"✅ Fetched {len(df)} recent sessions")
        return df
    except Exception as e:
        print(f"❌ Error fetching data: {e}")
        return pd.DataFrame()

# Add technical indicators to DataFrame
def add_technical_indicators(df):
    df['close/open'] = df['Close'] / df['Open'] - 1
    df['high-low'] = (df['High'] - df['Low']) / df['Close']
    df['ema12'] = ta.trend.ema_indicator(close=df['Close'], window=12) / df['Close'] - 1
    df['ema26'] = ta.trend.ema_indicator(close=df['Close'], window=26) / df['Close'] - 1
    df['macd'] = (df['ema12'] - df['ema26']) / df['Close']
    df['rsi14'] = ta.momentum.RSIIndicator(close=df['Close'], window=14).rsi() / 50 - 1
    df['stoch_rsi'] = ta.momentum.StochRSIIndicator(close=df['Close'], window=14).stochrsi() * 2 - 1
    bb = ta.volatility.BollingerBands(close=df['Close'], window=14)
    df['bb_width'] = (bb.bollinger_hband() - bb.bollinger_lband()) / bb.bollinger_mavg()
    return df.dropna()

# Load AI model (run once)
model = tf.keras.models.load_model('transformer_model_balanced.keras')

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

def fetch_mongo_data():
    try:
        cursor = collection.find({}, {'_id': 0}).sort('Date', -1).limit(130)
        df = pd.DataFrame(cursor).sort_values('Date')
        return df
    except Exception as e:
        print(f"❌ Error fetching MongoDB data: {e}")
        return pd.DataFrame()

# Main loop
def main_loop():
    while True:
        start_time = time.time()

        # Step 1: Fetch 5 recent sessions
        df = fetch_data()
        if not df.empty:
            # Step 2: Update MongoDB with new data
            update_mongo(df)

        # Step 3: Fetch 130 latest sessions from MongoDB
        df_mongo = fetch_mongo_data()
        if not df_mongo.empty:
            # Step 4: Add technical indicators
            df_mongo = add_technical_indicators(df_mongo)
            
            # Step 5: Run AI model predictions
            df_mongo = predict_with_model(df_mongo)
            
            # print(df_mongo['confidence'])
            
            # Step 6: Update MongoDB with predictions
            update_mongo(df_mongo)
            
        else:
            print("❌ Skipping predictions due to empty MongoDB data")

        end_time = time.time()
        elapsed = end_time - start_time
        print(f"✅ Loop completed in {elapsed:.2f} seconds")

        # Wait for next iteration (minimum 2 seconds)
        if elapsed < 2:
            time.sleep(2 - elapsed)

# Create FastAPI app
app = FastAPI()

@app.get("/ping")
async def ping():
    return {"message": "Server alive"}

if __name__ == "__main__":
    # Run main_loop in a separate thread
    thread = threading.Thread(target=main_loop, daemon=True)
    thread.start()

    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)