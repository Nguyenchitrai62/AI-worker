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
import gc

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

# Predict with LSTM model (Updated for LSTM)
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

    # LSTM model prediction - returns probability for 2 classes
    pred_probs = model.predict(X_seq, verbose=0)
    confidence = pred_probs[:, 1]  # Probability for class 1
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
        return {"updated": updated, "inserted": inserted}
    except Exception as e:
        print(f"❌ Error updating MongoDB: {e}")
        return {"error": str(e)}

def fetch_mongo_data(count = 130):
    try:
        cursor = collection.find({}, {'_id': 0}).sort('Date', -1).limit(count)
        df = pd.DataFrame(cursor).sort_values('Date')
        return df
    except Exception as e:
        print(f"❌ Error fetching MongoDB data: {e}")
        return pd.DataFrame()

# Load LSTM model at startup (Updated model name)
try:
    model = tf.keras.models.load_model('lstm_model.keras')
    print("✅ LSTM model loaded successfully")
except Exception as e:
    print(f"❌ Error loading LSTM model: {e}")
    model = None

def run_prediction_pipeline():
    """
    Thực hiện toàn bộ pipeline: crawl data -> tính chỉ báo -> AI prediction -> update mongo
    """
    start_time = time.time()
    
    try:
        # Step 1: Fetch new data from Binance
        df_new = fetch_data(5)
        if df_new.empty:
            return {"error": "Failed to fetch data from Binance"}
        
        # Step 2: Update MongoDB with new raw data
        update_result = update_mongo(df_new)
        
        # Step 3: Fetch comprehensive data from MongoDB for prediction
        df_mongo = fetch_mongo_data(200)
        if df_mongo.empty:
            return {"error": "No data available in MongoDB for prediction"}
        
        # Step 4: Add technical indicators
        df_mongo = add_technical_indicators(df_mongo)
        
        # Step 5: AI Prediction with LSTM
        if model is not None:
            df_mongo = predict_with_model(df_mongo)
            
            # Step 6: Update MongoDB with predictions
            prediction_update = update_mongo(df_mongo)
        else:
            return {"error": "LSTM model not loaded"}
        
        elapsed = time.time() - start_time
        
        # Get latest prediction info
        latest_data = df_mongo.tail(1).iloc[0] if not df_mongo.empty else None
        
        return {
            "status": "success",
            "execution_time": f"{elapsed:.2f}s",
            "data_update": update_result,
            "prediction_update": prediction_update,
            "model_type": "LSTM",
            "latest_prediction": {
                "date": str(latest_data['Date']) if latest_data is not None else None,
                "prediction": int(latest_data['predict']) if latest_data is not None and not pd.isna(latest_data['predict']) else None,
                "confidence": float(latest_data['confidence']) if latest_data is not None and not pd.isna(latest_data['confidence']) else None
            } if latest_data is not None else None
        }
        
    except Exception as e:
        print(f"❌ Pipeline error: {e}")
        return {"error": f"Pipeline failed: {str(e)}"}

# Create FastAPI app
app = FastAPI(title="Crypto LSTM Prediction API", version="1.0.0")

@app.get("/")
async def root():
    return {"message": "Crypto LSTM Prediction API is running"}

@app.get("/ping")
async def ping():
    """
    Endpoint để trigger pipeline prediction
    """
    result = run_prediction_pipeline()
    return result

@app.get("/status")
async def status():
    """
    Endpoint để check trạng thái của model và database
    """
    try:
        # Check database connection
        db_status = "connected" if client.admin.command('ping') else "disconnected"
        
        # Check model status
        model_status = "loaded" if model is not None else "not_loaded"
        
        return {
            "database": db_status,
            "model": model_status,
            "model_type": "LSTM",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "database": "error",
            "model": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)