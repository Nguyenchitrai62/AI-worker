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

# Load biến môi trường
load_dotenv()
password = os.getenv('MONGO_PASSWORD')

# Kết nối MongoDB Atlas
uri = f"mongodb+srv://trainguyenchi30:{password}@cryptodata.t2i1je2.mongodb.net/?retryWrites=true&w=majority&appName=CryptoData"
client = MongoClient(uri, server_api=ServerApi('1'))

# Truy cập DB và Collection
db = client['my_database']
collection = db['my_collection']

# Kết nối với Binance
binance = ccxt.binance({
    'apiKey': '',
    'secret': '',
})

symbol = 'BTC/USDT'
limit = 1000
total_limit = 1000
num_requests = total_limit // limit

# Hàm để fetch dữ liệu từ Binance
def fetch_data():
    current_time = int(datetime.now().timestamp() * 1000)
    ohlcv = []

    for i in range(num_requests):
        since = current_time - (i + 1) * limit * 60 * 60 * 55
        data = binance.fetch_ohlcv(symbol, timeframe='1h', limit=limit, since=since)
        if not data:
            break
        ohlcv[:0] = data
        print(f"{i+1} / {num_requests}")
        # time.sleep(binance.rateLimit / 1000)
    df = pd.DataFrame(ohlcv, columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])
    df['Date'] = pd.to_datetime(df['Date'], unit='ms')
    return df

# Hàm để thêm chỉ báo kỹ thuật vào dữ liệu
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

# Load mô hình AI (chạy 1 lần duy nhất)
model = tf.keras.models.load_model('transformer_model_balanced.keras')

# Hàm dự đoán với mô hình Transformer
def predict_with_model(df):
    features = ['close/open', 'high-low', 'ema12', 'ema26', 'macd', 'rsi14', 'stoch_rsi', 'bb_width']
    X_raw = df[features].values

    def create_input_sequences(X, timesteps=24):
        Xs = []
        for i in range(len(X) - timesteps):
            Xs.append(X[i:i+timesteps])
        return np.array(Xs)

    timesteps = 24
    X_seq = create_input_sequences(X_raw, timesteps)

    pred_probs = model.predict(X_seq, verbose=0)
    confidence = pred_probs[:, 1]
    pred_labels = (confidence > 0.5).astype(int)

    padding = [np.nan] * timesteps
    df['predict'] = padding + pred_labels.tolist()
    df['confidence'] = padding + confidence.tolist()

    return df.dropna()

# Hàm cập nhật MongoDB
def update_mongo(df):
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
    print(f"✅ Đã cập nhật {updated} dòng, chèn mới {inserted} dòng vào MongoDB.")

# Vòng lặp chính chạy liên tục
def main_loop():
    while True:
        start_time = time.time()

        df = fetch_data()
        df = add_technical_indicators(df)
        df = predict_with_model(df)
        
        update_mongo(df)

        end_time = time.time()
        elapsed = end_time - start_time
        print(f"✅ Vòng lặp hoàn thành trong {elapsed:.2f} giây.")
        
        # Nếu vòng lặp chạy chưa đủ 2 giây, chờ thêm
        if elapsed < 2:
            time.sleep(2 - elapsed)

# Tạo FastAPI app
app = FastAPI()

@app.get("/ping")
async def ping():
    return {"message": "Server alive"}

if __name__ == "__main__":
    # Chạy main_loop trong thread để không block API
    thread = threading.Thread(target=main_loop, daemon=True)
    thread.start()

    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
