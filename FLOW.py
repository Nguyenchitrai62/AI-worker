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

# Load biến môi trường
load_dotenv()
password = os.getenv('MONGO_PASSWORD')

# Kết nối MongoDB Atlas
uri = f"mongodb+srv://trainguyenchi30:{password}@cryptodata.t2i1je2.mongodb.net/?retryWrites=true&w=majority&appName=CryptoData"
client = MongoClient(uri, server_api=ServerApi('1'))

# Truy cập DB và Collection
db = client['my_database']
collection = db['AI_prediction']

# Kết nối với Binance
binance = ccxt.binance({
    'apiKey': '',
    'secret': '',
})

def fetch_data(count = 130):
    try:
        ohlcv = binance.fetch_ohlcv('BTC/USDT', timeframe='1h', limit=count)
        df = pd.DataFrame(ohlcv, columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])
        df['Date'] = pd.to_datetime(df['Date'], unit='ms')
        # print(f"✅ Fetched {len(df)} recent sessions")
        return df
    except Exception as e:
        print(f"❌ Error fetching data: {e}")
        return pd.DataFrame()

# Hàm để thêm chỉ báo kỹ thuật vào dữ liệu
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
    return df

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
    
# model = tf.keras.models.load_model('model_e.keras', custom_objects={'LinearPositionalEncoding': LinearPositionalEncoding})
model = tf.keras.models.load_model('transformer_model_balanced.keras')

# Hàm dự đoán với mô hình Transformer
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

# Hàm chính để chạy liên tục và đo thời gian mỗi vòng lặp
def main():
    while True:
        start_time = time.time()  # Lấy thời gian bắt đầu vòng lặp

        # Crawl dữ liệu từ Binance
        df = fetch_data(1000)

        # Thêm chỉ báo kỹ thuật
        df = add_technical_indicators(df)
        
        # Dự đoán với mô hình Transformer
        df = predict_with_model(df)
        
        print(df['confidence'])

        final_row_count = len(df)
        print(f"🔍 Số dòng sau khi dropna: {final_row_count}")    
        
        # Cập nhật MongoDB
        update_mongo(df)

        end_time = time.time()  # Lấy thời gian kết thúc vòng lặp

        loop_duration = end_time - start_time  # Tính thời gian vòng lặp
        print(f"✅ Vòng lặp hoàn thành trong {loop_duration:.2f} giây.")

if __name__ == "__main__":
    main()
