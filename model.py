import pandas as pd
import numpy as np
import ta
import tensorflow as tf
import ccxt

# Kết nối với Binance
binance = ccxt.binance({
    'apiKey': '',
    'secret': '',
})
# ==== 1. Đọc dữ liệu OHLCV gốc ====
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
    
df = fetch_data(200)

# ========= 2. Tính các chỉ báo kỹ thuật =========

df['close/open'] = df['Close'] / df['Open'] -1 
df['high-low'] = (df['High'] - df['Low']) / df['Close']
df['ema12'] = ta.trend.ema_indicator(close=df['Close'], window=12) /df['Close'] -1
df['ema26'] = ta.trend.ema_indicator(close=df['Close'], window=26) /df['Close'] -1
df['macd'] = (df['ema12'] - df['ema26'])
df['rsi14'] = ta.momentum.rsi(close=df['Close'], window=14) / 50 - 1
df['stoch_rsi'] = ta.momentum.stochrsi(close=df['Close'], window=14) * 2 -1 
bb = ta.volatility.BollingerBands(close=df['Close'], window=14)
df['bb_width'] = (bb.bollinger_hband() - bb.bollinger_lband()) / bb.bollinger_mavg()

# ==== 3. Loại bỏ các dòng NaN (do chỉ báo chưa đủ dữ liệu) ====
df = df.dropna()

# ==== 4. Load model đã huấn luyện ====
model = tf.keras.models.load_model('transformer_model_balanced.keras')

# ==== 5. Tạo dữ liệu đầu vào từ các đặc trưng ====
features = ['close/open', 'high-low', 'ema12', 'ema26', 'macd', 'rsi14', 'stoch_rsi', 'bb_width']
X_raw = df[features].values

# ==== 6. Tạo sequence theo timestep ====
timesteps = 100

def create_input_sequences(X, timesteps):
    Xs = []
    for i in range(len(X) - timesteps):
        Xs.append(X[i:i+timesteps])
    return np.array(Xs)

X_seq = create_input_sequences(X_raw, timesteps)

# ==== 7. Dự đoán với model ====
pred_probs = model.predict(X_seq, verbose=0)
confidence = pred_probs[:, 1]  # Xác suất lớp 1
pred_labels = (confidence > 0.5).astype(int)

# ==== 8. Gắn kết quả dự đoán vào dataframe gốc ====
# Thêm NaN vào đầu để khớp chiều dài
df['predict'] = [np.nan] * timesteps + pred_labels.tolist()
df['confidence'] = [np.nan] * timesteps + confidence.tolist()

last_conf = df['confidence'].iloc[-1]
print(f" độ tin cậy: {last_conf:.4f}")

# ==== 9. Lưu file kết quả (nếu cần) ====
df.to_csv('data_with_predictions.csv', index=False)
print("✅ Đã xử lý xong và lưu kết quả vào 'data_with_predictions.csv'")
