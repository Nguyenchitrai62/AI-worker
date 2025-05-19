import pandas as pd
import ta

# Đọc dữ liệu
df = pd.read_csv('data.csv', parse_dates=['Date'])
df = df.sort_values('Date')

df['close/open'] = df['Close'] / df['Open'] -1 
df['high-low'] = (df['High'] - df['Low']) / df['Close']
df['ema12'] = ta.trend.ema_indicator(close=df['Close'], window=12) /df['Close'] -1
df['ema26'] = ta.trend.ema_indicator(close=df['Close'], window=26) /df['Close'] -1
df['macd'] = (df['ema12'] - df['ema26'])
df['rsi14'] = ta.momentum.rsi(close=df['Close'], window=14) / 50 - 1
df['stoch_rsi'] = ta.momentum.stochrsi(close=df['Close'], window=14) * 2 -1 
bb = ta.volatility.BollingerBands(close=df['Close'], window=14)
df['bb_width'] = (bb.bollinger_hband() - bb.bollinger_lband()) / bb.bollinger_mavg()

# ========= ➎ VOLUME FEATURES =========
# df['obv'] = ta.volume.on_balance_volume(close=df['Close'], volume=df['Volume'])

df = df.dropna()

# ======= Xuất file kết quả =======
df.to_csv('data.csv', index=False)

# Xem trước
print(df.head(3))
