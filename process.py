import pandas as pd

# Load dữ liệu
data = pd.read_csv('OHLCV.csv')
data['Date'] = pd.to_datetime(data['Date'])

# Đảm bảo đúng thứ tự thời gian (tăng dần)
data = data.sort_values('Date').reset_index(drop=True)

window_size = 10

# Tính future_high và future_low đúng logic: lấy max/min của 10 phiên kế tiếp (không tính phiên hiện tại)
data['future_high'] = data['High'].shift(-1).rolling(window=window_size, min_periods=window_size).max().shift(-window_size + 1)
data['future_low'] = data['Low'].shift(-1).rolling(window=window_size, min_periods=window_size).min().shift(-window_size + 1)

# Tính tỉ lệ biến động giá
data['price_change_ratio'] = ((data['future_high'] - data['Close']) / (data['future_high'] - data['future_low'])).round(4)

# Gán nhãn
data['label'] = data['price_change_ratio'].apply(lambda x: 0 if x < 0.2 else (1 if x > 0.8 else -1))

# Xoá các dòng không đủ dữ liệu tương lai
# data = data.dropna(subset=['future_high', 'future_low'])

# Lưu lại
data.to_csv('data.csv', index=False)
