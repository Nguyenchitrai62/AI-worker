import tensorflow as tf
import pandas as pd
import numpy as np

# ==== Load lại model đã huấn luyện ====
model = tf.keras.models.load_model('transformer_model_balanced.keras')

# ==== Load dữ liệu gốc ====
data = pd.read_csv('data.csv')

# ==== Chọn các đặc trưng giống như khi huấn luyện ====
features = ['close/open', 'high-low', 'ema12', 'ema26', 'macd', 'rsi14', 'stoch_rsi', 'bb_width']
X_raw = data[features].values

# ==== Tạo sequence đầu vào ====
timesteps = 24

def create_input_sequences(X, timesteps=24):
    Xs = []
    for i in range(len(X) - timesteps):
        Xs.append(X[i:i+timesteps])
    return np.array(Xs)

X_seq = create_input_sequences(X_raw, timesteps)

# ==== Dự đoán ====
pred_probs = model.predict(X_seq, verbose=0)
confidence = pred_probs[:, 1]  # Xác suất lớp 1
pred_labels = (confidence > 0.5).astype(int)

# ==== Ghép kết quả vào dataframe ====
# Thêm NaN cho các dòng đầu bị mất do tạo sequence
padding = [np.nan] * timesteps
data['predict'] = padding + pred_labels.tolist()
data['confidence'] = padding + confidence.tolist()

# ==== Lưu lại kết quả ====
data.to_csv('data_with_predictions.csv', index=False)
print("Đã lưu kết quả vào 'data_with_predictions.csv'")
