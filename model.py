import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, LayerNormalization, Dropout, MultiHeadAttention, GlobalAveragePooling1D, Activation, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# ===== Load Data =====
data = pd.read_csv('/kaggle/input/data-test/data.csv')

# ==== Chọn các đặc trưng làm input ====
features = ['close/open', 'high-low', 'ema12', 'ema26', 'macd', 'rsi14', 'stoch_rsi', 'bb_width']
X_raw = data[features].values
y_raw = data['label'].values.astype(int)

# ===== Tạo sequences =====
timesteps = 100

def create_sequences(X, y, timesteps):
    Xs, ys = [], []
    for i in range(len(X) - timesteps):
        label = y[i + timesteps - 1]
        if label in [0, 1]:  # Chỉ lấy những mẫu có nhãn là 0 hoặc 1
            Xs.append(X[i:i+timesteps])
            ys.append(label)
    return np.array(Xs), np.array(ys)

X_seq, y_seq = create_sequences(X_raw, y_raw, timesteps)

# ===== Thống kê số lượng nhãn 0 và 1 =====
unique, counts = np.unique(y_seq, return_counts=True)
label_distribution = dict(zip(unique, counts))
print("Số lượng mẫu theo nhãn (dùng để huấn luyện):")
for label, count in label_distribution.items():
    print(f" - Nhãn {label}: {count} mẫu")

# ===== One-hot cho nhãn 2 lớp (0/1) =====
y_seq_cat = to_categorical(y_seq, num_classes=2)

# ===== Chia train/test =====
X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq_cat, test_size=0.2, random_state=42)

# ===== Transformer Block =====
def transformer_block(inputs, num_heads, ff_dim, dropout_rate=0.1):
    attention_output = MultiHeadAttention(num_heads=num_heads, key_dim=inputs.shape[-1])(inputs, inputs)
    attention_output = Dropout(dropout_rate)(attention_output)
    attention_output = LayerNormalization(epsilon=1e-6)(attention_output + inputs)

    ffn_output = Dense(ff_dim, activation="relu")(attention_output)
    ffn_output = Dense(inputs.shape[-1])(ffn_output)
    ffn_output = Dropout(dropout_rate)(ffn_output)
    return LayerNormalization(epsilon=1e-6)(ffn_output + attention_output)

# ===== Build Model =====
input_layer = Input(shape=(X_train.shape[1], X_train.shape[2]))

x = transformer_block(input_layer, num_heads=4, ff_dim=256)
x = transformer_block(x, num_heads=4, ff_dim=256)
x = GlobalAveragePooling1D()(x)

x = Dense(128, activation="relu")(x)
x = Dropout(0.1)(x)
x = Dense(64, activation="relu")(x)

# # use batchnorm
# x = Dense(128, use_bias=False)(x)
# x = BatchNormalization()(x)
# x = Activation("relu")(x)
# x = Dense(64, use_bias=False)(x)
# x = BatchNormalization()(x)
# x = Activation("relu")(x)

output_layer = Dense(2, activation="softmax")(x)

model = Model(inputs=input_layer, outputs=output_layer)

# ===== Compile Model with categorical crossentropy loss =====
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",  # Sử dụng crossentropy loss
    metrics=["accuracy"]
)

model.summary()

# ===== Setup ModelCheckpoint callback để lưu model tốt nhất =====
checkpoint = ModelCheckpoint(
    filepath='transformer_model_balanced.keras',
    monitor='val_accuracy',  # Theo dõi validation accuracy
    save_best_only=True,     # Chỉ lưu model tốt nhất
    mode='max',              # Lưu khi val_accuracy cao nhất
    verbose=1                # In thông báo khi lưu model
)

# ===== Train model =====
history = model.fit(
    X_train, y_train,
    epochs=200,
    batch_size=128,
    validation_data=(X_test, y_test),
    callbacks=[checkpoint],  # Thêm callback
    verbose=1
)

# ===== Load model tốt nhất để evaluate =====
best_model = tf.keras.models.load_model('transformer_model_balanced.keras')

# ===== Evaluate model =====
test_loss, test_accuracy = best_model.evaluate(X_test, y_test, verbose=0)
print(f"Test Loss:     {test_loss:.6f}")
print(f"Test Accuracy: {test_accuracy:.6f}")

# ===== Accuracy ngoài các khoảng xác suất =====
y_pred = best_model.predict(X_test, verbose=0)
y_true_labels = np.argmax(y_test, axis=1)
y_pred_probs = y_pred[:, 1]  # Lấy xác suất lớp 1

# Các khoảng xác suất cần kiểm tra (mẫu sẽ nằm ngoài các khoảng này)
intervals = [(0.4, 0.6), (0.35, 0.65), (0.3, 0.7), (0.25, 0.75), (0.2, 0.8), (0.15, 0.85),(0.1, 0.9)]

# Duyệt qua từng khoảng và tính độ chính xác cho các mẫu nằm ngoài các khoảng xác suất
for low, high in intervals:
    mask_outside = (y_pred_probs < low) | (y_pred_probs > high)  # Lọc các mẫu ngoài khoảng
    y_pred_outside = (y_pred_probs[mask_outside] > 0.5).astype(int)
    y_true_outside = y_true_labels[mask_outside]

    if len(y_true_outside) > 0:
        acc_outside = np.mean(y_pred_outside == y_true_outside)
        print(f"Accuracy outside range [{low}, {high}]: {acc_outside:.6f} on {len(y_true_outside)} samples")
    else:
        print(f"No confident predictions outside range [{low}, {high}].")

# ===== Plot history =====
plt.figure(figsize=(12, 5))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label="Train Accuracy")
plt.plot(history.history['val_accuracy'], label="Val Accuracy")
plt.title("Accuracy Curve")
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label="Train Loss")
plt.plot(history.history['val_loss'], label="Val Loss")
plt.title("Loss Curve")
plt.legend()
plt.show()