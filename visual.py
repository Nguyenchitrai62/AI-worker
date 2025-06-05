import ccxt
import pandas as pd
import numpy as np
import ta
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, TextBox
import matplotlib.patches as patches

# Kết nối với Binance
binance = ccxt.binance({
    'apiKey': '',
    'secret': '',
})

def fetch_data(count=500):
    try:
        ohlcv = binance.fetch_ohlcv('BTC/USDT', timeframe='1h', limit=count)
        df = pd.DataFrame(ohlcv, columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])
        df['Date'] = pd.to_datetime(df['Date'], unit='ms')
        print(f"✅ Fetched {len(df)} sessions")
        return df
    except Exception as e:
        print(f"❌ Error fetching data: {e}")
        return pd.DataFrame()

def add_technical_indicators(df):
    df['close/open'] = df['Close'] / df['Open'] - 1 
    df['high-low'] = (df['High'] - df['Low']) / df['Close']
    df['ema12'] = ta.trend.ema_indicator(close=df['Close'], window=12) / df['Close'] - 1
    df['ema26'] = ta.trend.ema_indicator(close=df['Close'], window=26) / df['Close'] - 1
    df['macd'] = (df['ema12'] - df['ema26'])
    df['rsi14'] = ta.momentum.rsi(close=df['Close'], window=14) / 50 - 1
    df['stoch_rsi'] = ta.momentum.stochrsi(close=df['Close'], window=14) * 2 - 1 
    bb = ta.volatility.BollingerBands(close=df['Close'], window=14)
    df['bb_width'] = (bb.bollinger_hband() - bb.bollinger_lband()) / bb.bollinger_mavg()
    return df

# Positional Encoding Classes
class LinearPositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(LinearPositionalEncoding, self).__init__(**kwargs)
    
    def call(self, inputs):
        seq_len = tf.shape(inputs)[1]
        feature_dim = tf.shape(inputs)[2]
        positions = tf.range(seq_len, dtype=tf.float32)
        position_weights = 0.05 + positions / tf.cast(seq_len - 1, tf.float32)
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

# Load model
model = tf.keras.models.load_model('transformer_model_balanced.keras')

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
    confidence = pred_probs[:, 1] * 100  # Convert to percentage

    padding = [np.nan] * timesteps
    df['ai_prediction'] = padding + confidence.tolist()
    return df.dropna()

class BTCPredictionChart:
    def __init__(self, df):
        self.df = df
        self.buy_threshold = 70
        self.sell_threshold = 30
        self.num_sessions = 200
        
        # Create figure and subplots
        self.fig = plt.figure(figsize=(16, 10))
        self.fig.suptitle('BTC/USDT AI Prediction Chart', fontsize=16, fontweight='bold')
        
        # Main chart area
        self.ax_price = plt.subplot(2, 1, 1)
        self.ax_ai = plt.subplot(2, 1, 2)
        
        # Control panel area
        plt.subplots_adjust(bottom=0.2)
        
        # Add controls
        self.add_controls()
        
        # Initial plot
        self.update_plot()
        
    def add_controls(self):
        # Buy threshold slider
        ax_buy = plt.axes([0.15, 0.1, 0.2, 0.03])
        self.slider_buy = Slider(ax_buy, 'Buy Threshold', 50, 95, valinit=self.buy_threshold, valfmt='%d')
        self.slider_buy.on_changed(self.update_buy_threshold)
        
        # Buy threshold text input
        ax_buy_text = plt.axes([0.37, 0.1, 0.05, 0.03])
        self.textbox_buy = TextBox(ax_buy_text, '', initial=str(int(self.buy_threshold)))
        self.textbox_buy.on_submit(self.on_buy_text_submit)
        
        # Sell threshold display (read-only)
        ax_sell = plt.axes([0.15, 0.05, 0.2, 0.03])
        self.slider_sell = Slider(ax_sell, 'Sell Threshold', 5, 50, valinit=self.sell_threshold, valfmt='%d')
        
        # Sessions slider
        ax_sessions = plt.axes([0.5, 0.1, 0.2, 0.03])
        max_sessions = min(len(self.df), 1000)
        self.slider_sessions = Slider(ax_sessions, 'Sessions', 200, max_sessions, valinit=self.num_sessions, valfmt='%d')
        self.slider_sessions.on_changed(self.update_sessions)
        
        # Sessions text input
        ax_sessions_text = plt.axes([0.72, 0.1, 0.05, 0.03])
        self.textbox_sessions = TextBox(ax_sessions_text, '', initial=str(int(self.num_sessions)))
        self.textbox_sessions.on_submit(self.on_sessions_text_submit)
        
        # Add text labels
        self.fig.text(0.15, 0.14, 'AI Prediction > Buy Threshold = BUY Signal', fontsize=10, color='green')
        self.fig.text(0.15, 0.01, 'Sell Threshold = 100 - Buy Threshold (Auto)', fontsize=10, color='red')
        self.fig.text(0.5, 0.14, 'Number of sessions to display', fontsize=10)
        
        # Input instructions
        self.fig.text(0.78, 0.12, 'Type value\n& press Enter', fontsize=8, color='gray')
        
    def update_buy_threshold(self, val):
        self.buy_threshold = self.slider_buy.val
        self.sell_threshold = 100 - self.buy_threshold
        
        # Update displays
        self.slider_sell.set_val(self.sell_threshold)
        self.textbox_buy.set_val(str(int(self.buy_threshold)))
        
        self.update_plot()
    
    def on_buy_text_submit(self, text):
        try:
            value = float(text)
            # Clamp value to valid range (50-95)
            value = max(50, min(95, value))
            
            # Update slider and recalculate
            self.slider_buy.set_val(value)
            self.buy_threshold = value
            self.sell_threshold = 100 - self.buy_threshold
            
            # Update displays
            self.slider_sell.set_val(self.sell_threshold)
            self.textbox_buy.set_val(str(int(value)))
            
            self.update_plot()
            
        except ValueError:
            # If invalid input, reset to current value
            self.textbox_buy.set_val(str(int(self.buy_threshold)))
    
    def on_sessions_text_submit(self, text):
        try:
            value = int(text)
            max_sessions = min(len(self.df), 1000)
            # Clamp value to valid range (200-max_sessions)
            value = max(200, min(max_sessions, value))
            
            # Update slider and value
            self.slider_sessions.set_val(value)
            self.num_sessions = value
            self.textbox_sessions.set_val(str(value))
            
            self.update_plot()
            
        except ValueError:
            # If invalid input, reset to current value
            self.textbox_sessions.set_val(str(int(self.num_sessions)))
        
    def update_thresholds(self, val):
        # This function is no longer used
        pass
        
    def update_sessions(self, val):
        self.num_sessions = int(self.slider_sessions.val)
        self.update_plot()
        
    def update_plot(self):
        # Clear previous plots
        self.ax_price.clear()
        self.ax_ai.clear()
        
        # Get data for specified number of sessions
        df_plot = self.df.tail(self.num_sessions).copy()
        
        # Generate signals based on current thresholds
        df_plot['signal'] = 'HOLD'
        df_plot.loc[df_plot['ai_prediction'] > self.buy_threshold, 'signal'] = 'BUY'
        df_plot.loc[df_plot['ai_prediction'] < self.sell_threshold, 'signal'] = 'SELL'
        
        # Plot 1: Price chart with signals
        self.ax_price.plot(df_plot['Date'], df_plot['Close'], color='black', linewidth=1, label='BTC Price')
        
        # Buy signals
        buy_signals = df_plot[df_plot['signal'] == 'BUY']
        if not buy_signals.empty:
            self.ax_price.scatter(buy_signals['Date'], buy_signals['Close'], 
                                color='green', marker='^', s=60, alpha=0.8, label=f'Buy Signal ({len(buy_signals)})', zorder=5)
        
        # Sell signals
        sell_signals = df_plot[df_plot['signal'] == 'SELL']
        if not sell_signals.empty:
            self.ax_price.scatter(sell_signals['Date'], sell_signals['Close'], 
                                color='red', marker='v', s=60, alpha=0.8, label=f'Sell Signal ({len(sell_signals)})', zorder=5)
        
        self.ax_price.set_title(f'BTC/USDT Price with AI Signals (Last {self.num_sessions} sessions)', fontweight='bold')
        self.ax_price.set_ylabel('Price (USDT)')
        self.ax_price.legend()
        self.ax_price.grid(True, alpha=0.3)
        
        # Plot 2: AI Prediction with thresholds
        self.ax_ai.plot(df_plot['Date'], df_plot['ai_prediction'], color='blue', linewidth=2, label='AI Prediction')
        
        # Threshold lines
        self.ax_ai.axhline(y=self.buy_threshold, color='green', linestyle='--', linewidth=2, 
                          label=f'Buy Threshold ({self.buy_threshold}%)', alpha=0.7)
        self.ax_ai.axhline(y=self.sell_threshold, color='red', linestyle='--', linewidth=2, 
                          label=f'Sell Threshold ({self.sell_threshold}%)', alpha=0.7)
        
        # Fill areas
        self.ax_ai.fill_between(df_plot['Date'], self.buy_threshold, 100, alpha=0.1, color='green', label='Buy Zone')
        self.ax_ai.fill_between(df_plot['Date'], 0, self.sell_threshold, alpha=0.1, color='red', label='Sell Zone')
        
        # Color the prediction line based on signal
        for i in range(len(df_plot)):
            if df_plot.iloc[i]['signal'] == 'BUY':
                self.ax_ai.plot(df_plot.iloc[i]['Date'], df_plot.iloc[i]['ai_prediction'], 'go', markersize=4, alpha=0.6)
            elif df_plot.iloc[i]['signal'] == 'SELL':
                self.ax_ai.plot(df_plot.iloc[i]['Date'], df_plot.iloc[i]['ai_prediction'], 'ro', markersize=4, alpha=0.6)
        
        self.ax_ai.set_title('AI Prediction Score with Thresholds', fontweight='bold')
        self.ax_ai.set_ylabel('AI Prediction (%)')
        self.ax_ai.set_xlabel('Date')
        self.ax_ai.set_ylim(0, 100)
        self.ax_ai.legend()
        self.ax_ai.grid(True, alpha=0.3)
        
        # Statistics
        total_signals = len(buy_signals) + len(sell_signals)
        hold_signals = len(df_plot) - total_signals
        
        stats_text = f"📊 Signals: BUY({len(buy_signals)}) | SELL({len(sell_signals)}) | HOLD({hold_signals}) | Buy+Sell={self.buy_threshold + self.sell_threshold}%"
        current_price = df_plot['Close'].iloc[-1]
        current_prediction = df_plot['ai_prediction'].iloc[-1]
        current_signal = df_plot['signal'].iloc[-1]
        
        stats_text += f"\n💰 Current: ${current_price:,.2f} | AI: {current_prediction:.1f}% | Signal: {current_signal}"
        
        self.fig.text(0.02, 0.95, stats_text, fontsize=10, verticalalignment='top',
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
        
        # Refresh the plot
        self.fig.canvas.draw()

def main():
    print("🚀 Starting BTC AI Prediction Analysis...")
    
    # Fetch data
    print("📊 Fetching data from Binance...")
    df = fetch_data(1000)  # Fetch maximum data
    
    if df.empty:
        print("❌ Could not fetch data!")
        return
    
    # Add technical indicators
    print("🔧 Calculating technical indicators...")
    df = add_technical_indicators(df)
    
    # AI predictions
    print("🤖 Running AI model predictions...")
    df = predict_with_model(df)
    
    print(f"✅ Completed! Generated {len(df)} predictions.")
    
    # Create interactive chart
    print("📈 Creating interactive chart...")
    chart = BTCPredictionChart(df)
    
    plt.show()
    
    return df

if __name__ == "__main__":
    result_df = main()