import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
import matplotlib.pyplot as plt


class ChartPatternAnalyzer:
    def __init__(self, csv_file='OHLCV.csv'):
        self.df = pd.read_csv(csv_file, parse_dates=['Date'])
        self.df['signal'] = -1  # Mặc định signal là -1
        self.highs = None
        self.lows = None
    
    def find_peaks_and_troughs(self, order=5):
        highs = argrelextrema(self.df['High'].values, np.greater, order=order)[0]
        lows = argrelextrema(self.df['Low'].values, np.less, order=order)[0]
        self.highs = highs
        self.lows = lows
        return highs, lows
    
    def detect_head_and_shoulders(self, window=10, threshold=0.01):
        if self.highs is None or self.lows is None:
            raise ValueError("Cần chạy find_peaks_and_troughs() trước")
        
        signals = np.full(len(self.df), -1)
        
        # Sửa lỗi: Tìm các đỉnh và đáy phù hợp để tạo thành mô hình H&S
        for i in range(2, len(self.highs) - 2):
            left_shoulder = self.highs[i-2]
            head = self.highs[i]
            right_shoulder = self.highs[i+2]
            
            # Tìm các đáy gần nhất giữa các đỉnh để làm neckline
            # Tìm đáy giữa left_shoulder và head
            neck_left_candidates = [low for low in self.lows if left_shoulder < low < head]
            if not neck_left_candidates:
                continue
            neck_left = neck_left_candidates[np.argmin([self.df['Low'][idx] for idx in neck_left_candidates])]
            
            # Tìm đáy giữa head và right_shoulder
            neck_right_candidates = [low for low in self.lows if head < low < right_shoulder]
            if not neck_right_candidates:
                continue
            neck_right = neck_right_candidates[np.argmin([self.df['Low'][idx] for idx in neck_right_candidates])]
            
            # Điều kiện H&S: Đầu cao hơn vai, hai vai gần bằng nhau, neckline hợp lý
            if (self.df['High'][head] > self.df['High'][left_shoulder] and 
                self.df['High'][head] > self.df['High'][right_shoulder] and 
                abs(self.df['High'][left_shoulder] - self.df['High'][right_shoulder]) / self.df['High'][head] < threshold and
                abs(self.df['Low'][neck_left] - self.df['Low'][neck_right]) / self.df['Low'][neck_left] < threshold):
                signals[head] = 0  # Tín hiệu giảm
                
        return signals
    
    def detect_inverted_head_and_shoulders(self, window=10, threshold=0.01):
        """Phát hiện mô hình Inverted Head and Shoulders (tín hiệu tăng)"""
        if self.highs is None or self.lows is None:
            raise ValueError("Cần chạy find_peaks_and_troughs() trước")
        
        signals = np.full(len(self.df), -1)
        
        # Tương tự H&S nhưng ngược lại - tìm mô hình với đáy thấp nhất ở giữa
        for i in range(2, len(self.lows) - 2):
            left_shoulder = self.lows[i-2]
            head = self.lows[i]
            right_shoulder = self.lows[i+2]
            
            # Tìm các đỉnh gần nhất giữa các đáy để làm neckline
            # Tìm đỉnh giữa left_shoulder và head
            neck_left_candidates = [high for high in self.highs if left_shoulder < high < head]
            if not neck_left_candidates:
                continue
            neck_left = neck_left_candidates[np.argmax([self.df['High'][idx] for idx in neck_left_candidates])]
            
            # Tìm đỉnh giữa head và right_shoulder
            neck_right_candidates = [high for high in self.highs if head < high < right_shoulder]
            if not neck_right_candidates:
                continue
            neck_right = neck_right_candidates[np.argmax([self.df['High'][idx] for idx in neck_right_candidates])]
            
            # Điều kiện Inverted H&S: Đầu thấp hơn vai, hai vai gần bằng nhau, neckline hợp lý
            if (self.df['Low'][head] < self.df['Low'][left_shoulder] and 
                self.df['Low'][head] < self.df['Low'][right_shoulder] and 
                abs(self.df['Low'][left_shoulder] - self.df['Low'][right_shoulder]) / self.df['Low'][head] < threshold and
                abs(self.df['High'][neck_left] - self.df['High'][neck_right]) / self.df['High'][neck_left] < threshold):
                signals[head] = 1  # Tín hiệu tăng
                
        return signals
    
    def detect_double_top(self, window=10, threshold=0.01):
        if self.highs is None:
            raise ValueError("Cần chạy find_peaks_and_troughs() trước")
        
        signals = np.full(len(self.df), -1)
        
        for i in range(len(self.highs)-1):
            top1, top2 = self.highs[i], self.highs[i+1]
            if top2 - top1 < window:
                continue
            
            # Điều kiện: Hai đỉnh gần bằng nhau
            if abs(self.df['High'][top1] - self.df['High'][top2]) / self.df['High'][top1] < threshold:
                signals[top2] = 0  # Tín hiệu giảm
                
        return signals
    
    def detect_double_bottom(self, window=10, threshold=0.01):
        if self.lows is None:
            raise ValueError("Cần chạy find_peaks_and_troughs() trước")
        
        signals = np.full(len(self.df), -1)
        
        for i in range(len(self.lows)-1):
            bottom1, bottom2 = self.lows[i], self.lows[i+1]
            if bottom2 - bottom1 < window:
                continue
            
            # Điều kiện: Hai đáy gần bằng nhau
            if abs(self.df['Low'][bottom1] - self.df['Low'][bottom2]) / self.df['Low'][bottom1] < threshold:
                signals[bottom2] = 1  # Tín hiệu tăng
                
        return signals
    
    def combine_signals(self, *signal_arrays):
        """Kết hợp nhiều mảng tín hiệu"""
        for signals in signal_arrays:
            self.df['signal'] = np.where(signals != -1, signals, self.df['signal'])
    
    def validate_predictions(self, future_window=12):
        self.df['future_close'] = self.df['Close'].shift(-future_window)
        self.df['actual_change'] = self.df['future_close'] - self.df['Close']
        self.df['prediction_correct'] = -1
        
        # Kiểm tra tín hiệu mua (signal = 1)
        self.df.loc[self.df['signal'] == 1, 'prediction_correct'] = (
            self.df['actual_change'] > 0
        ).astype(int)
        
        # Kiểm tra tín hiệu bán (signal = 0)
        self.df.loc[self.df['signal'] == 0, 'prediction_correct'] = (
            self.df['actual_change'] < 0
        ).astype(int)
    
    def calculate_accuracy(self, model_signal):
        model_df = self.df[(self.df['signal'] == model_signal) & (self.df['future_close'].notna())]
        if len(model_df) == 0:
            return 0, 0
        
        correct = model_df['prediction_correct'].sum()
        total = len(model_df)
        accuracy = correct / total if total > 0 else 0
        return accuracy, total
    
    def get_accuracy_report(self, hs_signals, ihs_signals):
        # Head and Shoulders (Bearish)
        hs_accuracy, hs_count = self.calculate_accuracy_for_pattern(hs_signals, 0)
        
        # Inverted Head and Shoulders (Bullish)
        ihs_accuracy, ihs_count = self.calculate_accuracy_for_pattern(ihs_signals, 1)
        
        # Double Bottom (Bullish)
        db_accuracy, db_count = self.calculate_accuracy_for_pattern_excluding_others(1, [ihs_signals])
        
        # Double Top (Bearish) - loại trừ Head and Shoulders
        dt_accuracy, dt_count = self.calculate_accuracy_for_pattern_excluding_others(0, [hs_signals])
        
        return {
            'head_shoulders': {'accuracy': hs_accuracy, 'count': hs_count},
            'inverted_head_shoulders': {'accuracy': ihs_accuracy, 'count': ihs_count},
            'double_top': {'accuracy': dt_accuracy, 'count': dt_count},
            'double_bottom': {'accuracy': db_accuracy, 'count': db_count}
        }
    
    def calculate_accuracy_for_pattern(self, pattern_signals, signal_value):
        """Tính độ chính xác cho một mô hình cụ thể"""
        pattern_df = self.df[(pattern_signals == signal_value) & (self.df['future_close'].notna())]
        if len(pattern_df) == 0:
            return 0, 0
        
        correct = pattern_df['prediction_correct'].sum()
        total = len(pattern_df)
        accuracy = correct / total if total > 0 else 0
        return accuracy, total
    
    def calculate_accuracy_for_pattern_excluding_others(self, signal_value, exclude_patterns):
        """Tính độ chính xác cho một mô hình, loại trừ các mô hình khác"""
        mask = (self.df['signal'] == signal_value) & (self.df['future_close'].notna())
        
        # Loại trừ các mô hình khác
        for pattern_signals in exclude_patterns:
            mask = mask & (pattern_signals == -1)
        
        pattern_df = self.df[mask]
        if len(pattern_df) == 0:
            return 0, 0
        
        correct = pattern_df['prediction_correct'].sum()
        total = len(pattern_df)
        accuracy = correct / total if total > 0 else 0
        return accuracy, total
    
    def plot_chart_patterns(self, num_sessions=1000):
        df_plot = self.df.iloc[-num_sessions:].copy()

        plt.figure(figsize=(15, 7))
        
        # Vẽ đường giá
        plt.plot(df_plot['Date'], df_plot['Close'], label='Close Price', color='black')
        
        # Phân loại các điểm tín hiệu
        correct_buy = df_plot[(df_plot['signal'] == 1) & (df_plot['prediction_correct'] == 1)]
        wrong_buy = df_plot[(df_plot['signal'] == 1) & (df_plot['prediction_correct'] == 0)]
        correct_sell = df_plot[(df_plot['signal'] == 0) & (df_plot['prediction_correct'] == 1)]
        wrong_sell = df_plot[(df_plot['signal'] == 0) & (df_plot['prediction_correct'] == 0)]

        # Vẽ các điểm tín hiệu
        plt.scatter(correct_buy['Date'], correct_buy['Low'], 
                   color='green', label='Buy Correct', s=50, zorder=5)
        plt.scatter(correct_sell['Date'], correct_sell['High'], 
                   color='red', label='Sell Correct', s=50, zorder=5)
        plt.scatter(wrong_buy['Date'], wrong_buy['Low'], 
                   color='blue', label='Buy Wrong', s=50, zorder=5)
        plt.scatter(wrong_sell['Date'], wrong_sell['High'], 
                   color='gold', label='Sell Wrong', s=50, zorder=5)

        plt.title(f'BTC Price Chart with Signals (Last {num_sessions} Sessions)')
        plt.xlabel('Date')
        plt.ylabel('Price (USD)')
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    
    def save_results(self, filename='OHLCV_with_signals.csv'):
        self.df.to_csv(filename, index=False)
        print(f"Đã lưu kết quả vào file '{filename}'")
    
    def run_full_analysis(self, order=5, window=10, threshold=0.01, future_window=12):
        # Bước 1: Tìm đỉnh và đáy
        self.find_peaks_and_troughs(order=order)
        
        print(f"Tìm thấy {len(self.highs)} đỉnh và {len(self.lows)} đáy")
        
        # Bước 2: Nhận diện các mô hình
        hs_signals = self.detect_head_and_shoulders(window=window, threshold=threshold)
        ihs_signals = self.detect_inverted_head_and_shoulders(window=window, threshold=threshold)
        dt_signals = self.detect_double_top(window=window, threshold=threshold)
        db_signals = self.detect_double_bottom(window=window, threshold=threshold)
        
        # Bước 3: Kết hợp tín hiệu
        self.combine_signals(hs_signals, ihs_signals, dt_signals, db_signals)
        
        # Bước 4: Kiểm tra độ chính xác
        self.validate_predictions(future_window=future_window)
        
        # Bước 5: Tạo báo cáo
        report = self.get_accuracy_report(hs_signals, ihs_signals)
        
        return report


def print_analysis_report(report):
    print("Kết quả nhận diện mô hình và độ chính xác:")
    print(f"Head and Shoulders (Dự đoán giảm):")
    print(f"  - Số lần phát hiện: {report['head_shoulders']['count']}")
    print(f"  - Độ chính xác: {report['head_shoulders']['accuracy']:.2%}")
    
    print(f"Inverted Head and Shoulders (Dự đoán tăng):")
    print(f"  - Số lần phát hiện: {report['inverted_head_shoulders']['count']}")
    print(f"  - Độ chính xác: {report['inverted_head_shoulders']['accuracy']:.2%}")
    
    print(f"Double Top (Dự đoán giảm):")
    print(f"  - Số lần phát hiện: {report['double_top']['count']}")
    print(f"  - Độ chính xác: {report['double_top']['accuracy']:.2%}")
    
    print(f"Double Bottom (Dự đoán tăng):")
    print(f"  - Số lần phát hiện: {report['double_bottom']['count']}")
    print(f"  - Độ chính xác: {report['double_bottom']['accuracy']:.2%}")


def main():
    # Khởi tạo analyzer
    analyzer = ChartPatternAnalyzer('OHLCV.csv')
    
    # Chạy phân tích đầy đủ với các tham số mặc định
    report = analyzer.run_full_analysis(
        order=5,
        window=10,
        threshold=0.01,
        future_window=4
    )
    
    # In báo cáo
    print_analysis_report(report)
    
    # Vẽ biểu đồ
    analyzer.plot_chart_patterns(num_sessions=1000)
    
    # Lưu kết quả
    analyzer.save_results()


if __name__ == "__main__":
    main()