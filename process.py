import pandas as pd

# Load the CSV file
data = pd.read_csv('OHLCV.csv')

# Ensure the Date column is in datetime format
data['Date'] = pd.to_datetime(data['Date'])

# Define a variable for the rolling window size
window_size = 10

# Calculate the highest high and lowest low for the next rows based on the window size
data['future_high'] = data['High'].rolling(window=window_size, min_periods=1).max().shift(-(window_size - 1))
data['future_low'] = data['Low'].rolling(window=window_size, min_periods=1).min().shift(-(window_size - 1))

# Calculate the price change ratio with -1 as strong decrease and 1 as strong increase
data['price_change_ratio'] = ((data['future_high'] - data['Close']) / (data['future_high'] - data['future_low'])).round(4)

# Add a new column 'label' based on the condition
data['label'] = data['price_change_ratio'].apply(lambda x: 0 if x < 0.3 else (1 if x > 0.7 else -1))

# Save the updated data to a new CSV file
data.to_csv('data.csv', index=False)