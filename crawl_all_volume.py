import ccxt
import pandas as pd
from datetime import datetime, timedelta
import csv

# Fetch all available exchanges from ccxt
all_exchanges = ccxt.exchanges

# Symbol and timeframe
symbol = 'BTC/USDT'
timeframe = '1h'
limit = 1000  # Maximum records per API call

# Initialize a dictionary to store data from all exchanges
all_data = {}

# Initialize a dictionary to store average volumes
average_volumes = {}

for exchange_id in all_exchanges:
    try:
        # Initialize exchange
        exchange = getattr(ccxt, exchange_id)({
            'apiKey': '',
            'secret': '',
        })

        # Fetch OHLCV data
        current_time = int(datetime.now().timestamp() * 1000)
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit, since=current_time - limit * 60 * 60 * 1000)

        # Convert to DataFrame
        df = pd.DataFrame(ohlcv, columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])

        # Convert timestamp to datetime
        df['Date'] = pd.to_datetime(df['Date'], unit='ms')

        # Adjust to UTC+7
        df['Date'] = df['Date'] + timedelta(hours=7)

        # Store data in dictionary
        all_data[exchange_id] = df

        # Calculate average volume
        average_volume = df['Volume'].mean()
        average_volumes[exchange_id] = average_volume

        print(f"Fetched {len(df)} records from {exchange_id} with average volume: {average_volume}")

    except Exception as e:
        print(f"Error fetching data from {exchange_id}: {e}")

# Determine the exchange with the highest average volume
highest_volume_exchange = max(average_volumes, key=average_volumes.get)
print(f"Exchange with the highest average volume: {highest_volume_exchange} ({average_volumes[highest_volume_exchange]})")

# Define CSV file path
output_csv_path = 'exchange_volumes.csv'

# Update average volume calculation to use total volume divided by fetched records
csv_data = []
for exchange_id, df in all_data.items():
    try:
        # Use the actual number of fetched records as Max Fetch Limit
        max_fetch_limit = len(df)

        # Calculate total volume and average volume
        total_volume = df['Volume'].sum()
        average_volume = total_volume / max_fetch_limit

        csv_data.append({
            'Exchange': exchange_id,
            'Max Fetch Limit': max_fetch_limit,
            'Average Volume': average_volume
        })
    except Exception as e:
        print(f"Error processing data for {exchange_id}: {e}")

# Write data to CSV
with open(output_csv_path, mode='w', newline='', encoding='utf-8') as csv_file:
    fieldnames = ['Exchange', 'Max Fetch Limit', 'Average Volume']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

    # Write header
    writer.writeheader()

    # Write rows
    writer.writerows(csv_data)

print(f"Data saved to {output_csv_path}")

