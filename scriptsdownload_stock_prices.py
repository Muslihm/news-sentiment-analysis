"""
Download historical stock price data for stocks in the news dataset
Stocks: AAPL, AMZN, GOOG, NVDA, MET
"""

import yfinance as yf
import pandas as pd
import os
from datetime import datetime

# Create output directory
os.makedirs('data/raw', exist_ok=True)

# Stocks from the news dataset
stocks = ['AAPL', 'AMZN', 'GOOG', 'NVDA', 'MET']

# Date range (match news data: 2020-01-01 to 2024-12-31)
start_date = '2020-01-01'
end_date = '2024-12-31'

print("=" * 60)
print("DOWNLOADING STOCK PRICE DATA")
print("=" * 60)

all_data = {}

for stock in stocks:
    print(f"\nDownloading {stock}...")
    try:
        ticker = yf.Ticker(stock)
        df = ticker.history(start=start_date, end=end_date)
        
        if len(df) > 0:
            # Reset index to make Date a column
            df = df.reset_index()
            df['Stock'] = stock
            
            all_data[stock] = df
            print(f"  ✓ Downloaded {len(df)} records for {stock}")
            print(f"    Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")
        else:
            print(f"  ✗ No data for {stock}")
            
    except Exception as e:
        print(f"  ✗ Error downloading {stock}: {e}")

# Combine all data
if all_data:
    combined_df = pd.concat(all_data.values(), ignore_index=True)
    
    # Save to CSV
    output_path = 'data/raw/stock_prices.csv'
    combined_df.to_csv(output_path, index=False)
    print(f"\n✓ Saved combined data to {output_path}")
    print(f"  Total records: {len(combined_df)}")
    print(f"  Columns: {combined_df.columns.tolist()}")
else:
    print("\n✗ No data downloaded")

print("\n" + "=" * 60)
