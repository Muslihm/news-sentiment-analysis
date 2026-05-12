import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

print("Running Task 2: Technical Analysis")

np.random.seed(42)
stocks = ['AAPL', 'AMZN', 'GOOG', 'NVDA', 'MET']
dates = pd.date_range(start='2020-01-01', end='2024-12-31', freq='D')

# Create sample stock data
all_data = []
base_prices = {'AAPL': 70, 'AMZN': 1800, 'GOOG': 1300, 'NVDA': 50, 'MET': 40}

for stock in stocks:
    base = base_prices[stock]
    returns = np.random.normal(0.0003, 0.02, len(dates))
    prices = base * np.exp(np.cumsum(returns))
    
    for i, date in enumerate(dates):
        all_data.append({'Date': date, 'Stock': stock, 'Close': prices[i]})

df = pd.DataFrame(all_data)

# Calculate indicators
df['SMA_20'] = df.groupby('Stock')['Close'].transform(lambda x: x.rolling(20, min_periods=1).mean())
df['SMA_50'] = df.groupby('Stock')['Close'].transform(lambda x: x.rolling(50, min_periods=1).mean())
df['Daily_Return'] = df.groupby('Stock')['Close'].pct_change() * 100

# Create output directory
os.makedirs('outputs/task2', exist_ok=True)

# Plot for a sample stock
stock_data = df[df['Stock'] == 'AAPL'].copy()

fig, axes = plt.subplots(2, 1, figsize=(12, 8))
axes[0].plot(stock_data['Date'], stock_data['Close'], label='Close Price')
axes[0].plot(stock_data['Date'], stock_data['SMA_20'], label='SMA 20', alpha=0.7)
axes[0].plot(stock_data['Date'], stock_data['SMA_50'], label='SMA 50', alpha=0.7)
axes[0].set_title('AAPL - Price with Moving Averages')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].hist(df['Daily_Return'].dropna(), bins=50, alpha=0.7, edgecolor='black')
axes[1].set_title('Daily Returns Distribution')
axes[1].set_xlabel('Return (%)')
axes[1].set_ylabel('Frequency')

plt.tight_layout()
plt.savefig('outputs/task2/technical_dashboard.png', dpi=150)
plt.close()
print("✓ Saved: outputs/task2/technical_dashboard.png")

print("Task 2 completed!")
