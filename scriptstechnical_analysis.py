import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("TECHNICAL ANALYSIS - WORKING VERSION")
print("=" * 60)

# ============================================
# CREATE SAMPLE DATA (SINCE WE DON'T HAVE REAL DATA YET)
# ============================================

print("\n[1/4] Creating sample stock data...")

np.random.seed(42)

# Stocks from your news data
stocks = ['AAPL', 'AMZN', 'GOOG', 'NVDA', 'MET']
dates = pd.date_range(start='2020-01-01', end='2024-12-31', freq='D')

# Base prices for each stock
base_prices = {'AAPL': 70, 'AMZN': 1800, 'GOOG': 1300, 'NVDA': 50, 'MET': 40}
trends = {'AAPL': 0.0003, 'AMZN': 0.0003, 'GOOG': 0.0002, 'NVDA': 0.0006, 'MET': 0.0001}

all_data = []

for stock in stocks:
    # Generate price series
    returns = np.random.normal(trends[stock], 0.02, len(dates))
    prices = base_prices[stock] * np.exp(np.cumsum(returns))
    
    for i, date in enumerate(dates):
        all_data.append({
            'Date': date,
            'Stock': stock,
            'Open': prices[i] * (1 + np.random.normal(0, 0.01)),
            'High': prices[i] * (1 + abs(np.random.normal(0, 0.02))),
            'Low': prices[i] * (1 - abs(np.random.normal(0, 0.02))),
            'Close': prices[i],
            'Volume': np.random.randint(1000000, 100000000)
        })

df = pd.DataFrame(all_data)
print(f"✓ Created data for {len(df):,} rows")
print(f"  Stocks: {', '.join(stocks)}")
print(f"  Date range: {dates[0].date()} to {dates[-1].date()}")

# ============================================
# CALCULATE TECHNICAL INDICATORS
# ============================================

print("\n[2/4] Calculating technical indicators...")

# Sort by date
df = df.sort_values(['Stock', 'Date'])

# 1. MOVING AVERAGES
df['SMA_20'] = df.groupby('Stock')['Close'].transform(
    lambda x: x.rolling(window=20, min_periods=1).mean()
)
df['SMA_50'] = df.groupby('Stock')['Close'].transform(
    lambda x: x.rolling(window=50, min_periods=1).mean()
)
df['SMA_200'] = df.groupby('Stock')['Close'].transform(
    lambda x: x.rolling(window=200, min_periods=1).mean()
)

# 2. RSI (Relative Strength Index)
def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

df['RSI_14'] = df.groupby('Stock')['Close'].transform(lambda x: calculate_rsi(x, 14))

# 3. MACD
def calculate_macd(prices, fast=12, slow=26, signal=9):
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

for stock in stocks:
    mask = df['Stock'] == stock
    macd, sig, hist = calculate_macd(df.loc[mask, 'Close'])
    df.loc[mask, 'MACD'] = macd
    df.loc[mask, 'MACD_Signal'] = sig
    df.loc[mask, 'MACD_Histogram'] = hist

# 4. BOLLINGER BANDS
def calculate_bollinger_bands(prices, period=20, std_dev=2):
    sma = prices.rolling(window=period).mean()
    std = prices.rolling(window=period).std()
    upper = sma + (std * std_dev)
    lower = sma - (std * std_dev)
    return upper, sma, lower

for stock in stocks:
    mask = df['Stock'] == stock
    upper, middle, lower = calculate_bollinger_bands(df.loc[mask, 'Close'])
    df.loc[mask, 'BB_Upper'] = upper
    df.loc[mask, 'BB_Middle'] = middle
    df.loc[mask, 'BB_Lower'] = lower

# 5. VOLATILITY
df['Daily_Return'] = df.groupby('Stock')['Close'].pct_change()
df['Volatility'] = df.groupby('Stock')['Daily_Return'].transform(
    lambda x: x.rolling(window=20).std() * np.sqrt(252)
)

print("✓ All indicators calculated successfully!")

# ============================================
# CREATE VISUALIZATIONS
# ============================================

print("\n[3/4] Creating visualizations...")

# Create output folder
os.makedirs('outputs/task2', exist_ok=True)

# Pick the first stock for detailed analysis
primary_stock = stocks[0]
stock_data = df[df['Stock'] == primary_stock].copy()

print(f"  Analyzing {primary_stock}")

# FIGURE 1: Technical Dashboard
fig, axes = plt.subplots(4, 1, figsize=(14, 12))
fig.suptitle(f'{primary_stock} - Technical Analysis Dashboard', fontsize=14, fontweight='bold')

# Plot 1: Price with Moving Averages
axes[0].plot(stock_data['Date'], stock_data['Close'], label='Close Price', linewidth=1.5, color='black')
axes[0].plot(stock_data['Date'], stock_data['SMA_20'], label='SMA 20', linewidth=1, alpha=0.7)
axes[0].plot(stock_data['Date'], stock_data['SMA_50'], label='SMA 50', linewidth=1, alpha=0.7)
axes[0].plot(stock_data['Date'], stock_data['SMA_200'], label='SMA 200', linewidth=1, alpha=0.7)
axes[0].fill_between(stock_data['Date'], stock_data['BB_Lower'], stock_data['BB_Upper'], 
                      alpha=0.2, color='gray', label='Bollinger Bands')
axes[0].set_title('Price with Moving Averages')
axes[0].set_ylabel('Price ($)')
axes[0].legend(loc='upper left')
axes[0].grid(True, alpha=0.3)

# Plot 2: RSI
axes[1].plot(stock_data['Date'], stock_data['RSI_14'], label='RSI (14)', linewidth=1.5, color='purple')
axes[1].axhline(y=70, color='red', linestyle='--', label='Overbought (70)')
axes[1].axhline(y=30, color='green', linestyle='--', label='Oversold (30)')
axes[1].fill_between(stock_data['Date'], 70, 100, alpha=0.1, color='red')
axes[1].fill_between(stock_data['Date'], 0, 30, alpha=0.1, color='green')
axes[1].set_title('Relative Strength Index (RSI)')
axes[1].set_ylabel('RSI')
axes[1].set_ylim(0, 100)
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Plot 3: MACD
axes[2].plot(stock_data['Date'], stock_data['MACD'], label='MACD', linewidth=1.5, color='blue')
axes[2].plot(stock_data['Date'], stock_data['MACD_Signal'], label='Signal', linewidth=1.5, color='red')
colors = ['green' if x >= 0 else 'red' for x in stock_data['MACD_Histogram']]
axes[2].bar(stock_data['Date'], stock_data['MACD_Histogram'], label='Histogram', alpha=0.5, color=colors)
axes[2].set_title('MACD')
axes[2].set_ylabel('MACD')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

# Plot 4: Volatility
axes[3].plot(stock_data['Date'], stock_data['Volatility'], linewidth=1.5, color='orange')
axes[3].set_title('20-Day Rolling Volatility (Annualized)')
axes[3].set_xlabel('Date')
axes[3].set_ylabel('Volatility')
axes[3].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/task2/technical_dashboard.png', dpi=150, bbox_inches='tight')
print("  ✓ Saved: technical_dashboard.png")

# FIGURE 2: Stock Comparison
fig2, axes = plt.subplots(2, 2, figsize=(14, 10))
fig2.suptitle('Stock Comparison', fontsize=14, fontweight='bold')

# Get latest data for each stock
latest = df.groupby('Stock').last().reset_index()

# Current Price
axes[0,0].bar(latest['Stock'], latest['Close'], color='steelblue')
axes[0,0].set_title('Current Price')
axes[0,0].set_ylabel('Price ($)')

# RSI
colors_rsi = ['red' if x > 70 else 'green' if x < 30 else 'blue' for x in latest['RSI_14']]
axes[0,1].bar(latest['Stock'], latest['RSI_14'], color=colors_rsi)
axes[0,1].axhline(y=70, color='red', linestyle='--', label='Overbought')
axes[0,1].axhline(y=30, color='green', linestyle='--', label='Oversold')
axes[0,1].set_title('RSI (14)')
axes[0,1].set_ylabel('RSI')
axes[0,1].set_ylim(0, 100)
axes[0,1].legend()

# Volatility
axes[1,0].bar(latest['Stock'], latest['Volatility'], color='orange')
axes[1,0].set_title('Annualized Volatility')
axes[1,0].set_ylabel('Volatility')

# Returns Distribution
all_returns = df['Daily_Return'].dropna()
axes[1,1].hist(all_returns, bins=50, alpha=0.7, color='green', edgecolor='black')
axes[1,1].axvline(x=0, color='red', linestyle='--')
axes[1,1].set_title('Daily Returns Distribution (All Stocks)')
axes[1,1].set_xlabel('Daily Return')
axes[1,1].set_ylabel('Frequency')

plt.tight_layout()
plt.savefig('outputs/task2/stock_comparison.png', dpi=150, bbox_inches='tight')
print("  ✓ Saved: stock_comparison.png")

# FIGURE 3: Price History for All Stocks
fig3, ax = plt.subplots(figsize=(14, 8))

for stock in stocks:
    stock_data_plot = df[df['Stock'] == stock]
    # Normalize prices to start at 100
    normalized = (stock_data_plot['Close'] / stock_data_plot['Close'].iloc[0]) * 100
    ax.plot(stock_data_plot['Date'], normalized, label=stock, linewidth=1.5)

ax.set_title('Normalized Price Performance (Base 100)')
ax.set_xlabel('Date')
ax.set_ylabel('Normalized Price')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/task2/price_comparison.png', dpi=150, bbox_inches='tight')
print("  ✓ Saved: price_comparison.png")

# ============================================
# SUMMARY
# ============================================

print("\n[4/4] Summary Report")
print("=" * 60)

print("\n📊 CURRENT MARKET CONDITIONS:")
for stock in stocks:
    latest_row = df[df['Stock'] == stock].iloc[-1]
    rsi_status = "OVERBOUGHT" if latest_row['RSI_14'] > 70 else "OVERSOLD" if latest_row['RSI_14'] < 30 else "NEUTRAL"
    print(f"\n  {stock}:")
    print(f"    Price: ${latest_row['Close']:.2f}")
    print(f"    RSI: {latest_row['RSI_14']:.1f} ({rsi_status})")
    print(f"    Volatility: {latest_row['Volatility']*100:.1f}%")

print("\n📁 VISUALIZATIONS SAVED IN 'outputs/task2/':")
print("  • technical_dashboard.png - Full technical analysis for one stock")
print("  • stock_comparison.png - Compare all stocks")
print("  • price_comparison.png - Normalized price history")

print("\n" + "=" * 60)
print("✓ ANALYSIS COMPLETED SUCCESSFULLY!")
print("=" * 60)
