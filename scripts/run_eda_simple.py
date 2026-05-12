import pandas as pd
import matplotlib.pyplot as plt
import os

print("Running Task 1: EDA Analysis")

# Load data
df = pd.read_excel('data/raw/raw_analyst_ratings.xlsx', sheet_name='raw_analyst_ratings1')
df['date'] = pd.to_datetime(df['date'], utc=True, errors='coerce')
df = df.dropna(subset=['date'])

print(f"Loaded {len(df)} rows")

# Create output directory
os.makedirs('outputs', exist_ok=True)

# Basic statistics
df['headline_length'] = df['headline'].fillna('').astype(str).str.len()

# Plot headline length
plt.figure(figsize=(10, 6))
plt.hist(df['headline_length'], bins=50, edgecolor='black', alpha=0.7)
plt.title('Headline Length Distribution')
plt.xlabel('Characters')
plt.ylabel('Frequency')
plt.savefig('outputs/eda_summary.png', dpi=150)
plt.close()
print("✓ Saved: outputs/eda_summary.png")

# Stock distribution
stock_counts = df['stock'].value_counts()
plt.figure(figsize=(10, 8))
stock_counts.head(8).plot(kind='pie', autopct='%1.1f%%')
plt.title('Article Distribution by Stock')
plt.ylabel('')
plt.savefig('outputs/stock_distribution.png', dpi=150)
plt.close()
print("✓ Saved: outputs/stock_distribution.png")

print("Task 1 completed!")
