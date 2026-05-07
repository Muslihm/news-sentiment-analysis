"""
Financial News EDA - Task 1
Simplified working version
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import sys

print("="*60)
print("RUNNING EDA ANALYSIS")
print("="*60)
print(f"Python version: {sys.version}")
print(f"Pandas version: {pd.__version__}")

# Create sample data
print("\n📊 Creating sample dataset...")

np.random.seed(42)

headlines = [
    "Fed raises interest rates by 25 basis points",
    "Apple beats earnings expectations", 
    "Oil prices surge on supply concerns",
    "Tesla stock drops after recall",
    "FDA approves new cancer drug"
]

publishers = ['Reuters', 'Bloomberg', 'WSJ', 'CNBC']

# Create 100 sample articles
df = pd.DataFrame({
    'headline': np.random.choice(headlines, 100),
    'publisher': np.random.choice(publishers, 100),
    'timestamp': pd.date_range(end=datetime.now(), periods=100, freq='h')
})

print(f"✅ Created {len(df)} articles")

# 1. Descriptive Statistics
print("\n" + "="*60)
print("📝 1. DESCRIPTIVE STATISTICS")
print("="*60)

df['char_count'] = df['headline'].str.len()
df['word_count'] = df['headline'].str.split().str.len()

print(f"Average characters per headline: {df['char_count'].mean():.1f}")
print(f"Average words per headline: {df['word_count'].mean():.1f}")
print(f"Shortest headline: {df['word_count'].min()} words")
print(f"Longest headline: {df['word_count'].max()} words")

# 2. Publisher Analysis
print("\n" + "="*60)
print("📰 2. PUBLISHER ANALYSIS")
print("="*60)

publisher_counts = df['publisher'].value_counts()
for pub, count in publisher_counts.items():
    percentage = (count / len(df)) * 100
    print(f"{pub}: {count} articles ({percentage:.1f}%)")

# Find most active publisher
most_active = publisher_counts.index[0]
print(f"\n🏆 Most active publisher: {most_active}")

# 3. Time Series Analysis
print("\n" + "="*60)
print("⏰ 3. TIME SERIES ANALYSIS")
print("="*60)

df['hour'] = df['timestamp'].dt.hour
df['day'] = df['timestamp'].dt.day_name()

hourly_counts = df['hour'].value_counts()
peak_hour = hourly_counts.index[0]

print(f"Peak publishing hour: {peak_hour}:00")
print(f"Most active day: {df['day'].mode()[0]}")
print(f"Unique days in dataset: {df['day'].nunique()}")

# 4. Topic Modeling (Simple)
print("\n" + "="*60)
print("🔑 4. TOP KEYWORDS (Simple Analysis)")
print("="*60)

# Simple word frequency
all_words = ' '.join(df['headline']).lower().split()
common_words = pd.Series(all_words).value_counts().head(10)

print("Most common words in headlines:")
for word, count in common_words.items():
    print(f"  - '{word}': {count} times")

# 5. Create Visualizations
print("\n" + "="*60)
print("📊 5. CREATING VISUALIZATIONS")
print("="*60)

# Visualization 1: Headline length distribution
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.hist(df['char_count'], bins=15, edgecolor='black', alpha=0.7)
plt.xlabel('Character Count')
plt.ylabel('Frequency')
plt.title('Headline Length Distribution')
plt.axvline(df['char_count'].mean(), color='red', linestyle='--', label=f'Mean: {df["char_count"].mean():.0f}')
plt.legend()

# Visualization 2: Publisher bar chart
plt.subplot(1, 2, 2)
publisher_counts.plot(kind='bar', color='coral', edgecolor='black')
plt.xlabel('Publisher')
plt.ylabel('Number of Articles')
plt.title('Articles per Publisher')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('eda_visualizations.png', dpi=100, bbox_inches='tight')
print("✅ Saved visualization to: eda_visualizations.png")

# Display the plot (if running interactively)
plt.show()

print("\n" + "="*60)
print("✅ EDA ANALYSIS COMPLETE!")
print("="*60)
print("\n📁 Files generated:")
print("   - eda_visualizations.png (visualization chart)")
print("\n💡 To view the image:")
print("   - Windows: start eda_visualizations.png")
print("   - Mac: open eda_visualizations.png")
print("   - Linux: xdg-open eda_visualizations.png")
