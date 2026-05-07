"""
Financial News EDA - Task 1
Complete implementation with descriptive statistics, publisher analysis, and time series
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("📊 FINANCIAL NEWS EDA - TASK 1: DESCRIPTIVE STATISTICS")
print("="*70)

# ============================================
# CREATE SAMPLE DATASET (Replace with real data)
# ============================================
np.random.seed(42)

# Realistic financial headlines
headlines = [
    "Fed raises interest rates by 25 basis points",
    "Apple beats Q4 earnings expectations",
    "Oil prices surge amid Middle East tensions",
    "Tesla stock drops 5% on production concerns",
    "FDA approves groundbreaking Alzheimer's treatment",
    "Amazon announces stock split",
    "Job market shows unexpected strength in March",
    "Bank of America price target raised to $45",
    "Inflation data comes in lower than expected",
    "Fed signals potential rate cuts in 2024",
    "Microsoft acquires AI startup for $1.5B",
    "Crypto market rallies after regulatory clarity",
    "Goldman Sachs downgrades tech sector",
    "Retail sales beat estimates in holiday season",
    "China's economy shows signs of recovery",
    "European Central Bank holds rates steady",
    "JP Morgan reports record quarterly profits",
    "Bitcoin surpasses $60,000 milestone",
    "Housing market cools as mortgage rates rise",
    "Supply chain disruptions ease in manufacturing"
]

publishers = ['Reuters', 'Bloomberg', 'Wall Street Journal', 'CNBC', 
              'Financial Times', 'Yahoo Finance', 'MarketWatch']

# Create dates (simpler approach to avoid errors)
dates = []
market_hours = list(range(9, 18))  # 9 AM to 5 PM
off_hours = list(range(0, 24))  # All hours

for i in range(300):
    base_date = datetime.now() - timedelta(days=np.random.randint(1, 60))
    # 70% chance of market hours, 30% chance of off hours
    if np.random.random() < 0.7:
        hour = np.random.choice(market_hours)
    else:
        hour = np.random.choice(off_hours)
    dates.append(base_date.replace(hour=hour, minute=np.random.randint(0, 59)))

df = pd.DataFrame({
    'headline': np.random.choice(headlines, 300),
    'publisher': np.random.choice(publishers, 300),
    'timestamp': dates
})

# Add market event spikes
event_dates = [
    datetime.now() - timedelta(days=45),  # Fed meeting
    datetime.now() - timedelta(days=30),  # Earnings season
    datetime.now() - timedelta(days=15),  # Jobs report
]
for event_date in event_dates:
    for _ in range(15):  # Add extra articles on event days
        new_row = pd.DataFrame({
            'headline': [np.random.choice([
                "Fed announces surprise rate hike",
                "Jobs report beats expectations dramatically",
                "Major earnings beat across tech sector"
            ])],
            'publisher': [np.random.choice(publishers)],
            'timestamp': [event_date]
        })
        df = pd.concat([df, new_row], ignore_index=True)

df = df.sort_values('timestamp').reset_index(drop=True)

print(f"\n📅 Dataset Info:")
print(f"   Total articles: {len(df)}")
print(f"   Date range: {df['timestamp'].min().strftime('%Y-%m-%d')} to {df['timestamp'].max().strftime('%Y-%m-%d')}")
print(f"   Time span: {(df['timestamp'].max() - df['timestamp'].min()).days} days")

# ============================================
# 2a. TEXTUAL LENGTHS - DESCRIPTIVE STATISTICS
# ============================================
print("\n" + "="*70)
print("📝 2a. TEXTUAL LENGTHS ANALYSIS")
print("="*70)

df['char_count'] = df['headline'].str.len()
df['word_count'] = df['headline'].str.split().str.len()

print("\n📊 HEADLINE LENGTH STATISTICS:")
print("-" * 55)
print(f"{'Metric':<25} {'Characters':<15} {'Words':<15}")
print("-" * 55)
print(f"{'Mean (Average)':<25} {df['char_count'].mean():<15.1f} {df['word_count'].mean():<15.1f}")
print(f"{'Median':<25} {df['char_count'].median():<15.0f} {df['word_count'].median():<15.0f}")
print(f"{'Mode':<25} {df['char_count'].mode()[0]:<15} {df['word_count'].mode()[0]:<15}")
print(f"{'Standard Deviation':<25} {df['char_count'].std():<15.1f} {df['word_count'].std():<15.1f}")
print(f"{'Minimum':<25} {df['char_count'].min():<15} {df['word_count'].min():<15}")
print(f"{'25th Percentile':<25} {df['char_count'].quantile(0.25):<15.0f} {df['word_count'].quantile(0.25):<15.0f}")
print(f"{'75th Percentile':<25} {df['char_count'].quantile(0.75):<15.0f} {df['word_count'].quantile(0.75):<15.0f}")
print(f"{'Maximum':<25} {df['char_count'].max():<15} {df['word_count'].max():<15}")

# ============================================
# 2b. PUBLISHER ANALYSIS
# ============================================
print("\n" + "="*70)
print("📰 2b. PUBLISHER ANALYSIS")
print("="*70)

publisher_counts = df['publisher'].value_counts()
publisher_pct = (publisher_counts / len(df) * 100).round(2)

print("\n🏢 ARTICLES PER PUBLISHER (RANKED):")
print("-" * 55)
print(f"{'Rank':<6} {'Publisher':<25} {'Articles':<10} {'%':<8}")
print("-" * 55)
for i, (pub, count) in enumerate(publisher_counts.items(), 1):
    print(f"{i:<6} {pub:<25} {count:<10} {publisher_pct[pub]:<8}%")

print(f"\n🏆 MOST ACTIVE PUBLISHER: {publisher_counts.index[0]} ({publisher_counts.values[0]} articles)")
print(f"📉 LEAST ACTIVE: {publisher_counts.index[-1]} ({publisher_counts.values[-1]} articles)")

# ============================================
# 2c. TIME SERIES & MARKET EVENTS
# ============================================
print("\n" + "="*70)
print("⏰ 2c. TIME SERIES ANALYSIS")
print("="*70)

# Extract time features
df['date'] = df['timestamp'].dt.date
df['hour'] = df['timestamp'].dt.hour
df['day_of_week'] = df['timestamp'].dt.day_name()

daily_volume = df.groupby('date').size()

print(f"\n📈 DAILY PUBLICATION VOLUME:")
print("-" * 40)
print(f"Average daily articles: {daily_volume.mean():.1f}")
print(f"Median daily articles: {daily_volume.median():.0f}")
print(f"Max daily articles: {daily_volume.max()} ({daily_volume.idxmax()})")
print(f"Min daily articles: {daily_volume.min()} ({daily_volume.idxmin()})")

# Detect spikes (days with > 2 standard deviations)
threshold = daily_volume.mean() + (2 * daily_volume.std())
spike_days = daily_volume[daily_volume > threshold]

print(f"\n⚠️ NEWS VOLUME SPIKES:")
print("-" * 40)
if len(spike_days) > 0:
    for date, vol in spike_days.items():
        pct_above = (vol - daily_volume.mean()) / daily_volume.mean() * 100
        print(f"   📅 {date}: {vol} articles ({pct_above:.0f}% above avg)")
    print("\n   🔍 Possible causes: Fed meetings, earnings season, jobs reports")
else:
    print("   No significant spikes detected (volume within normal range)")

# Hourly patterns
hourly_volume = df.groupby('hour').size()
peak_hour = hourly_volume.idxmax()

print(f"\n⏰ HOURLY PATTERNS:")
print("-" * 40)
print(f"Peak publishing hour: {peak_hour}:00 ({hourly_volume[peak_hour]} articles)")
print(f"Quietest hour: {hourly_volume.idxmin()}:00 ({hourly_volume[hourly_volume.idxmin()]} articles)")

# Weekday patterns
weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
weekday_volume = df['day_of_week'].value_counts().reindex(weekday_order)
print(f"\n📅 WEEKDAY PATTERNS:")
print("-" * 40)
print(f"Most active: {weekday_volume.idxmax()} ({weekday_volume.max()} articles)")
print(f"Least active: {weekday_volume.idxmin()} ({weekday_volume.min()} articles)")

# ============================================
# CREATE VISUALIZATIONS
# ============================================
print("\n" + "="*70)
print("📊 GENERATING VISUALIZATIONS...")
print("="*70)

fig = plt.figure(figsize=(15, 10))

# 1. Character count distribution
plt.subplot(2, 3, 1)
plt.hist(df['char_count'], bins=25, edgecolor='black', alpha=0.7, color='steelblue')
plt.axvline(df['char_count'].mean(), color='red', linestyle='--', label=f'Mean: {df["char_count"].mean():.0f}')
plt.xlabel('Characters')
plt.ylabel('Frequency')
plt.title('Headline Character Count')
plt.legend()

# 2. Word count distribution
plt.subplot(2, 3, 2)
plt.hist(df['word_count'], bins=15, edgecolor='black', alpha=0.7, color='coral')
plt.axvline(df['word_count'].mean(), color='red', linestyle='--', label=f'Mean: {df["word_count"].mean():.1f}')
plt.xlabel('Words')
plt.ylabel('Frequency')
plt.title('Headline Word Count')
plt.legend()

# 3. Publisher bar chart
plt.subplot(2, 3, 3)
publisher_counts.plot(kind='bar', color='teal', edgecolor='black')
plt.xlabel('Publisher')
plt.ylabel('Articles')
plt.title('Articles per Publisher')
plt.xticks(rotation=45, ha='right')

# 4. Daily volume with spikes
plt.subplot(2, 3, 4)
plt.plot(daily_volume.index, daily_volume.values, marker='o', markersize=3, linewidth=1)
plt.axhline(y=daily_volume.mean(), color='green', linestyle='--', label=f'Mean: {daily_volume.mean():.1f}')
if len(spike_days) > 0:
    for date in spike_days.index:
        plt.axvline(x=date, color='red', alpha=0.3, linewidth=2)
plt.xlabel('Date')
plt.ylabel('Articles')
plt.title('Daily News Volume (Red lines = spikes)')
plt.legend()
plt.xticks(rotation=45)

# 5. Hourly distribution
plt.subplot(2, 3, 5)
bars = plt.bar(hourly_volume.index, hourly_volume.values, edgecolor='black', alpha=0.7, color='lightgreen')
bars[peak_hour].set_color('red')
plt.xlabel('Hour of Day')
plt.ylabel('Articles')
plt.title(f'Publication Time (Peak: {peak_hour}:00)')
plt.xticks(range(0, 24, 3))

# 6. Weekday distribution
plt.subplot(2, 3, 6)
weekday_volume.plot(kind='bar', color='purple', edgecolor='black', alpha=0.7)
plt.xlabel('Day of Week')
plt.ylabel('Articles')
plt.title('Articles by Day of Week')
plt.xticks(rotation=45)

plt.suptitle('Financial News EDA - Task 1: Complete Analysis', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('eda_complete_analysis.png', dpi=150, bbox_inches='tight')
print("✅ Saved: eda_complete_analysis.png")

# ============================================
# FINAL SUMMARY
# ============================================
print("\n" + "="*70)
print("📋 TASK 1 - COMPLETE SUMMARY")
print("="*70)

print(f"\n✅ 2a. TEXTUAL LENGTHS:")
print(f"     Average: {df['word_count'].mean():.1f} words / {df['char_count'].mean():.1f} chars")

print(f"\n✅ 2b. PUBLISHER RANKING:")
print(f"     1. {publisher_counts.index[0]} ({publisher_counts.values[0]} articles)")
if len(publisher_counts) > 1:
    print(f"     2. {publisher_counts.index[1]} ({publisher_counts.values[1]} articles)")
if len(publisher_counts) > 2:
    print(f"     3. {publisher_counts.index[2]} ({publisher_counts.values[2]} articles)")

print(f"\n✅ 2c. TIME SERIES:")
print(f"     Peak hour: {peak_hour}:00")
print(f"     Busiest day: {weekday_volume.idxmax()}")
print(f"     Spikes detected: {len(spike_days)}")
print(f"     Weekend vs Weekday: {weekday_volume.loc['Saturday':'Sunday'].sum()} vs {weekday_volume.loc['Monday':'Friday'].sum()} articles")

print("\n" + "="*70)
print("✅ TASK 1 COMPLETE! View eda_complete_analysis.png for charts")
print("="*70)