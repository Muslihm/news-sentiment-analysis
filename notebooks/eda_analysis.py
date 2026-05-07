"""
Financial News EDA - Task 1
Complete analysis with descriptive stats, topic modeling, time series, and publisher analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

print("="*60)
print("Financial News EDA - Task 1")
print("="*60)

# Create dataset
np.random.seed(42)

headlines = [
    "Fed raises interest rates by 25 basis points",
    "Apple beats Q4 earnings expectations",
    "Oil prices surge on supply concerns",
    "Tesla stock drops on production issues",
    "FDA approves new cancer treatment"
]

publishers = ['Reuters', 'Bloomberg', 'WSJ', 'CNBC']

# Create 200 sample articles
dates = []
for i in range(200):
    base_date = datetime.now() - timedelta(days=np.random.randint(1, 30))
    hour = np.random.choice(range(8, 18))
    dates.append(base_date.replace(hour=hour, minute=np.random.randint(0, 59)))

df = pd.DataFrame({
    'headline': np.random.choice(headlines, 200),
    'publisher': np.random.choice(publishers, 200),
    'timestamp': dates
})

# Add email field
email_map = {
    'Reuters': 'news@reuters.com',
    'Bloomberg': 'editor@bloomberg.net',
    'WSJ': 'tips@wsj.com',
    'CNBC': 'news@cnbc.com'
}
df['email'] = df['publisher'].map(email_map)

print(f"Dataset: {len(df)} articles")
print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

# 1. Descriptive Statistics
df['char_count'] = df['headline'].str.len()
df['word_count'] = df['headline'].str.split().str.len()

print("\n" + "="*60)
print("1. DESCRIPTIVE STATISTICS")
print("="*60)
print(f"Average words per headline: {df['word_count'].mean():.1f}")
print(f"Average characters: {df['char_count'].mean():.1f}")
print(f"Median word count: {df['word_count'].median():.0f}")

# 2. Publisher Analysis
print("\n" + "="*60)
print("2. PUBLISHER ANALYSIS")
print("="*60)
pub_counts = df['publisher'].value_counts()
for pub, count in pub_counts.items():
    print(f"{pub}: {count} articles ({count/len(df)*100:.1f}%)")

print(f"\nMost active: {pub_counts.index[0]}")

# Email domain analysis
df['domain'] = df['email'].str.split('@').str[1]
print(f"\nUnique email domains: {df['domain'].nunique()}")

# 3. Time Series Analysis
print("\n" + "="*60)
print("3. TIME SERIES ANALYSIS")
print("="*60)

df['hour'] = df['timestamp'].dt.hour
df['day'] = df['timestamp'].dt.day_name()

hourly = df['hour'].value_counts().sort_index()
peak_hour = hourly.idxmax()

print(f"Peak publishing hour: {peak_hour}:00 ({hourly[peak_hour]} articles)")
print(f"Most active day: {df['day'].mode()[0]}")

# 4. Topic Modeling
print("\n" + "="*60)
print("4. TOPIC MODELING")
print("="*60)

stop_words = set(stopwords.words('english'))
tfidf = TfidfVectorizer(max_features=15, stop_words=list(stop_words))
tfidf_matrix = tfidf.fit_transform(df['headline'])
keywords = tfidf.get_feature_names_out()

print("Top keywords:")
for kw in keywords[:10]:
    print(f"  - {kw}")

# 5. Visualizations
print("\n" + "="*60)
print("5. CREATING VISUALIZATIONS")
print("="*60)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot 1: Word count histogram
axes[0,0].hist(df['word_count'], bins=15, edgecolor='black', alpha=0.7)
axes[0,0].set_xlabel('Word Count')
axes[0,0].set_ylabel('Frequency')
axes[0,0].set_title('Headline Word Count Distribution')

# Plot 2: Publisher bar chart
pub_counts.plot(kind='bar', ax=axes[0,1], color='coral', edgecolor='black')
axes[0,1].set_xlabel('Publisher')
axes[0,1].set_ylabel('Articles')
axes[0,1].set_title('Articles per Publisher')
axes[0,1].tick_params(axis='x', rotation=45)

# Plot 3: Hourly pattern
hourly.plot(kind='bar', ax=axes[1,0], color='lightgreen', edgecolor='black')
axes[1,0].set_xlabel('Hour of Day')
axes[1,0].set_ylabel('Articles')
axes[1,0].set_title('Publication Time Distribution')

# Plot 4: Daily volume
daily_vol = df.groupby(df['timestamp'].dt.date).size()
axes[1,1].plot(daily_vol.index, daily_vol.values, marker='o', linewidth=1)
axes[1,1].set_xlabel('Date')
axes[1,1].set_ylabel('Articles')
axes[1,1].set_title('Daily News Volume')
axes[1,1].tick_params(axis='x', rotation=45)

plt.suptitle('Financial News EDA - Task 1', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('eda_complete_analysis.png', dpi=150)
print("Saved: eda_complete_analysis.png")

print("\n" + "="*60)
print("EDA COMPLETE!")
print("="*60)