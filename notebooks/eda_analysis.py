
"""
Financial News EDA - Task 1
Complete implementation with Topic Modeling, Time Series & Publisher Analysis
"""

from datetime import datetime, timedelta
# NLP Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import nltk
from nltk.corpus import stopwords
import re
import warnings
warnings.filterwarnings('ignore')

# Download NLTK data
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

 task-2
print("="*70)
print("📊 FINANCIAL NEWS EDA - TASK 1: COMPLETE ANALYSIS")
print("Sections: 1. Descriptive Stats | 2. Topic Modeling | 3. Time Series | 4. Publisher Analysis")
print("="*70)

# ============================================
# CREATE ENHANCED DATASET
# ============================================
np.random.seed(42)

# Rich financial headlines for topic modeling
headlines = [
    # Interest rates & Fed
    "Fed raises interest rates by 25 basis points",
    "Federal Reserve signals rate cuts coming in 2024",
    "Fed holds rates steady amid inflation concerns",
    "Powell hints at aggressive rate hikes",
    
    # Earnings reports
    "Apple beats Q4 earnings expectations by wide margin",
    "Tesla quarterly earnings miss estimates on production delays",
    "Amazon Web Services drives earnings beat",
    "Microsoft cloud revenue boosts earnings surprise",
    
    # FDA & healthcare
    "FDA approves breakthrough Alzheimer's treatment",
    "FDA grants fast track status to cancer drug",
    "FDA approval paves way for new weight loss drug",
    
    # Price targets & analyst ratings
    "Bank of America price target raised to $65",
    "Goldman Sachs upgrades Apple to buy with $200 target",
    "Analysts lower price target on Tesla citing competition",
    
    # Market movements
    "Oil prices surge 5% on supply concerns",
    "Bitcoin rallies past $60000 on ETF inflows",
    "Tech stocks lead market rally on AI optimism",
    
    # Economic data
    "Jobs report shows unexpected strength in March",
    "CPI inflation data comes in lower than expected",
    "Retail sales beat estimates signaling strong economy",
    
    # Mergers & acquisitions
    "Microsoft acquires AI startup for $1.5 billion",
    "Amazon to acquire healthcare provider for $3.9B",
    "Cisco announces $28 billion acquisition",
    
    # Stock movements
    "Tesla stock drops 8% on delivery concerns",
    "Nvidia shares surge on AI chip demand",
    "Meta stock rallies after strong earnings"
]

publishers = ['Reuters', 'Bloomberg', 'Wall Street Journal', 'CNBC', 
              'Financial Times', 'Yahoo Finance', 'MarketWatch', 'Seeking Alpha']

# Create realistic dates (last 90 days)
dates = []
for i in range(500):
    base_date = datetime.now() - timedelta(days=np.random.randint(1, 90))
    # More news during market hours (9 AM - 5 PM)
    if np.random.random() < 0.7:
        hour = np.random.choice(range(9, 17))
    else:
        hour = np.random.choice(range(0, 24))
    dates.append(base_date.replace(hour=hour, minute=np.random.randint(0, 59)))

df = pd.DataFrame({
    'headline': np.random.choice(headlines, 500),
    'publisher': np.random.choice(publishers, 500),
    'timestamp': dates
})

# Add email addresses for some publishers
email_mapping = {
    'Reuters': 'news@reuters.com',
    'Bloomberg': 'editor@bloomberg.net',
    'Wall Street Journal': 'tips@wsj.com',
    'CNBC': 'news@cnbc.com',
    'Financial Times': 'editors@ft.com',
    'Yahoo Finance': 'tips@yahooinc.com'
}
df['email'] = df['publisher'].map(email_mapping).fillna('unknown@news.com')

# Add market event spikes
event_dates = {
    datetime.now() - timedelta(days=60): "Fed Meeting",
    datetime.now() - timedelta(days=45): "Jobs Report",
    datetime.now() - timedelta(days=30): "CPI Release",
    datetime.now() - timedelta(days=15): "Earnings Season Peak"
}
for event_date, event_name in event_dates.items():
    for _ in range(20):  # Add spike in volume
        new_row = pd.DataFrame({
            'headline': [f"{event_name}: Market reacts to new data"],
            'publisher': [np.random.choice(publishers)],
            'timestamp': [event_date],
            'email': [np.random.choice(list(email_mapping.values()))]
        })
        df = pd.concat([df, new_row], ignore_index=True)

df = df.sort_values('timestamp').reset_index(drop=True)
df['char_count'] = df['headline'].str.len()
df['word_count'] = df['headline'].str.split().str.len()

print(f"\n📅 Dataset Overview:")
print(f"   Total articles: {len(df)}")
print(f"   Date range: {df['timestamp'].min().strftime('%Y-%m-%d')} to {df['timestamp'].max().strftime('%Y-%m-%d')}")
print(f"   Unique publishers: {df['publisher'].nunique()}")
print(f"   Time span: {(df['timestamp'].max() - df['timestamp'].min()).days} days")

# ============================================
# SECTION 3: TEXT ANALYSIS - TOPIC MODELING
# ============================================
print("\n" + "="*70)
print("🔍 SECTION 3: TEXT ANALYSIS & TOPIC MODELING")
print("="*70)

# Prepare stop words
stop_words = set(stopwords.words('english'))
finance_stopwords = {'said', 'says', 'will', 'can', 'may', 'would', 'could', 'also', 'get', 'make'}
stop_words.update(finance_stopwords)

# 3a. TF-IDF for Keyword Extraction
print("\n📝 3a. TF-IDF KEYWORD EXTRACTION")
print("-" * 50)

tfidf_vectorizer = TfidfVectorizer(max_features=25, stop_words=list(stop_words), ngram_range=(1, 2))
tfidf_matrix = tfidf_vectorizer.fit_transform(df['headline'])
feature_names = tfidf_vectorizer.get_feature_names_out()
tfidf_scores = tfidf_matrix.sum(axis=0).A1

keyword_df = pd.DataFrame({'keyword': feature_names, 'score': tfidf_scores})
keyword_df = keyword_df.sort_values('score', ascending=False)

print("\n🏆 TOP 15 KEYWORDS & PHRASES (TF-IDF):")
for i, row in keyword_df.head(15).iterrows():
    print(f"   {row['keyword']:<35} Score: {row['score']:.4f}")

# 3b. Extract Bigrams (Common Phrases)
print("\n📚 3b. COMMON PHRASES (BIGRAMS)")
print("-" * 50)

bigram_vectorizer = CountVectorizer(ngram_range=(2, 2), max_features=20, stop_words=list(stop_words))
bigram_matrix = bigram_vectorizer.fit_transform(df['headline'])
bigram_counts = bigram_matrix.sum(axis=0).A1
bigram_features = bigram_vectorizer.get_feature_names_out()

bigram_df = pd.DataFrame({'phrase': bigram_features, 'count': bigram_counts})
bigram_df = bigram_df.sort_values('count', ascending=False)

print("\n🔑 TOP 10 PHRASES (Bigrams):")
for i, row in bigram_df.head(10).iterrows():
    print(f"   \"{row['phrase']}\" appears {row['count']} times")

# 3c. LDA Topic Modeling
print("\n🎯 3c. LDA TOPIC MODELING")
print("-" * 50)

count_vectorizer = CountVectorizer(max_features=100, stop_words=list(stop_words))
doc_term_matrix = count_vectorizer.fit_transform(df['headline'])

n_topics = 5
lda = LatentDirichletAllocation(n_components=n_topics, random_state=42, max_iter=100)
lda.fit(doc_term_matrix)
feature_names_lda = count_vectorizer.get_feature_names_out()

print("\n📊 DISCOVERED TOPICS (LDA):")
for topic_idx, topic in enumerate(lda.components_):
    top_words_idx = topic.argsort()[:-8:-1]
    top_words = [feature_names_lda[i] for i in top_words_idx]
    
    # Identify topic theme
    if any(word in top_words for word in ['rate', 'fed', 'rates']):
        theme = "Central Bank / Interest Rates"
    elif any(word in top_words for word in ['earnings', 'beat', 'miss']):
        theme = "Corporate Earnings"
    elif any(word in top_words for word in ['fda', 'approves', 'drug']):
        theme = "FDA / Healthcare"
    elif any(word in top_words for word in ['price', 'target', 'analyst']):
        theme = "Price Targets / Analyst Ratings"
    elif any(word in top_words for word in ['acquires', 'acquisition', 'merger']):
        theme = "Mergers & Acquisitions"
    else:
        theme = "Market Movements"
    
    print(f"\n   Topic {topic_idx + 1}: {theme}")
    print(f"   Keywords: {', '.join(top_words[:6])}")

# ============================================
# SECTION 4: TIME SERIES ANALYSIS
# ============================================
print("\n" + "="*70)
print("⏰ SECTION 4: TIME SERIES ANALYSIS")
print("="*70)

# Extract time features
df['date'] = df['timestamp'].dt.date
df['hour'] = df['timestamp'].dt.hour
df['day_of_week'] = df['timestamp'].dt.day_name()
df['week'] = df['timestamp'].dt.isocalendar().week

daily_volume = df.groupby('date').size()
hourly_volume = df.groupby('hour').size()
weekday_volume = df['day_of_week'].value_counts()

# 4a. Publication frequency over time
print("\n📈 4a. PUBLICATION VOLUME OVER TIME")
print("-" * 50)
print(f"Total time period: {df['timestamp'].min().strftime('%Y-%m-%d')} to {df['timestamp'].max().strftime('%Y-%m-%d')}")
print(f"Average daily articles: {daily_volume.mean():.1f}")
print(f"Median daily articles: {daily_volume.median():.0f}")
print(f"Peak day: {daily_volume.idxmax()} ({daily_volume.max()} articles)")
print(f"Lowest day: {daily_volume.idxmin()} ({daily_volume.min()} articles)")

# 4b. Identify spikes and relate to market events
print("\n⚠️ 4b. VOLUME SPIKES & MARKET EVENTS")
print("-" * 50)

mean_vol = daily_volume.mean()
std_vol = daily_volume.std()
spike_threshold = mean_vol + (1.5 * std_vol)
spike_days = daily_volume[daily_volume > spike_threshold]

print(f"Normal volume range: {mean_vol - std_vol:.1f} to {mean_vol + std_vol:.1f} articles")
print(f"Spike threshold: {spike_threshold:.1f} articles")
print(f"Spikes detected: {len(spike_days)} days")

if len(spike_days) > 0:
    print("\n🔍 SPIKE DAYS AND CORRELATED MARKET EVENTS:")
    spike_count = 0
    for date, volume in spike_days.items():
        if spike_count >= 5:  # Limit to 5 spikes for display
            break
        pct_increase = ((volume - mean_vol) / mean_vol) * 100
        
        # Check if this date is near any event date
        event_found = None
        for event_date, event_name in event_dates.items():
            date_diff = abs((datetime.combine(date, datetime.min.time()) - event_date).days)
            if date_diff <= 2:  # Within 2 days of event
                event_found = event_name
                break
        
        if event_found:
            print(f"   📅 {date}: {volume} articles (+{pct_increase:.0f}%) → {event_found}")
        else:
            print(f"   📅 {date}: {volume} articles (+{pct_increase:.0f}%) → Possible: Scheduled economic release or breaking news")
        spike_count += 1
    
    if len(spike_days) > 5:
        print(f"   ... and {len(spike_days) - 5} more spike days")
else:
    print("   No significant spikes detected")

# 4c. Publishing times analysis
print("\n⏰ 4c. PUBLISHING TIME PATTERNS")
print("-" * 50)

peak_hour = hourly_volume.idxmax()
quiet_hour = hourly_volume.idxmin()
market_open_volume = hourly_volume.get(9, 0)
market_close_volume = hourly_volume.get(16, 0)

print(f"Peak publishing time: {peak_hour}:00 ({hourly_volume[peak_hour]} articles)")
print(f"Quietest time: {quiet_hour}:00 ({hourly_volume[quiet_hour]} articles)")

# News distribution by time blocks
morning = sum(hourly_volume[h] for h in range(6, 12))
afternoon = sum(hourly_volume[h] for h in range(12, 17))
evening = sum(hourly_volume[h] for h in range(17, 21))
night = sum(hourly_volume[h] for h in range(21, 24)) + sum(hourly_volume[h] for h in range(0, 6))

print(f"\n📊 News Distribution by Time Block:")
print(f"   Morning (6 AM - 12 PM): {morning} articles ({morning/len(df)*100:.1f}%)")
print(f"   Afternoon (12 PM - 5 PM): {afternoon} articles ({afternoon/len(df)*100:.1f}%)")
print(f"   Evening (5 PM - 9 PM): {evening} articles ({evening/len(df)*100:.1f}%)")
print(f"   Night (9 PM - 6 AM): {night} articles ({night/len(df)*100:.1f}%)")

# ============================================
# SECTION 5: PUBLISHER ANALYSIS
# ============================================
print("\n" + "="*70)
print("📰 SECTION 5: PUBLISHER ANALYSIS")
print("="*70)

# 5a. Most active publishers and their coverage
print("\n🏢 5a. MOST ACTIVE PUBLISHERS")
print("-" * 50)

publisher_counts = df['publisher'].value_counts()
publisher_pct = (publisher_counts / len(df) * 100).round(2)

print(f"{'Rank':<6} {'Publisher':<25} {'Articles':<10} {'Share':<10} {'Avg Words':<12}")
print("-" * 65)
for i, (pub, count) in enumerate(publisher_counts.head(7).items(), 1):
    avg_words = df[df['publisher'] == pub]['word_count'].mean()
    print(f"{i:<6} {pub:<25} {count:<10} {publisher_pct[pub]:<9}% {avg_words:.1f}")

# Characterize publisher coverage (topic focus)
print("\n📊 PUBLISHER COVERAGE CHARACTERISTICS:")
print("-" * 50)

for pub in publisher_counts.head(5).index:
    pub_headlines = df[df['publisher'] == pub]['headline'].str.lower().str.cat()
    topics = []
    if 'rate' in pub_headlines or 'fed' in pub_headlines:
        topics.append("Central Bank")
    if 'earnings' in pub_headlines or 'beat' in pub_headlines:
        topics.append("Earnings")
    if 'fda' in pub_headlines or 'approves' in pub_headlines:
        topics.append("Healthcare")
    if 'price' in pub_headlines or 'target' in pub_headlines:
        topics.append("Analysis")
    if 'acquisition' in pub_headlines or 'acquires' in pub_headlines:
        topics.append("M&A")
    
    if not topics:
        topics = ["General Market"]
    
    print(f"   {pub}: Focus on {', '.join(topics[:3])}")

# 5b. Email domain analysis
print("\n📧 5b. EMAIL DOMAIN ANALYSIS")
print("-" * 50)

df['domain'] = df['email'].str.split('@').str[1]

# Identify unique domains
domains = df['domain'].value_counts()
print(f"Unique domains detected: {len(domains)}")
print(f"\nTop domains by article count:")
for domain, count in domains.head(8).items():
    org_type = "News Agency" if 'reuters' in str(domain) else "Financial Media" if 'bloomberg' in str(domain) else "Business Publication" if 'wsj' in str(domain) else "News Network" if 'cnbc' in str(domain) else "Other"
    print(f"   {domain:<25} {count:>3} articles ({org_type})")

# Publisher collaboration patterns
print("\n🔗 PUBLISHER INSIGHTS:")
print("-" * 50)
print(f"Most diverse coverage: {publisher_counts.index[0]} ({publisher_counts.values[0]} articles)")
print(f"Most specialized: {publisher_counts.index[-1]} (smallest volume)")
print(f"Average articles per publisher: {len(df)/df['publisher'].nunique():.1f}")

# ============================================
# COMPREHENSIVE VISUALIZATIONS
# ============================================
print("\n" + "="*70)
print("📊 GENERATING COMPREHENSIVE VISUALIZATIONS...")
print("="*70)

fig = plt.figure(figsize=(18, 14))

# 1. Topic Modeling - Top Keywords
plt.subplot(3, 3, 1)
top_keywords = keyword_df.head(12)
plt.barh(top_keywords['keyword'], top_keywords['score'], color='teal', edgecolor='black')
plt.xlabel('TF-IDF Score')
plt.title('Top Keywords & Phrases')
plt.gca().invert_yaxis()

# 2. LDA Topic Distribution
plt.subplot(3, 3, 2)
topic_scores = []
for topic_idx in range(n_topics):
    topic_scores.append(lda.components_[topic_idx].sum())
plt.bar(range(1, n_topics+1), topic_scores, color='coral', edgecolor='black')
plt.xlabel('Topic Number')
plt.ylabel('Topic Weight')
plt.title('LDA Topic Importance')
plt.xticks(range(1, n_topics+1))

# 3. Daily Publication Volume
plt.subplot(3, 3, 3)
plt.plot(daily_volume.index, daily_volume.values, marker='o', markersize=3, linewidth=1, alpha=0.7)
plt.axhline(y=mean_vol, color='green', linestyle='--', label=f'Mean: {mean_vol:.1f}')
if len(spike_days) > 0:
    for date in list(spike_days.index)[:3]:
        plt.axvline(x=date, color='red', alpha=0.3, linewidth=2)
plt.xlabel('Date')
plt.ylabel('Articles')
plt.title('Daily News Volume (Red = Spikes)')
plt.legend()
plt.xticks(rotation=45)

# 4. Hourly Publication Pattern
plt.subplot(3, 3, 4)
bars = plt.bar(hourly_volume.index, hourly_volume.values, edgecolor='black', alpha=0.7, color='lightgreen')
bars[peak_hour].set_color('red')
plt.xlabel('Hour of Day (24h)')
plt.ylabel('Articles')
plt.title(f'Publication Time (Peak: {peak_hour}:00)')
plt.xticks(range(0, 24, 3))

# 5. Publisher Market Share - Pie Chart
plt.subplot(3, 3, 5)
top_pubs = publisher_counts.head(6)
other_count = publisher_counts.iloc[6:].sum()
if other_count > 0:
    top_pubs['Other'] = other_count
plt.pie(top_pubs.values, labels=top_pubs.index, autopct='%1.1f%%', startangle=90)
plt.title('Publisher Market Share')

# 6. Weekday vs Weekend
plt.subplot(3, 3, 6)
weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
weekday_vol = df['day_of_week'].value_counts().reindex(weekday_order)
colors = ['steelblue']*5 + ['lightgray']*2
plt.bar(weekday_vol.index, weekday_vol.values, color=colors, edgecolor='black')
plt.xlabel('Day of Week')
plt.ylabel('Articles')
plt.title('Weekday vs Weekend Publication')
plt.xticks(rotation=45)

# 7. Publisher Average Headline Length
plt.subplot(3, 3, 7)
publisher_avg_words = df.groupby('publisher')['word_count'].mean().sort_values(ascending=False).head(8)
plt.barh(publisher_avg_words.index, publisher_avg_words.values, color='purple', edgecolor='black', alpha=0.7)
plt.xlabel('Average Words per Headline')
plt.title('Publisher Writing Style (Verbosity)')
plt.gca().invert_yaxis()

# 8. Time Series Heatmap (Hour vs Day)
plt.subplot(3, 3, 8)
pivot_table = df.pivot_table(index='hour', columns='day_of_week', values='headline', aggfunc='count', fill_value=0)
pivot_table = pivot_table[weekday_order]
im = plt.imshow(pivot_table.values, aspect='auto', cmap='YlOrRd')
plt.colorbar(im, label='Article Count')
plt.xlabel('Day of Week')
plt.ylabel('Hour of Day')
plt.title('Publication Heatmap: Hour vs Day')
plt.xticks(range(7), weekday_order, rotation=45)
plt.yticks(range(0, 24, 3), range(0, 24, 3))

# 9. Keyword Timeline
plt.subplot(3, 3, 9)
df['date_week'] = df['timestamp'].dt.to_period('W')
keywords_over_time = {}
for kw in keyword_df.head(5)['keyword'].values:
    mask = df['headline'].str.contains(kw, case=False, na=False)
    keywords_over_time[kw] = df[mask].groupby('date_week').size()
for kw, series in keywords_over_time.items():
    if len(series) > 0:
        plt.plot([str(p) for p in series.index], series.values, marker='o', label=kw, linewidth=1)
plt.xlabel('Week')
plt.ylabel('Frequency')
plt.title('Top Keywords Over Time')
plt.legend()
plt.xticks(rotation=45)

plt.suptitle('Financial News EDA - Task 1: Complete Analysis\n(Topic Modeling + Time Series + Publisher Analysis)', 
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('eda_complete_analysis.png', dpi=150, bbox_inches='tight')
print("✅ Saved: eda_complete_analysis.png")

# ============================================
# FINAL SUMMARY REPORT
# ============================================
print("\n" + "="*70)
print("📋 TASK 1 - COMPLETE SUMMARY REPORT")
print("="*70)

print("\n✅ SECTION 3: TOPIC MODELING")
print(f"   • Top keyword: '{keyword_df.iloc[0]['keyword']}' (score: {keyword_df.iloc[0]['score']:.3f})")
print(f"   • Top phrase: \"{bigram_df.iloc[0]['phrase']}\" (appears {bigram_df.iloc[0]['count']}x)")
print(f"   • {n_topics} major topics identified using LDA")

print("\n✅ SECTION 4: TIME SERIES")
print(f"   • Peak news time: {peak_hour}:00 ({hourly_volume[peak_hour]} articles)")
print(f"   • Morning news {morning/len(df)*100:.1f}% / Afternoon {afternoon/len(df)*100:.1f}%")
print(f"   • {len(spike_days)} volume spikes detected")

print("\n✅ SECTION 5: PUBLISHER ANALYSIS")
print(f"   • Most active: {publisher_counts.index[0]} ({publisher_counts.values[0]} articles, {publisher_pct[publisher_counts.index[0]]}%)")
print(f"   • Unique publishers: {df['publisher'].nunique()}")
print(f"   • Email domains identified: {df['domain'].nunique()}")

print("\n" + "="*70)
print("✅ TASK 1 FULLY COMPLETE! View eda_complete_analysis.png for all visualizations")
print("="*70)
=======
# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("= - eda_analysis.py:22" * 70)
print("FINANCIAL NEWS SENTIMENT ANALYSIS  TASK 1: EXPLORATORY DATA ANALYSIS - eda_analysis.py:23")
print("= - eda_analysis.py:24" * 70)

# ============================================================================
# 1. DATA LOADING
# ============================================================================
print("\n - eda_analysis.py:29" + "="*50)
print("1. DATA LOADING - eda_analysis.py:30")
print("= - eda_analysis.py:31"*50)

df = pd.read_excel('data/raw/raw_analyst_ratings.xlsx', sheet_name='raw_analyst_ratings1')
print(f"Dataset shape: {df.shape} - eda_analysis.py:34")
print(f"Columns: {df.columns.tolist()} - eda_analysis.py:35")

# Convert date
df['date'] = pd.to_datetime(df['date'], errors='coerce')
print(f"Date range: {df['date'].min()} to {df['date'].max()} - eda_analysis.py:39")
print(f"Unique stocks: {df['stock'].nunique()} - eda_analysis.py:40")
print(f"Stocks: {df['stock'].unique()} - eda_analysis.py:41")

# ============================================================================
# 2. DESCRIPTIVE STATISTICS
# ============================================================================
print("\n - eda_analysis.py:46" + "="*50)
print("2. DESCRIPTIVE STATISTICS - eda_analysis.py:47")
print("= - eda_analysis.py:48"*50)

df['headline_length'] = df['headline'].fillna('').astype(str).str.len()
df['word_count'] = df['headline'].fillna('').astype(str).str.split().str.len()

print(f"\nHeadline Character Count: - eda_analysis.py:53")
print(f"Mean: {df['headline_length'].mean():.1f} - eda_analysis.py:54")
print(f"Median: {df['headline_length'].median():.0f} - eda_analysis.py:55")
print(f"Std Dev: {df['headline_length'].std():.1f} - eda_analysis.py:56")
print(f"Min: {df['headline_length'].min()} - eda_analysis.py:57")
print(f"Max: {df['headline_length'].max()} - eda_analysis.py:58")

print(f"\nHeadline Word Count: - eda_analysis.py:60")
print(f"Mean: {df['word_count'].mean():.1f} - eda_analysis.py:61")
print(f"Median: {df['word_count'].median():.0f} - eda_analysis.py:62")
print(f"Std Dev: {df['word_count'].std():.1f} - eda_analysis.py:63")
print(f"Min: {df['word_count'].min()} - eda_analysis.py:64")
print(f"Max: {df['word_count'].max()} - eda_analysis.py:65")

# ============================================================================
# 3. PUBLISHER ANALYSIS
# ============================================================================
print("\n - eda_analysis.py:70" + "="*50)
print("3. PUBLISHER ANALYSIS - eda_analysis.py:71")
print("= - eda_analysis.py:72"*50)

publisher_stats = df['publisher'].value_counts()
print(f"\nTotal unique publishers: {df['publisher'].nunique()} - eda_analysis.py:75")
print(f"\nTop 10 most active publishers: - eda_analysis.py:76")
print(publisher_stats.head(10))

# Check for email domains
email_pattern = r'[\w\.-]+@[\w\.-]+\.\w+'
email_publishers = df[df['publisher'].str.contains(email_pattern, na=False, regex=True)]
print(f"\nPublishers with email format: {len(email_publishers)} - eda_analysis.py:82")

# ============================================================================
# 4. TIME SERIES ANALYSIS
# ============================================================================
print("\n - eda_analysis.py:87" + "="*50)
print("4. TIME SERIES ANALYSIS - eda_analysis.py:88")
print("= - eda_analysis.py:89"*50)

df['date_only'] = df['date'].dt.date
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['hour'] = df['date'].dt.hour
df['weekday'] = df['date'].dt.day_name()

daily_counts = df.groupby('date_only').size().reset_index(name='count')

print(f"\nArticles per day: - eda_analysis.py:99")
print(f"Mean: {daily_counts['count'].mean():.1f} - eda_analysis.py:100")
print(f"Median: {daily_counts['count'].median():.0f} - eda_analysis.py:101")
print(f"Max: {daily_counts['count'].max()} - eda_analysis.py:102")
print(f"Min: {daily_counts['count'].min()} - eda_analysis.py:103")

# Identify spikes (95th percentile)
threshold = daily_counts['count'].quantile(0.95)
spikes = daily_counts[daily_counts['count'] > threshold].sort_values('count', ascending=False)

print(f"\nPublication Spikes (days with >95th percentile volume): - eda_analysis.py:109")
for _, row in spikes.head(10).iterrows():
    print(f"{row['date_only']}: {row['count']} articles - eda_analysis.py:111")

# Hourly distribution
print(f"\nPeak publication hours: - eda_analysis.py:114")
hourly_counts = df.groupby('hour').size()
top_hours = hourly_counts.nlargest(5)
for hour, count in top_hours.items():
    print(f"{hour:02d}:00  {count} articles - eda_analysis.py:118")

# Weekday distribution
weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
weekday_counts = df['weekday'].value_counts().reindex(weekday_order)
print(f"\nPublications by weekday: - eda_analysis.py:123")
for day, count in weekday_counts.items():
    print(f"{day}: {count} articles ({count/len(df)*100:.1f}%) - eda_analysis.py:125")

# ============================================================================
# 5. TEXT ANALYSIS - KEYWORD EXTRACTION
# ============================================================================
print("\n - eda_analysis.py:130" + "="*50)
print("5. TEXT ANALYSIS  KEYWORD EXTRACTION - eda_analysis.py:131")
print("= - eda_analysis.py:132"*50)

# Define stopwords
stop_words = set(stopwords.words('english'))
custom_stops = {'said', 'says', 'will', 'could', 'would', 'share', 'shares', 'stock',
                'company', 'companies', 'quarter', 'year', 'day', 'week', 'month', 'today',
                'new', 'also', 'still', 'even', 'may', 'just', 'like', 'get', 'amp',
                'one', 'two', 'three', 'first', 'second', 'third', 'last', 'next'}
stop_words.update(custom_stops)

# Clean headlines
def clean_text(text):
    if pd.isna(text):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return ' '.join([word for word in text.split() if word not in stop_words and len(word) > 2])

df['cleaned_headline'] = df['headline'].apply(clean_text)

# TF-IDF vectorization
tfidf_vectorizer = TfidfVectorizer(max_features=100, ngram_range=(1, 2))
tfidf_matrix = tfidf_vectorizer.fit_transform(df['cleaned_headline'].fillna(''))

# Get top keywords
feature_names = tfidf_vectorizer.get_feature_names_out()
tfidf_scores = tfidf_matrix.sum(axis=0).A1
top_keywords = sorted(zip(feature_names, tfidf_scores), key=lambda x: x[1], reverse=True)

print("\nTop 20 keywords/phrases across all headlines: - eda_analysis.py:161")
for i, (keyword, score) in enumerate(top_keywords[:20], 1):
    print(f"{i:2d}. {keyword}: {score:.4f} - eda_analysis.py:163")

# ============================================================================
# 6. STOCK-SPECIFIC ANALYSIS
# ============================================================================
print("\n - eda_analysis.py:168" + "="*50)
print("6. STOCKSPECIFIC ANALYSIS - eda_analysis.py:169")
print("= - eda_analysis.py:170"*50)

stock_counts = df['stock'].value_counts()
print(f"\nArticle distribution by stock: - eda_analysis.py:173")
for stock, count in stock_counts.head(10).items():
    print(f"{stock}: {count} articles ({count/len(df)*100:.1f}%) - eda_analysis.py:175")

# Top keywords per stock
print("\nTop keywords for top 5 stocks: - eda_analysis.py:178")
for stock in stock_counts.head(5).index:
    stock_df = df[df['stock'] == stock]
    stock_corpus = ' '.join(stock_df['cleaned_headline'].fillna(''))
    
    if len(stock_corpus) > 100:
        vec = TfidfVectorizer(max_features=10, ngram_range=(1, 2))
        vec.fit([stock_corpus])
        keywords = vec.get_feature_names_out()
        print(f"\n  {stock}: - eda_analysis.py:187")
        for kw in keywords[:8]:
            print(f"{kw} - eda_analysis.py:189")

# ============================================================================
# 7. TOPIC MODELING
# ============================================================================
print("\n - eda_analysis.py:194" + "="*50)
print("7. TOPIC MODELING (LDA) - eda_analysis.py:195")
print("= - eda_analysis.py:196"*50)

# Prepare corpus for LDA
vectorizer = CountVectorizer(max_features=500, stop_words=list(stop_words))
dtm = vectorizer.fit_transform(df['cleaned_headline'].fillna(''))

# Apply LDA
lda = LatentDirichletAllocation(n_components=5, random_state=42)
lda.fit(dtm)

# Display topics
feature_names = vectorizer.get_feature_names_out()

def display_topic(lda, feature_names, topic_idx, n_words=10):
    top_words_idx = lda.components_[topic_idx].argsort()[:-n_words-1:-1]
    return [feature_names[i] for i in top_words_idx]

print("\nExtracted topics from headlines: - eda_analysis.py:213")
for i in range(5):
    topic_words = display_topic(lda, feature_names, i, 10)
    print(f"\n  Topic {i+1}: {', '.join(topic_words)} - eda_analysis.py:216")

# ============================================================================
# 8. VISUALIZATIONS
# ============================================================================
print("\n - eda_analysis.py:221" + "="*50)
print("8. GENERATING VISUALIZATIONS - eda_analysis.py:222")
print("= - eda_analysis.py:223"*50)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Headline length distribution
axes[0,0].hist(df['headline_length'], bins=50, edgecolor='black', alpha=0.7)
axes[0,0].axvline(df['headline_length'].mean(), color='red', linestyle='--', 
                  label=f'Mean: {df["headline_length"].mean():.0f}')
axes[0,0].axvline(df['headline_length'].median(), color='green', linestyle='--', 
                  label=f'Median: {df["headline_length"].median():.0f}')
axes[0,0].set_xlabel('Character Count')
axes[0,0].set_ylabel('Frequency')
axes[0,0].set_title('Distribution of Headline Length')
axes[0,0].legend()

# Plot 2: Top publishers
top_publishers = publisher_stats.head(15)
axes[0,1].barh(range(len(top_publishers)), top_publishers.values, color='skyblue')
axes[0,1].set_yticks(range(len(top_publishers)))
axes[0,1].set_yticklabels([name[:20] + '...' if len(str(name)) > 20 else name 
                           for name in top_publishers.index])
axes[0,1].set_xlabel('Number of Articles')
axes[0,1].set_title('Top 15 Most Active Publishers')
axes[0,1].invert_yaxis()

# Plot 3: Daily publication volume
axes[1,0].plot(daily_counts['date_only'], daily_counts['count'], alpha=0.7, linewidth=0.8)
axes[1,0].set_xlabel('Date')
axes[1,0].set_ylabel('Number of Articles')
axes[1,0].set_title('Daily Article Publication Volume')
axes[1,0].tick_params(axis='x', rotation=45)

# Plot 4: Hourly distribution
hourly_counts = df.groupby('hour').size()
axes[1,1].bar(hourly_counts.index, hourly_counts.values, color='coral')
axes[1,1].set_xlabel('Hour of Day')
axes[1,1].set_ylabel('Number of Articles')
axes[1,1].set_title('Publication Times by Hour')

plt.tight_layout()
plt.savefig('eda_visualizations_summary.png', dpi=150, bbox_inches='tight')
print("\nSaved: eda_visualizations_summary.png - eda_analysis.py:264")

# Additional figure: weekday vs stock
fig2, ax = plt.subplots(figsize=(12, 6))

# Pivot table for top stocks by weekday
top_stocks = stock_counts.head(5).index
df_top = df[df['stock'].isin(top_stocks)]
weekday_stock = pd.crosstab(df_top['weekday'], df_top['stock'], normalize='columns')
weekday_stock = weekday_stock.reindex(weekday_order)

weekday_stock.plot(kind='bar', ax=ax, stacked=True, width=0.8)
ax.set_xlabel('Weekday')
ax.set_ylabel('Proportion of Articles')
ax.set_title('Stock Coverage Distribution by Weekday')
ax.legend(title='Stock', bbox_to_anchor=(1.05, 1), loc='upper left')
ax.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('stock_weekday_distribution.png', dpi=150, bbox_inches='tight')
print("Saved: stock_weekday_distribution.png - eda_analysis.py:284")

print("\n - eda_analysis.py:286" + "="*70)
print("EDA COMPLETED SUCCESSFULLY! - eda_analysis.py:287")
print("= - eda_analysis.py:288"*70)
print("\nKey Findings Summary: - eda_analysis.py:289")
print(f"Dataset contains {len(df)} news articles across {df['stock'].nunique()} stocks - eda_analysis.py:290")
print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()} - eda_analysis.py:291")
print(f"Most active publisher: {publisher_stats.index[0]} ({publisher_stats.iloc[0]} articles) - eda_analysis.py:292")
print(f"Peak publication hour: {hourly_counts.idxmax()}:00 ({hourly_counts.max()} articles) - eda_analysis.py:293")
print(f"Most covered stock: {stock_counts.index[0]} ({stock_counts.iloc[0]} articles) - eda_analysis.py:294")
print(f"Average headline length: {df['headline_length'].mean():.0f} characters - eda_analysis.py:295")

