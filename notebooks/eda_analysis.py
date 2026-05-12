import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
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