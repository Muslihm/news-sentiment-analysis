#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
News Sentiment Analysis - Task 1: Exploratory Data Analysis
Run this script to perform complete EDA on the financial news dataset.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_data(file_path):
    """Load the Excel data file"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")
    
    df = pd.read_excel(file_path, sheet_name='raw_analyst_ratings1')
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    return df

def clean_text(text):
    """Clean and preprocess headline text"""
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

def display_topic(lda, feature_names, topic_idx, n_words=10):
    """Extract top words from LDA topic"""
    top_words_idx = lda.components_[topic_idx].argsort()[:-n_words-1:-1]
    return [feature_names[i] for i in top_words_idx]

# ============================================================================
# MAIN ANALYSIS
# ============================================================================

def main():
    print("= - run_eda.py:53" * 70)
    print("FINANCIAL NEWS SENTIMENT ANALYSIS  TASK 1: EXPLORATORY DATA ANALYSIS - run_eda.py:54")
    print("= - run_eda.py:55" * 70)
    
    # Load data
    print("\n[1/7] Loading data... - run_eda.py:58")
    file_path = 'data/raw/raw_analyst_ratings.xlsx'
    
    if not os.path.exists(file_path):
        print(f"\nERROR: Could not find {file_path} - run_eda.py:62")
        print("Please place the raw_analyst_ratings.xlsx file in data/raw/ directory - run_eda.py:63")
        return
    
    df = load_data(file_path)
    print(f"✓ Loaded {len(df)} rows, {df['stock'].nunique()} unique stocks - run_eda.py:67")
    print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()} - run_eda.py:68")
    
    # ========================================================================
    # 1. Descriptive Statistics
    # ========================================================================
    print("\n[2/7] Calculating descriptive statistics... - run_eda.py:73")
    
    df['headline_length'] = df['headline'].fillna('').astype(str).str.len()
    df['word_count'] = df['headline'].fillna('').astype(str).str.split().str.len()
    
    print("\n  Headline Statistics: - run_eda.py:78")
    print(f"Character count  Mean: {df['headline_length'].mean():.1f}, - run_eda.py:79"
          f"Median: {df['headline_length'].median():.0f}, "
          f"Std: {df['headline_length'].std():.1f}")
    print(f"Word count  Mean: {df['word_count'].mean():.1f}, - run_eda.py:82"
          f"Median: {df['word_count'].median():.0f}")
    
    # ========================================================================
    # 2. Publisher Analysis
    # ========================================================================
    print("\n[3/7] Analyzing publishers... - run_eda.py:88")
    
    publisher_stats = df['publisher'].value_counts()
    print(f"Total unique publishers: {df['publisher'].nunique()} - run_eda.py:91")
    print(f"\n  Top 10 most active publishers: - run_eda.py:92")
    for i, (pub, count) in enumerate(publisher_stats.head(10).items(), 1):
        pub_name = str(pub)[:40] + '...' if len(str(pub)) > 40 else str(pub)
        print(f"{i:2d}. {pub_name}: {count} articles - run_eda.py:95")
    
    # ========================================================================
    # 3. Time Series Analysis
    # ========================================================================
    print("\n[4/7] Analyzing time series patterns... - run_eda.py:100")
    
    df['hour'] = df['date'].dt.hour
    df['weekday'] = df['date'].dt.day_name()
    df['date_only'] = df['date'].dt.date
    
    daily_counts = df.groupby('date_only').size()
    
    print(f"\n  Daily publication volume: - run_eda.py:108")
    print(f"Mean: {daily_counts.mean():.1f} articles/day - run_eda.py:109")
    print(f"Max: {daily_counts.max()} articles on {daily_counts.idxmax()} - run_eda.py:110")
    print(f"Min: {daily_counts.min()} articles - run_eda.py:111")
    
    # Peak hours
    hourly_counts = df.groupby('hour').size()
    peak_hour = hourly_counts.idxmax()
    print(f"\n  Peak publication hour: {peak_hour}:00 ({hourly_counts.max()} articles) - run_eda.py:116")
    
    # Weekday distribution
    weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    weekday_counts = df['weekday'].value_counts().reindex(weekday_order)
    busiest_day = weekday_counts.idxmax()
    print(f"\n  Busiest day: {busiest_day} ({weekday_counts.max()} articles, - run_eda.py:122"
          f"{weekday_counts.max()/len(df)*100:.1f}% of total)")
    
    # ========================================================================
    # 4. Stock Analysis
    # ========================================================================
    print("\n[5/7] Analyzing stock coverage... - run_eda.py:128")
    
    stock_counts = df['stock'].value_counts()
    print(f"\n  Most covered stocks: - run_eda.py:131")
    for i, (stock, count) in enumerate(stock_counts.head(10).items(), 1):
        print(f"{i:2d}. {stock}: {count} articles ({count/len(df)*100:.1f}%) - run_eda.py:133")
    
    # ========================================================================
    # 5. Text Analysis - Keyword Extraction
    # ========================================================================
    print("\n[6/7] Extracting keywords from headlines... - run_eda.py:138")
    
    # Import NLP libraries (lazy import)
    from sklearn.feature_extraction.text import TfidfVectorizer
    from nltk.corpus import stopwords
    import nltk
    nltk.download('stopwords', quiet=True)
    
    # Preprocess
    stop_words = set(stopwords.words('english'))
    custom_stops = {'said', 'says', 'will', 'could', 'would', 'share', 'shares', 
                    'stock', 'company', 'quarter', 'year', 'day', 'week', 'month',
                    'today', 'new', 'also', 'still', 'even', 'may', 'just', 'get',
                    'amp', 'like', 'one', 'two', 'three', 'first', 'second'}
    stop_words.update(custom_stops)
    
    def clean_for_keywords(text):
        if pd.isna(text):
            return ""
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        return ' '.join([w for w in text.split() if w not in stop_words and len(w) > 2])
    
    df['cleaned'] = df['headline'].apply(clean_for_keywords)
    
    # TF-IDF vectorization
    tfidf = TfidfVectorizer(max_features=30, ngram_range=(1, 2))
    tfidf_matrix = tfidf.fit_transform(df['cleaned'].fillna(''))
    
    feature_names = tfidf.get_feature_names_out()
    tfidf_scores = tfidf_matrix.sum(axis=0).A1
    top_keywords = sorted(zip(feature_names, tfidf_scores), key=lambda x: x[1], reverse=True)
    
    print("\n  Top 20 keywords/phrases across all headlines: - run_eda.py:171")
    for i, (keyword, score) in enumerate(top_keywords[:20], 1):
        print(f"{i:2d}. {keyword}: {score:.4f} - run_eda.py:173")
    
    # Keywords per top stock
    print("\n  Top keywords for top 5 stocks: - run_eda.py:176")
    for stock in stock_counts.head(5).index:
        stock_text = ' '.join(df[df['stock'] == stock]['cleaned'].fillna(''))
        if len(stock_text) > 100:
            vec = TfidfVectorizer(max_features=8, ngram_range=(1, 2))
            vec.fit([stock_text])
            keywords = vec.get_feature_names_out()
            print(f"{stock}: {', '.join(keywords[:6])} - run_eda.py:183")
    
    # ========================================================================
    # 6. Visualizations
    # ========================================================================
    print("\n[7/7] Generating visualizations... - run_eda.py:188")
    
    # Create output directory for images
    os.makedirs('outputs', exist_ok=True)
    
    # Figure 1: Headline length distribution
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    axes[0,0].hist(df['headline_length'], bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    axes[0,0].axvline(df['headline_length'].mean(), color='red', linestyle='--', 
                      label=f'Mean: {df["headline_length"].mean():.0f}')
    axes[0,0].axvline(df['headline_length'].median(), color='green', linestyle='--', 
                      label=f'Median: {df["headline_length"].median():.0f}')
    axes[0,0].set_xlabel('Character Count')
    axes[0,0].set_ylabel('Frequency')
    axes[0,0].set_title('Distribution of Headline Length')
    axes[0,0].legend()
    
    # Figure 2: Top publishers
    top_publishers = publisher_stats.head(12)
    axes[0,1].barh(range(len(top_publishers)), top_publishers.values, color='coral')
    axes[0,1].set_yticks(range(len(top_publishers)))
    pub_labels = [str(p)[:25] + '...' if len(str(p)) > 25 else str(p) 
                  for p in top_publishers.index]
    axes[0,1].set_yticklabels(pub_labels)
    axes[0,1].set_xlabel('Number of Articles')
    axes[0,1].set_title('Top 12 Most Active Publishers')
    axes[0,1].invert_yaxis()
    
    # Figure 3: Daily publication volume
    axes[1,0].plot(daily_counts.index, daily_counts.values, alpha=0.7, linewidth=0.8, color='steelblue')
    axes[1,0].set_xlabel('Date')
    axes[1,0].set_ylabel('Number of Articles')
    axes[1,0].set_title('Daily Article Publication Volume')
    axes[1,0].tick_params(axis='x', rotation=45)
    
    # Figure 4: Hourly distribution
    axes[1,1].bar(hourly_counts.index, hourly_counts.values, color='seagreen')
    axes[1,1].set_xlabel('Hour of Day (24h format)')
    axes[1,1].set_ylabel('Number of Articles')
    axes[1,1].set_title('Publication Times by Hour')
    axes[1,1].axvline(peak_hour, color='red', linestyle='--', alpha=0.5, label=f'Peak: {peak_hour}:00')
    axes[1,1].legend()
    
    plt.tight_layout()
    plt.savefig('outputs/eda_summary.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: outputs/eda_summary.png - run_eda.py:234")
    
    # Figure 5: Stock distribution pie chart
    fig2, ax = plt.subplots(figsize=(10, 8))
    top_8_stocks = stock_counts.head(8)
    other_count = stock_counts[8:].sum()
    
    pie_data = pd.concat([top_8_stocks, pd.Series({'Other': other_count})])
    colors = sns.color_palette('Set3', len(pie_data))
    
    wedges, texts, autotexts = ax.pie(pie_data.values, labels=pie_data.index, 
                                       autopct='%1.1f%%', colors=colors, startangle=90)
    ax.set_title('Article Distribution by Stock')
    
    plt.tight_layout()
    plt.savefig('outputs/stock_distribution.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: outputs/stock_distribution.png - run_eda.py:250")
    
    # Figure 6: Weekday vs Stock heatmap
    fig3, ax = plt.subplots(figsize=(12, 6))
    top_stocks = stock_counts.head(5).index
    df_top = df[df['stock'].isin(top_stocks)]
    
    weekday_stock = pd.crosstab(df_top['weekday'], df_top['stock'], normalize='index')
    weekday_stock = weekday_stock.reindex(weekday_order)
    
    weekday_stock.plot(kind='bar', ax=ax, width=0.8, colormap='Set2')
    ax.set_xlabel('Weekday')
    ax.set_ylabel('Proportion of Articles')
    ax.set_title('Stock Coverage Distribution by Weekday')
    ax.legend(title='Stock', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('outputs/stock_weekday_distribution.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: outputs/stock_weekday_distribution.png - run_eda.py:269")
    
    # ========================================================================
    # 7. Summary Report
    # ========================================================================
    print("\n - run_eda.py:274" + "=" * 70)
    print("EDA COMPLETED SUCCESSFULLY! - run_eda.py:275")
    print("= - run_eda.py:276" * 70)
    print("\nKEY FINDINGS: - run_eda.py:277")
    print(f"• Dataset contains {len(df):,} news articles across {df['stock'].nunique()} stocks - run_eda.py:278")
    print(f"• Date range: {df['date'].min().date()} to {df['date'].max().date()} - run_eda.py:279")
    print(f"• Most active publisher: {publisher_stats.index[0]} - run_eda.py:280")
    print(f"• Peak publication hour: {peak_hour}:00 ({hourly_counts.max()} articles) - run_eda.py:281")
    print(f"• Busiest weekday: {busiest_day} - run_eda.py:282")
    print(f"• Most covered stock: {stock_counts.index[0]} ({stock_counts.iloc[0]} articles) - run_eda.py:283")
    print(f"• Average headline length: {df['headline_length'].mean():.0f} characters - run_eda.py:284")
    print(f"\nVisualizations saved in 'outputs/' directory - run_eda.py:285")
    print("\n - run_eda.py:286" + "=" * 70)

if __name__ == "__main__":
    main()