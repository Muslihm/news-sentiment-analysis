#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""News Sentiment Analysis - Task 1: Exploratory Data Analysis"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

def check_dependencies():
    """Check if required packages are installed"""
    missing = []
    try:
        import pandas
        print(f"✓ pandas {pandas.__version__} - run_eda_fixed.py:15")
    except ImportError:
        missing.append("pandas")
    
    try:
        import openpyxl
        print(f"✓ openpyxl {openpyxl.__version__} - run_eda_fixed.py:21")
    except ImportError:
        missing.append("openpyxl")
    
    try:
        import matplotlib
        print(f"✓ matplotlib {matplotlib.__version__} - run_eda_fixed.py:27")
    except ImportError:
        missing.append("matplotlib")
    
    try:
        import sklearn
        print(f"✓ scikitlearn {sklearn.__version__} - run_eda_fixed.py:33")
    except ImportError:
        missing.append("scikit-learn")
    
    try:
        import nltk
        print(f"✓ nltk {nltk.__version__} - run_eda_fixed.py:39")
    except ImportError:
        missing.append("nltk")
    
    if missing:
        print(f"\n❌ Missing packages: {', '.join(missing)} - run_eda_fixed.py:44")
        print("\nInstall with: - run_eda_fixed.py:45")
        print(f"pip install {' '.join(missing)} - run_eda_fixed.py:46")
        return False
    return True

def main():
    print("= - run_eda_fixed.py:51" * 70)
    print("CHECKING DEPENDENCIES - run_eda_fixed.py:52")
    print("= - run_eda_fixed.py:53" * 70)
    
    if not check_dependencies():
        sys.exit(1)
    
    # Now import everything
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import re
    
    from sklearn.feature_extraction.text import TfidfVectorizer
    from nltk.corpus import stopwords
    import nltk
    
    # Download NLTK data
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
    
    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
    
    print("\n - run_eda_fixed.py:77" + "=" * 70)
    print("FINANCIAL NEWS SENTIMENT ANALYSIS  TASK 1: EDA - run_eda_fixed.py:78")
    print("= - run_eda_fixed.py:79" * 70)
    
    # Load data
    file_path = 'data/raw/raw_analyst_ratings.xlsx'
    
    if not os.path.exists(file_path):
        print(f"\n❌ ERROR: Could not find {file_path} - run_eda_fixed.py:85")
        print("Please place the raw_analyst_ratings.xlsx file in data/raw/ directory - run_eda_fixed.py:86")
        return
    
    print("\n[1/5] Loading data... - run_eda_fixed.py:89")
    df = pd.read_excel(file_path, sheet_name='raw_analyst_ratings1')
    
    # Fix date parsing - handle mixed timezones by using utc=True
    print("Parsing dates (this may take a moment)... - run_eda_fixed.py:93")
    try:
        # Try with utc=True to handle mixed timezones
        df['date'] = pd.to_datetime(df['date'], utc=True, errors='coerce')
    except Exception as e:
        print(f"First attempt failed: {e} - run_eda_fixed.py:98")
        # Fallback: convert to string first, then parse
        df['date_str'] = df['date'].astype(str)
        df['date'] = pd.to_datetime(df['date_str'], errors='coerce')
        df.drop('date_str', axis=1, inplace=True)
    
    # Remove rows with invalid dates
    initial_rows = len(df)
    df = df.dropna(subset=['date'])
    print(f"Removed {initial_rows  len(df)} rows with invalid dates - run_eda_fixed.py:107")
    
    print(f"✓ Loaded {len(df):,} rows - run_eda_fixed.py:109")
    print(f"Date range: {df['date'].min()} to {df['date'].max()} - run_eda_fixed.py:110")
    print(f"Unique stocks: {df['stock'].nunique()} - run_eda_fixed.py:111")
    
    # Basic statistics
    print("\n[2/5] Calculating statistics... - run_eda_fixed.py:114")
    
    df['headline_length'] = df['headline'].fillna('').astype(str).str.len()
    df['word_count'] = df['headline'].fillna('').astype(str).str.split().str.len()
    
    print(f"\n  Headline Statistics: - run_eda_fixed.py:119")
    print(f"Average length: {df['headline_length'].mean():.0f} characters - run_eda_fixed.py:120")
    print(f"Average words: {df['word_count'].mean():.1f} - run_eda_fixed.py:121")
    print(f"Max length: {df['headline_length'].max()} characters - run_eda_fixed.py:122")
    print(f"Min length: {df['headline_length'].min()} characters - run_eda_fixed.py:123")
    
    # Publisher analysis
    print("\n[3/5] Analyzing publishers... - run_eda_fixed.py:126")
    publisher_stats = df['publisher'].value_counts()
    print(f"Total publishers: {df['publisher'].nunique()} - run_eda_fixed.py:128")
    print(f"\n  Top 10 publishers: - run_eda_fixed.py:129")
    for i, (pub, count) in enumerate(publisher_stats.head(10).items(), 1):
        pub_name = str(pub)[:60] + '...' if len(str(pub)) > 60 else str(pub)
        print(f"{i:2d}. {pub_name}: {count} articles - run_eda_fixed.py:132")
    
    # Time analysis
    print("\n[4/5] Analyzing time patterns... - run_eda_fixed.py:135")
    
    # Extract time components (use .dt accessor)
    df['hour'] = df['date'].dt.hour
    df['weekday'] = df['date'].dt.day_name()
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['date_only'] = df['date'].dt.date
    
    hourly = df.groupby('hour').size()
    peak_hour = hourly.idxmax()
    print(f"Peak publication hour: {peak_hour}:00 ({hourly.max()} articles) - run_eda_fixed.py:146")
    
    weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    weekday_counts = df['weekday'].value_counts().reindex(weekday_order)
    busiest_day = weekday_counts.idxmax()
    print(f"Busiest day: {busiest_day} ({weekday_counts.max():,} articles, {weekday_counts.max()/len(df)*100:.1f}%) - run_eda_fixed.py:151")
    
    # Yearly trend
    yearly_counts = df['year'].value_counts().sort_index()
    print(f"\n  Yearly publication trends: - run_eda_fixed.py:155")
    for year, count in yearly_counts.items():
        print(f"{year}: {count:,} articles - run_eda_fixed.py:157")
    
    # Stock analysis
    print("\n[5/5] Analyzing stocks... - run_eda_fixed.py:160")
    stock_counts = df['stock'].value_counts()
    print(f"\n  Top 10 stocks by coverage: - run_eda_fixed.py:162")
    for i, (stock, count) in enumerate(stock_counts.head(10).items(), 1):
        print(f"{i:2d}. {stock}: {count:,} articles ({count/len(df)*100:.1f}%) - run_eda_fixed.py:164")
    
    # Generate visualizations
    print("\n - run_eda_fixed.py:167" + "=" * 50)
    print("GENERATING VISUALIZATIONS - run_eda_fixed.py:168")
    print("= - run_eda_fixed.py:169" * 50)
    
    os.makedirs('outputs', exist_ok=True)
    
    # Figure 1: Distribution plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Headline length histogram
    axes[0,0].hist(df['headline_length'], bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    axes[0,0].axvline(df['headline_length'].mean(), color='red', linestyle='--', 
                      label=f'Mean: {df["headline_length"].mean():.0f}')
    axes[0,0].axvline(df['headline_length'].median(), color='green', linestyle='--', 
                      label=f'Median: {df["headline_length"].median():.0f}')
    axes[0,0].set_title('Headline Length Distribution', fontsize=12)
    axes[0,0].set_xlabel('Number of Characters')
    axes[0,0].set_ylabel('Frequency')
    axes[0,0].legend()
    
    # Plot 2: Word count distribution
    axes[0,1].hist(df['word_count'], bins=30, edgecolor='black', alpha=0.7, color='coral')
    axes[0,1].axvline(df['word_count'].mean(), color='red', linestyle='--', 
                      label=f'Mean: {df["word_count"].mean():.1f}')
    axes[0,1].set_title('Word Count Distribution', fontsize=12)
    axes[0,1].set_xlabel('Number of Words')
    axes[0,1].set_ylabel('Frequency')
    axes[0,1].legend()
    
    # Plot 3: Top publishers
    top_publishers = publisher_stats.head(12)
    bars = axes[1,0].barh(range(len(top_publishers)), top_publishers.values, color='lightgreen')
    axes[1,0].set_yticks(range(len(top_publishers)))
    pub_labels = [str(p)[:25] + '...' if len(str(p)) > 25 else str(p) 
                  for p in top_publishers.index]
    axes[1,0].set_yticklabels(pub_labels)
    axes[1,0].set_xlabel('Number of Articles')
    axes[1,0].set_title('Top 12 Most Active Publishers', fontsize=12)
    axes[1,0].invert_yaxis()
    
    # Plot 4: Hourly distribution
    axes[1,1].bar(hourly.index, hourly.values, color='steelblue')
    axes[1,1].set_title('Publications by Hour of Day', fontsize=12)
    axes[1,1].set_xlabel('Hour (24h format)')
    axes[1,1].set_ylabel('Number of Articles')
    axes[1,1].axvline(peak_hour, color='red', linestyle='--', alpha=0.7, 
                      label=f'Peak: {peak_hour}:00')
    axes[1,1].legend()
    
    plt.tight_layout()
    plt.savefig('outputs/eda_summary.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: outputs/eda_summary.png - run_eda_fixed.py:218")
    
    # Figure 2: Time series
    fig2, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Daily publication volume
    daily_counts = df.groupby('date_only').size()
    axes[0,0].plot(daily_counts.index, daily_counts.values, linewidth=0.8, color='steelblue')
    axes[0,0].set_title('Daily Publication Volume', fontsize=12)
    axes[0,0].set_xlabel('Date')
    axes[0,0].set_ylabel('Number of Articles')
    axes[0,0].tick_params(axis='x', rotation=45)
    
    # Weekly distribution
    axes[0,1].bar(weekday_counts.index, weekday_counts.values, color='coral')
    axes[0,1].set_title('Publications by Day of Week', fontsize=12)
    axes[0,1].set_xlabel('Weekday')
    axes[0,1].set_ylabel('Number of Articles')
    axes[0,1].tick_params(axis='x', rotation=45)
    
    # Monthly trend
    monthly_counts = df.groupby(df['date'].dt.to_period('M')).size()
    axes[1,0].plot(range(len(monthly_counts)), monthly_counts.values, marker='o', linewidth=2, color='steelblue')
    axes[1,0].set_title('Monthly Publication Trend', fontsize=12)
    axes[1,0].set_xlabel('Month Sequence')
    axes[1,0].set_ylabel('Number of Articles')
    
    # Yearly comparison
    years = sorted(df['year'].unique())
    yearly_data = []
    for year in years:
        year_df = df[df['year'] == year]
        monthly = year_df.groupby(year_df['date'].dt.month).size()
        yearly_data.append(monthly)
    
    for i, year in enumerate(years):
        axes[1,1].plot(range(1, 13), yearly_data[i], marker='o', label=str(year), linewidth=2)
    axes[1,1].set_title('Monthly Comparison by Year', fontsize=12)
    axes[1,1].set_xlabel('Month')
    axes[1,1].set_ylabel('Number of Articles')
    axes[1,1].legend()
    axes[1,1].set_xticks(range(1, 13))
    
    plt.tight_layout()
    plt.savefig('outputs/time_series_analysis.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: outputs/time_series_analysis.png - run_eda_fixed.py:263")
    
    # Figure 3: Stock distribution
    fig3, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Pie chart
    top_8 = stock_counts.head(8)
    other = pd.Series({'Other': stock_counts[8:].sum()})
    pie_data = pd.concat([top_8, other])
    
    colors = sns.color_palette('Set3', len(pie_data))
    axes[0].pie(pie_data.values, labels=pie_data.index, autopct='%1.1f%%', 
                colors=colors, startangle=90)
    axes[0].set_title('Article Distribution by Stock', fontsize=12)
    
    # Bar chart of top stocks
    top_15 = stock_counts.head(15)
    axes[1].barh(range(len(top_15)), top_15.values, color='steelblue')
    axes[1].set_yticks(range(len(top_15)))
    axes[1].set_yticklabels(top_15.index)
    axes[1].set_xlabel('Number of Articles')
    axes[1].set_title('Top 15 Most Covered Stocks', fontsize=12)
    axes[1].invert_yaxis()
    
    plt.tight_layout()
    plt.savefig('outputs/stock_analysis.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: outputs/stock_analysis.png - run_eda_fixed.py:289")
    
    # Text analysis
    print("\n - run_eda_fixed.py:292" + "=" * 50)
    print("TEXT ANALYSIS - run_eda_fixed.py:293")
    print("= - run_eda_fixed.py:294" * 50)
    
    # Define stopwords
    stop_words = set(stopwords.words('english'))
    custom_stops = {'said', 'says', 'will', 'could', 'would', 'share', 'shares', 
                    'stock', 'company', 'quarter', 'year', 'day', 'week', 'month',
                    'today', 'new', 'also', 'still', 'even', 'may', 'just', 'get',
                    'amp', 'like', 'one', 'two', 'three', 'first', 'second',
                    'inc', 'corp', 'co', 'nyse', 'nasdaq'}
    stop_words.update(custom_stops)
    
    def clean_text(text):
        if pd.isna(text):
            return ""
        text = str(text).lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        return ' '.join([w for w in text.split() if w not in stop_words and len(w) > 2])
    
    df['cleaned'] = df['headline'].apply(clean_text)
    
    # TF-IDF analysis
    tfidf = TfidfVectorizer(max_features=30, ngram_range=(1, 2))
    tfidf_matrix = tfidf.fit_transform(df['cleaned'].fillna(''))
    
    feature_names = tfidf.get_feature_names_out()
    tfidf_scores = tfidf_matrix.sum(axis=0).A1
    top_keywords = sorted(zip(feature_names, tfidf_scores), key=lambda x: x[1], reverse=True)
    
    print("\nTop 20 keywords/phrases across all headlines: - run_eda_fixed.py:322")
    for i, (keyword, score) in enumerate(top_keywords[:20], 1):
        print(f"{i:2d}. {keyword}: {score:.4f} - run_eda_fixed.py:324")
    
    # Keywords per top stock
    print("\nTop keywords for top 5 stocks: - run_eda_fixed.py:327")
    for stock in stock_counts.head(5).index:
        stock_text = ' '.join(df[df['stock'] == stock]['cleaned'].fillna(''))
        if len(stock_text) > 100:
            vec = TfidfVectorizer(max_features=6, ngram_range=(1, 2))
            vec.fit([stock_text])
            keywords = vec.get_feature_names_out()
            print(f"{stock}: {', '.join(keywords)} - run_eda_fixed.py:334")
        else:
            print(f"{stock}: insufficient data - run_eda_fixed.py:336")
    
    print("\n - run_eda_fixed.py:338" + "=" * 70)
    print("✓ EDA COMPLETED SUCCESSFULLY! - run_eda_fixed.py:339")
    print("= - run_eda_fixed.py:340" * 70)
    print("\n - run_eda_fixed.py:341" + "=" * 50)
    print("FINDINGS SUMMARY - run_eda_fixed.py:342")
    print("= - run_eda_fixed.py:343" * 50)
    print(f"• Dataset: {len(df):,} articles, {df['stock'].nunique()} stocks - run_eda_fixed.py:344")
    print(f"• Date range: {df['date'].min().date()} to {df['date'].max().date()} - run_eda_fixed.py:345")
    print(f"• Most active publisher: {publisher_stats.index[0]} - run_eda_fixed.py:346")
    print(f"({publisher_stats.iloc[0]:,} articles) - run_eda_fixed.py:347")
    print(f"• Peak publication hour: {peak_hour}:00 - run_eda_fixed.py:348")
    print(f"• Busiest day: {busiest_day} - run_eda_fixed.py:349")
    print(f"• Top stock: {stock_counts.index[0]} ({stock_counts.iloc[0]:,} articles) - run_eda_fixed.py:350")
    print(f"• Average headline: {df['headline_length'].mean():.0f} characters - run_eda_fixed.py:351")
    print(f"\n📁 Visualizations saved in 'outputs/' directory - run_eda_fixed.py:352")
    print("= - run_eda_fixed.py:353" * 70)

if __name__ == "__main__":
    main()