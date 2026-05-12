import pytest
import pandas as pd
import os

def test_data_file_exists():
    """Test that the data file exists"""
    assert os.path.exists('data/raw/raw_analyst_ratings.xlsx'), \
        "Data file not found. Please place raw_analyst_ratings.xlsx in data/raw/"
def test_data_loading():
    """Test data loading functionality"""
    df = pd.read_excel('data/raw/raw_analyst_ratings.xlsx', sheet_name='raw_analyst_ratings1')
    assert df is not None
    assert len(df) > 0
    assert 'headline' in df.columns
    assert 'date' in df.columns
    assert 'stock' in df.columns
def test_date_conversion():
    """Test date column conversion"""
    df = pd.read_excel('data/raw/raw_analyst_ratings.xlsx', sheet_name='raw_analyst_ratings1')
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    assert df['date'].notna().any()
