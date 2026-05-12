"""Test script to verify all dependencies are installed"""

import sys
print(f"Python version: {sys.version}")

# Test imports
try:
    import pandas as pd
    print(f"✓ pandas {pd.__version__}")
except ImportError as e:
    print(f"✗ pandas failed: {e}")

try:
    import openpyxl
    print(f"✓ openpyxl {openpyxl.__version__}")
except ImportError as e:
    print(f"✗ openpyxl failed: {e}")

try:
    import matplotlib
    print(f"✓ matplotlib {matplotlib.__version__}")
except ImportError as e:
    print(f"✗ matplotlib failed: {e}")

try:
    import sklearn
    print(f"✓ scikit-learn {sklearn.__version__}")
except ImportError as e:
    print(f"✗ scikit-learn failed: {e}")

try:
    import nltk
    print(f"✓ nltk {nltk.__version__}")
except ImportError as e:
    print(f"✗ nltk failed: {e}")

# Test file reading
file_path = 'data/raw/raw_analyst_ratings.xlsx'
try:
    df = pd.read_excel(file_path, sheet_name='raw_analyst_ratings1')
    print(f"\n✓ Successfully loaded {len(df)} rows from Excel file")
    print(f"  Columns: {df.columns.tolist()}")
except FileNotFoundError:
    print(f"\n✗ File not found: {file_path}")
    print("  Please place the Excel file in data/raw/ directory")
except Exception as e:
    print(f"\n✗ Error loading file: {e}")

print("\nSetup check complete!")
