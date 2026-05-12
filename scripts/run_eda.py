# scripts/run_eda.py
# This script performs the complete EDA analysis
# Place the Excel file in data/raw/ before running
import os
import sys
# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
def main():
print("Starting EDA Analysis...")
print("Make sure raw_analyst_ratings.xlsx is in data/raw/ directory")
# The full code from above goes here
if __name__ == "__main__":
    main()
