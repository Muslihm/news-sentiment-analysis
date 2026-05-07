"""Simple test to verify setup"""

def test_imports():
    """Test that required packages are installed"""
    try:
        import pandas
        import numpy
        import sklearn
        assert True
    except ImportError:
        assert False
